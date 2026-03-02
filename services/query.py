"""
Query service — filter building, retrieval orchestration, generation.

Full query pipeline:
  1. QueryAnalyzer    — decompose + extract metadata filters in one LLM call
  2. build_filters    — merge LLM-extracted hints + explicit request filters
  3. run_retrieval / retrieve_with_crag — retrieval with optional HyDE and CRAG
  4. swap_to_parent_content — replace child chunks with full parent sections
  5. ColBERT second-pass reranker (optional)
  6. run_generation   — prompt_builder → LLM → answer_builder

Feature flags (controlled via .env):
  HYDE_ENABLED  — hypothetical document embedding for dense retrieval
  CRAG_ENABLED  — corrective re-retrieval on low relevance score
"""

import logging
from dataclasses import dataclass
from datetime import date as date_type, datetime, time, timezone
from typing import Any

from haystack.dataclasses import Document, GeneratedAnswer

from components.query_analyzer import AnalysisResult, QueryAnalyzer
from config import Settings
from models.api import QueryRequest, SourceDocument
from pipelines.retrieval import swap_to_parent_content

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class RetrievalError(Exception):
    """Raised when retrieval fails for a sub-question."""


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------

@dataclass
class QueryContext:
    """Encapsulates the full retrieval result passed to the generation step."""
    merged_docs:    list[Document]
    sub_questions:  list[str]
    is_compound:    bool
    low_confidence: bool
    analysis:       AnalysisResult


# ---------------------------------------------------------------------------
# Filter helpers (private)
# ---------------------------------------------------------------------------

_BARE_FIELDS       = {"id", "content"}
_LOGICAL_OPERATORS = {"AND", "OR", "NOT"}
_SET_OPERATORS     = {"in", "not in"}


def _is_scalar_filter_value(value: Any) -> bool:
    return isinstance(value, (str, int, float, bool))


def _normalize_filter_fields(f: dict) -> dict:
    """Recursively ensure all metadata field names carry the ``meta.`` prefix."""
    operator = f.get("operator")
    if operator in _LOGICAL_OPERATORS:
        return {**f, "conditions": [_normalize_filter_fields(c) for c in f.get("conditions", [])]}
    field = f.get("field", "")
    if field and not field.startswith("meta.") and field not in _BARE_FIELDS:
        return {**f, "field": f"meta.{field}"}
    return f


def _normalize_field_name(field: str) -> str:
    if field and not field.startswith("meta.") and field not in _BARE_FIELDS:
        return f"meta.{field}"
    return field


def _is_valid_filter_expression(f: Any) -> bool:
    if not isinstance(f, dict):
        return False
    operator = f.get("operator")
    if operator in _LOGICAL_OPERATORS:
        conditions = f.get("conditions")
        if not isinstance(conditions, list) or not conditions:
            return False
        return all(_is_valid_filter_expression(c) for c in conditions)
    if "field" in f and "operator" in f and "value" in f:
        field    = f.get("field")
        operator = f.get("operator")
        value    = f.get("value")
        if not isinstance(field, str) or not field.strip() or not isinstance(operator, str):
            return False
        if operator.lower() in _SET_OPERATORS:
            return isinstance(value, list) and bool(value) and all(_is_scalar_filter_value(v) for v in value)
        return _is_scalar_filter_value(value)
    return False


def _coerce_request_filters(raw_filters: dict[str, Any] | None) -> dict | None:
    """
    Accept either:
      1) Native Haystack filter expression, or
      2) Flat shorthand mapping, e.g. {"source": "report.pdf", "language": "de"}.

    Shorthand mappings are converted to equality comparisons combined with AND.
    """
    if not raw_filters:
        return None

    looks_like_expression = any(k in raw_filters for k in ("field", "operator", "conditions", "value"))
    normalized = _normalize_filter_fields(raw_filters)
    if _is_valid_filter_expression(normalized):
        return normalized
    if looks_like_expression:
        raise ValueError(
            "Invalid filters format. Use Haystack syntax "
            "({'field','operator','value'} / logical {'operator','conditions'}) "
            "or shorthand mapping {'source': 'file.pdf'}."
        )

    conditions: list[dict] = []
    for key, value in raw_filters.items():
        if value is None:
            continue
        field = _normalize_field_name(str(key))
        if isinstance(value, dict):
            if not value:
                continue
            raise ValueError(
                f"Invalid shorthand filter value for '{key}': nested objects are not supported."
            )
        if isinstance(value, list):
            if not value:
                continue
            if not all(_is_scalar_filter_value(v) for v in value):
                raise ValueError(
                    f"Invalid shorthand filter value for '{key}': list items must be scalar values."
                )
            conditions.append({"field": field, "operator": "in", "value": value})
        else:
            conditions.append({"field": field, "operator": "==", "value": value})

    if not conditions:
        return None
    if len(conditions) == 1:
        return conditions[0]
    return {"operator": "AND", "conditions": conditions}


async def _reformulate_query(query: str, settings: Settings) -> str:
    """LLM rephrases query to improve retrieval. Falls back to original on error."""
    try:
        from openai import AsyncOpenAI
        client_kwargs: dict[str, Any] = {"api_key": settings.openai_api_key}
        if settings.openai_url:
            client_kwargs["base_url"] = settings.openai_url
        client = AsyncOpenAI(**client_kwargs)
        response = await client.chat.completions.create(
            model    = settings.llm_model,
            messages = [{
                "role": "user",
                "content": (
                    "Rephrase the following question to improve document retrieval. "
                    "Return ONLY the rephrased question, nothing else.\n\n"
                    f"Original: {query}"
                ),
            }],
            temperature = 0.3,
            max_tokens  = 100,
        )
        rephrased = (response.choices[0].message.content or "").strip()
        if rephrased:
            logger.debug("CRAG: reformulated '%s' → '%s'", query, rephrased)
            return rephrased
    except Exception as exc:
        logger.warning("CRAG: query reformulation failed (%s), using original", exc)
    return query


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_filters(
    request: QueryRequest,
    analysis: AnalysisResult | None = None,
) -> dict | None:
    """
    Merge LLM-extracted filter hints (lowest priority) with explicit request
    filters (highest priority) into a single Haystack filter expression.

    Build order (later entries override earlier ones logically):
      1. LLM-extracted date range (analysis.date_from / date_to)
      2. LLM-extracted metadata (classification, language, source filename)
      3. Explicit request.date_from / request.date_to
      4. Explicit request.filters (always takes priority over everything)
    """
    conditions: list[dict] = []

    if analysis:
        if analysis.date_from and not request.date_from:
            try:
                d  = date_type.fromisoformat(analysis.date_from)
                ts = int(datetime.combine(d, time.min, tzinfo=timezone.utc).timestamp())
                conditions.append({"field": "meta.indexed_at_ts", "operator": ">=", "value": ts})
            except ValueError:
                pass
        if analysis.date_to and not request.date_to:
            try:
                d  = date_type.fromisoformat(analysis.date_to)
                ts = int(datetime.combine(d, time.max, tzinfo=timezone.utc).timestamp())
                conditions.append({"field": "meta.indexed_at_ts", "operator": "<=", "value": ts})
            except ValueError:
                pass
        if analysis.classification:
            conditions.append({
                "field": "meta.classification", "operator": "==",
                "value": analysis.classification,
            })
        if analysis.language:
            conditions.append({
                "field": "meta.language", "operator": "==",
                "value": analysis.language,
            })
        if analysis.source:
            conditions.append({
                "field": "meta.source", "operator": "==",
                "value": analysis.source,
            })

    if request.filters:
        explicit = _coerce_request_filters(request.filters)
        if explicit is not None:
            conditions.append(explicit)
    if request.date_from:
        ts = int(datetime.combine(request.date_from, time.min, tzinfo=timezone.utc).timestamp())
        conditions.append({"field": "meta.indexed_at_ts", "operator": ">=", "value": ts})
    if request.date_to:
        ts = int(datetime.combine(request.date_to, time.max, tzinfo=timezone.utc).timestamp())
        conditions.append({"field": "meta.indexed_at_ts", "operator": "<=", "value": ts})

    if not conditions:
        return None
    if len(conditions) == 1:
        return conditions[0]
    return {"operator": "AND", "conditions": conditions}


def analysis_to_filter_dict(analysis: AnalysisResult) -> dict[str, Any] | None:
    hints = {
        "date_from":      analysis.date_from,
        "date_to":        analysis.date_to,
        "classification": analysis.classification,
        "language":       analysis.language,
        "source":         analysis.source,
    }
    non_null = {k: v for k, v in hints.items() if v is not None}
    return non_null or None


async def retrieve_with_crag(
    pipeline,
    sub_q: str,
    filters: dict | None,
    settings: Settings,
    hyde_generator,
    use_hyde: bool,
    attempt: int = 0,
) -> tuple[list, bool]:
    """
    Retrieval with CRAG retry loop.
    Returns (reranked_docs, low_confidence).
    """
    docs      = await run_retrieval(pipeline, sub_q, filters, hyde_generator, use_hyde)
    top_score = (getattr(docs[0], "score", None) or 0.0) if docs else 0.0
    sufficient = top_score >= settings.crag_score_threshold

    if sufficient or attempt >= settings.crag_max_retries:
        if not sufficient:
            logger.info(
                "CRAG: low confidence (score=%.3f, threshold=%.3f, attempts=%d)",
                top_score, settings.crag_score_threshold, attempt + 1,
            )
        return docs, not sufficient

    reformulated = await _reformulate_query(sub_q, settings)
    return await retrieve_with_crag(
        pipeline, reformulated, filters, settings, hyde_generator, use_hyde, attempt + 1,
    )


_HIDDEN_META = {"parent_content", "original_content", "doc_beginning"}


def format_source_docs(docs: list[Document]) -> list[SourceDocument]:
    """Convert retrieved Document objects to SourceDocument response models."""
    return [
        SourceDocument(
            content = doc.meta.get("original_content") or doc.content or "",
            score   = getattr(doc, "score", None),
            meta    = {k: v for k, v in doc.meta.items() if k not in _HIDDEN_META},
        )
        for doc in docs
    ]


async def run_retrieval(
    pipeline,
    sub_q: str,
    filters: dict | None,
    hyde_generator,
    use_hyde: bool,
) -> list:
    """Run the hybrid retrieval pipeline asynchronously."""
    if use_hyde and hyde_generator is not None:
        dense_text = await hyde_generator.generate(sub_q)
        logger.info("  HyDE: hypothetical doc generated (%d chars): %s", len(dense_text), dense_text[:300])
    else:
        dense_text = sub_q

    run_input: dict = {
        "dense_embedder":  {"text": dense_text},
        "sparse_embedder": {"text": sub_q},
        "reranker":        {"query": sub_q},
    }
    if filters:
        run_input["dense_retriever"]  = {"filters": filters}
        run_input["sparse_retriever"] = {"filters": filters}
    if "colbert_reranker" in pipeline.graph.nodes:
        run_input["colbert_reranker"] = {"query": sub_q}

    result    = await pipeline.run_async(run_input)
    docs      = (result.get("colbert_reranker") or result.get("reranker", {})).get("documents", [])
    top_score = getattr(docs[0], "score", None) if docs else None
    logger.info(
        "  → %d doc(s) after reranking%s",
        len(docs),
        f" | top_score={top_score:.3f}" if top_score is not None else "",
    )
    return docs


async def run_generation(
    pipeline,
    documents: list[Document],
    questions: list[str],
    query: str,
) -> list[GeneratedAnswer]:
    """Run prompt_builder → llm → answer_builder (async pipeline)."""
    result = await pipeline.run_async(
        {
            "prompt_builder": {"documents": documents, "questions": questions},
            "answer_builder": {"query": query, "documents": documents},
        }
    )
    return result.get("answer_builder", {}).get("answers", [])


async def prepare_context(
    request: QueryRequest,
    settings: Settings,
    pipeline,
    analyzer: QueryAnalyzer,
    hyde_generator,
) -> QueryContext:
    """
    Full retrieval phase: analyze → build filters → retrieve per sub-question
    → deduplicate → parent-content swap → top_k cut.
    ColBERT second-pass reranking (if enabled) runs inside the pipeline.

    Raises:
        ValueError       — invalid filter expression in request.filters
        RetrievalError   — retrieval failure for a sub-question
    """
    # 1. Analyze: decompose + extract metadata filters
    analysis      = await analyzer.analyze(request.query)
    sub_questions = analysis.sub_questions
    is_compound   = analysis.is_compound
    logger.info(
        "Analyze  → compound=%s | sub_questions=%d: %s",
        is_compound, len(sub_questions), sub_questions,
    )

    # 2. Build merged filter set (raises ValueError on invalid filters)
    filters  = build_filters(request, analysis)
    use_hyde = settings.hyde_enabled or request.use_hyde
    logger.info("Filters  → %s | HyDE=%s", filters or "none", use_hyde)

    # 3. Retrieve per sub-question, deduplicate by doc.id
    all_docs_by_id: dict[str, Any] = {}
    low_confidence = False

    for i, sub_q in enumerate(sub_questions):
        logger.info(
            "Retrieve sub_q[%d/%d]: %r%s",
            i + 1, len(sub_questions), sub_q,
            " (CRAG)" if settings.crag_enabled else "",
        )
        try:
            if settings.crag_enabled:
                docs, lc = await retrieve_with_crag(
                    pipeline, sub_q, filters, settings, hyde_generator, use_hyde,
                )
                low_confidence = low_confidence or lc
            else:
                docs = await run_retrieval(pipeline, sub_q, filters, hyde_generator, use_hyde)
        except Exception as exc:
            raise RetrievalError(f"Retrieval failed for '{sub_q}': {exc}") from exc

        for doc in docs:
            if doc.id not in all_docs_by_id:
                all_docs_by_id[doc.id] = doc

    # 4. Parent-content swap; budget keeps evidence per sub-question
    candidate_budget = request.top_k * max(1, len(sub_questions))
    before_swap      = len(all_docs_by_id)
    merged_docs      = swap_to_parent_content(list(all_docs_by_id.values()))
    merged_docs      = merged_docs[:candidate_budget]
    logger.info(
        "Parent   → %d merged → %d unique sections (budget=%d)",
        before_swap, len(merged_docs), candidate_budget,
    )

    merged_docs = merged_docs[:request.top_k]

    return QueryContext(
        merged_docs    = merged_docs,
        sub_questions  = sub_questions,
        is_compound    = is_compound,
        low_confidence = low_confidence,
        analysis       = analysis,
    )
