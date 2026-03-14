"""
Query service — filter building, retrieval orchestration, generation.

Full query pipeline:
  1. QueryAnalyzer    — decompose + extract metadata filters in one LLM call
  2. build_filters    — merge LLM-extracted hints + explicit request filters
  3. run_retrieval / retrieve_with_crag — retrieval with optional HyDE and CRAG
     Pipeline order: RRF joiner → AutoMergingRetriever → ColBERT (optional) → cross-encoder
  4. run_generation   — prompt_builder → LLM → answer_builder

Feature flags (controlled via .env):
  HYDE_ENABLED  — hypothetical document embedding for dense retrieval
  CRAG_ENABLED  — corrective re-retrieval on low relevance score
"""

import logging
from collections import OrderedDict
from dataclasses import dataclass
from datetime import date as date_type
from datetime import datetime, time, timezone
from math import ceil
from typing import Any
from urllib.parse import quote

from haystack.dataclasses import Document, GeneratedAnswer

from components.query_analyzer import AnalysisResult, QueryAnalyzer
from config import Settings
from models.api import QueryRequest, SourceDocument

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
    source_docs:    list[Document]
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
    if not isinstance(f, dict):
        return f
    operator = f.get("operator")
    if operator in _LOGICAL_OPERATORS:
        conditions = f.get("conditions", [])
        if not isinstance(conditions, list):
            return f
        return {**f, "conditions": [_normalize_filter_fields(c) for c in conditions]}
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
            model    = settings.effective_instruct_model,
            messages = [{
                "role": "user",
                "content": (
                    "Rephrase the following question to improve document retrieval. "
                    "Return ONLY the rephrased question, nothing else.\n\n"
                    f"Original: {query}"
                ),
            }],
            temperature = 0.3,
            max_tokens  = 150,
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
      1. LLM-extracted semantic date range (document date / covered period)
      2. LLM-extracted metadata (classification, document_type, language, source filename)
      3. Explicit request.date_from / request.date_to
      4. Explicit request.filters (always takes priority over everything)
    """
    conditions: list[dict] = []
    inferred_date_from: date_type | None = None
    inferred_date_to: date_type | None = None

    if analysis:
        if analysis.date_from and not request.date_from:
            try:
                inferred_date_from = date_type.fromisoformat(analysis.date_from)
            except ValueError:
                pass
        if analysis.date_to and not request.date_to:
            try:
                inferred_date_to = date_type.fromisoformat(analysis.date_to)
            except ValueError:
                pass
        # Skip LLM-extracted metadata hints when explicit filters are provided —
        # they would AND-conflict and the explicit filters take priority.
        if not request.filters:
            if analysis.classification:
                conditions.append({
                    "field": "meta.classification", "operator": "==",
                    "value": analysis.classification,
                })
            if analysis.document_type:
                conditions.append({
                    "field": "meta.document_type", "operator": "==",
                    "value": analysis.document_type,
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

    date_filter = _build_semantic_date_filter(
        request.date_from or inferred_date_from,
        request.date_to or inferred_date_to,
    )
    if date_filter is not None:
        conditions.append(date_filter)

    if request.filters:
        explicit = _coerce_request_filters(request.filters)
        if explicit is not None:
            conditions.append(explicit)

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
        "document_type":  analysis.document_type,
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
    attempt: int = 0,
) -> tuple[list, list, bool]:
    """
    Retrieval with CRAG retry loop.
    Returns (generation_docs, source_docs, low_confidence).
    """
    docs, source_docs = await run_retrieval(pipeline, sub_q, filters)
    top_score = (getattr(docs[0], "score", None) or 0.0) if docs else 0.0
    sufficient = top_score >= settings.crag_score_threshold

    if sufficient or attempt >= settings.crag_max_retries:
        if not sufficient:
            logger.info(
                "CRAG: low confidence (score=%.3f, threshold=%.3f, attempts=%d)",
                top_score, settings.crag_score_threshold, attempt + 1,
            )
        return docs, source_docs, not sufficient

    reformulated = await _reformulate_query(sub_q, settings)
    return await retrieve_with_crag(
        pipeline, reformulated, filters, settings, attempt + 1,
    )


_HIDDEN_META = {"original_content", "doc_beginning"}


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


def attach_download_urls(
    sources: list[SourceDocument],
    api_base_url: str,
) -> list[SourceDocument]:
    """Attach stable backend download URLs based on the stored MinIO object key."""
    base_url = api_base_url.rstrip("/")

    enriched: list[SourceDocument] = []
    for source in sources:
        meta = dict(source.meta)
        minio_key = meta.get("minio_key")

        if minio_key:
            meta["download_url"] = f"{base_url}/documents/download/{quote(str(minio_key), safe='')}"

        enriched.append(source.model_copy(update={"meta": meta}))

    return enriched


async def run_retrieval(
    pipeline,
    sub_q: str,
    filters: dict | None,
) -> tuple[list, list]:
    """Run the hybrid retrieval pipeline asynchronously."""
    run_input: dict = {
        "dense_embedder":  {"text": sub_q},
        "sparse_embedder": {"text": sub_q},
        "reranker":        {"query": sub_q},
    }

    if "hyde_generator" in pipeline.graph.nodes:
        run_input["hyde_generator"] = {"query": sub_q}

    if filters:
        run_input["dense_retriever"]  = {"filters": filters}
        run_input["sparse_retriever"] = {"filters": filters}
        if "dense_retriever_hyde" in pipeline.graph.nodes:
            run_input["dense_retriever_hyde"] = {"filters": filters}
    if "colbert_reranker" in pipeline.graph.nodes:
        run_input["colbert_reranker"] = {"query": sub_q}

    result      = await pipeline.run_async(run_input, include_outputs_from={"joiner"})
    docs        = result.get("reranker", {}).get("documents", [])
    source_docs = result.get("joiner", {}).get("documents", [])
    top_score = getattr(docs[0], "score", None) if docs else None
    logger.info(
        "  → %d doc(s) after reranking%s | %d source doc(s)",
        len(docs),
        f" | top_score={top_score:.3f}" if top_score is not None else "",
        len(source_docs),
    )
    return docs, source_docs


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
) -> QueryContext:
    """
    Full retrieval phase: analyze → build filters → retrieve per sub-question
    → score-aware deduplicate → context budget cut.
    Parent-content swap is handled inside the pipeline by AutoMergingRetriever.
    HyDE, ColBERT, and CRAG are controlled via .env flags.

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
    has_hyde = "hyde_generator" in pipeline.graph.nodes
    logger.info("Filters  → %s | HyDE=%s", filters or "none", has_hyde)

    # 3. Retrieve per sub-question and merge with score-aware deduplication
    sub_question_results: list[list[Any]] = []
    source_results: list[list[Any]] = []
    low_confidence = False

    for i, sub_q in enumerate(sub_questions):
        logger.info(
            "Retrieve sub_q[%d/%d]: %r%s",
            i + 1, len(sub_questions), sub_q,
            " (CRAG)" if settings.crag_enabled else "",
        )
        try:
            if settings.crag_enabled:
                docs, source_docs, lc = await retrieve_with_crag(
                    pipeline, sub_q, filters, settings,
                )
                low_confidence = low_confidence or lc
            else:
                docs, source_docs = await run_retrieval(pipeline, sub_q, filters)
        except Exception as exc:
            raise RetrievalError(f"Retrieval failed for '{sub_q}': {exc}") from exc

        sub_question_results.append(docs)
        source_results.append(source_docs)

    # 4. Merge with per-question coverage first, then fill by best global score.
    context_budget = max(settings.final_top_k, request.top_k, len(sub_questions) * 2)
    merged_docs = _merge_sub_question_results(sub_question_results, context_budget)
    source_budget = max(request.top_k, len(sub_questions))
    merged_source_docs = _merge_sub_question_results(source_results, source_budget)
    logger.info(
        "Merged   → %d generation doc(s) (budget=%d) | %d source doc(s)",
        len(merged_docs), context_budget,
        len(merged_source_docs),
    )

    return QueryContext(
        merged_docs    = merged_docs,
        source_docs    = merged_source_docs,
        sub_questions  = sub_questions,
        is_compound    = is_compound,
        low_confidence = low_confidence,
        analysis       = analysis,
    )


def _build_semantic_date_filter(
    date_from: date_type | None,
    date_to: date_type | None,
) -> dict | None:
    if not date_from and not date_to:
        return None

    start_ts = (
        int(datetime.combine(date_from, time.min, tzinfo=timezone.utc).timestamp())
        if date_from else None
    )
    end_ts = (
        int(datetime.combine(date_to, time.max, tzinfo=timezone.utc).timestamp())
        if date_to else None
    )

    period_conditions: list[dict[str, Any]] = []
    document_date_conditions: list[dict[str, Any]] = []

    if start_ts is not None:
        period_conditions.append({"field": "meta.period_end_ts", "operator": ">=", "value": start_ts})
        document_date_conditions.append({"field": "meta.document_date_ts", "operator": ">=", "value": start_ts})
    if end_ts is not None:
        period_conditions.append({"field": "meta.period_start_ts", "operator": "<=", "value": end_ts})
        document_date_conditions.append({"field": "meta.document_date_ts", "operator": "<=", "value": end_ts})

    semantic_conditions: list[dict[str, Any]] = []
    if period_conditions:
        if len(period_conditions) == 1:
            semantic_conditions.append(period_conditions[0])
        else:
            semantic_conditions.append({"operator": "AND", "conditions": period_conditions})
    if document_date_conditions:
        if len(document_date_conditions) == 1:
            semantic_conditions.append(document_date_conditions[0])
        else:
            semantic_conditions.append({"operator": "AND", "conditions": document_date_conditions})

    if not semantic_conditions:
        return None
    if len(semantic_conditions) == 1:
        return semantic_conditions[0]
    return {"operator": "OR", "conditions": semantic_conditions}


def _merge_sub_question_results(
    sub_question_results: list[list[Any]],
    context_budget: int,
) -> list[Any]:
    if not sub_question_results:
        return []

    per_question_budget = max(1, ceil(context_budget / len(sub_question_results)))
    selected: OrderedDict[str, Any] = OrderedDict()
    global_best: dict[str, Any] = {}

    for docs in sub_question_results:
        for doc in docs:
            _keep_best_doc(global_best, doc)
        for doc in docs[:per_question_budget]:
            _keep_best_doc(selected, doc)

    if len(selected) < context_budget:
        for doc in sorted(global_best.values(), key=_doc_score, reverse=True):
            _keep_best_doc(selected, doc)
            if len(selected) >= context_budget:
                break

    return sorted(selected.values(), key=_doc_score, reverse=True)[:context_budget]


def _keep_best_doc(target: dict[str, Any], candidate: Any) -> None:
    existing = target.get(candidate.id)
    if existing is None or _doc_score(candidate) > _doc_score(existing):
        target[candidate.id] = candidate


def _doc_score(doc: Any) -> float:
    return float(getattr(doc, "score", 0.0) or 0.0)
