"""
Query router — handles both simple and compound (multi-question) queries.

Enhanced pipeline
─────────────────
1. QueryAnalyzer — decompose + extract metadata filters in one LLM call.
2. _build_filters — merge LLM-extracted hints + explicit request filters.
3. _retrieve_simple / _retrieve_with_crag — retrieval with optional HyDE and CRAG.
4. swap_to_parent_content — replace child chunks with full parent sections.
5. ColBERT second-pass reranker (optional, COLBERT_ENABLED).
6. Single LLM call over merged document set → answer.

Feature flags (controlled via .env):
  HYDE_ENABLED      — hypothetical document embedding for dense retrieval
  CRAG_ENABLED      — corrective re-retrieval on low relevance score
  COLBERT_ENABLED   — ColBERT late-interaction second-pass reranker

Per-request HyDE override:
  request.use_hyde=true enables HyDE for a single request even if HYDE_ENABLED=false.
"""

import asyncio
import logging
from datetime import datetime, time, timezone
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request, status

from components.query_analyzer import AnalysisResult, QueryAnalyzer
from config import Settings
from models.schemas import QueryRequest, QueryResponse, SourceDocument
from pipelines.retrieval import swap_to_parent_content
from routers._deps import get_colbert_reranker, get_generation_pipeline, get_hyde_generator, get_query_analyzer, get_retrieval_pipeline

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/query", tags=["query"])


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_BARE_FIELDS = {"id", "content"}  # Qdrant top-level fields that need no prefix
_LOGICAL_OPERATORS = {"AND", "OR", "NOT"}
_SET_OPERATORS = {"in", "not in"}


def _is_scalar_filter_value(value: Any) -> bool:
    return isinstance(value, (str, int, float, bool))


def _normalize_filter_fields(f: dict) -> dict:
    """
    Recursively ensure all metadata field names carry the ``meta.`` prefix.

    Users may pass ``{"field": "source", ...}`` or ``{"field": "meta.source", ...}``;
    both are accepted.  The special Qdrant top-level fields ``id`` and ``content``
    are left unchanged.
    """
    operator = f.get("operator")
    if operator in ("AND", "OR", "NOT"):
        return {**f, "conditions": [_normalize_filter_fields(c) for c in f.get("conditions", [])]}
    # Comparison operator — normalise field name
    field = f.get("field", "")
    if field and not field.startswith("meta.") and field not in _BARE_FIELDS:
        return {**f, "field": f"meta.{field}"}
    return f


def _normalize_field_name(field: str) -> str:
    if field and not field.startswith("meta.") and field not in _BARE_FIELDS:
        return f"meta.{field}"
    return field


def _is_valid_filter_expression(f: Any) -> bool:
    """Validate Haystack-like filter structures recursively."""
    if not isinstance(f, dict):
        return False

    operator = f.get("operator")
    if operator in _LOGICAL_OPERATORS:
        conditions = f.get("conditions")
        if not isinstance(conditions, list) or not conditions:
            return False
        return all(_is_valid_filter_expression(c) for c in conditions)

    # Comparison expression: {"field": "...", "operator": "...", "value": ...}
    if "field" in f and "operator" in f and "value" in f:
        field = f.get("field")
        operator = f.get("operator")
        value = f.get("value")
        if not isinstance(field, str) or not field.strip() or not isinstance(operator, str):
            return False
        op = operator.lower()
        if op in _SET_OPERATORS:
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

    # Shorthand map mode: convert key/value pairs into equality filters.
    # Example: {"source": "a.pdf", "classification": "financial"}
    #      -> {"operator":"AND","conditions":[...]}.
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


def _build_filters(
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

    # --- LLM-extracted filters (lower priority) ---
    if analysis:
        if analysis.date_from and not request.date_from:
            try:
                from datetime import date as date_type
                d = date_type.fromisoformat(analysis.date_from)
                ts = int(datetime.combine(d, time.min, tzinfo=timezone.utc).timestamp())
                conditions.append({"field": "meta.indexed_at_ts", "operator": ">=", "value": ts})
            except ValueError:
                pass
        if analysis.date_to and not request.date_to:
            try:
                from datetime import date as date_type
                d = date_type.fromisoformat(analysis.date_to)
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

    # --- Explicit request filters (higher priority) ---
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


def _analysis_to_filter_dict(analysis: AnalysisResult) -> dict[str, Any] | None:
    hints = {
        "date_from":      analysis.date_from,
        "date_to":        analysis.date_to,
        "classification": analysis.classification,
        "language":       analysis.language,
        "source":         analysis.source,
    }
    non_null = {k: v for k, v in hints.items() if v is not None}
    return non_null or None


def _retrieve_simple(
    pipeline,
    sub_q: str,
    filters: dict | None,
    hyde_generator,
    use_hyde: bool,
) -> list:
    """Single retrieval pass — returns reranked documents."""
    if use_hyde and hyde_generator is not None:
        dense_text = hyde_generator.generate(sub_q)
        logger.info("  HyDE: hypothetical doc generated (%d chars)", len(dense_text))
    else:
        dense_text = sub_q

    # Only run retrieval stages (embedders → retrievers → joiner → reranker).
    # PromptBuilder requires "questions" — omitting it stops the pipeline there,
    # so the LLM is NOT called here.  Generation happens once in _run_generation_only.
    run_input: dict = {
        "dense_embedder":  {"text": dense_text},
        "sparse_embedder": {"text": sub_q},
        "reranker":        {"query": sub_q},
    }
    if filters:
        run_input["dense_retriever"]  = {"filters": filters}
        run_input["sparse_retriever"] = {"filters": filters}

    result = pipeline.run(run_input)
    docs = result.get("reranker", {}).get("documents", [])
    top_score = getattr(docs[0], "score", None) if docs else None
    logger.info(
        "  → %d doc(s) after reranking%s",
        len(docs),
        f" | top_score={top_score:.3f}" if top_score is not None else "",
    )
    return docs


def _reformulate_query(query: str, settings: Settings) -> str:
    """LLM rephrases query to improve retrieval. Falls back to original on error."""
    try:
        from openai import OpenAI
        client_kwargs: dict[str, Any] = {"api_key": settings.openai_api_key}
        if settings.openai_base_url:
            client_kwargs["base_url"] = settings.openai_base_url
        client = OpenAI(**client_kwargs)
        response = client.chat.completions.create(
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


async def _retrieve_with_crag(
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
    docs = await asyncio.to_thread(
        _retrieve_simple, pipeline, sub_q, filters, hyde_generator, use_hyde,
    )
    top_score = (getattr(docs[0], "score", None) or 0.0) if docs else 0.0
    sufficient = top_score >= settings.crag_score_threshold

    if sufficient or attempt >= settings.crag_max_retries:
        if not sufficient:
            logger.info(
                "CRAG: low confidence (score=%.3f, threshold=%.3f, attempts=%d)",
                top_score, settings.crag_score_threshold, attempt + 1,
            )
        return docs, not sufficient

    reformulated = _reformulate_query(sub_q, settings)
    return await _retrieve_with_crag(
        pipeline, reformulated, filters, settings, hyde_generator, use_hyde, attempt + 1,
    )


def _run_generation_only(gen_pipeline, documents: list, questions: list[str], query: str) -> dict:
    """Run prompt_builder → llm → answer_builder (pre-built generation pipeline)."""
    return gen_pipeline.run({
        "prompt_builder": {"documents": documents, "questions": questions},
        "answer_builder": {"query": query, "documents": documents},
    })


# ---------------------------------------------------------------------------
# Main endpoint
# ---------------------------------------------------------------------------

@router.post(
    "",
    response_model=QueryResponse,
    summary="Query the RAG system",
    description=(
        "Send a natural-language question — or multiple questions in one message. "
        "The system automatically detects compound queries and metadata filters, "
        "retrieves focused evidence via hybrid search (dense + SPLADE + cross-encoder), "
        "and generates a single coherent answer. "
        "Optional: HyDE, CRAG, ColBERT second-pass reranking (controlled via .env flags)."
    ),
)
async def query_rag(
    request: QueryRequest,
    req: Request,
    pipeline=Depends(get_retrieval_pipeline),
    gen_pipeline=Depends(get_generation_pipeline),
    analyzer: QueryAnalyzer = Depends(get_query_analyzer),
    hyde_generator=Depends(get_hyde_generator),
    colbert_reranker=Depends(get_colbert_reranker),
) -> QueryResponse:
    settings: Settings = req.app.state.settings

    logger.info("── QUERY ─────────────────────────────────────────────────────")
    logger.info("Query: %r", request.query)

    # ── 1. Analyze: decompose + extract filters ───────────────────────────────
    analysis = analyzer.analyze(request.query)
    sub_questions = analysis.sub_questions
    is_compound = len(sub_questions) > 1
    logger.info(
        "[1/7] Analyze  → compound=%s | sub_questions=%d: %s",
        is_compound,
        len(sub_questions),
        sub_questions,
    )

    # ── 2. Build merged filter set ────────────────────────────────────────────
    try:
        filters = _build_filters(request, analysis)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        ) from exc
    logger.info("[2/7] Filters  → %s", filters or "none")

    # ── 3. Decide HyDE ───────────────────────────────────────────────────────
    use_hyde = settings.hyde_enabled or request.use_hyde
    logger.info(
        "[3/7] HyDE     → %s",
        "enabled" if use_hyde else "disabled",
    )

    # ── 4. Retrieve per sub-question (with optional CRAG) ─────────────────────
    all_docs_by_id: dict[str, Any] = {}
    low_confidence = False

    for i, sub_q in enumerate(sub_questions):
        logger.info(
            "[4/7] Retrieve sub_q[%d/%d]: %r%s",
            i + 1, len(sub_questions), sub_q,
            " (CRAG)" if settings.crag_enabled else "",
        )
        try:
            if settings.crag_enabled:
                docs, lc = await _retrieve_with_crag(
                    pipeline, sub_q, filters, settings, hyde_generator, use_hyde,
                )
                low_confidence = low_confidence or lc
            else:
                docs = await asyncio.to_thread(
                    _retrieve_simple, pipeline, sub_q, filters, hyde_generator, use_hyde,
                )
        except Exception as exc:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Retrieval failed for '{sub_q}': {exc}",
            ) from exc

        for doc in docs:
            if doc.id not in all_docs_by_id:
                all_docs_by_id[doc.id] = doc

    if not all_docs_by_id:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No relevant documents found for your query.",
        )

    # ── 5. Parent-content swap ────────────────────────────────────────────────
    # Keep top_k * sub_questions budget so each sub-question retains evidence.
    # Final cut to top_k happens after ColBERT (or here if ColBERT is disabled).
    candidate_budget = request.top_k * max(1, len(sub_questions))
    before_swap = len(all_docs_by_id)
    merged_docs = swap_to_parent_content(list(all_docs_by_id.values()))
    merged_docs = merged_docs[:candidate_budget]
    logger.info(
        "[5/7] Parent   → %d merged → %d unique sections (budget=%d)",
        before_swap, len(merged_docs), candidate_budget,
    )

    # ── 6. Optional ColBERT second-pass ───────────────────────────────────────
    if settings.colbert_enabled and colbert_reranker is not None:
        before_colbert = len(merged_docs)
        merged_docs = colbert_reranker.rerank(request.query, merged_docs)
        logger.info("[6/7] ColBERT  → %d → %d doc(s)", before_colbert, len(merged_docs))
    else:
        logger.info("[6/7] ColBERT  → disabled")

    # Final cut to top_k for the LLM context window
    merged_docs = merged_docs[: request.top_k]

    # ── 7. Generate answer ────────────────────────────────────────────────────
    logger.info("[7/7] Generate → %d doc(s) as context …", len(merged_docs))
    try:
        gen_result = await asyncio.to_thread(
            _run_generation_only, gen_pipeline, merged_docs, sub_questions, request.query,
        )
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Answer generation failed: {exc}",
        ) from exc

    answers = gen_result.get("answer_builder", {}).get("answers", [])
    if not answers:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="LLM returned no answer.",
        )

    best = answers[0]
    sources = [
        SourceDocument(
            content = doc.meta.get("original_content") or doc.content or "",
            score   = getattr(doc, "score", None),
            meta    = {
                k: v for k, v in doc.meta.items()
                if k not in {"parent_content", "original_content", "doc_beginning"}
            },
        )
        for doc in (best.documents or merged_docs)[: request.top_k]
    ]

    logger.info(
        "Done  → answer=%d chars | sources=%d%s",
        len(best.data or ""),
        len(sources),
        " | low_confidence=true" if low_confidence else "",
    )
    logger.info("──────────────────────────────────────────────────────────────")

    return QueryResponse(
        answer            = best.data or "",
        sources           = sources,
        query             = request.query,
        sub_questions     = sub_questions if is_compound else [],
        is_compound       = is_compound,
        low_confidence    = low_confidence,
        extracted_filters = _analysis_to_filter_dict(analysis) if analysis else None,
    )
