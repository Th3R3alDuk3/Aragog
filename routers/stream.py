"""
SSE streaming query endpoint — POST /query/stream

Same retrieval logic as POST /query, but the LLM answer is streamed
token-by-token as Server-Sent Events (SSE).

SSE event protocol:
  event: token   data: <token text>     (one per LLM output token)
  event: sources data: <JSON array>     (SourceDocument list, sent after stream ends)
  event: done    data: ""               (stream termination signal)

Retrieval runs synchronously via asyncio.to_thread() to avoid blocking the
async event loop.  The LLM generation uses AsyncOpenAI.chat.completions.create
with stream=True for true async streaming.

The RAG_PROMPT Jinja2 template is imported directly from pipelines/retrieval_pipeline.py
and rendered with the standard jinja2 library (Haystack transitive dependency).
"""

import asyncio
import json
import logging
from datetime import datetime, time, timezone
from typing import Any, AsyncGenerator

from fastapi import APIRouter, Depends, HTTPException, Request, status
from sse_starlette.sse import EventSourceResponse

from components.query_analyzer import QueryAnalyzer
from config import Settings
from models.schemas import QueryRequest, SourceDocument
from pipelines.retrieval import RAG_PROMPT, swap_to_parent_content
from routers._deps import get_colbert_reranker, get_hyde_generator, get_query_analyzer, get_retrieval_pipeline
from routers.query import (
    _analysis_to_filter_dict,
    _build_filters,
    _retrieve_simple,
    _retrieve_with_crag,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/query", tags=["query"])


@router.post(
    "/stream",
    summary="Query the RAG system with streaming response",
    description=(
        "Same as POST /query but streams the LLM answer token-by-token via SSE. "
        "Connect with EventSource or any SSE client. "
        "Events: 'token' (text chunks), 'sources' (JSON), 'done' (end of stream)."
    ),
)
async def query_stream(
    request: QueryRequest,
    req: Request,
    pipeline=Depends(get_retrieval_pipeline),
    analyzer: QueryAnalyzer = Depends(get_query_analyzer),
    hyde_generator=Depends(get_hyde_generator),
    colbert_reranker=Depends(get_colbert_reranker),
) -> EventSourceResponse:
    settings: Settings = req.app.state.settings

    # ── Analyze + filter + retrieve (same as /query) ──────────────────────────
    analysis = analyzer.analyze(request.query)
    sub_questions = analysis.sub_questions
    try:
        filters = _build_filters(request, analysis)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        ) from exc
    use_hyde = settings.hyde_enabled or request.use_hyde

    all_docs_by_id: dict[str, Any] = {}
    low_confidence = False

    for sub_q in sub_questions:
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

    candidate_budget = request.top_k * max(1, len(sub_questions))
    merged_docs = swap_to_parent_content(list(all_docs_by_id.values()))
    merged_docs = merged_docs[:candidate_budget]

    if settings.colbert_enabled and colbert_reranker is not None:
        merged_docs = colbert_reranker.rerank(request.query, merged_docs)

    # Final cut to top_k for the LLM context window
    merged_docs = merged_docs[: request.top_k]

    # ── Build sources payload for the final SSE event ─────────────────────────
    sources_payload = [
        {
            "content": doc.meta.get("original_content") or doc.content or "",
            "score":   getattr(doc, "score", None),
            "meta": {
                k: v for k, v in doc.meta.items()
                if k not in {"parent_content", "original_content", "doc_beginning"}
            },
        }
        for doc in merged_docs
    ]

    # ── Render the prompt ─────────────────────────────────────────────────────
    try:
        from jinja2 import Environment
        prompt_text = (
            Environment()
            .from_string(RAG_PROMPT)
            .render(documents=merged_docs, questions=sub_questions)
        )
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prompt rendering failed: {exc}",
        ) from exc

    return EventSourceResponse(
        _stream_generator(prompt_text, sources_payload, settings),
        media_type="text/event-stream",
    )


async def _stream_generator(
    prompt_text: str,
    sources_payload: list[dict],
    settings: Settings,
) -> AsyncGenerator[dict, None]:
    """Async generator that yields SSE-formatted dicts for EventSourceResponse."""
    try:
        from openai import AsyncOpenAI

        client_kwargs: dict[str, Any] = {"api_key": settings.openai_api_key}
        if settings.openai_base_url:
            client_kwargs["base_url"] = settings.openai_base_url

        client = AsyncOpenAI(**client_kwargs)

        stream = await client.chat.completions.create(
            model    = settings.llm_model,
            messages = [{"role": "user", "content": prompt_text}],
            stream   = True,
        )

        async for chunk in stream:
            delta = chunk.choices[0].delta.content if chunk.choices else None
            if delta:
                yield {"event": "token", "data": delta}

    except Exception as exc:
        logger.error("SSE stream: LLM streaming failed: %s", exc)
        yield {"event": "error", "data": str(exc)}
        return

    # Send sources as final structured event
    yield {"event": "sources", "data": json.dumps(sources_payload, default=str)}
    yield {"event": "done", "data": ""}
