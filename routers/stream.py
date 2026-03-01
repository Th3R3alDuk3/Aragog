"""
SSE streaming query endpoint — POST /query/stream

Same retrieval logic as POST /query, but the LLM answer is streamed
token-by-token as Server-Sent Events (SSE).

SSE event protocol:
  event: token   data: <token text>     (one per LLM output token)
  event: sources data: <JSON array>     (SourceDocument list, sent after stream ends)
  event: done    data: ""               (stream termination signal)
  event: error   data: <error message>  (on LLM streaming failure)
"""

import json
import logging
from typing import Any, AsyncGenerator

from fastapi import APIRouter, Depends, HTTPException, status
from jinja2 import Environment
from sse_starlette.sse import EventSourceResponse

from components.query_analyzer import QueryAnalyzer
from config import Settings
from models.schemas import QueryRequest
from pipelines.generation import RAG_PROMPT
from routers._deps import get_colbert_reranker, get_hyde_generator, get_query_analyzer, get_retrieval_pipeline, get_settings
from services import query as query_service

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
    pipeline=Depends(get_retrieval_pipeline),
    analyzer: QueryAnalyzer = Depends(get_query_analyzer),
    hyde_generator=Depends(get_hyde_generator),
    colbert_reranker=Depends(get_colbert_reranker),
    settings: Settings = Depends(get_settings),
) -> EventSourceResponse:
    try:
        ctx = await query_service.prepare_context(
            request, settings, pipeline, analyzer, hyde_generator, colbert_reranker,
        )
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)) from exc
    except query_service.RetrievalError as exc:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc)) from exc

    if not ctx.merged_docs:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No relevant documents found for your query.",
        )

    sources_payload = [
        s.model_dump() for s in query_service.format_source_docs(ctx.merged_docs)
    ]

    prompt_text = (
        Environment()
        .from_string(RAG_PROMPT)
        .render(documents=ctx.merged_docs, questions=ctx.sub_questions)
    )

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
        stream  = await client.chat.completions.create(
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

    yield {"event": "sources", "data": json.dumps(sources_payload, default=str)}
    yield {"event": "done",    "data": ""}
