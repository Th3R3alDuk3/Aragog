from json import dumps
from logging import getLogger
from typing import Any, AsyncGenerator

from fastapi import APIRouter, Depends, HTTPException, status
from haystack.core.pipeline.async_pipeline import AsyncPipeline
from jinja2 import Environment
from sse_starlette.sse import EventSourceResponse

from components.hyde_generator import HyDEGenerator
from components.query_analyzer import QueryAnalyzer
from config import Settings
from models.schemas import QueryRequest
from pipelines.generation import RAG_PROMPT
from routers._deps import (
    get_hyde_generator,
    get_query_analyzer,
    get_retrieval_pipeline,
    get_settings,
)
from services import query as query_service

logger = getLogger(__name__)


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
    settings: Settings = Depends(get_settings),
    retrieval_pipeline: AsyncPipeline = Depends(get_retrieval_pipeline),
    query_analyzer: QueryAnalyzer = Depends(get_query_analyzer),
    hyde_generator: HyDEGenerator | None = Depends(get_hyde_generator),
) -> EventSourceResponse:
    try:
        ctx = await query_service.prepare_context(
            request,
            settings,
            retrieval_pipeline,
            query_analyzer,
            hyde_generator,
        )
    except ValueError as error:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Invalid query or filter parameters: {error}",
        ) from error
    except query_service.RetrievalError as error:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Document retrieval failed: {error}",
        ) from error

    if not ctx.merged_docs:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No relevant documents found for your query.",
        )

    sources = query_service.format_source_docs(ctx.merged_docs[: request.top_k])

    sources_payload = [source.model_dump() for source in sources]

    prompt_text = (
        Environment()
        .from_string(RAG_PROMPT)
        .render(documents=ctx.merged_docs, questions=ctx.sub_questions),
    )

    return EventSourceResponse(
        _stream_generator(settings, prompt_text, sources_payload),
        media_type="text/event-stream",
    )


async def _stream_generator(
    settings: Settings,
    prompt_text: str,
    sources_payload: list[dict],
) -> AsyncGenerator[dict, None]:
    """Async generator that yields SSE-formatted dicts for EventSourceResponse."""
    try:
        from openai import AsyncOpenAI

        client = AsyncOpenAI(
            base_url=settings.openai_url,
            api_key=settings.openai_api_key,
        )

        stream = await client.chat.completions.create(
            model=settings.llm_model,
            messages=[{"role": "user", "content": prompt_text}],
            stream=True,
        )

        async for chunk in stream:
            delta = chunk.choices[0].delta.content if chunk.choices else None
            if delta:
                yield {"event": "token", "data": delta}

    except Exception as error:
        logger.error("SSE stream: LLM streaming failed: %s", error)
        yield {
            "event": "error",
            "data": f"LLM streaming failed: {error}",
        }
        return

    yield {"event": "sources", "data": dumps(sources_payload, default=str)}
    yield {"event": "done", "data": ""}
