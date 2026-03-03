from logging import getLogger

from fastapi import APIRouter, Depends, HTTPException, status
from haystack.core.pipeline.async_pipeline import AsyncPipeline

from components.query_analyzer import QueryAnalyzer
from config import Settings
from models.api import QueryRequest, QueryResponse
from routers._deps import (
    get_generation_pipeline,
    get_query_analyzer,
    get_retrieval_pipeline,
    get_settings,
)
from services import query as query_service

logger = getLogger(__name__)


router = APIRouter(prefix="/query", tags=["query"])


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
    settings: Settings = Depends(get_settings),
    retrieval_pipeline: AsyncPipeline = Depends(get_retrieval_pipeline),
    generation_pipeline: AsyncPipeline = Depends(get_generation_pipeline),
    query_analyzer: QueryAnalyzer = Depends(get_query_analyzer),
) -> QueryResponse:

    logger.info("── QUERY ─────────────────────────────────────────────────────")
    logger.info("Query: %r", request.query)

    try:
        ctx = await query_service.prepare_context(
            request,
            settings,
            retrieval_pipeline,
            query_analyzer,
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

    logger.info("Generate → %d doc(s) as context...", len(ctx.merged_docs))

    try:
        answers = await query_service.run_generation(
            generation_pipeline,
            ctx.merged_docs,
            ctx.sub_questions,
            request.query,
        )
    except Exception as error:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Answer generation failed: {error}",
        ) from error

    if not answers:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="LLM returned no answer.",
        )

    answer = answers[0]

    sources = query_service.format_source_docs(
        (answer.documents or ctx.merged_docs)[: request.top_k]
    )

    logger.info(
        "Done  → answer=%d chars | sources=%d%s",
        len(answer.data or ""),
        len(sources),
        " | low_confidence=true" if ctx.low_confidence else "",
    )

    logger.info("──────────────────────────────────────────────────────────────")

    return QueryResponse(
        answer=answer.data or "",
        sources=sources,
        query=request.query,
        sub_questions=ctx.sub_questions if ctx.is_compound else [],
        is_compound=ctx.is_compound,
        low_confidence=ctx.low_confidence,
        extracted_filters=query_service.analysis_to_filter_dict(ctx.analysis)
        if ctx.analysis
        else None,
    )
