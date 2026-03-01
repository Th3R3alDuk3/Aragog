import logging

from fastapi import APIRouter, Depends, HTTPException, status

from components.query_analyzer import QueryAnalyzer
from config import Settings
from models.schemas import QueryRequest, QueryResponse
from routers._deps import (
    get_colbert_reranker,
    get_generation_pipeline,
    get_hyde_generator,
    get_query_analyzer,
    get_retrieval_pipeline,
    get_settings,
)
from services import query as query_service

logger = logging.getLogger(__name__)

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
    pipeline=Depends(get_retrieval_pipeline),
    gen_pipeline=Depends(get_generation_pipeline),
    analyzer: QueryAnalyzer = Depends(get_query_analyzer),
    hyde_generator=Depends(get_hyde_generator),
    colbert_reranker=Depends(get_colbert_reranker),
    settings: Settings = Depends(get_settings),
) -> QueryResponse:
    logger.info("── QUERY ─────────────────────────────────────────────────────")
    logger.info("Query: %r", request.query)

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

    logger.info("Generate → %d doc(s) as context …", len(ctx.merged_docs))
    try:
        gen_result = await query_service.run_generation(
            gen_pipeline, ctx.merged_docs, ctx.sub_questions, request.query,
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

    best    = answers[0]
    sources = query_service.format_source_docs(
        (best.documents or ctx.merged_docs)[: request.top_k]
    )

    logger.info(
        "Done  → answer=%d chars | sources=%d%s",
        len(best.data or ""),
        len(sources),
        " | low_confidence=true" if ctx.low_confidence else "",
    )
    logger.info("──────────────────────────────────────────────────────────────")

    return QueryResponse(
        answer            = best.data or "",
        sources           = sources,
        query             = request.query,
        sub_questions     = ctx.sub_questions if ctx.is_compound else [],
        is_compound       = ctx.is_compound,
        low_confidence    = ctx.low_confidence,
        extracted_filters = query_service.analysis_to_filter_dict(ctx.analysis) if ctx.analysis else None,
    )
