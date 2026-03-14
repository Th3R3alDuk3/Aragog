from logging import getLogger

from fastapi import APIRouter, Depends, HTTPException, status
from haystack.core.pipeline.async_pipeline import AsyncPipeline

from config import Settings
from models.api import EvaluationRequest, EvaluationResponse
from routers._deps import get_generation_pipeline, get_retrieval_pipeline, get_settings
from services import evaluation as evaluation_service

logger = getLogger(__name__)

router = APIRouter(prefix="/evaluation", tags=["evaluation"])


@router.post(
    "/run",
    response_model=EvaluationResponse,
    summary="Run RAGAS evaluation",
    description=(
        "Evaluate retrieval quality on a test set using RAGAS metrics "
        "(faithfulness, answer_relevancy, context_precision). "
        "Requires RAGAS_ENABLED=true in .env."
    ),
)
async def evaluate_rag(
    request: EvaluationRequest,
    settings: Settings = Depends(get_settings),
    retrieval_pipeline: AsyncPipeline = Depends(get_retrieval_pipeline),
    generation_pipeline: AsyncPipeline = Depends(get_generation_pipeline),
) -> EvaluationResponse:

    if not settings.ragas_enabled:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="RAGAS evaluation is disabled. Set RAGAS_ENABLED=true in .env to enable.",
        )

    try:
        return await evaluation_service.run_evaluation(
            request,
            settings,
            retrieval_pipeline,
            generation_pipeline,
        )
    except ImportError as error:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"ragas or datasets package not installed: {error}",
        ) from error
    except Exception as error:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"RAGAS evaluation failed: {error}",
        ) from error
