from fastapi import APIRouter, Depends, HTTPException, status

from adapters.api.deps import get_runtime, get_settings
from adapters.api.models.evaluation import EvaluationRequest, EvaluationResponse
from core.models.evaluation import EvaluationInput, EvaluationSample
from core.services.evaluation_service import run_evaluation

router = APIRouter(prefix="/evaluation", tags=["evaluation"])


@router.post("/run", response_model=EvaluationResponse, summary="Run RAGAS evaluation")
async def evaluate_rag(
    request: EvaluationRequest,
    settings=Depends(get_settings),
    runtime=Depends(get_runtime),
) -> EvaluationResponse:
    if not settings.ragas_enabled:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="RAGAS evaluation is disabled.",
        )
    if runtime.query_engine is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Query engine is not initialized.",
        )

    try:
        result = await run_evaluation(
            EvaluationInput(
                samples=[
                    EvaluationSample(
                        question=sample.question,
                        ground_truth=sample.ground_truth,
                    )
                    for sample in request.samples
                ],
                top_k=request.top_k,
            ),
            settings,
            runtime.query_engine,
        )
        return EvaluationResponse(
            scores=result.scores,
            aggregate=result.aggregate,
            num_samples=result.num_samples,
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
