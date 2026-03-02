from asyncio import to_thread
from logging import getLogger
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, status
from haystack.core.pipeline.async_pipeline import AsyncPipeline

from config import Settings
from models.schemas import EvaluationRequest, EvaluationResponse
from pipelines.retrieval import swap_to_parent_content
from routers._deps import get_generation_pipeline, get_retrieval_pipeline, get_settings
from services import query as query_service

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
async def run_evaluation(
    request: EvaluationRequest,
    settings: Settings = Depends(get_settings),
    retrieval_pipeline: AsyncPipeline = Depends(get_retrieval_pipeline),
    generation_pipeline: AsyncPipeline = Depends(get_generation_pipeline),
) -> EvaluationResponse:

    if not settings.ragas_enabled:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=(
                "RAGAS evaluation is disabled. ",
                "Set RAGAS_ENABLED=true in .env to enable.",
            ),
        )

    # Lazy imports — never loaded at startup when RAGAS_ENABLED=false
    try:
        from datasets import Dataset
        from ragas import evaluate
        from ragas.metrics import answer_relevancy, context_precision, faithfulness
    except ImportError as error:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"ragas or datasets package not installed: {error}",
        ) from error

    rows: list[dict[str, Any]] = []

    for sample in request.samples:
        try:
            # Retrieve
            docs = await query_service.retrieve(
                retrieval_pipeline,
                sample.question,
                None,  # no filters for evaluation
                None,  # no HyDE generator
                False,  # use_hyde=False
            )

            docs = swap_to_parent_content(docs)[: request.top_k]

            # Generate answer
            generation_result = await query_service.run_generation(
                generation_pipeline,
                docs,
                [sample.question],
                sample.question,
            )
            
            answers = generation_result.get("answer_builder", {}).get("answers", [])
            answer_text = answers[0].data if answers else ""

            rows.append(
                {
                    "question": sample.question,
                    "answer": answer_text,
                    "contexts": [doc.content or "" for doc in docs],
                    "ground_truth": sample.ground_truth,
                }
            )

        except Exception as exc:
            logger.warning(
                "Evaluation: skipping sample '%s': %s", sample.question[:60], exc
            )
            rows.append(
                {
                    "question": sample.question,
                    "answer": "",
                    "contexts": [],
                    "ground_truth": sample.ground_truth,
                }
            )

    if not rows:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="No samples could be processed.",
        )

    dataset = Dataset.from_list(rows)

    def _run_ragas():
        from langchain_community.embeddings import HuggingFaceEmbeddings
        from langchain_openai import ChatOpenAI
        from ragas.embeddings import LangchainEmbeddingsWrapper
        from ragas.llms import LangchainLLMWrapper

        ragas_llm = LangchainLLMWrapper(ChatOpenAI(
            model=settings.llm_model,
            api_key=settings.openai_api_key or "dummy",
            base_url=settings.openai_url or None,
        ))
        ragas_embeddings = LangchainEmbeddingsWrapper(HuggingFaceEmbeddings(
            model_name=settings.embedding_model,
        ))
        return evaluate(
            dataset,
            metrics=[faithfulness, answer_relevancy, context_precision],
            llm=ragas_llm,
            embeddings=ragas_embeddings,
        )

    try:
        result = await to_thread(_run_ragas)
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"RAGAS evaluation failed: {exc}",
        ) from exc

    df = result.to_pandas()
    per_question_scores = df.to_dict(orient="records")

    aggregate = {
        metric: float(df[metric].mean())
        for metric in ("faithfulness", "answer_relevancy", "context_precision")
        if metric in df.columns
    }

    return EvaluationResponse(
        scores=per_question_scores,
        aggregate=aggregate,
        num_samples=len(rows),
    )
