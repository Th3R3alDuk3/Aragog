"""
RAGAS evaluation endpoint — POST /evaluation/run

Runs the full RAG pipeline on a set of test questions, then scores the results
using the RAGAS evaluation framework.

Metrics computed:
  faithfulness        — does the answer contain only claims supported by the context?
  answer_relevancy    — how relevant is the answer to the question?
  context_precision   — are the retrieved chunks actually relevant to the question?

Guard: the endpoint returns HTTP 403 unless RAGAS_ENABLED=true in .env.
This prevents accidental load of the heavy ragas + datasets dependencies
(several hundred MB) during normal operation.

Usage:
  POST /evaluation/run
  {
    "samples": [
      {"question": "What is the EBITDA?", "ground_truth": "The EBITDA is 12M EUR."},
      ...
    ],
    "top_k": 5
  }
"""

import asyncio
import logging
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request, status

from models.schemas import EvaluationRequest, EvaluationResponse
from routers._deps import get_retrieval_pipeline
from routers.query import _retrieve_simple, _run_generation_only
from pipelines.retrieval import swap_to_parent_content

logger = logging.getLogger(__name__)

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
    req: Request,
    pipeline=Depends(get_retrieval_pipeline),
) -> EvaluationResponse:
    settings = req.app.state.settings

    if not settings.ragas_enabled:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="RAGAS evaluation is disabled. Set RAGAS_ENABLED=true in .env to enable.",
        )

    # Lazy imports — never loaded at startup when RAGAS_ENABLED=false
    try:
        from ragas import evaluate
        from ragas.metrics import answer_relevancy, context_precision, faithfulness
        from datasets import Dataset
    except ImportError as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"ragas or datasets package not installed: {exc}",
        ) from exc

    rows: list[dict[str, Any]] = []

    for sample in request.samples:
        try:
            # Retrieve
            docs = await asyncio.to_thread(
                _retrieve_simple,
                pipeline,
                sample.question,
                None,       # no filters for evaluation
                None,       # no HyDE generator
                False,      # use_hyde=False
            )
            docs = swap_to_parent_content(docs)[: request.top_k]

            # Generate answer
            gen_result = await asyncio.to_thread(
                _run_generation_only,
                pipeline,
                docs,
                [sample.question],
                sample.question,
            )
            answers = gen_result.get("answer_builder", {}).get("answers", [])
            answer_text = answers[0].data if answers else ""

            rows.append({
                "question":     sample.question,
                "answer":       answer_text,
                "contexts":     [doc.content or "" for doc in docs],
                "ground_truth": sample.ground_truth,
            })

        except Exception as exc:
            logger.warning("Evaluation: skipping sample '%s': %s", sample.question[:60], exc)
            rows.append({
                "question":     sample.question,
                "answer":       "",
                "contexts":     [],
                "ground_truth": sample.ground_truth,
            })

    if not rows:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="No samples could be processed.",
        )

    dataset = Dataset.from_list(rows)

    try:
        result = evaluate(
            dataset,
            metrics=[faithfulness, answer_relevancy, context_precision],
        )
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
        scores      = per_question_scores,
        aggregate   = aggregate,
        num_samples = len(rows),
    )
