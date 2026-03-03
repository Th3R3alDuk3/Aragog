"""
Evaluation service — RAGAS evaluation orchestration.

Handles all business logic for running RAGAS on a test set:
  collect_rows → build_dataset → run_ragas → aggregate_scores

Raises:
  ImportError   — ragas or datasets package not installed
  Exception     — no processable samples, or RAGAS evaluation failure
"""

import logging
from asyncio import to_thread
from typing import Any

from haystack.core.pipeline.async_pipeline import AsyncPipeline

from config import Settings
from models.api import EvaluationRequest, EvaluationResponse
from pipelines.retrieval import swap_to_parent_content
from services import query as query_service

logger = logging.getLogger(__name__)

_RAGAS_METRICS = ("faithfulness", "answer_relevancy", "context_precision")


async def run_evaluation(
    request: EvaluationRequest,
    settings: Settings,
    retrieval_pipeline: AsyncPipeline,
    generation_pipeline: AsyncPipeline,
) -> EvaluationResponse:
    """Run RAGAS evaluation on a test set.

    Lazy-imports ragas and datasets so they are never loaded when
    RAGAS_ENABLED=false.

    Raises:
        ImportError  — ragas or datasets package not installed.
        RuntimeError — no samples could be processed.
        Exception    — RAGAS evaluation failed.
    """
    # Lazy imports — never loaded at startup when RAGAS_ENABLED=false
    from datasets import Dataset
    from ragas import evaluate
    from ragas.metrics import answer_relevancy, context_precision, faithfulness

    rows: list[dict[str, Any]] = []

    for sample in request.samples:
        try:
            docs = await query_service.run_retrieval(
                retrieval_pipeline,
                sample.question,
                None,   # no filters for evaluation
            )
            docs = swap_to_parent_content(docs)[: request.top_k]

            answers = await query_service.run_generation(
                generation_pipeline,
                docs,
                [sample.question],
                sample.question,
            )
            
            answer_text = answers[0].data if answers else ""

            rows.append({
                "question":     sample.question,
                "answer":       answer_text,
                "contexts":     [doc.content or "" for doc in docs],
                "ground_truth": sample.ground_truth,
            })

        except Exception as exc:
            logger.warning(
                "Evaluation: skipping sample '%s': %s", sample.question[:60], exc
            )
            rows.append({
                "question":     sample.question,
                "answer":       "",
                "contexts":     [],
                "ground_truth": sample.ground_truth,
            })

    if not rows:
        raise RuntimeError("No samples could be processed.")

    dataset = Dataset.from_list(rows)

    def _run_ragas():
        from langchain_community.embeddings import HuggingFaceEmbeddings
        from langchain_openai import ChatOpenAI
        from ragas.embeddings import LangchainEmbeddingsWrapper
        from ragas.llms import LangchainLLMWrapper

        ragas_llm = LangchainLLMWrapper(ChatOpenAI(
            base_url=settings.openai_url,
            api_key=settings.openai_api_key,
            model=settings.llm_model,
        ))
        ragas_embeddings = LangchainEmbeddingsWrapper(HuggingFaceEmbeddings(
            model_name=settings.embedding_model,
            model_kwargs={"device": "cpu"},
        ))
        return evaluate(
            dataset=dataset,
            metrics=[faithfulness, answer_relevancy, context_precision],
            llm=ragas_llm,
            embeddings=ragas_embeddings,
        )

    result = await to_thread(_run_ragas)

    df = result.to_pandas()
    per_question_scores = df.to_dict(orient="records")
    aggregate = {
        metric: float(df[metric].mean())
        for metric in _RAGAS_METRICS
        if metric in df.columns
    }

    return EvaluationResponse(
        scores=per_question_scores,
        aggregate=aggregate,
        num_samples=len(rows),
    )
