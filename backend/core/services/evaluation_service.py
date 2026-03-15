import logging
from asyncio import to_thread
from typing import Any

from core.config import Settings
from core.models.evaluation import EvaluationInput, EvaluationResult
from core.models.query import QueryInput
from core.services.query_engine import QueryEngine

logger = logging.getLogger(__name__)

_RAGAS_METRICS = ("faithfulness", "answer_relevancy", "context_precision")


async def run_evaluation(
    request: EvaluationInput,
    settings: Settings,
    query_engine: QueryEngine,
) -> EvaluationResult:
    from datasets import Dataset
    from ragas import evaluate
    from ragas.metrics import answer_relevancy, context_precision, faithfulness

    rows: list[dict[str, Any]] = []

    for sample in request.samples:
        try:
            result = await query_engine.query(
                QueryInput(
                    query=sample.question,
                    top_k=request.top_k,
                )
            )
            rows.append(
                {
                    "question": sample.question,
                    "answer": result.answer,
                    "contexts": [source.content for source in result.sources[: request.top_k]],
                    "ground_truth": sample.ground_truth,
                }
            )
        except Exception as error:
            logger.warning("Evaluation: skipping sample '%s': %s", sample.question[:60], error)
            rows.append(
                {
                    "question": sample.question,
                    "answer": "",
                    "contexts": [],
                    "ground_truth": sample.ground_truth,
                }
            )

    if not rows:
        raise RuntimeError("No samples could be processed.")

    dataset = Dataset.from_list(rows)

    def _run_ragas():
        from langchain_community.embeddings import HuggingFaceEmbeddings
        from langchain_openai import ChatOpenAI
        from ragas.embeddings import LangchainEmbeddingsWrapper
        from ragas.llms import LangchainLLMWrapper

        ragas_llm = LangchainLLMWrapper(
            ChatOpenAI(
                base_url=settings.openai_url,
                api_key=settings.openai_api_key,
                model=settings.llm_model,
            )
        )
        ragas_embeddings = LangchainEmbeddingsWrapper(
            HuggingFaceEmbeddings(
                model_name=settings.embedding_model,
                model_kwargs={"device": "cpu"},
            )
        )
        return evaluate(
            dataset=dataset,
            metrics=[faithfulness, answer_relevancy, context_precision],
            llm=ragas_llm,
            embeddings=ragas_embeddings,
        )

    result = await to_thread(_run_ragas)
    frame = result.to_pandas()

    return EvaluationResult(
        scores=frame.to_dict(orient="records"),
        aggregate={
            metric: float(frame[metric].mean())
            for metric in _RAGAS_METRICS
            if metric in frame.columns
        },
        num_samples=len(rows),
    )
