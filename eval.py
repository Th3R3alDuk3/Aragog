from argparse import ArgumentParser

from dotenv import load_dotenv
load_dotenv()

from asyncio import run
from json import loads
from logging import INFO, basicConfig, getLogger
from pathlib import Path

from haystack import AsyncPipeline, Document
from haystack.components.evaluators import (
    DocumentMAPEvaluator,
    DocumentMRREvaluator,
    DocumentRecallEvaluator,
)
from haystack.components.evaluators.document_recall import RecallMode

from pipelines._factories import build_document_store
from pipelines.retrieval import (
    build_dense_retrieval_pipeline,
    build_hybrid_retrieval_pipeline,
    build_sparse_retrieval_pipeline,
)


basicConfig(level=INFO)
logger = getLogger(__name__)


#--------------------------------------------
# GLOBALS
#--------------------------------------------


BUILDERS = {
    "dense": build_dense_retrieval_pipeline,
    "sparse": build_sparse_retrieval_pipeline,
    "hybrid": build_hybrid_retrieval_pipeline,
}


#--------------------------------------------
# HELPERS
#--------------------------------------------


async def _retrieve(
    pipeline: AsyncPipeline,
    mode: str,
    query: str,
    top_k_before: int,
    top_k_after: int,
) -> list[Document]:

    inputs = {
        "embedder": {"text": query},
        "retriever": {"top_k": top_k_before},
        "reranker": {"query": query, "top_k": top_k_after},
    }

    if mode == "hybrid":
        inputs = {
            "dense_embedder": {"text": query},
            "sparse_embedder": {"text": query},
            "dense_retriever": {"top_k": top_k_before},
            "sparse_retriever": {"top_k": top_k_before},
            "reranker": {"query": query, "top_k": top_k_after},
        }

    result = await pipeline.run_async(inputs)
    return result["reranker"]["documents"]


def _score(
    ground_truth: list[list[Document]],
    retrieved: list[list[Document]],
) -> dict[str, float]:

    evaluators = {
        "recall@k (single)": DocumentRecallEvaluator(mode=RecallMode.SINGLE_HIT, document_comparison_field="id"),
        "recall@k (multi)": DocumentRecallEvaluator(mode=RecallMode.MULTI_HIT, document_comparison_field="id"),
        "mrr": DocumentMRREvaluator(document_comparison_field="id"),
        "map": DocumentMAPEvaluator(document_comparison_field="id"),
    }

    return {
        name: evaluator.run(ground_truth, retrieved)["score"]
        for name, evaluator in evaluators.items()
    }


def _report(
    metrics_by_mode: dict[str, dict[str, float]],
    n_queries: int,
    top_k_after: int,
) -> None:

    metrics = list(next(iter(metrics_by_mode.values())))
    header = f"{'mode':10}" + "".join(f"{metric:>20}" for metric in metrics)

    print(f"\nRetrieval eval — {n_queries} queries, k={top_k_after}\n")
    print(header)
    print("-" * len(header))
    for mode, scores in metrics_by_mode.items():
        print(f"{mode:10}" + "".join(f"{scores[metric]:>20.4f}" for metric in metrics))
    print()


#--------------------------------------------
# EVALUATION
#--------------------------------------------


async def evaluate(
    golden_set_path: Path,
    modes: list[str],
    top_k_before: int,
    top_k_after: int,
) -> None:

    golden = loads(golden_set_path.read_text(encoding="utf-8"))
    queries = [item["query"] for item in golden]
    ground_truth = [
        [Document(id=chunk_id) for chunk_id in item["relevant_chunk_ids"]]
        for item in golden
    ]

    document_store = build_document_store()

    metrics_by_mode: dict[str, dict[str, float]] = {}

    for mode in modes:
        pipeline = BUILDERS[mode](document_store)
        retrieved = [
            await _retrieve(pipeline, mode, query, top_k_before, top_k_after)
            for query in queries
        ]
        metrics_by_mode[mode] = _score(ground_truth, retrieved)
        logger.info(f"Evaluated {mode} over {len(queries)} queries")

    _report(metrics_by_mode, len(queries), top_k_after)


async def main():

    parser = ArgumentParser(
        description="Evaluate retrieval (Recall / MRR / MAP) against a golden set")

    parser.add_argument("golden_set", nargs="?", type=Path, default=Path("eval-golden-set.json"),
        help='JSON golden set: [{"query": ..., "relevant_chunk_ids": [...]}, ...]. Defaults to eval-golden-set.json')
    parser.add_argument("--modes", nargs="+", choices=tuple(BUILDERS), default=list(BUILDERS),
        help="Retrieval modes to evaluate. Defaults to all.")
    parser.add_argument("--top-k-before", type=int, default=30,
        help="Candidate chunks retrieved before reranking. Defaults to 30.")
    parser.add_argument("--top-k-after", type=int, default=5,
        help="Chunks kept after reranking (the k in recall@k). Defaults to 5.")

    args = parser.parse_args()

    await evaluate(args.golden_set, args.modes, args.top_k_before, args.top_k_after)


if __name__ == "__main__":
    run(main())
