from dotenv import load_dotenv
load_dotenv()

from argparse import ArgumentParser
from asyncio import Semaphore, gather, run
from json import dumps, loads
from pathlib import Path
from random import Random

from haystack import AsyncPipeline, Document
from haystack.components.evaluators import (
    DocumentMAPEvaluator,
    DocumentMRREvaluator,
    DocumentRecallEvaluator,
)
from haystack.components.evaluators.document_recall import RecallMode
from haystack.components.generators.chat import OpenAIChatGenerator
from haystack.dataclasses import ChatMessage
from haystack.utils import Secret

from config import get_settings
from pipelines._factories import build_document_store
from pipelines.retrieval import (
    build_dense_retrieval_pipeline,
    build_hybrid_retrieval_pipeline,
    build_sparse_retrieval_pipeline,
)


#--------------------------------------------
# GLOBALS
#--------------------------------------------


settings = get_settings()

document_store = build_document_store()

BUILDERS = {
    "dense": build_dense_retrieval_pipeline,
    "sparse": build_sparse_retrieval_pipeline,
    "hybrid": build_hybrid_retrieval_pipeline,
}


#--------------------------------------------
# GOLDEN SET
#--------------------------------------------


def build_question_generator() -> OpenAIChatGenerator:
    return OpenAIChatGenerator(
        api_base_url=settings.enricher_url,
        api_key=Secret.from_token(settings.enricher_token),
        model=settings.enricher_model,
        timeout=settings.enricher_timeout,
        generation_kwargs={"temperature": 0},
    )


async def generate_question(content: str, generator: OpenAIChatGenerator,
        semaphore: Semaphore) -> str:

    prompt = (
        f"Below is an excerpt from a document. Write ONE specific question in "
        f"{settings.enricher_language} that is answerable solely from this excerpt — "
        f"the kind of question a real user would ask. Output only the question.\n\n"
        f"<excerpt>\n{content}\n</excerpt>"
    )

    async with semaphore:
        result = await generator.run_async(messages=[ChatMessage.from_user(prompt)])

    return result["replies"][0].text.strip()


async def generate(output: Path, num_questions: int, seed: int,
        min_chars: int, concurrency: int) -> None:

    documents = await document_store.filter_documents_async()
    documents = [
        doc for doc in documents
        if doc.content and len(doc.content.strip()) >= min_chars
    ]

    if not documents:
        print("No usable chunks in the index — index some documents first.")
        return

    Random(seed).shuffle(documents)
    sample = documents[:num_questions]

    generator = build_question_generator()
    semaphore = Semaphore(concurrency)
    questions = await gather(*[
        generate_question(doc.content, generator, semaphore) for doc in sample
    ])

    golden = [
        {"query": question, "relevant_chunk_ids": [doc.id]}
        for question, doc in zip(questions, sample)
        if question
    ]

    output.write_text(dumps(golden, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {len(golden)} query/chunk pairs to {output} "
        f"(language: {settings.enricher_language})")


#--------------------------------------------
# EVALUATION
#--------------------------------------------


async def retrieve(pipeline: AsyncPipeline, mode: str, query: str,
        top_k_before: int, top_k_after: int) -> list[Document]:

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


def score(ground_truth: list[list[Document]],
        retrieved: list[list[Document]]) -> dict[str, float]:

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


def report(metrics_by_mode: dict[str, dict[str, float]],
        n_queries: int, top_k_after: int) -> None:

    metrics = list(next(iter(metrics_by_mode.values())))
    header = f"{'mode':10}" + "".join(f"{metric:>20}" for metric in metrics)

    print(f"\nRetrieval eval — {n_queries} queries, k={top_k_after}\n")
    print(header)
    print("-" * len(header))
    for mode, scores in metrics_by_mode.items():
        print(f"{mode:10}" + "".join(f"{scores[metric]:>20.4f}" for metric in metrics))
    print()


async def evaluate(golden_set_path: Path, modes: list[str],
        top_k_before: int, top_k_after: int) -> None:

    golden = loads(golden_set_path.read_text(encoding="utf-8"))
    queries = [item["query"] for item in golden]
    ground_truth = [
        [Document(id=chunk_id) for chunk_id in item["relevant_chunk_ids"]]
        for item in golden
    ]

    metrics_by_mode: dict[str, dict[str, float]] = {}

    for mode in modes:
        pipeline = BUILDERS[mode](document_store)
        retrieved = [
            await retrieve(pipeline, mode, query, top_k_before, top_k_after)
            for query in queries
        ]
        metrics_by_mode[mode] = score(ground_truth, retrieved)
        print(f"Evaluated {mode} over {len(queries)} queries")

    report(metrics_by_mode, len(queries), top_k_after)


#--------------------------------------------
# CLI
#--------------------------------------------


async def main():

    parser = ArgumentParser(
        description="Generate a synthetic golden set, then evaluate retrieval against it")
    subparsers = parser.add_subparsers(dest="command", required=True)

    gen = subparsers.add_parser("generate",
        help="Generate a synthetic golden set (query -> chunk) from the live index")
    gen.add_argument("-o", "--output", type=Path, default=Path("eval-golden-set.json"),
        help="Where to write the golden set")
    gen.add_argument("-n", "--num-questions", type=int, default=50,
        help="Number of chunks to sample into questions")
    gen.add_argument("-s", "--seed", type=int, default=42,
        help="Sampling seed for reproducibility")
    gen.add_argument("--min-chars", type=int, default=100,
        help="Skip chunks shorter than this many characters")
    gen.add_argument("--concurrency", type=int, default=5,
        help="Max concurrent LLM calls")

    ev = subparsers.add_parser("run",
        help="Evaluate retrieval (Recall / MRR / MAP) against a golden set")
    ev.add_argument("golden_set", nargs="?", type=Path, default=Path("eval-golden-set.json"),
        help="JSON golden set")
    ev.add_argument("--modes", nargs="+", choices=tuple(BUILDERS), default=list(BUILDERS),
        help="Retrieval modes to evaluate")
    ev.add_argument("--top-k-before", type=int, default=30,
        help="Candidate chunks retrieved before reranking")
    ev.add_argument("--top-k-after", type=int, default=5,
        help="Chunks kept after reranking (the k in recall@k)")

    args = parser.parse_args()

    if args.command == "generate":
        await generate(args.output, args.num_questions, args.seed,
            args.min_chars, args.concurrency)
    else:
        await evaluate(args.golden_set, args.modes, args.top_k_before, args.top_k_after)


if __name__ == "__main__":
    run(main())
