from dotenv import load_dotenv

load_dotenv()

from argparse import ArgumentParser
from asyncio import Semaphore, gather, run
from datetime import UTC, datetime
from itertools import batched
from pathlib import Path

from haystack import Pipeline

from pipelines._factories import build_document_store, build_rustfs_store
from pipelines.indexing import build_indexing_pipeline
from services.rustfs import RustfsStore


#-----------------------------------------------------
# Indexing
#-----------------------------------------------------


async def index_batch(
    rustfs_store: RustfsStore,
    indexing_pipeline: Pipeline,
    file_paths: list[Path],
    semaphore: Semaphore,
    batch_num: int,
    total_batches: int,
) -> bool:

    async with semaphore:

        try:

            for file_path in file_paths:
                await rustfs_store.upload(file_path)

            result = await indexing_pipeline.run_async({
                "converter": {
                    "sources": file_paths,
                    "meta": [{
                        "source": file_path.name,
                        "modified_at": datetime.fromtimestamp(
                            file_path.stat().st_mtime, tz=UTC).isoformat(),
                    } for file_path in file_paths],
                },
            })

        except Exception as error:
            print(f"[{batch_num}/{total_batches}] {len(file_paths)} file(s) "
                f"→ FAILED: {error}")
            return False

        chunks_written = result.get("writer", {}).get("documents_written", 0)
        chunks_failed = len(
            result.get("chunk_enricher", {}).get("failed_documents", []))

        summary = (f"[{batch_num}/{total_batches}] {len(file_paths)} file(s) "
            f"→ {chunks_written} chunk(s)")
        if chunks_failed:
            summary += f", {chunks_failed} failed enrichment"
        print(summary)
        return chunks_failed == 0


async def main():

    parser = ArgumentParser(description="Index documents into the Qdrant document store")

    parser.add_argument("file_paths", nargs="+", type=Path,
        help="Paths to the files to be indexed")
    parser.add_argument("-c", "--concurrency", type=int, default=3,
        help="Number of batches to index concurrently")
    parser.add_argument("-b", "--batch-size", type=int, default=3,
        help="Number of files per indexing batch")

    args = parser.parse_args()

    if not_files := [str(path) for path in args.file_paths if not path.is_file()]:
        parser.error(f"not a file: {', '.join(not_files)}")

    rustfs_store = build_rustfs_store()
    document_store = build_document_store()
    indexing_pipeline = build_indexing_pipeline(document_store)

    semaphore = Semaphore(args.concurrency)
    batches = [list(batch) for batch in batched(args.file_paths, args.batch_size)]

    results = await gather(*[
        index_batch(rustfs_store, indexing_pipeline, batch, semaphore, batch_num, len(batches))
        for batch_num, batch in enumerate(batches, 1)
    ])

    if not all(results):
        raise SystemExit(1)


if __name__ == "__main__":
    run(main())
