from dotenv import load_dotenv
load_dotenv()

from argparse import ArgumentParser
from asyncio import Semaphore, gather, run
from datetime import datetime, timezone
from itertools import batched
from pathlib import Path

from config import get_settings
from pipelines.indexing import build_indexing_pipeline
from services.storage import MinioStore


#--------------------------------------------
# GLOBALS
#--------------------------------------------


settings = get_settings()

minio_store = MinioStore(
    settings.minio_endpoint,
    settings.minio_user,
    settings.minio_password,
    settings.minio_bucket,
)

indexing_pipeline = build_indexing_pipeline()


#--------------------------------------------
# INDEXING
#--------------------------------------------


async def index(file_paths: list[Path], semaphore: Semaphore, idx: int, total: int) -> None:

    async with semaphore:

        for file_path in file_paths:
            await minio_store.upload(file_path, file_path.name)

        result = await indexing_pipeline.run_async({
            "converter": {
                "sources": file_paths,
                "meta": [{
                    "source": file_path.name,
                    "created_at": datetime.fromtimestamp(
                        file_path.stat().st_mtime, tz=timezone.utc).isoformat(),
                } for file_path in file_paths],
            },
        })

        chunks_written = result.get("writer", {}).get("documents_written", 0)
        print("+---------------------------------------------------------------")
        print(f"| [{idx}/{total}] {len(file_paths)} file(s) → {chunks_written} chunk(s)")
        print("+---------------------------------------------------------------")


async def main():

    parser = ArgumentParser(description="Index documents into the Qdrant document store")

    parser.add_argument("file_paths", nargs="+", type=Path,
        help="Paths to the files to be indexed")
    parser.add_argument("-c", "--concurrency", type=int, default=3,
        help="Number of batches to index concurrently")
    parser.add_argument("-b", "--batch-size", type=int, default=3,
        help="Number of files per indexing batch")

    args = parser.parse_args()

    semaphore = Semaphore(args.concurrency)
    batches = [list(batch) for batch in batched(args.file_paths, args.batch_size)]
    total = len(batches)

    await gather(*[
        index(batch, semaphore, idx, total)
        for idx, batch in enumerate(batches, 1)
    ])


if __name__ == "__main__":
    run(main())
