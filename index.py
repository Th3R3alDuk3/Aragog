from argparse import ArgumentParser

from dotenv import load_dotenv
load_dotenv()

from asyncio import run
from datetime import datetime, timezone
from logging import INFO, basicConfig, getLogger
from pathlib import Path

from config import get_settings
from pipelines.indexing import build_indexing_pipeline
from services.storage import MinioStore


basicConfig(level=INFO)
logger = getLogger(__name__)


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


async def index(file_paths: list[Path]) -> None:

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
    logger.info(f"Indexed {len(file_paths)} file(s) -> {chunks_written} chunk(s)")


async def main():

    parser = ArgumentParser(description="Index documents into the Qdrant document store")

    parser.add_argument("file_paths", nargs="+", type=Path,
        help="Paths to the files to be indexed")

    args = parser.parse_args()

    await index(args.file_paths)


if __name__ == "__main__":
    run(main())
