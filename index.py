from argparse import ArgumentParser
from asyncio import Semaphore, gather, run
from hashlib import sha256
from itertools import batched
from json import dumps, loads
from pathlib import Path
from typing import Any

from haystack import Pipeline
from haystack_integrations.document_stores.qdrant import QdrantDocumentStore
from qdrant_client.http.models import FieldCondition, Filter, MatchValue

from pipelines._factories import build_document_store
from pipelines.indexing import build_indexing_pipeline


#-----------------------------------------------------
# Indexing
#-----------------------------------------------------


async def index_batch(
    document_store: QdrantDocumentStore,
    indexing_pipeline: Pipeline,
    exports: dict[Path, dict[str, Any]],
    semaphore: Semaphore,
    batch_num: int,
    total_batches: int,
) -> bool:

    async with semaphore:

        try:

            result = await indexing_pipeline.run_async({
                "converter": {"exports": list(exports.values())},
            })

            chunks_failed = len(
                result.get("enricher", {}).get("failed_documents", []))
            chunks_replaced = 0

            # only a fully enriched version may replace the previous one
            if not chunks_failed:
                for export in exports.values():
                    chunks_replaced += await document_store.delete_by_filter_async(
                        Filter(
                            must=[FieldCondition(
                                key="meta.url",
                                match=MatchValue(value=export["path"]),
                            )],
                            must_not=[FieldCondition(
                                key="meta.hash",
                                match=MatchValue(value=export["hash"]),
                            )],
                        ))

        except Exception as error:
            print(f"[{batch_num}/{total_batches}] {len(exports)} export(s) "
                f"→ FAILED: {error}")
            return False

        chunks_written = result.get("writer", {}).get("documents_written", 0)

        summary = (f"[{batch_num}/{total_batches}] {len(exports)} export(s) "
            f"→ {chunks_written} chunk(s)")
        if chunks_failed:
            summary += f", {chunks_failed} failed enrichment"
        if chunks_replaced:
            summary += f", {chunks_replaced} of older versions removed"
        print(summary)
        return chunks_failed == 0


async def main():

    parser = ArgumentParser(
        description="Index JSON document exports into the Qdrant document store")

    parser.add_argument("file_paths", nargs="+", type=Path,
        help="Paths to the JSON exports to be indexed")
    parser.add_argument("-c", "--concurrency", type=int, default=3,
        help="Number of batches to index concurrently")
    parser.add_argument("-b", "--batch-size", type=int, default=3,
        help="Number of exports per indexing batch")

    args = parser.parse_args()

    if args.concurrency < 1 or args.batch_size < 1:
        parser.error("concurrency and batch size must be at least 1")

    if not_exports := [
        str(path) for path in args.file_paths
        if not path.is_file() or path.suffix.lower() != ".json"
    ]:
        parser.error(f"not a JSON export: {', '.join(not_exports)}")

    exports = {
        file_path: loads(file_path.read_text(encoding="utf-8"))
        for file_path in args.file_paths
    }

    # sha256 over the export without its hash, canonical as in RFC 8785
    if wrong_hashes := [
        str(file_path) for file_path, export in exports.items()
        if export.get("hash") != sha256(dumps(
            {key: value for key, value in export.items() if key != "hash"},
            ensure_ascii=False, sort_keys=True, separators=(",", ":"),
        ).encode()).hexdigest()
    ]:
        parser.error(f"hash does not match the content: {', '.join(wrong_hashes)}")

    urls = [export["path"] for export in exports.values()]
    if duplicates := sorted({url for url in urls if urls.count(url) > 1}):
        parser.error(f"several files export the same path: {', '.join(duplicates)}")

    document_store = build_document_store()
    indexing_pipeline = build_indexing_pipeline(document_store)

    pending: dict[Path, dict[str, Any]] = {}

    # the hash identifies a version; a partly enriched one is indexed again
    for file_path, export in exports.items():
        indexed = await document_store.filter_documents_async(filters=Filter(must=[
            FieldCondition(key="meta.hash", match=MatchValue(value=export["hash"])),
        ]))
        if indexed and len(indexed) == indexed[0].meta["total_chunks"]:
            print(f"{file_path.name} → unchanged, skipped")
        else:
            pending[file_path] = export

    semaphore = Semaphore(args.concurrency)
    batches = [dict(batch) for batch in batched(pending.items(), args.batch_size)]

    results = await gather(*[
        index_batch(document_store, indexing_pipeline,
            batch, semaphore, batch_num, len(batches))
        for batch_num, batch in enumerate(batches, 1)
    ])

    await indexing_pipeline.close_async()
    await document_store.close_async()

    if not all(results):
        raise SystemExit(1)


if __name__ == "__main__":
    run(main())
