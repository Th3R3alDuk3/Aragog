"""
Indexing service — pipeline orchestration for document indexing.

Handles all business logic for indexing documents into Qdrant:
  deleting_stale → uploading_minio → converting → enriching_metadata
  → cleaning → splitting_headers → splitting_chunks → enriching_chunks
  → analyzing_content → [summarizing_raptor] → embedding_dense
  → embedding_sparse → writing → done
"""

import asyncio
import hashlib
import logging
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from haystack.core.pipeline.async_pipeline import AsyncPipeline
from haystack_integrations.document_stores.qdrant import QdrantDocumentStore

from models.schemas import IndexResponse, TaskState
from services.minio_store import MinioStore

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _advance(task: TaskState, step: str) -> None:
    task.step = step
    task.updated_at = datetime.now(timezone.utc)


def _run_component(pipeline: AsyncPipeline, task: TaskState, step: str, component_name: str, **kwargs):
    """Advance task step, warm up component if needed, then run it."""
    _advance(task, step)
    comp = pipeline.get_component(component_name)
    if hasattr(comp, "warm_up"):
        comp.warm_up()
    return comp.run(**kwargs)


def _sync_index(
    task: TaskState,
    pipeline: AsyncPipeline,
    document_store: QdrantDocumentStore,
    minio_store,
    file_bytes: bytes,
    original_name: str,
) -> None:
    """
    Synchronous indexing — intended to run inside a thread-pool executor
    so it does not block the asyncio event loop.
    """
    try:
        task.status = "running"
        task.updated_at = datetime.now(timezone.utc)

        # ── Pre-pipeline: delete stale chunks ────────────────────────────────
        _advance(task, "deleting_stale")
        try:
            stale = document_store.filter_documents(
                filters={"field": "meta.source", "operator": "==", "value": original_name}
            )
            if stale:
                document_store.delete_documents([doc.id for doc in stale])
        except Exception as exc:
            logger.warning("Could not delete stale chunks for '%s': %s", original_name, exc)

        # ── Pre-pipeline: upload to MinIO ─────────────────────────────────────
        _advance(task, "uploading_minio")
        minio_url: str | None = None
        minio_key: str | None = None
        file_hash = hashlib.sha256(file_bytes).hexdigest()[:16]
        minio_key = f"{file_hash}-{original_name}"
        try:
            with tempfile.NamedTemporaryFile(
                delete=False, suffix=Path(original_name).suffix
            ) as tmp:
                tmp.write(file_bytes)
                tmp_upload_path = tmp.name
            minio_url = minio_store.upload(tmp_upload_path, minio_key)
            Path(tmp_upload_path).unlink(missing_ok=True)
        except Exception as exc:
            logger.warning("MinIO upload failed: %s", exc)
            minio_url = None
            minio_key = None

        extra_meta: dict = {}
        if minio_url:
            extra_meta["minio_url"] = minio_url
        if minio_key:
            extra_meta["minio_key"] = minio_key

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir) / original_name
            tmp_path.write_bytes(file_bytes)

            # Stage 1 — DoclingConverter: PDF/DOCX → markdown
            docs = _run_component(
                pipeline, task, "converting", "converter",
                paths=[str(tmp_path)],
            )["documents"]

            # Stage 2 — MetadataEnricher: doc_id, title, language, …
            docs = _run_component(
                pipeline, task, "enriching_metadata", "meta_enricher",
                documents=docs,
                extra_meta=extra_meta or None,
            )["documents"]

            # Stage 3 — DocumentCleaner: normalise whitespace
            docs = _run_component(
                pipeline, task, "cleaning", "cleaner",
                documents=docs,
            )["documents"]

            # Stage 4 — MarkdownHeaderSplitter: split at H1-H6 boundaries
            docs = _run_component(
                pipeline, task, "splitting_headers", "header_splitter",
                documents=docs,
            )["documents"]

            # Stage 5 — ParentChildSplitter: child chunks + parent_content meta
            docs = _run_component(
                pipeline, task, "splitting_chunks", "parent_child_splitter",
                documents=docs,
            )["documents"]

            # Stage 6 — ChunkContextEnricher: chunk_index, section_path, …
            docs = _run_component(
                pipeline, task, "enriching_chunks", "chunk_enricher",
                documents=docs,
            )["documents"]

            # Stage 7 — ContentAnalyzer: one LLM call/chunk (parallelised)
            docs = _run_component(
                pipeline, task, "analyzing_content", "analyzer",
                documents=docs,
            )["documents"]

            # Stage 8 — RaptorSummarizer (optional)
            try:
                docs = _run_component(
                    pipeline, task, "summarizing_raptor", "raptor",
                    documents=docs,
                )["documents"]
            except ValueError:
                pass  # component not present — RAPTOR_ENABLED=false

            # Stage 9 — SentenceTransformersDocumentEmbedder: dense vectors
            docs = _run_component(
                pipeline, task, "embedding_dense", "dense_embedder",
                documents=docs,
            )["documents"]

            # Stage 10 — FastembedSparseDocumentEmbedder: sparse vectors
            docs = _run_component(
                pipeline, task, "embedding_sparse", "sparse_embedder",
                documents=docs,
            )["documents"]

            # Stage 11 — DocumentWriter: persist to Qdrant
            written: int = _run_component(
                pipeline, task, "writing", "writer",
                documents=docs,
            ).get("documents_written", 0)

        task.status = "done"
        task.step = "done"
        task.result = IndexResponse(indexed=written, source=original_name, minio_url=minio_url)
        task.updated_at = datetime.now(timezone.utc)

    except Exception as exc:
        logger.exception("Indexing task failed for '%s'", original_name)
        task.status = "failed"
        task.step = "failed"
        task.error = str(exc)
        task.updated_at = datetime.now(timezone.utc)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

async def run_indexing(
    task: TaskState,
    document_store: QdrantDocumentStore,
    minio_store: MinioStore,
    pipeline: AsyncPipeline,
    file_name: str,
    file_bytes: bytes,
) -> None:
    """Async entry point — offloads synchronous indexing to a thread-pool executor."""
    loop = asyncio.get_running_loop()
    await loop.run_in_executor(
        None,
        _sync_index,
        task, pipeline, document_store, minio_store, file_bytes, file_name,
    )
