"""
Indexing service — pipeline orchestration for document indexing.

Handles all business logic for indexing documents into Qdrant:
  uploading_minio → converting → enriching_metadata → deleting_stale
  → cleaning → splitting_chunks → [writing_parents]
  → enriching_chunks → analyzing_content → [summarizing_raptor]
  → embedding_dense → embedding_sparse → writing → done
"""

import asyncio
import hashlib
import logging
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from haystack import Document
from haystack.core.pipeline.async_pipeline import AsyncPipeline
from haystack_integrations.document_stores.qdrant import QdrantDocumentStore

from models.api import IndexResponse, TaskState, TaskStep
from services.minio_store import MinioStore

logger = logging.getLogger(__name__)


INDEXING_STEP_LABELS = [
    ("uploading_minio", "Datei ablegen"),
    ("converting", "Dokument konvertieren"),
    ("enriching_metadata", "Metadaten anreichern"),
    ("deleting_stale", "Vorherige Version entfernen"),
    ("cleaning", "Dokument bereinigen"),
    ("splitting_chunks", "Parent-Child-Chunks erzeugen"),
    ("writing_parents", "Parent-Chunks speichern"),
    ("enriching_chunks", "Chunk-Kontext anreichern"),
    ("analyzing_content", "Chunk-Inhalte analysieren"),
    ("summarizing_raptor", "Hierarchische Zusammenfassungen erzeugen"),
    ("embedding_dense", "Dense Embeddings berechnen"),
    ("embedding_sparse", "Sparse Embeddings berechnen"),
    ("writing", "Child-Chunks schreiben"),
]


# ---------------------------------------------------------------------------
# Task definition
# ---------------------------------------------------------------------------

def build_indexing_steps(pipeline: AsyncPipeline) -> list[TaskStep]:
    include_raptor = _has_component(pipeline, "raptor")
    steps: list[TaskStep] = []

    for key, label in INDEXING_STEP_LABELS:
        if key == "summarizing_raptor" and not include_raptor:
            continue
        steps.append(
            TaskStep(
                key=key,
                label=label,
                index=len(steps),
                status="pending",
            )
        )

    return steps


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _has_component(pipeline: AsyncPipeline, component_name: str) -> bool:
    try:
        pipeline.get_component(component_name)
        return True
    except Exception:
        return False


def _step_index(task: TaskState, step: str) -> int:
    for item in task.steps:
        if item.key == step:
            return item.index
    return -1


def _apply_step_statuses(
    task: TaskState,
    *,
    running: int | None = None,
    failed: int | None = None,
    done_through: int = -1,
) -> None:
    for item in task.steps:
        if failed is not None and item.index == failed:
            item.status = "failed"
        elif item.index <= done_through:
            item.status = "done"
        elif running is not None and item.index == running:
            item.status = "running"
        else:
            item.status = "pending"


def _advance(task: TaskState, step: str) -> None:
    step_index = _step_index(task, step)
    task.step = step
    task.current_step_index = step_index
    _apply_step_statuses(task, running=step_index, done_through=step_index - 1)
    task.updated_at = datetime.now(timezone.utc)


def _finish(task: TaskState) -> None:
    last_index = len(task.steps) - 1
    task.status = "done"
    task.step = "done"
    task.current_step_index = last_index
    _apply_step_statuses(task, done_through=last_index)
    task.updated_at = datetime.now(timezone.utc)


def _fail(task: TaskState, error: Exception) -> None:
    failed_index = task.current_step_index

    if failed_index < 0 and task.steps:
        failed_index = 0

    task.status = "failed"
    task.step = "failed"
    task.error = str(error)
    task.current_step_index = failed_index
    _apply_step_statuses(task, failed=failed_index, done_through=failed_index - 1)
    task.updated_at = datetime.now(timezone.utc)


def _run_component(
    pipeline: AsyncPipeline,
    task: TaskState,
    step: str,
    component_name: str,
    **kwargs: Any,
) -> dict[str, Any]:
    """Advance task step, warm up component if needed, then run it.

    Handles both sync and async component run() methods. asyncio.run() is safe
    here because _run_component is only ever called from _sync_index, which runs
    inside a thread-pool executor (no running event loop in that thread).
    """
    _advance(task, step)
    comp = pipeline.get_component(component_name)
    if hasattr(comp, "warm_up"):
        comp.warm_up()
    result = comp.run(**kwargs)
    if asyncio.iscoroutine(result):
        result = asyncio.run(result)
    return result


def _delete_stale_documents(
    stores: tuple[QdrantDocumentStore, QdrantDocumentStore],
    doc_id: str,
) -> None:
    stale_filter = {"field": "meta.doc_id", "operator": "==", "value": doc_id}
    for store in stores:
        try:
            stale = store.filter_documents(filters=stale_filter)
            if stale:
                store.delete_documents([doc.id for doc in stale])
        except Exception as exc:
            logger.warning(
                "Could not delete stale docs in '%s' for doc_id '%s': %s",
                store.index,
                doc_id,
                exc,
            )


def _stamp_ingestion_metadata(
    docs: list[Document],
    indexed_at: datetime,
    extra_meta: dict[str, str],
) -> None:
    indexed_at_iso = indexed_at.isoformat()
    indexed_at_ts = int(indexed_at.timestamp())

    for doc in docs:
        doc.meta["indexed_at"] = indexed_at_iso
        doc.meta["indexed_at_ts"] = indexed_at_ts
        for key, value in extra_meta.items():
            doc.meta[key] = value


def _sync_index(
    task: TaskState,
    pipeline: AsyncPipeline,
    children_store: QdrantDocumentStore,
    parents_store: QdrantDocumentStore,
    minio_store: MinioStore,
    file_bytes: bytes,
    original_name: str,
) -> None:
    """
    Synchronous indexing — intended to run inside a thread-pool executor
    so it does not block the asyncio event loop.
    """
    try:
        task.status = "running"
        indexed_at = datetime.now(timezone.utc)
        task.updated_at = indexed_at

        # ── Pre-pipeline: upload to MinIO ─────────────────────────────────────
        _advance(task, "uploading_minio")
        minio_key: str | None = None
        file_hash = hashlib.sha256(file_bytes).hexdigest()[:16]
        minio_key = f"{file_hash}-{original_name}"
        try:
            with tempfile.NamedTemporaryFile(
                delete=False, suffix=Path(original_name).suffix
            ) as tmp:
                tmp.write(file_bytes)
                tmp_upload_path = tmp.name
            try:
                minio_store.upload(tmp_upload_path, minio_key)
            finally:
                Path(tmp_upload_path).unlink(missing_ok=True)
        except Exception as exc:
            logger.warning("MinIO upload failed: %s", exc)
            minio_key = None

        extra_meta: dict = {}
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
            )["documents"]

            # Delete only the exact same logical document, not every file that
            # happens to share the same filename.
            _advance(task, "deleting_stale")
            for doc in docs:
                doc_id = doc.meta.get("doc_id")
                if doc_id:
                    _delete_stale_documents((children_store, parents_store), doc_id)

            # Stage 3 — DocumentCleaner: normalise whitespace
            docs = _run_component(
                pipeline, task, "cleaning", "cleaner",
                documents=docs,
            )["documents"]

            # Stage 4 — ParentChildSplitter: children (→ children collection) + parents (→ parents collection)
            split_result = _run_component(
                pipeline, task, "splitting_chunks", "parent_child_splitter",
                documents=docs,
            )
            children = split_result["children"]
            parents  = split_result["parents"]

            _stamp_ingestion_metadata(parents, indexed_at, extra_meta)
            _stamp_ingestion_metadata(children, indexed_at, extra_meta)

            # ── Parents branch: write to parents collection (no embedding needed) ────
            # Strip EphemeralMeta fields before writing — parents skip ContentAnalyzer
            # which normally excludes doc_beginning via model_dump(exclude={"doc_beginning"}).
            for p in parents:
                p.meta.pop("doc_beginning", None)
            _run_component(
                pipeline, task, "writing_parents", "parents_writer",
                documents=parents,
            )

            # ── Children branch: enrich → analyze → [raptor] → embed → write ─

            # Stage 5 — ChunkContextEnricher: chunk_index, section_path, …
            docs = _run_component(
                pipeline, task, "enriching_chunks", "chunk_enricher",
                documents=children,
            )["documents"]

            # Stage 6 — ContentAnalyzer: one LLM call/chunk (parallelised)
            docs = _run_component(
                pipeline, task, "analyzing_content", "analyzer",
                documents=docs,
            )["documents"]

            # Stage 7 — RaptorSummarizer (optional)
            try:
                docs = _run_component(
                    pipeline, task, "summarizing_raptor", "raptor",
                    documents=docs,
                )["documents"]
            except ValueError:
                pass  # component not present — RAPTOR_ENABLED=false

            # Stage 8 — SentenceTransformersDocumentEmbedder: dense vectors
            docs = _run_component(
                pipeline, task, "embedding_dense", "dense_embedder",
                documents=docs,
            )["documents"]

            # Stage 9 — FastembedSparseDocumentEmbedder: sparse vectors
            docs = _run_component(
                pipeline, task, "embedding_sparse", "sparse_embedder",
                documents=docs,
            )["documents"]

            # Stage 10 — DocumentWriter: persist children to Qdrant
            written: int = _run_component(
                pipeline, task, "writing", "children_writer",
                documents=docs,
            ).get("documents_written", 0)

        _finish(task)
        task.result = IndexResponse(indexed=written, source=original_name)

    except Exception as exc:
        logger.exception("Indexing task failed for '%s'", original_name)
        _fail(task, exc)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

async def run_indexing(
    task: TaskState,
    children_store: QdrantDocumentStore,
    parents_store: QdrantDocumentStore,
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
        task, pipeline, children_store, parents_store, minio_store, file_bytes, file_name,
    )
