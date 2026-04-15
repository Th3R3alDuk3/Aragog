import asyncio
import hashlib
import logging
import tempfile
from asyncio import Semaphore
from datetime import datetime, timezone
from inspect import isawaitable
from pathlib import Path
from uuid import uuid4

from haystack import Document
from haystack.core.pipeline.async_pipeline import AsyncPipeline

from core.models.indexing import IndexCommand, IndexResponse
from core.models.tasks import TaskInfo, TaskState, TaskStep
from core.storage.minio_store import MinioStore
from core.storage.qdrant_store import QdrantStores
from core.storage.task_store import TaskStore

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
    ("injecting_context", "Kontext-Präfix injizieren"),
    ("embedding_sparse", "Sparse Embeddings berechnen"),
    ("embedding_dense", "Dense Embeddings berechnen"),
    ("writing", "Child-Chunks schreiben"),
]


class IndexingService:
    def __init__(
        self,
        pipeline: AsyncPipeline,
        qdrant_stores: QdrantStores,
        minio_store: MinioStore,
        task_store: TaskStore,
        semaphore: Semaphore,
    ) -> None:
        self.pipeline = pipeline
        self.qdrant_stores = qdrant_stores
        self.minio_store = minio_store
        self.task_store = task_store
        self.semaphore = semaphore
        self._commands: dict[str, IndexCommand] = {}
        self._run_options: dict[str, dict] = {}

    def enqueue(self, command: IndexCommand, *, use_raptor: bool = True) -> TaskInfo:
        now = datetime.now(timezone.utc)
        task = TaskState(
            task_id=str(uuid4()),
            status="pending",
            step="pending",
            current_step_index=-1,
            steps=self.build_steps(use_raptor=use_raptor),
            source=command.file_name,
            created_at=now,
            updated_at=now,
        )
        self.task_store[task.task_id] = task
        self._commands[task.task_id] = command
        self._run_options[task.task_id] = {"use_raptor": use_raptor}
        return TaskInfo(task_id=task.task_id, source=command.file_name)

    async def run(self, task_id: str) -> None:
        task = self.task_store.get(task_id)
        command = self._commands.get(task_id)
        if task is None or command is None:
            raise KeyError(f"Task '{task_id}' not found.")

        options = self._run_options.pop(task_id, {})
        try:
            async with self.semaphore:
                await self._index(task, command, use_raptor=options.get("use_raptor", True))
        finally:
            self._commands.pop(task_id, None)

    def build_steps(self, *, use_raptor: bool = True) -> list[TaskStep]:
        include_raptor = use_raptor and self._has_component("raptor")
        steps: list[TaskStep] = []

        for key, label in INDEXING_STEP_LABELS:
            if key == "summarizing_raptor" and not include_raptor:
                continue
            steps.append(TaskStep(key=key, label=label, index=len(steps), status="pending"))

        return steps

    def get_task(self, task_id: str) -> TaskState | None:
        return self.task_store.get(task_id)

    def _has_component(self, component_name: str) -> bool:
        try:
            self.pipeline.get_component(component_name)
            return True
        except Exception:
            return False

    def _step_index(self, task: TaskState, step: str) -> int:
        for item in task.steps:
            if item.key == step:
                return item.index
        return -1

    def _apply_step_statuses(
        self,
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

    def _advance(self, task: TaskState, step: str) -> None:
        step_index = self._step_index(task, step)
        task.step = step
        task.current_step_index = step_index
        self._apply_step_statuses(task, running=step_index, done_through=step_index - 1)
        task.updated_at = datetime.now(timezone.utc)

    def _finish(self, task: TaskState) -> None:
        last_index = len(task.steps) - 1
        task.status = "done"
        task.step = "done"
        task.current_step_index = last_index
        self._apply_step_statuses(task, done_through=last_index)
        task.updated_at = datetime.now(timezone.utc)

    def _fail(self, task: TaskState, error: Exception) -> None:
        failed_index = task.current_step_index
        if failed_index < 0 and task.steps:
            failed_index = 0

        task.status = "failed"
        task.step = "failed"
        task.error = str(error)
        task.current_step_index = failed_index
        self._apply_step_statuses(task, failed=failed_index, done_through=failed_index - 1)
        task.updated_at = datetime.now(timezone.utc)

    async def _run_component(self, task: TaskState, step: str, component_name: str, **kwargs):
        self._advance(task, step)
        comp = self.pipeline.get_component(component_name)
        result = await asyncio.to_thread(self._call_component, comp, **kwargs)
        if isawaitable(result):
            return await result
        return result

    def _call_component(self, component, **kwargs):
        if hasattr(component, "warm_up"):
            component.warm_up()
        return component.run(**kwargs)

    def _delete_stale_documents(self, doc_id: str) -> None:
        stale_filter = {"field": "meta.doc_id", "operator": "==", "value": doc_id}
        for store in (self.qdrant_stores.children, self.qdrant_stores.parents):
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
        self,
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

    async def _index(self, task: TaskState, command: IndexCommand, *, use_raptor: bool = True) -> None:
        try:
            task.status = "running"
            indexed_at = datetime.now(timezone.utc)
            task.updated_at = indexed_at

            self._advance(task, "uploading_minio")
            file_hash = hashlib.sha256(command.file_bytes).hexdigest()[:16]
            blob_key = f"{file_hash}-{command.file_name}"
            try:
                with tempfile.NamedTemporaryFile(delete=False, suffix=Path(command.file_name).suffix) as tmp:
                    tmp.write(command.file_bytes)
                    tmp_upload_path = tmp.name
                try:
                    await asyncio.to_thread(self.minio_store.upload, tmp_upload_path, blob_key)
                finally:
                    Path(tmp_upload_path).unlink(missing_ok=True)
            except Exception as exc:
                logger.warning("Blob upload failed: %s", exc)
                blob_key = ""

            extra_meta: dict[str, str] = {}
            if blob_key:
                extra_meta["minio_key"] = blob_key

            with tempfile.TemporaryDirectory() as tmpdir:
                tmp_path = Path(tmpdir) / command.file_name
                tmp_path.write_bytes(command.file_bytes)

                docs = (await self._run_component(task, "converting", "converter", paths=[str(tmp_path)]))["documents"]
                docs = (
                    await self._run_component(task, "enriching_metadata", "document_analyzer", documents=docs)
                )["documents"]

                self._advance(task, "deleting_stale")
                for doc in docs:
                    doc_id = doc.meta.get("doc_id")
                    if doc_id:
                        await asyncio.to_thread(self._delete_stale_documents, doc_id)

                docs = (await self._run_component(task, "cleaning", "cleaner", documents=docs))["documents"]
                split_result = await self._run_component(
                    task,
                    "splitting_chunks",
                    "parent_child_splitter",
                    documents=docs,
                )
                children = split_result["children"]
                parents = split_result["parents"]

                self._stamp_ingestion_metadata(parents, indexed_at, extra_meta)
                self._stamp_ingestion_metadata(children, indexed_at, extra_meta)

                for parent in parents:
                    parent.meta.pop("doc_beginning", None)
                await self._run_component(task, "writing_parents", "parents_writer", documents=parents)

                docs = (
                    await self._run_component(task, "enriching_chunks", "chunk_annotator", documents=children)
                )["documents"]
                docs = (await self._run_component(task, "analyzing_content", "chunk_analyzer", documents=docs))[
                    "documents"
                ]

                if use_raptor:
                    try:
                        pre_raptor_ids = {d.id for d in docs}
                        docs = (
                            await self._run_component(task, "summarizing_raptor", "raptor", documents=docs)
                        )["documents"]
                        # ChunkAnalyzer ran before RAPTOR — apply it to the newly
                        # generated summary chunks so they get context_prefix,
                        # summary, keywords, classification and NER too.
                        raptor_new = [d for d in docs if d.id not in pre_raptor_ids]
                        if raptor_new:
                            analyzer = self.pipeline.get_component("chunk_analyzer")
                            analyzed = await analyzer.run(documents=raptor_new)
                            originals = [d for d in docs if d.id in pre_raptor_ids]
                            docs = originals + analyzed["documents"]
                    except ValueError:
                        pass

                docs = (
                    await self._run_component(task, "injecting_context", "context_injector", documents=docs)
                )["documents"]
                docs = (await self._run_component(task, "embedding_sparse", "sparse_embedder", documents=docs))[
                    "documents"
                ]
                docs = (await self._run_component(task, "embedding_dense", "dense_embedder", documents=docs))[
                    "documents"
                ]
                written = (
                    await self._run_component(task, "writing", "children_writer", documents=docs)
                ).get("documents_written", 0)

            self._finish(task)
            task.result = IndexResponse(indexed=written, source=command.file_name)
        except Exception as exc:
            logger.exception("Indexing task failed for '%s'", command.file_name)
            self._fail(task, exc)
