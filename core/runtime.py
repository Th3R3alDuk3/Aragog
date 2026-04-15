from __future__ import annotations

from asyncio import Semaphore
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from logging import getLogger

from core.components.query_analyzer import QueryAnalyzer
from core.config import Settings, get_settings
from core.pipelines.generation import build_generation_pipeline
from core.pipelines.indexing import build_indexing_pipeline
from core.pipelines.retrieval import build_retrieval_pipeline
from core.services.indexing_service import IndexingService
from core.services.query_engine import QueryEngine
from core.services.retrieval_engine import RetrievalEngine
from core.storage.minio_store import MinioStore
from core.storage.qdrant_store import QdrantStores
from core.storage.task_store import TaskStore

logger = getLogger(__name__)


class RagRuntime:
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self.minio_store: MinioStore | None = None
        self.task_store = TaskStore(self.settings.task_store_size)
        self.retrieval_engine: RetrievalEngine | None = None
        self.query_engine: QueryEngine | None = None
        self.indexing_service: IndexingService | None = None
        self._indexing_semaphore = Semaphore(self.settings.indexing_max_concurrent)

    async def startup(self) -> None:
        indexing_pipeline, children_store, parents_store = build_indexing_pipeline(self.settings)
        retrieval_pipeline = build_retrieval_pipeline(self.settings, children_store, parents_store)
        retrieval_pipeline_hyde = (
            build_retrieval_pipeline(self.settings, children_store, parents_store, with_hyde=True)
            if self.settings.hyde_enabled
            else retrieval_pipeline
        )
        generation_pipeline = build_generation_pipeline(self.settings)

        logger.info(
            "Qdrant connected (%s, children: %s, parents: %s)",
            self.settings.qdrant_url,
            self.settings.qdrant_children_collection,
            self.settings.qdrant_parents_collection,
        )

        minio_store = MinioStore(
            endpoint=self.settings.minio_endpoint,
            access_key=self.settings.minio_access_key,
            secret_key=self.settings.minio_secret_key,
            bucket=self.settings.minio_bucket,
            secure=self.settings.minio_secure,
        )
        logger.info(
            "MinIO connected (%s, bucket: %s)",
            self.settings.minio_endpoint,
            self.settings.minio_bucket,
        )

        query_analyzer = QueryAnalyzer(
            openai_url=self.settings.openai_url,
            openai_api_key=self.settings.openai_api_key,
            llm_model=self.settings.effective_instruct_model,
            taxonomy=self.settings.classification_taxonomy,
        )
        if self.settings.hyde_enabled:
            logger.info("HyDE enabled (model: %s)", self.settings.effective_instruct_model)

        qdrant_stores = QdrantStores(children=children_store, parents=parents_store)
        self.minio_store = minio_store
        self.retrieval_engine = RetrievalEngine(
            settings=self.settings,
            retrieval_pipeline=retrieval_pipeline,
            retrieval_pipeline_hyde=retrieval_pipeline_hyde,
            analyzer=query_analyzer,
        )
        self.query_engine = QueryEngine(
            settings=self.settings,
            generation_pipeline=generation_pipeline,
            retrieval_engine=self.retrieval_engine,
        )
        self.indexing_service = IndexingService(
            pipeline=indexing_pipeline,
            qdrant_stores=qdrant_stores,
            minio_store=minio_store,
            task_store=self.task_store,
            semaphore=self._indexing_semaphore,
        )

    async def shutdown(self) -> None:
        return None


@asynccontextmanager
async def managed_runtime(settings: Settings | None = None) -> AsyncIterator[RagRuntime]:
    runtime = RagRuntime(settings)
    await runtime.startup()
    try:
        yield runtime
    finally:
        await runtime.shutdown()
