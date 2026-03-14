from asyncio import Semaphore
from contextlib import asynccontextmanager
from logging import INFO, basicConfig, getLogger

from fastapi.middleware.cors import CORSMiddleware
from fastapi_offline import FastAPIOffline as FastAPI
from uvicorn import run

from components.query_analyzer import QueryAnalyzer
from config import get_settings
from pipelines.generation import build_generation_pipeline
from pipelines.indexing import build_indexing_pipeline
from pipelines.retrieval import build_retrieval_pipeline
from routers.documents import router as documents_router
from routers.evaluation import router as evaluation_router
from routers.query import router as query_router
from routers.stream import router as stream_router
from routers.tasks import router as tasks_router
from services.minio_store import MinioStore
from services.tasks import BoundedTaskStore

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------


basicConfig(
    format="%(asctime)s %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
    level=INFO,
)


logger = getLogger(__name__)


# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan — runs startup code before the first request,
    and shutdown code (after the ``yield``) when the server stops.
    """

    logger.info("Advanced RAG API starting up")
    settings = get_settings()

    # Qdrant: document store and indexing, retrieval, generation pipelines

    indexing_pipeline, children_store, parents_store = build_indexing_pipeline(settings)
    retrieval_pipeline = build_retrieval_pipeline(settings, children_store, parents_store)
    generation_pipeline = build_generation_pipeline(settings)

    logger.info(
        "Qdrant connected (%s, children: %s, parents: %s)",
        settings.qdrant_url,
        settings.qdrant_children_collection,
        settings.qdrant_parents_collection,
    )

    # MinIO: stores original uploaded documents.

    minio_store = MinioStore(
        endpoint=settings.minio_endpoint,
        access_key=settings.minio_access_key,
        secret_key=settings.minio_secret_key,
        bucket=settings.minio_bucket,
        secure=settings.minio_secure,
    )

    logger.info(
        "MinIO connected (%s, bucket: %s)",
        settings.minio_endpoint,
        settings.minio_bucket,
    )

    # QueryAnalyzer: decompose + extract metadata filters in one LLM call.

    query_analyzer = QueryAnalyzer(
        openai_url=settings.openai_url,
        openai_api_key=settings.openai_api_key,
        llm_model=settings.effective_instruct_model,
        taxonomy=settings.classification_taxonomy,
    )

    if settings.hyde_enabled:
        logger.info("HyDE enabled (model: %s)", settings.effective_instruct_model)

    app.state.settings = settings
    app.state.children_store = children_store
    app.state.parents_store = parents_store
    app.state.indexing_pipeline = indexing_pipeline
    app.state.indexing_semaphore = Semaphore(settings.indexing_max_concurrent)
    app.state.retrieval_pipeline = retrieval_pipeline
    app.state.generation_pipeline = generation_pipeline
    app.state.minio_store = minio_store
    app.state.query_analyzer = query_analyzer
    app.state.tasks = BoundedTaskStore(settings.task_store_size)

    yield

    logger.info("Advanced RAG API stopped")


app = FastAPI(
    lifespan=lifespan,
    title="Advanced Hybrid RAG API",
    version="0.1.0",
    description=(
        "Hybrid Retrieval-Augmented Generation service built on FastAPI and Haystack. "
        "Documents are indexed into Qdrant using dense and sparse embeddings "
        "with Anthropic Contextual Retrieval chunking (contextual prefix + parent-child). "
        "Queries are answered via hybrid retrieval, cross-encoder reranking, "
        "and an OpenAI-compatible LLM. "
        "Optional: HyDE · RAPTOR · CRAG · ColBERT · RAGAS · SSE streaming."
    ),
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=get_settings().cors_origins_list,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(documents_router)
app.include_router(query_router)
app.include_router(stream_router)
app.include_router(evaluation_router)
app.include_router(tasks_router)


if __name__ == "__main__":
    _settings = get_settings()
    run(app, host=_settings.app_host, port=_settings.app_port)
