from asyncio import Semaphore
from contextlib import asynccontextmanager
from logging import INFO, basicConfig, getLogger
from fastapi.middleware.cors import CORSMiddleware
from fastapi_offline import FastAPIOffline as FastAPI
from services.minio_store import MinioStore
from components.query_analyzer import QueryAnalyzer
from services.tasks import BoundedTaskStore
from config import get_settings
from pipelines.generation import build_generation_pipeline
from pipelines.indexing import build_indexing_pipeline
from pipelines.retrieval import build_retrieval_pipeline
from routers.documents import router as documents_router
from routers.evaluation import router as evaluation_router
from routers.query import router as query_router
from routers.stream import router as stream_router
from routers.tasks import router as tasks_router


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------


basicConfig(
    format="%(asctime)s %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%H:%M:%S", level=INFO
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

    indexing_pipeline, document_store = build_indexing_pipeline(settings)
    retrieval_pipeline                = build_retrieval_pipeline(settings, document_store)
    generation_pipeline               = build_generation_pipeline(settings)

    # MinIO: stores original uploaded documents.

    minio_store = MinioStore(
        endpoint   = settings.minio_endpoint,
        access_key = settings.minio_access_key,
        secret_key = settings.minio_secret_key,
        bucket     = settings.minio_bucket,
        secure     = settings.minio_secure,
    )

    logger.info("MinIO connected (%s, bucket: %s)", 
        settings.minio_endpoint, settings.minio_bucket)

    # QueryAnalyzer: decompose + extract metadata filters in one LLM call.

    query_analyzer = QueryAnalyzer(
        openai_base_url = settings.openai_base_url,
        openai_api_key  = settings.openai_api_key,
        llm_model       = settings.llm_model,
        taxonomy        = settings.classification_taxonomy,
    )

    # HyDE: optional — only instantiated when HYDE_ENABLED=true.

    hyde_generator = None

    if settings.hyde_enabled:

        from components.hyde_generator import HyDEGenerator
        hyde_generator = HyDEGenerator(
            openai_base_url = settings.openai_base_url,
            openai_api_key  = settings.openai_api_key,
            llm_model       = settings.llm_model,
        )

        logger.info("HyDE enabled (model: %s)",
            settings.llm_model)

    # ColBERT: optional — loads ~500 MB model when COLBERT_ENABLED=true.

    colbert_reranker = None

    if settings.colbert_enabled:

        from components.colbert_reranker import ColBERTReranker
        colbert_reranker = ColBERTReranker(
            model_name = settings.colbert_model,
            top_k      = settings.colbert_top_k,
            device     = settings.colbert_device,
        )

        logger.info("ColBERT reranker enabled (model: %s)",
            settings.colbert_model)

    app.state.settings            = settings
    app.state.document_store      = document_store
    app.state.minio_store         = minio_store
    app.state.indexing_pipeline   = indexing_pipeline
    app.state.indexing_semaphore  = Semaphore(settings.indexing_max_concurrent)
    app.state.retrieval_pipeline  = retrieval_pipeline
    app.state.generation_pipeline = generation_pipeline
    app.state.query_analyzer      = query_analyzer
    app.state.hyde_generator      = hyde_generator
    app.state.colbert_reranker    = colbert_reranker
    app.state.tasks               = BoundedTaskStore(settings.task_store_size)

    yield

    logger.info("Advanced RAG API stopped")


app = FastAPI(
    lifespan=lifespan,
    title="Advanced Hybrid RAG API",
    version="0.1.0",
    description=(
        "Hybrid Retrieval-Augmented Generation service built on FastAPI and Haystack. "
        "Documents are indexed into Qdrant using dense (BGE-M3) and sparse (SPLADE) embeddings "
        "with Anthropic Contextual Retrieval chunking (contextual prefix + parent-child). "
        "Queries are answered via hybrid retrieval, cross-encoder reranking, "
        "and an OpenAI-compatible LLM. "
        "Optional: HyDE · RAPTOR · CRAG · ColBERT · RAGAS · SSE streaming."
    ),
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=True,
)

app.include_router(documents_router)
app.include_router(query_router)
app.include_router(stream_router)
app.include_router(evaluation_router)
app.include_router(tasks_router)
