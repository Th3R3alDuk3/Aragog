import logging
from contextlib import asynccontextmanager

from uvicorn import run
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from components.query_analyzer import QueryAnalyzer
from config import get_settings
from pipelines.ingestion import build_indexing_pipeline
from pipelines.retrieval import build_query_pipeline
from routers.documents import router as documents_router
from routers.evaluation import router as evaluation_router
from routers.query import router as query_router
from routers.stream import router as stream_router

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------
# basicConfig configures the root logger so all app loggers (which propagate
# to root by default) emit INFO messages.  log_config=None in uvicorn.run()
# prevents uvicorn from overriding this config with its own dictConfig.

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)

logger = logging.getLogger(__name__)


#


settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan — runs startup code before the first request,
    and shutdown code (after the ``yield``) when the server stops.

    Using the ``lifespan`` parameter is the modern FastAPI approach;
    the old ``@app.on_event("startup")`` decorator is deprecated since 0.93.
    """

    logger.info("Initialising Qdrant document store and pipelines …")

    # build_indexing_pipeline creates the QdrantDocumentStore internally and
    # returns it — reuse the same instance for the query pipeline to avoid
    # creating two separate Qdrant connections / collections.
    ingestion_pipeline, document_store = build_indexing_pipeline(settings)
    retrieval_pipeline, generation_pipeline = build_query_pipeline(settings, document_store)

    # QueryAnalyzer: decompose + extract metadata filters in one LLM call.
    # QueryDecomposer is an alias for backward compatibility.
    query_analyzer = QueryAnalyzer(
        openai_api_key  = settings.openai_api_key,
        llm_model       = settings.llm_model,
        openai_base_url = settings.openai_base_url,
        taxonomy        = settings.classification_taxonomy,
    )

    # HyDE: optional — only instantiated when HYDE_ENABLED=true.
    hyde_generator = None
    if settings.hyde_enabled:
        from components.hyde_generator import HyDEGenerator
        hyde_generator = HyDEGenerator(
            openai_api_key  = settings.openai_api_key,
            llm_model       = settings.hyde_model or settings.llm_model,
            openai_base_url = settings.openai_base_url,
        )
        logger.info("HyDE enabled (model: %s)", settings.hyde_model or settings.llm_model)

    # ColBERT: optional — loads ~500 MB model when COLBERT_ENABLED=true.
    colbert_reranker = None
    if settings.colbert_enabled:
        from components.colbert_reranker import ColBERTReranker
        colbert_reranker = ColBERTReranker(
            model_name = settings.colbert_model,
            top_k      = settings.colbert_top_k,
            device     = settings.colbert_device,
        )
        logger.info("ColBERT reranker enabled (model: %s)", settings.colbert_model)

    # MinIO: stores original uploaded documents.
    minio_store = None
    try:
        from components.minio_store import MinioStore
        minio_store = MinioStore(
            endpoint   = settings.minio_endpoint,
            access_key = settings.minio_access_key,
            secret_key = settings.minio_secret_key,
            bucket     = settings.minio_bucket,
            secure     = settings.minio_secure,
        )
        logger.info("MinIO connected (%s, bucket: %s)", settings.minio_endpoint, settings.minio_bucket)
    except Exception as exc:
        logger.warning("MinIO unavailable — file storage disabled (%s)", exc)

    app.state.settings             = settings
    app.state.document_store       = document_store
    app.state.ingestion_pipeline   = ingestion_pipeline
    app.state.retrieval_pipeline   = retrieval_pipeline
    app.state.generation_pipeline  = generation_pipeline
    app.state.query_analyzer       = query_analyzer
    app.state.hyde_generator       = hyde_generator
    app.state.colbert_reranker     = colbert_reranker
    app.state.minio_store          = minio_store

    features = []
    if settings.hyde_enabled:    features.append("HyDE")
    if settings.raptor_enabled:  features.append("RAPTOR")
    if settings.crag_enabled:    features.append("CRAG")
    if settings.colbert_enabled: features.append("ColBERT")
    if settings.ragas_enabled:   features.append("RAGAS")

    logger.info(
        "All pipelines ready. Active features: %s",
        ", ".join(features) if features else "none (all flags disabled)",
    )
    yield
    logger.info("Shutting down Advanced RAG API.")


app = FastAPI(
    lifespan=lifespan,
    title="Advanced Hybrid RAG API",
    description=(
        "FastAPI + Haystack 2.x · Qdrant (dense + SPLADE sparse) · "
        "Contextual Retrieval (Anthropic Option A) · "
        "Cross-encoder reranking · Multi-question decomposition · "
        "HyDE · RAPTOR · CRAG · ColBERT · RAGAS evaluation · "
        "SSE streaming · Document extraction via docling-serve."
    ),
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(documents_router)
app.include_router(query_router)
app.include_router(stream_router)
app.include_router(evaluation_router)


if __name__ == "__main__":
    run(
        "main:app",
        host=settings.app_host,
        port=settings.app_port,
        log_level=settings.app_log_level,
        log_config=None,
        reload=True,
    )
