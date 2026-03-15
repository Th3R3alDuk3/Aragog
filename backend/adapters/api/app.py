from contextlib import asynccontextmanager
from logging import INFO, basicConfig, getLogger

from fastapi.middleware.cors import CORSMiddleware
from fastapi_offline import FastAPIOffline as FastAPI
from uvicorn import run

from adapters.api.routes.documents import router as documents_router
from adapters.api.routes.evaluation import router as evaluation_router
from adapters.api.routes.query import router as query_router
from adapters.api.routes.tasks import router as tasks_router
from core.config import get_settings
from core.runtime import managed_runtime

basicConfig(
    format="%(asctime)s %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
    level=INFO,
)

logger = getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Advanced RAG API starting up")
    async with managed_runtime() as runtime:
        app.state.runtime = runtime
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
        "Optional: HyDE · RAPTOR · CRAG · ColBERT · RAGAS."
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
app.include_router(evaluation_router)
app.include_router(tasks_router)


def main() -> None:
    settings = get_settings()
    run(app, host=settings.app_host, port=settings.app_port)
