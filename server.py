from dotenv import load_dotenv

load_dotenv()

from asyncio import Semaphore
from collections.abc import AsyncIterator

from fastmcp import FastMCP
from fastmcp.server.auth.providers.jwt import JWTVerifier
from fastmcp.server.dependencies import get_access_token
from fastmcp.server.lifespan import lifespan as composable_lifespan
from fastmcp.server.middleware.rate_limiting import RateLimitingMiddleware
from fastmcp.utilities.logging import configure_logging

from config import get_settings
from pipelines._factories import build_document_store, build_rustfs_store
from pipelines.retrieval import (
    build_dense_retrieval_pipeline,
    build_hybrid_retrieval_pipeline,
    build_sparse_retrieval_pipeline,
)
from tools import TOOLS

configure_logging()


#-----------------------------------------------------
# Globals
#-----------------------------------------------------


settings = get_settings()


#-----------------------------------------------------
# Server
#-----------------------------------------------------


@composable_lifespan
async def lifespan(server: FastMCP) -> AsyncIterator[dict]:

    document_store = build_document_store()
    dense_pipeline = build_dense_retrieval_pipeline(document_store)
    sparse_pipeline = build_sparse_retrieval_pipeline(document_store)
    hybrid_pipeline = build_hybrid_retrieval_pipeline(document_store)

    try:
        yield {
            "document_store": document_store,
            "rustfs_store": build_rustfs_store(),
            "dense_pipeline": dense_pipeline,
            "sparse_pipeline": sparse_pipeline,
            "hybrid_pipeline": hybrid_pipeline,
            "search_limiter": Semaphore(settings.search_max_concurrency),
        }
    finally:

        await document_store.close_async()
        # releases the embedder and reranker http clients
        await dense_pipeline.close_async()
        await sparse_pipeline.close_async()
        await hybrid_pipeline.close_async()


INSTRUCTIONS = """\
A-RAG-OG exposes tools to search and read a document knowledge base.

Workflow: use `keyword_and_semantic_search` for most queries (combines meaning + exact
terms, the recommended default); use `semantic_search` (by meaning) or `keyword_search`
(by exact terms) only when you specifically want one modality, or `filtered_search` to
restrict by keywords, entities, content types or date. Use `find_related` to pull more
chunks that mention the same entities as a promising hit (associative multi-hop). Each
search returns chunk ids with short snippets; call `read_chunks` to read promising chunks
in full, or `read_neighbors` to read the chunks immediately before and after a hit when
you need its surrounding context. Decompose complex questions and search in several
rounds. Ground every answer strictly in the retrieved chunks and cite their ids.
""".strip()

mcp = FastMCP(
    name="A-RAG-OG",
    instructions=INSTRUCTIONS,
    auth=JWTVerifier(
        public_key=settings.jwt_secret,
        algorithm=settings.jwt_algorithm,
    ),
    lifespan=lifespan,
)

mcp.add_middleware(RateLimitingMiddleware(
    max_requests_per_second=settings.rate_limit_rps,
    burst_capacity=settings.rate_limit_burst,
    # OpenWebUI JWTs carry the user in the `id` claim
    get_client_id=lambda context: (
        token.claims.get("id", "anonymous")
        if (token := get_access_token()) else "anonymous"
    ),
))

for tool in TOOLS:
    mcp.add_tool(tool)


if __name__ == "__main__":
    mcp.run(
        host="0.0.0.0",
        port=8000,
        transport="http",
        # no session state: any replica can serve any request
        stateless_http=True,
    )
