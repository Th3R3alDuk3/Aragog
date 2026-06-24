from dotenv import load_dotenv
load_dotenv()

from collections.abc import AsyncIterator

from fastmcp import FastMCP
from fastmcp.server.auth.providers.jwt import JWTVerifier
from fastmcp.server.lifespan import lifespan as composable_lifespan
from fastmcp.utilities.logging import configure_logging

from config import get_settings
from pipelines._factories import build_document_store
from pipelines.retrieval import (
    build_dense_retrieval_pipeline,
    build_sparse_retrieval_pipeline,
    build_hybrid_retrieval_pipeline,
)
from services.storage import MinioStore
from tools import TOOLS


configure_logging()


#--------------------------------------------
# GLOBALS
#--------------------------------------------


settings = get_settings()


#--------------------------------------------
# SERVER
#--------------------------------------------


@composable_lifespan
async def lifespan(server: FastMCP) -> AsyncIterator[dict]:

    document_store = build_document_store()

    yield {
        "document_store": document_store,
        "minio_store": MinioStore(
            settings.minio_endpoint,
            settings.minio_user,
            settings.minio_password,
            settings.minio_bucket,
        ),
        "dense_pipeline": build_dense_retrieval_pipeline(document_store),
        "sparse_pipeline": build_sparse_retrieval_pipeline(document_store),
        "hybrid_pipeline": build_hybrid_retrieval_pipeline(document_store),
    }


INSTRUCTIONS = """\
A-RAG-OG exposes tools to search and read a document knowledge base.

Workflow: use `keyword_and_semantic_search` for most queries (combines meaning + exact terms, the
recommended default); use `semantic_search` (by meaning) or `keyword_search` (by exact
terms) only when you specifically want one modality, or `filtered_search` to restrict
by keywords, entities, content types or date. Use `find_related` to pull more chunks that mention the
same entities as a promising hit (associative multi-hop). Each search returns chunk ids
with short snippets; call
`read_chunk` to read promising chunks in full, or `read_neighbors` to read the chunks
immediately before and after a hit when you need its surrounding context. Decompose complex questions and search
in several rounds. Ground every answer strictly in the retrieved chunks and cite their ids.
""".strip()

auth = JWTVerifier(
    public_key=settings.jwt_secret,
    algorithm=settings.jwt_algorithm,
)

mcp = FastMCP(
    name="A-RAG-OG",
    instructions=INSTRUCTIONS,
    auth=auth,
    lifespan=lifespan,
)


for tool in TOOLS:
    mcp.add_tool(tool)


if __name__ == "__main__":
    mcp.run(
        host=settings.host,
        port=settings.port,
        transport="streamable-http",
    )
