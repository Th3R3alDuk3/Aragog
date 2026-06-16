from dotenv import load_dotenv
load_dotenv()

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Annotated

from fastmcp import Context, FastMCP
from fastmcp.server.auth.providers.jwt import JWTVerifier
from fastmcp.utilities.logging import configure_logging
from haystack import Document
from mcp.types import ToolAnnotations
from pydantic import Field

from config import get_settings
from models.results import ChunkContent, ReadResult, SearchHit, SearchResult
from pipelines._factories import build_document_store
from pipelines.retrieval import (
    build_dense_retrieval_pipeline,
    build_sparse_retrieval_pipeline,
    build_hybrid_retrieval_pipeline,
)
from services.storage import MinioStore


configure_logging()


#--------------------------------------------
# GLOBALS
#--------------------------------------------


settings = get_settings()


#--------------------------------------------
# SERVER
#--------------------------------------------


@asynccontextmanager
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


#--------------------------------------------
# HELPERS
#--------------------------------------------


_ENTITY_FIELDS = (
    "ent_persons",
    "ent_organizations",
    "ent_products",
    "ent_locations",
)


def _search_hits(
    documents: list[Document],
    minio_store: MinioStore,
) -> list[SearchHit]:
    return [SearchHit(
        id=document.id,
        score=document.score,
        source=document.meta.get("source"),
        url=minio_store.presigned_url(
            document.meta.get("source"), settings.minio_url_expire),
        page=document.meta.get("page_number"),
        headings=document.meta.get("headings", []),
        snippet=document.meta.get("context") or (document.content or "")[:300],
    ) for document in documents]


def _chunk_contents(
    documents: list[Document],
    minio_store: MinioStore,
) -> list[ChunkContent]:
    return [ChunkContent(
        id=document.id,
        source=document.meta.get("source"),
        url=minio_store.presigned_url(
            document.meta.get("source"), settings.minio_url_expire),
        page=document.meta.get("page_number"),
        content=document.content,
    ) for document in documents]


def _search_response(
    documents: list[Document],
    minio_store: MinioStore,
) -> SearchResult:

    hint = (
        "No matches. Reformulate the query or narrow it with "
        "`filtered_search`."
    )
    
    if documents:
        hint = (
            "Open promising hits in full with `read_chunk` before answering. "
            "Use `find_related` to expand via a hit's entities, or "
            "`read_neighbors` for surrounding context."
        )        

    return SearchResult(
        hint=hint,
        hits=_search_hits(documents, minio_store),
    )


def _read_response(
    documents: list[Document],
    minio_store: MinioStore,
) -> ReadResult:

    hint = "No chunks found. Run a search first to get valid chunk ids."

    if documents:
        hint = (
            "Ground your answer in this content and cite source and page. "
            "Use `read_neighbors` or `find_related` to dig further."
        )

    return ReadResult(
        hint=hint,
        chunks=_chunk_contents(documents, minio_store),
    )


#--------------------------------------------
# TOOLS
#--------------------------------------------


@mcp.tool(
    name="keyword_and_semantic_search",
    description=(
        "Search the knowledge base with semantic (meaning) and keyword "
        "(exact-term) retrieval fused by the cross-encoder reranker — the "
        "recommended default; use it unless you specifically need a single "
        "modality or metadata filters. Decompose complex questions into "
        "several searches. Returns the top reranked chunks with id, source, "
        "page, headings, a short snippet and a temporary source URL."
    ),
    annotations=ToolAnnotations(readOnlyHint=True),
)
async def keyword_and_semantic_search(
    ctx: Context,
    query: Annotated[str, Field(
        description="A natural language query.",
    )],
    top_k_before: Annotated[int, Field(
        description=(
            "Number of candidate chunks to retrieve from each of dense and "
            "sparse before reranking. Defaults to 30."
        ),
    )] = 30,
    top_k_after: Annotated[int, Field(
        description="Number of chunks to return after reranking. Defaults to 5.",
    )] = 5,
) -> SearchResult:

    result = await ctx.lifespan_context["hybrid_pipeline"].run_async({
        "dense_embedder": {"text": query},
        "sparse_embedder": {"text": query},
        "dense_retriever": {"top_k": top_k_before},
        "sparse_retriever": {"top_k": top_k_before},
        "reranker": {"query": query, "top_k": top_k_after},
    })

    documents = result["reranker"]["documents"]
    return _search_response(documents, ctx.lifespan_context["minio_store"])


@mcp.tool(
    name="semantic_search",
    description=(
        "Search the knowledge base by meaning only (dense retrieval + "
        "cross-encoder reranking) — use when you want pure semantic matching "
        "rather than the default combined search. Returns the top reranked chunks with "
        "id, source, page, headings, a short snippet and a temporary source "
        "URL."
    ),
    annotations=ToolAnnotations(readOnlyHint=True),
)
async def semantic_search(
    ctx: Context,
    query: Annotated[str, Field(
        description="A natural language query.",
    )],
    top_k_before: Annotated[int, Field(
        description=(
            "Number of candidate chunks to retrieve before reranking. "
            "Defaults to 30."
        ),
    )] = 30,
    top_k_after: Annotated[int, Field(
        description="Number of chunks to return after reranking. Defaults to 5.",
    )] = 5,
) -> SearchResult:

    result = await ctx.lifespan_context["dense_pipeline"].run_async({
        "embedder": {"text": query},
        "retriever": {"top_k": top_k_before},
        "reranker": {"query": query, "top_k": top_k_after},
    })

    documents = result["reranker"]["documents"]
    return _search_response(documents, ctx.lifespan_context["minio_store"])


@mcp.tool(
    name="keyword_search",
    description=(
        "Search the knowledge base by exact terms (sparse/BM25 retrieval + "
        "cross-encoder reranking) — use for specific keywords, names or codes "
        "where exact wording matters rather than the default combined search. Returns "
        "the top reranked chunks with id, source, page, headings, a short "
        "snippet and a temporary source URL."
    ),
    annotations=ToolAnnotations(readOnlyHint=True),
)
async def keyword_search(
    ctx: Context,
    query: Annotated[str, Field(
        description="The keywords or exact terms to look up.",
    )],
    top_k_before: Annotated[int, Field(
        description=(
            "Number of candidate chunks to retrieve before reranking. "
            "Defaults to 30."
        ),
    )] = 30,
    top_k_after: Annotated[int, Field(
        description="Number of chunks to return after reranking. Defaults to 5.",
    )] = 5,
) -> SearchResult:

    result = await ctx.lifespan_context["sparse_pipeline"].run_async({
        "embedder": {"text": query},
        "retriever": {"top_k": top_k_before},
        "reranker": {"query": query, "top_k": top_k_after},
    })

    documents = result["reranker"]["documents"]
    return _search_response(documents, ctx.lifespan_context["minio_store"])


@mcp.tool(
    name="filtered_search",
    description=(
        "Search the knowledge base by meaning, restricted by metadata "
        "filters — use to constrain results by keywords, entities, content "
        "types or a date range. Combine any of the filters; all given must "
        "hold. Returns the top reranked chunks with id, source, page, "
        "headings, a short snippet and a temporary source URL."
    ),
    annotations=ToolAnnotations(readOnlyHint=True),
)
async def filtered_search(
    ctx: Context,
    query: Annotated[str, Field(
        description="A natural language query.",
    )],
    keywords: Annotated[list[str], Field(
        default_factory=list,
        description="A chunk matches if it carries any of these keywords.",
    )],
    entities: Annotated[list[str], Field(
        default_factory=list,
        description=(
            "A chunk matches if it mentions any of these entities "
            "(persons, organizations, products, locations)."
        ),
    )],
    content_types: Annotated[list[str], Field(
        default_factory=list,
        description=(
            "A chunk matches if it contains any of these structural element "
            "types (e.g. 'table', 'text', 'list_item', 'code', 'formula', "
            "'picture', 'section_header')."
        ),
    )],
    date_from: Annotated[str, Field(
        description="Earliest date (ISO YYYY-MM-DD) the chunk may refer to.",
    )] = "",
    date_to: Annotated[str, Field(
        description="Latest date (ISO YYYY-MM-DD) the chunk may refer to.",
    )] = "",
    created_from: Annotated[str, Field(
        description=(
            "Earliest creation date (ISO YYYY-MM-DD) of the source file the "
            "chunk comes from."
        ),
    )] = "",
    created_to: Annotated[str, Field(
        description=(
            "Latest creation date (ISO YYYY-MM-DD) of the source file, "
            "inclusive of the whole day."
        ),
    )] = "",
    top_k_before: Annotated[int, Field(
        description=(
            "Number of candidate chunks to retrieve before reranking. "
            "Defaults to 30."
        ),
    )] = 30,
    top_k_after: Annotated[int, Field(
        description="Number of chunks to return after reranking. Defaults to 5.",
    )] = 5,
) -> SearchResult:

    conditions: list[dict] = []

    if keywords:
        conditions.append(
            {"field": "meta.keywords", "operator": "in", "value": keywords})

    if entities:
        conditions.append({
            "operator": "OR",
            "conditions": [{
                "field": f"meta.{field}", "operator": "in", "value": entities
            } for field in _ENTITY_FIELDS]
        })

    if content_types:
        conditions.append(
            {"field": "meta.content_types", "operator": "in", "value": content_types})

    if date_from:
        conditions.append(
            {"field": "meta.dates", "operator": ">=", "value": date_from})

    if date_to:
        conditions.append(
            {"field": "meta.dates", "operator": "<=", "value": date_to})

    if created_from:
        conditions.append(
            {"field": "meta.created_at", "operator": ">=", "value": created_from})

    if created_to:
        upper = created_to if "T" in created_to else f"{created_to}T23:59:59"
        conditions.append(
            {"field": "meta.created_at", "operator": "<=", "value": upper})

    result = await ctx.lifespan_context["dense_pipeline"].run_async({
        "embedder": {"text": query},
        "retriever": {
            "top_k": top_k_before,
            "filters": {
                "operator": "AND", 
                "conditions": conditions
            } if conditions else None,
        },
        "reranker": {"query": query, "top_k": top_k_after},
    })

    documents = result["reranker"]["documents"]
    return _search_response(documents, ctx.lifespan_context["minio_store"])


@mcp.tool(
    name="find_related",
    description=(
        "Find more chunks that mention the same entities (persons, "
        "organizations, products, locations) as the given chunks — use to "
        "expand from earlier hits via shared entities (associative "
        "multi-hop). Ranked against the query by the cross-encoder reranker, "
        "excluding the given chunks. Returns the top reranked chunks with id, "
        "source, page, headings, a short snippet and a temporary source URL."
    ),
    annotations=ToolAnnotations(readOnlyHint=True),
)
async def find_related(
    ctx: Context,
    chunk_ids: Annotated[list[str], Field(
        description=(
            "Chunk ids (from a previous search result) whose entities define "
            "the expansion."
        ),
    )],
    query: Annotated[str, Field(
        description="A natural language query the related chunks are ranked against.",
    )],
    top_k_before: Annotated[int, Field(
        description=(
            "Number of candidate chunks to retrieve before reranking. "
            "Defaults to 30."
        ),
    )] = 30,
    top_k_after: Annotated[int, Field(
        description="Number of chunks to return after reranking. Defaults to 5.",
    )] = 5,
) -> SearchResult:

    seeds = await ctx.lifespan_context["document_store"].filter_documents_async(
        filters={"field": "id", "operator": "in", "value": chunk_ids})

    entities = sorted({
        entity
        for seed in seeds
        for field in _ENTITY_FIELDS
        for entity in seed.meta.get(field, [])
    })

    if not entities:
        return _search_response([], ctx.lifespan_context["minio_store"])

    result = await ctx.lifespan_context["dense_pipeline"].run_async({
        "embedder": {"text": query},
        "retriever": {
            "top_k": top_k_before,
            "filters": {
                "operator": "AND",
                "conditions": [
                    {"field": "id", "operator": "not in", "value": chunk_ids},
                    {"operator": "OR", "conditions": [
                        {"field": f"meta.{field}", "operator": "in", "value": entities}
                        for field in _ENTITY_FIELDS
                    ]},
                ],
            },
        },
        "reranker": {"query": query, "top_k": top_k_after},
    })

    documents = result["reranker"]["documents"]
    return _search_response(documents, ctx.lifespan_context["minio_store"])


@mcp.tool(
    name="read_chunk",
    description=(
        "Read the full content of chunks by their ids (from a search result). Returns "
        "each chunk with id, source, page, full content and a temporary source URL."
    ),
    annotations=ToolAnnotations(readOnlyHint=True),
)
async def read_chunk(
    ctx: Context,
    chunk_ids: Annotated[list[str], Field(
        description="The chunk ids to read in full.",
    )],
) -> ReadResult:

    documents = await ctx.lifespan_context["document_store"].filter_documents_async(
        filters={"field": "id", "operator": "in", "value": chunk_ids})

    return _read_response(documents, ctx.lifespan_context["minio_store"])


@mcp.tool(
    name="read_neighbors",
    description=(
        "Read the chunks surrounding given chunks within their source "
        "document — the contiguous passage of up to `window` chunks before "
        "and after each id, in document order. Use this to recover the "
        "context around a promising search hit. Returns each chunk with id, "
        "source, page, full content and a temporary source URL."
    ),
    annotations=ToolAnnotations(readOnlyHint=True),
)
async def read_neighbors(
    ctx: Context,
    chunk_ids: Annotated[list[str], Field(
        description=(
            "The chunk ids (from a search result) to read the surrounding context of."
        ),
    )],
    window: Annotated[int, Field(
        description=(
            "How many chunks before and after each id to include. Defaults to 1."
        ),
    )] = 1,
) -> ReadResult:

    document_store = ctx.lifespan_context["document_store"]

    seeds = await document_store.filter_documents_async(
        filters={"field": "id", "operator": "in", "value": chunk_ids})

    conditions: list[dict] = []

    for seed in seeds:

        index = seed.meta.get("chunk_index")
        if index is None:
            continue

        total_chunks = seed.meta.get("total_chunks", index + window + 1)
        conditions.append({
            "operator": "AND",
            "conditions": [
                {"field": "meta.source", "operator": "==", "value": seed.meta.get("source")},
                {"field": "meta.chunk_index", "operator": "in", "value": [
                    neighbor
                    for neighbor in range(index - window, index + window + 1)
                    if 0 <= neighbor < total_chunks
                ]},
            ],
        })

    if not conditions:
        return _read_response([], ctx.lifespan_context["minio_store"])

    neighbors = await document_store.filter_documents_async(
        filters={"operator": "OR", "conditions": conditions})

    neighbors.sort(key=lambda document: (
        document.meta.get("source") or "",
        document.meta.get("chunk_index", 0),
    ))

    return _read_response(neighbors, ctx.lifespan_context["minio_store"])


if __name__ == "__main__":
    mcp.run(
        host=settings.host,
        port=settings.port,
        transport="streamable-http",
    )
