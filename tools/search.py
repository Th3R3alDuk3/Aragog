from asyncio import Semaphore
from datetime import date
from typing import Annotated

from fastmcp import Context
from fastmcp.exceptions import ToolError
from fastmcp.tools import tool
from haystack import Document, Pipeline
from haystack.core.errors import PipelineRuntimeError
from mcp.types import ToolAnnotations
from pydantic import Field, StringConstraints

from config import get_settings
from schemas.enrichment import EnrichedMeta
from schemas.results import SearchResult
from tools._serializer import search_response

settings = get_settings()


_ENTITY_FIELDS = tuple(
    field for field in EnrichedMeta.model_fields
    if field.startswith("ent_")
)


Query = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=1, max_length=800),
    Field(description="Search query."),
]

KeywordQuery = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=1, max_length=800),
    Field(description="Exact names, codes, or terms; use short phrases, not questions."),
]


async def run_search(
    pipeline: Pipeline,
    inputs: dict,
    limiter: Semaphore,
) -> list[Document]:

    try:
        async with limiter:
            result = await pipeline.run_async(inputs)
    except (PipelineRuntimeError, TimeoutError) as error:
        raise ToolError(
            "The retrieval backend timed out or is temporarily unavailable. "
            "Retry this search in a moment."
        ) from error
    return result["reranker"]["documents"]


@tool(
    name="keyword_and_semantic_search",
    title="Keyword + semantic search",
    description=(
        "Search the knowledge base by meaning and exact terms at once. The "
        "default — use it unless you need a single modality or metadata "
        "filters. Decompose complex questions into several searches. "
        "Returns ranked chunks as ids with a short snippet; read promising ones with `read_chunks`."
    ),
    annotations=ToolAnnotations(readOnlyHint=True, openWorldHint=False),
    timeout=settings.tool_timeout,
)
async def keyword_and_semantic_search(
    ctx: Context,
    query: Query,
) -> SearchResult:

    hybrid_pipeline = ctx.lifespan_context["hybrid_pipeline"]
    search_limiter = ctx.lifespan_context["search_limiter"]

    documents = await run_search(hybrid_pipeline, {
        "dense_embedder": {"text": query},
        "sparse_embedder": {"text": query},
        "dense_retriever": {"top_k": settings.search_top_k_before},
        "sparse_retriever": {"top_k": settings.search_top_k_before},
        "reranker": {
            "query": query,
            "top_k": settings.search_top_k_after,
            "score_threshold": settings.reranker_score_threshold,
        },
    }, search_limiter)

    return search_response(documents)


@tool(
    name="semantic_search",
    title="Semantic search",
    description=(
        "Search by meaning only. Use when the wording varies but the concept "
        "is stable; otherwise prefer `keyword_and_semantic_search`. "
        "Returns ranked chunks as ids with a short snippet; read promising ones with `read_chunks`."
    ),
    annotations=ToolAnnotations(readOnlyHint=True, openWorldHint=False),
    timeout=settings.tool_timeout,
)
async def semantic_search(
    ctx: Context,
    query: Query,
) -> SearchResult:

    dense_pipeline = ctx.lifespan_context["dense_pipeline"]
    search_limiter = ctx.lifespan_context["search_limiter"]

    documents = await run_search(dense_pipeline, {
        "embedder": {"text": query},
        "retriever": {"top_k": settings.search_top_k_before},
        "reranker": {
            "query": query,
            "top_k": settings.search_top_k_after,
            "score_threshold": settings.reranker_score_threshold,
        },
    }, search_limiter)

    return search_response(documents)


@tool(
    name="keyword_search",
    title="Keyword search",
    description=(
        "Search by exact terms only (BM25). Use for names, codes or domain "
        "terms where the exact wording matters; otherwise prefer "
        "`keyword_and_semantic_search`. Returns ranked chunks as ids with a short snippet; read promising ones with `read_chunks`."
    ),
    annotations=ToolAnnotations(readOnlyHint=True, openWorldHint=False),
    timeout=settings.tool_timeout,
)
async def keyword_search(
    ctx: Context,
    query: KeywordQuery,
) -> SearchResult:

    sparse_pipeline = ctx.lifespan_context["sparse_pipeline"]
    search_limiter = ctx.lifespan_context["search_limiter"]

    documents = await run_search(sparse_pipeline, {
        "embedder": {"text": query},
        "retriever": {"top_k": settings.search_top_k_before},
        "reranker": {
            "query": query,
            "top_k": settings.search_top_k_after,
            "score_threshold": settings.reranker_score_threshold,
        },
    }, search_limiter)

    return search_response(documents)


@tool(
    name="filtered_search",
    title="Filtered search",
    description=(
        "Hybrid search restricted by metadata; all supplied filters must match. "
        "Returns ranked ids and previews; read promising hits with `read_chunks`."
    ),
    annotations=ToolAnnotations(readOnlyHint=True, openWorldHint=False),
    timeout=settings.tool_timeout,
)
async def filtered_search(
    ctx: Context,
    query: Query,
    keywords: Annotated[list[str], Field(
        default=[],
        max_length=5,
        description=(
            "Match any enriched keyword. Use known exact terms."
        ),
    )],
    entities: Annotated[list[str], Field(
        default=[],
        max_length=5,
        description=(
            "Match any enriched person, organization, product, or location."
        ),
    )],
    content_types: Annotated[list[str], Field(
        default=[],
        max_length=5,
        description=(
            "Match any structural type, e.g. text, table, list_item, or code."
        ),
    )],
    date_from: Annotated[date | None, Field(
        default=None,
        description="Earliest mentioned date, inclusive.",
    )],
    date_to: Annotated[date | None, Field(
        default=None,
        description="Latest mentioned date, inclusive; bounds may match different dates in one chunk.",
    )],
    modified_from: Annotated[date | None, Field(
        default=None,
        description="Earliest source modification date, inclusive.",
    )],
    modified_to: Annotated[date | None, Field(
        default=None,
        description="Latest source modification date, inclusive.",
    )],
) -> SearchResult:

    hybrid_pipeline = ctx.lifespan_context["hybrid_pipeline"]
    search_limiter = ctx.lifespan_context["search_limiter"]

    conditions: list[dict] = []

    if keywords:
        conditions.append({
            "field": "meta.keywords",
            "operator": "in",
            "value": keywords,
        })

    if entities:
        conditions.append({
            "operator": "OR",
            "conditions": [{
                "field": f"meta.{field}", "operator": "in", "value": entities,
            } for field in _ENTITY_FIELDS],
        })

    if content_types:
        conditions.append({
            "field": "meta.content_types",
            "operator": "in",
            "value": content_types,
        })

    if date_from:
        conditions.append({
            "field": "meta.dates",
            "operator": ">=",
            "value": date_from.isoformat(),
        })

    if date_to:
        conditions.append({
            "field": "meta.dates",
            "operator": "<=",
            "value": date_to.isoformat(),
        })

    if modified_from:
        conditions.append({
            "field": "meta.modified_at",
            "operator": ">=",
            "value": modified_from.isoformat(),
        })

    if modified_to:
        conditions.append({
            "field": "meta.modified_at",
            "operator": "<=",
            "value": f"{modified_to.isoformat()}T23:59:59",
        })

    filters = {
        "operator": "AND",
        "conditions": conditions,
    } if conditions else None

    documents = await run_search(hybrid_pipeline, {
        "dense_embedder": {"text": query},
        "sparse_embedder": {"text": query},
        "dense_retriever": {"top_k": settings.search_top_k_before, "filters": filters},
        "sparse_retriever": {"top_k": settings.search_top_k_before, "filters": filters},
        "reranker": {
            "query": query,
            "top_k": settings.search_top_k_after,
            "score_threshold": settings.reranker_score_threshold,
        },
    }, search_limiter)

    return search_response(documents)


@tool(
    name="find_related",
    title="Find related chunks",
    description=(
        "Find further chunks sharing entities (persons, organizations, "
        "products, locations) with the given ones — associative multi-hop "
        "from an earlier hit. Ranked against the query, excluding the given "
        "chunks. Returns ranked chunks as ids with a short snippet; read promising ones with `read_chunks`."
    ),
    annotations=ToolAnnotations(readOnlyHint=True, openWorldHint=False),
    timeout=settings.tool_timeout,
)
async def find_related(
    ctx: Context,
    chunk_ids: Annotated[list[str], Field(
        max_length=2,
        description=(
            "One or two search-hit ids whose entities define the expansion."
        ),
    )],
    query: Query,
) -> SearchResult:

    document_store = ctx.lifespan_context["document_store"]
    hybrid_pipeline = ctx.lifespan_context["hybrid_pipeline"]
    search_limiter = ctx.lifespan_context["search_limiter"]

    seeds = await document_store.filter_documents_async(
        filters={"field": "id", "operator": "in", "value": chunk_ids})

    if not seeds:
        return SearchResult(
            hint=(
                "None of the supplied chunk ids exists. Run a search first, "
                "then pass ids from its hits."
            ),
            hits=[],
        )

    entities = sorted({
        entity
        for seed in seeds
        for field in _ENTITY_FIELDS
        for entity in seed.meta.get(field, [])
    })

    if not entities:
        return SearchResult(
            hint=(
                "The supplied chunks contain no extracted entities. Choose a "
                "different hit, or continue with another search instead."
            ),
            hits=[],
        )

    filters = {
        "operator": "AND",
        "conditions": [
            {"field": "id", "operator": "not in", "value": chunk_ids},
            {"operator": "OR", "conditions": [
                {"field": f"meta.{field}", "operator": "in", "value": entities}
                for field in _ENTITY_FIELDS
            ]},
        ],
    }

    documents = await run_search(hybrid_pipeline, {
        "dense_embedder": {"text": query},
        "sparse_embedder": {"text": query},
        "dense_retriever": {"top_k": settings.search_top_k_before, "filters": filters},
        "sparse_retriever": {"top_k": settings.search_top_k_before, "filters": filters},
        "reranker": {
            "query": query,
            "top_k": settings.search_top_k_after,
            "score_threshold": settings.reranker_score_threshold,
        },
    }, search_limiter)

    return search_response(documents)
