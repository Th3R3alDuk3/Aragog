from asyncio import Semaphore
from datetime import UTC, date, datetime, time
from typing import Annotated, cast

from fastmcp import Context
from fastmcp.exceptions import ToolError
from fastmcp.tools import tool
from haystack import Document, Pipeline
from haystack.core.errors import PipelineRuntimeError
from mcp.types import ToolAnnotations
from pydantic import Field, StringConstraints
from qdrant_client.http.models import (
    Condition,
    DatetimeRange,
    FieldCondition,
    Filter,
    MatchAny,
)

from config import get_settings
from schemas.enrichment import ENTITY_FIELDS
from schemas.results import SearchHit, SearchResult

settings = get_settings()


Query = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=1, max_length=800),
    Field(description="Search query."),
]


async def _run_search(
    pipeline: Pipeline,
    inputs: dict,
    limiter: Semaphore,
) -> tuple[list[Document], int]:

    try:
        async with limiter:
            # reranker input: `joiner` when hybrid, `retriever` otherwise
            result = await pipeline.run_async(
                inputs, include_outputs_from={"joiner", "retriever"})
    except PipelineRuntimeError as error:
        raise ToolError(
            "The retrieval backend timed out or is temporarily unavailable. "
            "Retry this search in a moment."
        ) from error

    candidates = (result.get("joiner") or result["retriever"])["documents"]
    return result["reranker"]["documents"], len(candidates)


def _search_response(
    documents: list[Document],
    no_match_hint: str = (
        "No matches. Reformulate or broaden the query once more; if the topic "
        "is likely outside the knowledge base, say so instead of searching "
        "again. Use `filtered_search` only with metadata from chunks you read."
    ),
) -> SearchResult:
    # no url: the agent must read a chunk before it may cite one
    return SearchResult(
        hint="" if documents else no_match_hint,
        hits=[SearchHit(
            id=document.id,
            # reranked, so never None
            score=cast(float, document.score),
            source=document.meta["source"],
            page=document.meta.get("page_number"),
            headings=document.meta["headings"],
            snippet=(document.meta.get("context") or document.content or "")[:300],
        ) for document in documents],
    )


@tool(
    name="keyword_and_semantic_search",
    title="Keyword + semantic search",
    description=(
        "Search the knowledge base by meaning and exact terms at once. The "
        "default — use it unless you need a single modality or metadata "
        "filters. Decompose complex questions into several searches. Returns "
        "ranked chunks as ids with a short snippet; read promising ones with "
        "`read_chunks`."
    ),
    annotations=ToolAnnotations(read_only_hint=True, open_world_hint=False),
    timeout=settings.tool_timeout,
)
async def keyword_and_semantic_search(
    ctx: Context,
    query: Query,
) -> SearchResult:

    hybrid_pipeline = ctx.lifespan_context["hybrid_pipeline"]
    search_limiter = ctx.lifespan_context["search_limiter"]

    documents, _ = await _run_search(hybrid_pipeline, {
        "dense_embedder": {"text": query},
        "sparse_embedder": {"text": query},
        "reranker": {"query": query},
    }, search_limiter)

    return _search_response(documents)


@tool(
    name="semantic_search",
    title="Semantic search",
    description=(
        "Search by meaning only. Use when the wording varies but the concept "
        "is stable; otherwise prefer `keyword_and_semantic_search`. Returns "
        "ranked chunks as ids with a short snippet; read promising ones with "
        "`read_chunks`."
    ),
    annotations=ToolAnnotations(read_only_hint=True, open_world_hint=False),
    timeout=settings.tool_timeout,
)
async def semantic_search(
    ctx: Context,
    query: Query,
) -> SearchResult:

    dense_pipeline = ctx.lifespan_context["dense_pipeline"]
    search_limiter = ctx.lifespan_context["search_limiter"]

    documents, _ = await _run_search(dense_pipeline, {
        "embedder": {"text": query},
        "reranker": {"query": query},
    }, search_limiter)

    return _search_response(documents)


@tool(
    name="keyword_search",
    title="Keyword search",
    description=(
        "Search by exact terms only (BM25). Use for names, codes or domain "
        "terms where the exact wording matters; otherwise prefer "
        "`keyword_and_semantic_search`. Returns ranked chunks as ids with a "
        "short snippet; read promising ones with `read_chunks`."
    ),
    annotations=ToolAnnotations(read_only_hint=True, open_world_hint=False),
    timeout=settings.tool_timeout,
)
async def keyword_search(
    ctx: Context,
    query: Annotated[Query, Field(
        description=(
            "Exact names, codes, or terms in "
            f"{settings.sparse_embedding_language.title()}; use short phrases, "
            "not questions."
        ),
    )],
) -> SearchResult:

    sparse_pipeline = ctx.lifespan_context["sparse_pipeline"]
    search_limiter = ctx.lifespan_context["search_limiter"]

    documents, _ = await _run_search(sparse_pipeline, {
        "embedder": {"text": query},
        "reranker": {"query": query},
    }, search_limiter)

    return _search_response(documents)


@tool(
    name="filtered_search",
    title="Filtered search",
    description=(
        "Hybrid search restricted by metadata: within a list any value "
        "matches, across filters all must match. Take keyword and entity "
        "values from chunks read with `read_chunks`. Returns ranked chunks as "
        "ids with a short snippet; read promising ones with `read_chunks`."
    ),
    annotations=ToolAnnotations(read_only_hint=True, open_world_hint=False),
    timeout=settings.tool_timeout,
)
async def filtered_search(
    ctx: Context,
    query: Query,
    keywords: Annotated[list[str], Field(
        default=[],
        max_length=5,
        description="Match any enriched keyword. Use known exact terms.",
    )],
    entities: Annotated[list[str], Field(
        default=[],
        max_length=5,
        description=(
            "Match any exact person, organization, product, or location; "
            "list name variants."
        ),
    )],
    content_types: Annotated[list[str], Field(
        default=[],
        max_length=5,
        description="Match any structural type, e.g. text, table, list_item, or code.",
    )],
    date_from: Annotated[date | None, Field(
        default=None,
        description="Earliest mentioned date, inclusive.",
    )],
    date_to: Annotated[date | None, Field(
        default=None,
        description=(
            "Latest mentioned date, inclusive; bounds may match different "
            "dates in one chunk."
        ),
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

    conditions: list[Condition] = []

    if keywords:
        conditions.append(FieldCondition(
            key="meta.keywords",
            match=MatchAny(any=keywords),
        ))

    if entities:
        conditions.append(Filter(should=[
            FieldCondition(
                key=f"meta.{field}",
                match=MatchAny(any=entities),
            )
            for field in ENTITY_FIELDS
        ]))

    if content_types:
        conditions.append(FieldCondition(
            key="meta.content_types",
            match=MatchAny(any=content_types),
        ))

    if date_from:
        conditions.append(FieldCondition(
            key="meta.dates",
            range=DatetimeRange(gte=date_from),
        ))

    if date_to:
        conditions.append(FieldCondition(
            key="meta.dates",
            range=DatetimeRange(lte=date_to),
        ))

    if modified_from:
        conditions.append(FieldCondition(
            key="meta.modified_at",
            range=DatetimeRange(gte=datetime.combine(
                modified_from, time.min, tzinfo=UTC)),
        ))

    if modified_to:
        conditions.append(FieldCondition(
            key="meta.modified_at",
            range=DatetimeRange(lte=datetime.combine(
                modified_to, time.max, tzinfo=UTC)),
        ))

    filters = Filter(must=conditions) if conditions else None

    documents, candidates = await _run_search(hybrid_pipeline, {
        "dense_embedder": {"text": query},
        "sparse_embedder": {"text": query},
        "dense_retriever": {"filters": filters},
        "sparse_retriever": {"filters": filters},
        "reranker": {"query": query},
    }, search_limiter)

    if filters is None:
        return _search_response(documents)

    return _search_response(documents, no_match_hint=(
        "No chunk matches all filters. Take keyword and entity values from "
        "chunks you have read, or drop filters one at a time."
        if candidates == 0 else
        f"{candidates} chunk(s) match the filters, but none is relevant enough "
        "to the query. Rephrase the query or relax the filters."
    ))


@tool(
    name="find_related",
    title="Find related chunks",
    description=(
        "Find further chunks sharing entities (persons, organizations, "
        "products, locations) with the given ones — associative multi-hop "
        "from an earlier hit. Ranked against the query, excluding the given "
        "chunks. Returns ranked chunks as ids with a short snippet; read "
        "promising ones with `read_chunks`."
    ),
    annotations=ToolAnnotations(read_only_hint=True, open_world_hint=False),
    timeout=settings.tool_timeout,
)
async def find_related(
    ctx: Context,
    chunk_ids: Annotated[list[str], Field(
        min_length=1, max_length=2,
        description="One or two search-hit ids whose entities define the expansion.",
    )],
    query: Query,
) -> SearchResult:

    document_store = ctx.lifespan_context["document_store"]
    hybrid_pipeline = ctx.lifespan_context["hybrid_pipeline"]
    search_limiter = ctx.lifespan_context["search_limiter"]

    seeds = await document_store.get_documents_by_id_async(chunk_ids)

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
        for field in ENTITY_FIELDS
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

    filters = Filter(
        should=[
            FieldCondition(
                key=f"meta.{field}",
                match=MatchAny(any=entities),
            )
            for field in ENTITY_FIELDS
        ],
        must_not=[FieldCondition(
            key="id",
            match=MatchAny(any=chunk_ids),
        )],
    )

    documents, candidates = await _run_search(hybrid_pipeline, {
        "dense_embedder": {"text": query},
        "sparse_embedder": {"text": query},
        "dense_retriever": {"filters": filters},
        "sparse_retriever": {"filters": filters},
        "reranker": {"query": query},
    }, search_limiter)

    return _search_response(documents, no_match_hint=(
        "No other chunk mentions these entities. Continue with another search."
        if candidates == 0 else
        f"{candidates} chunk(s) mention the same entities, but none is relevant "
        "enough to the query. Rephrase the query toward what they would say."
    ))
