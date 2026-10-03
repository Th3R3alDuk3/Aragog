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
    MatchPhrase,
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
            "The retrieval backend timed out or failed. Retry this search once; "
            "if it fails again, report that instead of searching further."
        ) from error

    candidates = (result.get("joiner") or result["retriever"])["documents"]
    return result["reranker"]["documents"], len(candidates)


def _search_response(
    documents: list[Document],
    no_match_hint: str = (
        "No matches. Rephrase or broaden the query once more; if the topic is "
        "likely outside the knowledge base, say so instead of searching again."
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
        "Search the knowledge base by meaning and keywords at once. The "
        "default — use it unless you need a single modality, a verbatim phrase "
        "(`exact_search`) or metadata filters (`filtered_search`). Decompose "
        "complex questions into several searches. Returns ranked chunks as ids "
        "with a short snippet; read promising ones with `read_chunks`."
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
        "Search by meaning only — the dense half of "
        "`keyword_and_semantic_search`. Use when keyword hits crowd out "
        "paraphrases; otherwise prefer `keyword_and_semantic_search`. Returns "
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

    # dense retrieval always has candidates, so an empty result is a threshold miss
    return _search_response(documents, no_match_hint=(
        "No chunk is relevant enough by meaning alone. Try "
        "`keyword_and_semantic_search` once; if the topic is likely outside "
        "the knowledge base, say so instead of searching again."
    ))


@tool(
    name="keyword_search",
    title="Keyword search",
    description=(
        "Search by keywords only (BM25: stemmed, any order) — the sparse half "
        "of `keyword_and_semantic_search`. Use for single terms when "
        "meaning-based hits drift; for a name or exact word sequence use "
        "`exact_search`; otherwise prefer `keyword_and_semantic_search`. "
        "Returns ranked chunks as ids with a short snippet; read promising "
        "ones with `read_chunks`."
    ),
    annotations=ToolAnnotations(read_only_hint=True, open_world_hint=False),
    timeout=settings.tool_timeout,
)
async def keyword_search(
    ctx: Context,
    query: Annotated[Query, Field(
        description=(
            f"Keywords in {settings.sparse_embedding_language.title()}; "
            "single terms, not questions."
        ),
    )],
) -> SearchResult:

    sparse_pipeline = ctx.lifespan_context["sparse_pipeline"]
    search_limiter = ctx.lifespan_context["search_limiter"]

    documents, candidates = await _run_search(sparse_pipeline, {
        "embedder": {"text": query},
        "reranker": {"query": query},
    }, search_limiter)

    if candidates:
        return _search_response(documents)

    return _search_response(documents, no_match_hint=(
        "No chunk shares a searchable term with this query — stopwords do not "
        f"count, and terms must be {settings.sparse_embedding_language.title()} "
        "base forms. Try `keyword_and_semantic_search`."
    ))


@tool(
    name="exact_search",
    title="Exact phrase search",
    description=(
        "Search chunks that contain an exact word sequence — codes, "
        "identifiers, § references, names or quoted wording. Case-insensitive, "
        "punctuation is ignored, word order matters; matches are ranked by "
        "the query, or by the phrase itself without one. Returns ranked chunks "
        "as ids with a short snippet; read promising ones with `read_chunks`."
    ),
    annotations=ToolAnnotations(read_only_hint=True, open_world_hint=False),
    timeout=settings.tool_timeout,
)
async def exact_search(
    ctx: Context,
    phrase: Annotated[str, StringConstraints(
        strip_whitespace=True, min_length=1, max_length=200,
    ), Field(
        description=(
            "The word sequence as it appears in the text, e.g. `RX-7800-B`, "
            "`§ 823 BGB` or `Wilma Wundersinn`."
        ),
    )],
    query: Annotated[str, StringConstraints(
        strip_whitespace=True, max_length=800,
    ), Field(
        default="",
        description="What to rank the matching chunks by; defaults to the phrase.",
    )],
) -> SearchResult:

    hybrid_pipeline = ctx.lifespan_context["hybrid_pipeline"]
    search_limiter = ctx.lifespan_context["search_limiter"]

    text = query or phrase
    filters = Filter(must=[FieldCondition(
        key="content",
        match=MatchPhrase(phrase=phrase),
    )])

    documents, candidates = await _run_search(hybrid_pipeline, {
        "dense_embedder": {"text": text},
        "sparse_embedder": {"text": text},
        "dense_retriever": {"filters": filters},
        "sparse_retriever": {"filters": filters},
        # without a query the phrase match itself is the relevance criterion
        "reranker": {"query": text} if query else {"query": text, "score_threshold": 0.0},
    }, search_limiter)

    return _search_response(documents, no_match_hint=(
        "No chunk contains this exact word sequence. Check the spelling, "
        "shorten the phrase, or use `keyword_search` for single terms."
        if candidates == 0 else
        f"At least {candidates} chunk(s) contain the phrase, but none is "
        "relevant enough to the query. Rephrase the query or leave it empty."
    ))


@tool(
    name="filtered_search",
    title="Filtered search",
    description=(
        "Hybrid search restricted by metadata — needs at least one filter: "
        "within a list any value matches, across filters all must match. Take "
        "keyword and entity values from chunks read with `read_chunks`. "
        "Returns ranked chunks as ids with a short snippet; read promising "
        "ones with `read_chunks`."
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
        description="Match any enriched keyword; exact and case-sensitive.",
    )],
    entities: Annotated[list[str], Field(
        default=[],
        max_length=5,
        description=(
            "Match any person, organization, product, or location; exact and "
            "case-sensitive, so list name variants."
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
        description="Latest mentioned date, inclusive.",
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

    if (date_from and date_to and date_from > date_to) or (
            modified_from and modified_to and modified_from > modified_to):
        return SearchResult(
            hint="Inverted date range: the earliest date is after the latest.",
            hits=[],
        )

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

    if date_from or date_to:
        conditions.append(FieldCondition(
            key="meta.dates",
            # one condition, so both bounds apply to the same date of a chunk
            range=DatetimeRange(gte=date_from, lte=date_to),
        ))

    if modified_from or modified_to:
        conditions.append(FieldCondition(
            key="meta.modified_at",
            range=DatetimeRange(
                gte=datetime.combine(modified_from, time.min, tzinfo=UTC)
                if modified_from else None,
                lte=datetime.combine(modified_to, time.max, tzinfo=UTC)
                if modified_to else None,
            ),
        ))

    if not conditions:
        return SearchResult(
            hint=(
                "Provide at least one filter; without filters use "
                "`keyword_and_semantic_search`."
            ),
            hits=[],
        )

    filters = Filter(must=conditions)

    documents, candidates = await _run_search(hybrid_pipeline, {
        "dense_embedder": {"text": query},
        "sparse_embedder": {"text": query},
        "dense_retriever": {"filters": filters},
        "sparse_retriever": {"filters": filters},
        "reranker": {"query": query},
    }, search_limiter)

    return _search_response(documents, no_match_hint=(
        "No chunk matches all filters. Take keyword and entity values from "
        "chunks you have read, or drop filters one at a time."
        if candidates == 0 else
        f"At least {candidates} chunk(s) match the filters, but none is "
        "relevant enough to the query. Rephrase the query or relax the filters."
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
        f"At least {candidates} chunk(s) mention the same entities, but none "
        "is relevant enough to the query. Rephrase the query toward what they "
        "would say."
    ))
