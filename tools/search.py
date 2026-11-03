from asyncio import Semaphore
from datetime import date
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
    MatchValue,
)

from config import get_settings
from schemas.enrichment import ENTITY_FIELDS
from schemas.results import DocumentFields, SearchHit, SearchResult

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
            title=document.meta["title"],
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
        "Search chunk texts (not document metadata) for an exact word sequence "
        "— codes, identifiers, § references, names or quoted wording. "
        "Case-insensitive, punctuation is ignored, word order matters; matches "
        "are ranked by the query, or by the phrase itself without one. Returns "
        "ranked chunks as ids with a short snippet; read promising ones with "
        "`read_chunks`."
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

    documents, _ = await _run_search(hybrid_pipeline, {
        "dense_embedder": {"text": text},
        "sparse_embedder": {"text": text},
        "dense_retriever": {"filters": filters},
        "sparse_retriever": {"filters": filters},
        # the phrase match is the relevance criterion, the query only ranks
        "reranker": {"query": text, "score_threshold": 0.0},
    }, search_limiter)

    return _search_response(documents, no_match_hint=(
        "No chunk contains this exact word sequence. Check the spelling, "
        "shorten the phrase, or use `keyword_search` for single terms."
    ))


@tool(
    name="filtered_search",
    title="Filtered search",
    description=(
        "Hybrid search restricted by metadata — needs at least one filter: "
        "within a list any value matches, across filters all must match. Take "
        "keyword and entity values from chunks read with `read_chunks`, "
        "document metadata values from `list_document_fields`. Without a query "
        "it lists each matching document once, e.g. for 'which documents are …'. "
        "Returns ranked chunks as ids with a short snippet; read promising "
        "ones with `read_chunks`."
    ),
    annotations=ToolAnnotations(read_only_hint=True, open_world_hint=False),
    timeout=settings.tool_timeout,
)
async def filtered_search(
    ctx: Context,
    query: Annotated[str, StringConstraints(
        strip_whitespace=True, max_length=800,
    ), Field(
        default="",
        description=(
            "What to rank the matching chunks by. Leave empty to list the "
            f"matching documents instead (at most {settings.search_top_k_before})."
        ),
    )],
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
    document_meta: Annotated[dict[str, str], Field(
        default={},
        max_length=5,
        description=(
            "Match document metadata fields exactly, by the names and values "
            "`list_document_fields` returns; a list field matches any of its "
            "values, a field inside objects is named `parent.child`."
        ),
    )],
    date_from: Annotated[date | None, Field(
        default=None,
        description="Earliest mentioned date, inclusive.",
    )],
    date_to: Annotated[date | None, Field(
        default=None,
        description="Latest mentioned date, inclusive.",
    )],
) -> SearchResult:

    hybrid_pipeline = ctx.lifespan_context["hybrid_pipeline"]
    search_limiter = ctx.lifespan_context["search_limiter"]

    if date_from and date_to and date_from > date_to:
        return _search_response([], no_match_hint=(
            "Inverted date range: the earliest date is after the latest."
        ))

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

    for field, value in document_meta.items():
        conditions.append(FieldCondition(
            key="meta.document_facets",
            match=MatchValue(value=f"{field}={value}"),
        ))

    if date_from or date_to:
        conditions.append(FieldCondition(
            key="meta.dates",
            # one condition, so both bounds apply to the same date of a chunk
            range=DatetimeRange(gte=date_from, lte=date_to),
        ))

    if not conditions:
        return _search_response([], no_match_hint=(
            "Provide at least one filter; without filters use "
            "`keyword_and_semantic_search`."
        ))

    filters = Filter(must=conditions)
    text = query or " ".join(str(value) for value in (
        *keywords, *entities, *document_meta.values(), date_from, date_to,
    ) if value)

    documents, candidates = await _run_search(hybrid_pipeline, {
        "dense_embedder": {"text": text},
        "sparse_embedder": {"text": text},
        "dense_retriever": {"filters": filters},
        "sparse_retriever": {"filters": filters},
        # without a query the filters are the criterion, so every match is listed
        "reranker": {"query": text} if query else {
            "query": text,
            "score_threshold": 0.0,
            "top_k": settings.search_top_k_before,
        },
    }, search_limiter)

    if not query:
        # a listing names each document once, by its best-ranked chunk
        listed: dict[str, Document] = {}
        for document in documents:
            listed.setdefault(document.meta["hash"], document)
        documents = list(listed.values())

    return _search_response(documents, no_match_hint=(
        "No chunk matches all filters. Take keyword and entity values from "
        "chunks you have read, document metadata values from "
        "`list_document_fields`, or drop filters one at a time."
        if candidates == 0 else
        f"At least {candidates} chunk(s) match the filters, but none is "
        "relevant enough to the query. Rephrase the query or relax the filters."
    ))


@tool(
    name="list_document_fields",
    title="List document metadata fields",
    description=(
        "List the document metadata fields with their most common exact values "
        "— the names and values `filtered_search` takes in `document_meta`. "
        "Counts are chunks, not documents; five values per field unless `field` "
        "is given. Empty when no document carries metadata."
    ),
    annotations=ToolAnnotations(read_only_hint=True, open_world_hint=False),
    timeout=settings.tool_timeout,
)
async def list_document_fields(
    ctx: Context,
    field: Annotated[str, StringConstraints(
        strip_whitespace=True, max_length=200,
    ), Field(
        default="",
        description="Only this field and its `field.key` parts; empty lists all.",
    )],
) -> DocumentFields:

    qdrant_client = ctx.lifespan_context["qdrant_client"]

    # counted by the payload index, most common values first
    facets = await qdrant_client.facet(
        collection_name=settings.qdrant_collection,
        key="meta.document_facets",
        limit=1000,
        exact=True,
    )

    fields: dict[str, dict[str, int]] = {}
    for facet in facets.hits:
        name, _, value = facet.value.partition("=")
        if field and name != field and not name.startswith(f"{field}."):
            continue
        values = fields.setdefault(name, {})
        # a few examples per field, or every value of the requested one
        if len(values) < (50 if field else 5):
            values[value] = facet.count

    return DocumentFields(
        hint="" if fields else (
            f"No document has the field `{field}`. Call again without a field "
            "to see every field name."
            if field else
            "No document carries metadata; filter by keywords or entities instead."
        ),
        fields=fields,
    )
