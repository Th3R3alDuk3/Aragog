from typing import Annotated

from fastmcp import Context
from fastmcp.tools import tool
from mcp.types import ToolAnnotations
from pydantic import Field

from config import get_settings
from models.results import SearchResult
from tools._helpers import search_response


settings = get_settings()


_ENTITY_FIELDS = (
    "ent_persons",
    "ent_organizations",
    "ent_products",
    "ent_locations",
)


@tool(
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
        ge=20, le=60,
        description=(
            "Number of candidate chunks to retrieve from each of dense and "
            "sparse before reranking. Defaults to 30."
        ),
    )] = 30,
    top_k_after: Annotated[int, Field(
        ge=3, le=10,
        description="Number of chunks to return after reranking. Defaults to 5.",
    )] = 5,
) -> SearchResult:

    minio_store = ctx.lifespan_context["minio_store"]
    hybrid_pipeline = ctx.lifespan_context["hybrid_pipeline"]
    result = await hybrid_pipeline.run_async({
        "dense_embedder": {"text": query},
        "sparse_embedder": {"text": query},
        "dense_retriever": {"top_k": top_k_before},
        "sparse_retriever": {"top_k": top_k_before},
        "reranker": {
            "query": query,
            "top_k": top_k_after,
            "score_threshold": settings.reranker_score_threshold,
        },
    })

    documents = result["reranker"]["documents"]

    return search_response(documents, minio_store)


@tool(
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
        ge=20, le=60,
        description=(
            "Number of candidate chunks to retrieve before reranking. "
            "Defaults to 30."
        ),
    )] = 30,
    top_k_after: Annotated[int, Field(
        ge=3, le=10,
        description="Number of chunks to return after reranking. Defaults to 5.",
    )] = 5,
) -> SearchResult:

    minio_store = ctx.lifespan_context["minio_store"]
    dense_pipeline = ctx.lifespan_context["dense_pipeline"]
    result = await dense_pipeline.run_async({
        "embedder": {"text": query},
        "retriever": {"top_k": top_k_before},
        "reranker": {
            "query": query,
            "top_k": top_k_after,
            "score_threshold": settings.reranker_score_threshold,
        },
    })

    documents = result["reranker"]["documents"]

    return search_response(documents, minio_store)


@tool(
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
        description=(
            "The keywords or exact terms to look up. Use short, specific terms — "
            "entity names, codes, dates or domain terms (good: 'Siemens AG', "
            "'Garantiezeit', '2024-03-01'), not full questions or long phrases "
            "(bad: 'how long is the warranty?' → use 'Garantiezeit', "
            "'warranty period')."
        ),
    )],
    top_k_before: Annotated[int, Field(
        ge=20, le=60,
        description=(
            "Number of candidate chunks to retrieve before reranking. "
            "Defaults to 30."
        ),
    )] = 30,
    top_k_after: Annotated[int, Field(
        ge=3, le=10,
        description="Number of chunks to return after reranking. Defaults to 5.",
    )] = 5,
) -> SearchResult:

    minio_store = ctx.lifespan_context["minio_store"]
    sparse_pipeline = ctx.lifespan_context["sparse_pipeline"]
    result = await sparse_pipeline.run_async({
        "embedder": {"text": query},
        "retriever": {"top_k": top_k_before},
        "reranker": {
            "query": query,
            "top_k": top_k_after,
            "score_threshold": settings.reranker_score_threshold,
        },
    })

    documents = result["reranker"]["documents"]

    return search_response(documents, minio_store)


@tool(
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
        description=(
            "A chunk matches if it carries any of these keywords. Use exact "
            "domain terms as they appear in the documents (e.g. ['Garantiezeit', "
            "'Rahmenvertrag']), not full phrases."
        ),
    )],
    entities: Annotated[list[str], Field(
        default_factory=list,
        description=(
            "A chunk matches if it mentions any of these entities — persons, "
            "organizations, products or locations (e.g. ['Siemens AG', "
            "'Angela Merkel', 'Berlin'])."
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
        ge=20, le=60,
        description=(
            "Number of candidate chunks to retrieve before reranking. "
            "Defaults to 30."
        ),
    )] = 30,
    top_k_after: Annotated[int, Field(
        ge=3, le=10,
        description="Number of chunks to return after reranking. Defaults to 5.",
    )] = 5,
) -> SearchResult:

    minio_store = ctx.lifespan_context["minio_store"]
    dense_pipeline = ctx.lifespan_context["dense_pipeline"]
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
                "field": f"meta.{field}", "operator": "in", "value": entities
            } for field in _ENTITY_FIELDS]
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
            "value": date_from,
        })

    if date_to:
        conditions.append({
            "field": "meta.dates",
            "operator": "<=",
            "value": date_to
        })

    if created_from:
        conditions.append({
            "field": "meta.created_at",
            "operator": ">=",
            "value": created_from
        })

    if created_to:
        conditions.append({
            "field": "meta.created_at",
            "operator": "<=",
            "value": created_to if "T" in created_to else f"{created_to}T23:59:59"
        })

    result = await dense_pipeline.run_async({
        "embedder": {"text": query},
        "retriever": {
            "top_k": top_k_before,
            "filters": {
                "operator": "AND",
                "conditions": conditions
            } if conditions else None,
        },
        "reranker": {
            "query": query,
            "top_k": top_k_after,
            "score_threshold": settings.reranker_score_threshold,
        },
    })

    documents = result["reranker"]["documents"]

    return search_response(documents, minio_store)


@tool(
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
        ge=20, le=60,
        description=(
            "Number of candidate chunks to retrieve before reranking. "
            "Defaults to 30."
        ),
    )] = 30,
    top_k_after: Annotated[int, Field(
        ge=3, le=10,
        description="Number of chunks to return after reranking. Defaults to 5.",
    )] = 5,
) -> SearchResult:

    document_store = ctx.lifespan_context["document_store"]
    minio_store = ctx.lifespan_context["minio_store"]
    dense_pipeline = ctx.lifespan_context["dense_pipeline"]
    seeds = await document_store.filter_documents_async(
        filters={"field": "id", "operator": "in", "value": chunk_ids})

    entities = sorted({
        entity
        for seed in seeds
        for field in _ENTITY_FIELDS
        for entity in seed.meta.get(field, [])
    })

    if not entities:
        return search_response([], minio_store)

    result = await dense_pipeline.run_async({
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
        "reranker": {
            "query": query,
            "top_k": top_k_after,
            "score_threshold": settings.reranker_score_threshold,
        },
    })

    documents = result["reranker"]["documents"]

    return search_response(documents, minio_store)
