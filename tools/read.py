from typing import Annotated

from fastmcp import Context
from fastmcp.tools import tool
from haystack import Document
from mcp.types import ToolAnnotations
from pydantic import Field
from qdrant_client.http.models import (
    Condition,
    FieldCondition,
    Filter,
    MatchAny,
    MatchValue,
)

from config import get_settings
from schemas.results import ChunkContent, ReadResult

settings = get_settings()


def _read_response(
    documents: list[Document],
) -> ReadResult:
    return ReadResult(
        hint="" if documents else (
            "No chunks found. Pass the full ids exactly as returned by a search."
        ),
        chunks=[ChunkContent.model_validate({
            **document.meta,
            "id": document.id,
            "content": document.content,
        }) for document in documents],
    )


@tool(
    name="read_chunks",
    title="Read chunks",
    description=(
        "Read chunks in full by id (from a search result). Returns the complete "
        "text of each with title, link to cite and document metadata, and "
        "the keywords, entities and dates for `filtered_search`."
    ),
    annotations=ToolAnnotations(read_only_hint=True, open_world_hint=False),
    timeout=settings.tool_timeout,
)
async def read_chunks(
    ctx: Context,
    chunk_ids: Annotated[list[str], Field(
        min_length=1, max_length=5,
        description="The chunk ids to read in full. At most 5 search hits.",
    )],
) -> ReadResult:

    document_store = ctx.lifespan_context["document_store"]

    documents = await document_store.get_documents_by_id_async(chunk_ids)

    return _read_response(documents)


@tool(
    name="read_neighbors",
    title="Read surrounding chunks",
    description=(
        "Read the given chunks together with the chunks immediately before "
        "and after them in their document, in document order — "
        "recovers the context around a promising hit. Returns the complete "
        "text of each with title, link to cite and document metadata, and "
        "the keywords, entities and dates for `filtered_search`."
    ),
    annotations=ToolAnnotations(read_only_hint=True, open_world_hint=False),
    timeout=settings.tool_timeout,
)
async def read_neighbors(
    ctx: Context,
    chunk_ids: Annotated[list[str], Field(
        min_length=1, max_length=3,
        description=(
            "The chunk ids (from a search result) to read the surrounding "
            "context of. At most 3; call again for more."
        ),
    )],
    window: Annotated[int, Field(
        ge=1, le=2,
        default=1,
        description="Chunks before and after each id. Defaults to 1.",
    )],
) -> ReadResult:

    document_store = ctx.lifespan_context["document_store"]

    seeds = await document_store.get_documents_by_id_async(chunk_ids)

    if not seeds:
        return _read_response([])

    conditions: list[Condition] = []

    for seed in seeds:

        index = seed.meta["chunk_index"]
        conditions.append(Filter(must=[
            FieldCondition(
                key="meta.hash",
                match=MatchValue(value=seed.meta["hash"]),
            ),
            FieldCondition(
                key="meta.chunk_index",
                match=MatchAny(any=[
                    neighbor
                    for neighbor in range(index - window, index + window + 1)
                    if 0 <= neighbor < seed.meta["total_chunks"]
                ]),
            ),
        ]))

    neighbors = await document_store.filter_documents_async(
        filters=Filter(should=conditions))

    neighbors.sort(key=lambda document: (
        document.meta["hash"],
        document.meta["chunk_index"],
    ))

    return _read_response(neighbors)
