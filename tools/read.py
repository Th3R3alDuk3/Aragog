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
from services.rustfs import RustfsStore

settings = get_settings()


def _read_response(
    documents: list[Document],
    rustfs_store: RustfsStore,
) -> ReadResult:

    chunks: list[ChunkContent] = []

    for document in documents:

        url = rustfs_store.presigned_url(document.meta["source"])
        page = document.meta.get("page_number")
        chunks.append(ChunkContent.model_validate({
            **document.meta,
            "id": document.id,
            # fragment stays client-side, so the presigned signature is unaffected
            "url": f"{url}#page={page}" if page else url,
            "page": page,
            "content": document.content,
        }))

    return ReadResult(
        hint="" if chunks else (
            "No chunks found. Pass the full ids exactly as returned by a search."
        ),
        chunks=chunks,
    )


@tool(
    name="read_chunks",
    title="Read chunks",
    description=(
        "Read chunks in full by id (from a search result). Returns the complete "
        "text of each with source, page and modification date, the keywords, "
        "entities and dates for `filtered_search`, and a temporary link to cite."
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
    rustfs_store = ctx.lifespan_context["rustfs_store"]

    documents = await document_store.get_documents_by_id_async(chunk_ids)

    return _read_response(documents, rustfs_store)


@tool(
    name="read_neighbors",
    title="Read surrounding chunks",
    description=(
        "Read the given chunks together with the chunks immediately before "
        "and after them in their source document, in document order — "
        "recovers the context around a promising hit. Returns the complete "
        "text of each with source, page and modification date, the keywords, "
        "entities and dates for `filtered_search`, and a temporary link to cite."
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
    rustfs_store = ctx.lifespan_context["rustfs_store"]

    seeds = await document_store.get_documents_by_id_async(chunk_ids)

    if not seeds:
        return _read_response([], rustfs_store)

    conditions: list[Condition] = []

    for seed in seeds:

        index = seed.meta["chunk_index"]
        conditions.append(Filter(must=[
            FieldCondition(
                key="meta.source",
                match=MatchValue(value=seed.meta["source"]),
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
        document.meta["source"],
        document.meta["chunk_index"],
    ))

    return _read_response(neighbors, rustfs_store)
