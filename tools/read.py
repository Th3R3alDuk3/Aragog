from typing import Annotated

from fastmcp import Context
from fastmcp.tools import tool
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
from schemas.results import ReadResult
from tools._serializer import read_response

settings = get_settings()


@tool(
    name="read_chunks",
    title="Read chunks",
    description=(
        "Read chunks in full by id (from a search result). Returns the complete "
        "text of each, with its source, page and a temporary link to cite."
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

    documents = await document_store.filter_documents_async(
        filters=Filter(must=[FieldCondition(
            key="id",
            match=MatchAny(any=chunk_ids),
        )]))

    return read_response(documents, rustfs_store)


@tool(
    name="read_neighbors",
    title="Read surrounding chunks",
    description=(
        "Read the chunks immediately before and after the given ids within "
        "their source document, in document order — recovers the context "
        "around a promising hit. Returns the complete text of each, with its "
        "source, page and a temporary link to cite."
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
        description=(
            "Chunks before and after each id. Defaults to 1."
        ),
    )],
) -> ReadResult:

    document_store = ctx.lifespan_context["document_store"]
    rustfs_store = ctx.lifespan_context["rustfs_store"]

    seeds = await document_store.filter_documents_async(
        filters=Filter(must=[FieldCondition(
            key="id",
            match=MatchAny(any=chunk_ids),
        )]))

    if not seeds:
        return read_response([], rustfs_store)

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

    return read_response(neighbors, rustfs_store)
