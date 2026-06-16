from typing import Annotated

from fastmcp import Context
from fastmcp.tools import tool
from mcp.types import ToolAnnotations
from pydantic import Field

from models.results import ReadResult
from tools._helpers import read_response


@tool(
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

    document_store = ctx.lifespan_context["document_store"]
    minio_store = ctx.lifespan_context["minio_store"]

    documents = await document_store.filter_documents_async(
        filters={"field": "id", "operator": "in", "value": chunk_ids})

    return read_response(documents, minio_store)


@tool(
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
        ge=1, le=10,
        description=(
            "How many chunks before and after each id to include. Defaults to 1."
        ),
    )] = 1,
) -> ReadResult:

    document_store = ctx.lifespan_context["document_store"]
    minio_store = ctx.lifespan_context["minio_store"]

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
        return read_response([], minio_store)

    neighbors = await document_store.filter_documents_async(
        filters={"operator": "OR", "conditions": conditions})

    neighbors.sort(key=lambda document: (
        document.meta.get("source") or "",
        document.meta.get("chunk_index", 0),
    ))

    return read_response(neighbors, minio_store)

