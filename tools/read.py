from typing import Annotated

from fastmcp import Context
from fastmcp.tools import tool
from mcp.types import ToolAnnotations
from pydantic import Field

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
    annotations=ToolAnnotations(readOnlyHint=True, openWorldHint=False),
    timeout=settings.tool_timeout,
)
async def read_chunks(
    ctx: Context,
    chunk_ids: Annotated[list[str], Field(
        max_length=5,
        description="The chunk ids to read in full. At most 5 search hits.",
    )],
) -> ReadResult:

    document_store = ctx.lifespan_context["document_store"]
    minio_store = ctx.lifespan_context["minio_store"]

    documents = await document_store.filter_documents_async(
        filters={"field": "id", "operator": "in", "value": chunk_ids})

    return await read_response(documents, minio_store)


@tool(
    name="read_neighbors",
    title="Read surrounding chunks",
    description=(
        "Read the chunks immediately before and after the given ids within "
        "their source document, in document order — recovers the context "
        "around a promising hit. Returns the complete text of each, with its "
        "source, page and a temporary link to cite."
    ),
    annotations=ToolAnnotations(readOnlyHint=True, openWorldHint=False),
    timeout=settings.tool_timeout,
)
async def read_neighbors(
    ctx: Context,
    chunk_ids: Annotated[list[str], Field(
        max_length=3,
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
        return await read_response([], minio_store)

    neighbors = await document_store.filter_documents_async(
        filters={"operator": "OR", "conditions": conditions})

    neighbors.sort(key=lambda document: (
        document.meta.get("source") or "",
        document.meta.get("chunk_index", 0),
    ))

    return await read_response(neighbors, minio_store)
