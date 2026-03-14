import os
from datetime import date
from typing import Annotated, Any

import httpx
from mcp.server.fastmcp import FastMCP
from pydantic import Field

from config import get_settings
from models.api import QueryRequest, QueryResponse


def _backend_base_url() -> str:
    explicit_url = os.getenv("RAG_API_URL")
    if explicit_url:
        return explicit_url.rstrip("/")

    settings = get_settings()
    host = settings.app_host
    if host in {"0.0.0.0", "::"}:
        host = "127.0.0.1"
    return f"http://{host}:{settings.app_port}"


def _extract_error_detail(response: httpx.Response) -> str:
    try:
        payload = response.json()
    except ValueError:
        return response.text.strip() or f"HTTP {response.status_code}"

    detail = payload.get("detail")
    if isinstance(detail, str) and detail.strip():
        return detail
    return str(payload)


mcp = FastMCP(
    name="Advanced RAG Query",
    instructions=(
        "This MCP server exposes exactly one read-only tool: `rag_query`. "
        "Use it to answer questions against documents that have already been indexed "
        "by the Advanced RAG backend. The tool returns grounded answers with source "
        "snippets and supports optional metadata filters plus semantic date limits. "
        "It does not browse the web and it does not ingest new files."
    ),
    json_response=True,
)


@mcp.tool(
    name="rag_query",
    description=(
        "Answer a natural-language question against the indexed RAG knowledge base. "
        "Use this for grounded answers from already ingested documents. "
        "Supports optional metadata filters and semantic date constraints."
    ),
)
async def rag_query(
    query: Annotated[
        str,
        Field(
            min_length=1,
            description=(
                "The natural-language question to answer. "
                "Can be a simple question or a compound query with multiple sub-questions."
            ),
        ),
    ],
    top_k: Annotated[
        int,
        Field(
            ge=1,
            le=50,
            description="Maximum number of source documents to return alongside the answer.",
        ),
    ] = 5,
    filters: Annotated[
        dict[str, Any] | None,
        Field(
            description=(
                "Optional metadata filters. Accepts either a Haystack filter expression "
                "like {'field': 'source', 'operator': '==', 'value': 'report.pdf'} "
                "or a shorthand mapping like {'source': 'report.pdf', 'language': 'de'}."
            ),
        ),
    ] = None,
    date_from: Annotated[
        date | None,
        Field(
            description=(
                "Optional inclusive lower date bound in ISO format, for example 2024-01-01. "
                "Matches the document's semantic date or covered period."
            ),
        ),
    ] = None,
    date_to: Annotated[
        date | None,
        Field(
            description=(
                "Optional inclusive upper date bound in ISO format, for example 2024-12-31. "
                "Matches the document's semantic date or covered period."
            ),
        ),
    ] = None,
) -> QueryResponse:
    """Query the indexed document corpus and return a grounded answer with citations.

    Returns the generated answer, source snippets, compound-query metadata, and
    automatically extracted filters from the backend's query analysis step.
    """
    payload = QueryRequest(
        query=query,
        top_k=top_k,
        filters=filters,
        date_from=date_from,
        date_to=date_to,
    ).model_dump(mode="json", exclude_none=True)

    async with httpx.AsyncClient(
        base_url=_backend_base_url(),
        timeout=httpx.Timeout(120.0, connect=10.0),
    ) as client:
        response = await client.post("/query", json=payload)

    if response.is_error:
        detail = _extract_error_detail(response)
        raise RuntimeError(
            f"RAG backend query failed with HTTP {response.status_code}: {detail}"
        )

    return QueryResponse.model_validate(response.json())


def main() -> None:
    mcp.run()


if __name__ == "__main__":
    main()
