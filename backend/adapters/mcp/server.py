from logging import INFO, basicConfig

from fastmcp import FastMCP
from fastmcp.utilities.lifespan import combine_lifespans

from adapters.mcp.servers.query import lifespan as query_lifespan
from adapters.mcp.servers.query import mcp as query_mcp

basicConfig(
    format="%(asctime)s %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
    level=INFO,
)

lifespans = combine_lifespans(query_lifespan)

mcp = FastMCP(
    name="Advanced RAG Query",
    instructions=(
        "This MCP server exposes read tools for querying and retrieving passages from "
        "documents that were indexed by the Advanced RAG backend. It does not browse "
        "the web and it does not ingest files directly."
    ),
    lifespan=lifespans,
)

mcp.mount(query_mcp)


def main() -> None:
    mcp.run()
