from asyncio import Semaphore, timeout

from fastmcp.exceptions import ToolError
from haystack import Document, Pipeline

from config import get_settings

settings = get_settings()


async def run_search(
    pipeline: Pipeline,
    inputs: dict,
    limiter: Semaphore,
) -> list[Document]:

    try:
        async with timeout(settings.search_timeout):
            async with limiter:
                result = await pipeline.run_async(inputs)
        return result["reranker"]["documents"]
    except TimeoutError:
        raise ToolError(
            f"Search did not complete within {settings.search_timeout}s — "
            "the server may be busy; retry in a few seconds or narrow "
            "the query."
        ) from None
