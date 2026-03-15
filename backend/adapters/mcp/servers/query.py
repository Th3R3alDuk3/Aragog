from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import date
from typing import Annotated, Any

from fastmcp import Context, FastMCP
from pydantic import Field

from core.models.query import QueryInput
from core.models.retrieval import RetrievalInput
from core.runtime import RagRuntime, managed_runtime
from core.services.query_engine import GenerationError
from core.services.retrieval_engine import NoDocumentsFoundError, RetrievalError


@asynccontextmanager
async def lifespan(_: FastMCP[dict[str, RagRuntime]]) -> AsyncIterator[dict[str, RagRuntime]]:
    async with managed_runtime() as runtime:
        yield {"runtime": runtime}


mcp = FastMCP(name="Advanced RAG Query Tools", lifespan=lifespan)


def _require_runtime(ctx: Context) -> RagRuntime:
    runtime = ctx.lifespan_context.get("runtime")
    if runtime is None:
        raise RuntimeError("RAG runtime is not initialized.")
    return runtime


@mcp.tool(
    name="rag_query",
    description="Answer a natural-language question against the indexed RAG knowledge base.",
)
async def rag_query(
    ctx: Context,
    query: Annotated[str, Field(min_length=1)],
    top_k: Annotated[int, Field(ge=1, le=50)] = 5,
    filters: Annotated[dict[str, Any] | None, Field()] = None,
    date_from: Annotated[date | None, Field()] = None,
    date_to: Annotated[date | None, Field()] = None,
) -> dict[str, Any]:
    runtime = _require_runtime(ctx)
    if runtime.query_engine is None:
        raise RuntimeError("RAG runtime is not initialized.")

    try:
        result = await runtime.query_engine.query(
            QueryInput(
                query=query,
                top_k=top_k,
                filters=filters,
                date_from=date_from,
                date_to=date_to,
            )
        )
    except ValueError as error:
        raise RuntimeError(f"Invalid query or filter parameters: {error}") from error
    except RetrievalError as error:
        raise RuntimeError("Document retrieval failed.") from error
    except (NoDocumentsFoundError, GenerationError) as error:
        raise RuntimeError(str(error)) from error

    return {
        "answer": result.answer,
        "sources": [source.model_dump() for source in result.sources],
        "query": result.query,
        "sub_questions": result.sub_questions,
        "is_compound": result.is_compound,
        "low_confidence": result.low_confidence,
        "extracted_filters": result.extracted_filters,
    }


@mcp.tool(
    name="rag_retrieve",
    description="Retrieve relevant source passages without generating an answer.",
)
async def rag_retrieve(
    ctx: Context,
    query: Annotated[str, Field(min_length=1)],
    top_k: Annotated[int, Field(ge=1, le=50)] = 5,
    filters: Annotated[dict[str, Any] | None, Field()] = None,
    date_from: Annotated[date | None, Field()] = None,
    date_to: Annotated[date | None, Field()] = None,
) -> dict[str, Any]:
    runtime = _require_runtime(ctx)
    if runtime.retrieval_engine is None:
        raise RuntimeError("RAG runtime is not initialized.")

    try:
        result = await runtime.retrieval_engine.retrieve(
            RetrievalInput(
                query=query,
                top_k=top_k,
                filters=filters,
                date_from=date_from,
                date_to=date_to,
            )
        )
    except ValueError as error:
        raise RuntimeError(f"Invalid query or filter parameters: {error}") from error
    except RetrievalError as error:
        raise RuntimeError("Document retrieval failed.") from error
    except NoDocumentsFoundError as error:
        raise RuntimeError(str(error)) from error

    return {
        "query": result.query,
        "sources": [source.model_dump() for source in result.sources],
        "sub_questions": result.sub_questions,
        "is_compound": result.is_compound,
        "low_confidence": result.low_confidence,
        "extracted_filters": result.extracted_filters,
    }
