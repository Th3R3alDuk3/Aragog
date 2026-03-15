from fastapi import APIRouter, Depends, HTTPException, Request, status

from adapters.api.deps import get_query_engine
from adapters.api.models.query import QueryRequest, QueryResponse, SourceDocument
from core.models.query import QueryInput
from core.models.retrieval import RetrievedSource
from core.services.query_engine import GenerationError
from core.services.retrieval_engine import NoDocumentsFoundError, RetrievalError

router = APIRouter(prefix="/query", tags=["query"])


def _to_source_documents(sources: list[RetrievedSource], api_base_url: str) -> list[SourceDocument]:
    base_url = api_base_url.rstrip("/")
    response_sources: list[SourceDocument] = []

    for source in sources:
        meta = dict(source.meta)
        minio_key = meta.get("minio_key")
        if minio_key:
            meta["download_url"] = f"{base_url}/documents/download/{minio_key}"
        response_sources.append(SourceDocument(content=source.content, score=source.score, meta=meta))

    return response_sources


@router.post("", response_model=QueryResponse, summary="Query the RAG system")
async def query_rag(
    http_request: Request,
    request: QueryRequest,
    query_engine=Depends(get_query_engine),
) -> QueryResponse:
    try:
        result = await query_engine.query(
            QueryInput(
                query=request.query,
                top_k=request.top_k,
                filters=request.filters,
                date_from=request.date_from,
                date_to=request.date_to,
            )
        )
    except ValueError as error:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Invalid query or filter parameters: {error}",
        ) from error
    except RetrievalError as error:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Document retrieval failed.",
        ) from error
    except NoDocumentsFoundError as error:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(error),
        ) from error
    except GenerationError as error:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(error),
        ) from error

    return QueryResponse(
        answer=result.answer,
        sources=_to_source_documents(result.sources, str(http_request.base_url)),
        query=result.query,
        sub_questions=result.sub_questions,
        is_compound=result.is_compound,
        low_confidence=result.low_confidence,
        extracted_filters=result.extracted_filters,
    )
