from haystack import Document

from config import get_settings
from models.results import ChunkContent, ReadResult, SearchHit, SearchResult
from services.storage import MinioStore


settings = get_settings()


def _search_hits(
    documents: list[Document],
    minio_store: MinioStore,
) -> list[SearchHit]:
    return [SearchHit(
        id=document.id,
        score=document.score,
        source=document.meta.get("source"),
        url=minio_store.presigned_url(
            document.meta.get("source"), settings.minio_url_expire),
        page=document.meta.get("page_number"),
        headings=document.meta.get("headings", []),
        snippet=(
            document.meta.get("context")
            or (document.content or "")[:300]
        ),
    ) for document in documents]


def _chunk_contents(
    documents: list[Document],
    minio_store: MinioStore,
) -> list[ChunkContent]:
    return [ChunkContent(
        id=document.id,
        source=document.meta.get("source"),
        url=minio_store.presigned_url(
            document.meta.get("source"), settings.minio_url_expire),
        page=document.meta.get("page_number"),
        content=document.content,
    ) for document in documents]


def search_response(
    documents: list[Document],
    minio_store: MinioStore,
) -> SearchResult:

    hint = (
        "Open promising hits in full with `read_chunk` before answering. "
        "Use `find_related` to expand via a hit's entities, or "
        "`read_neighbors` for surrounding context."
    ) if documents else (
        "No matches. "
        "Reformulate the query or narrow it with `filtered_search`."
    )

    return SearchResult(
        hint=hint,
        hits=_search_hits(documents, minio_store),
    )


def read_response(
    documents: list[Document],
    minio_store: MinioStore,
) -> ReadResult:

    hint = (
        "Ground your answer in this content and cite source and page. "
        "Use `read_neighbors` or `find_related` to dig further."
    ) if documents else (
        "No chunks found. "
        "Run a search first to get valid chunk ids."
    )

    return ReadResult(
        hint=hint,
        chunks=_chunk_contents(documents, minio_store),
    )
