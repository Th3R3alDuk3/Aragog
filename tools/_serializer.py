from asyncio import gather

from haystack import Document

from config import get_settings
from schemas.results import ChunkContent, ReadResult, SearchHit, SearchResult
from services.minio import MinioStore

settings = get_settings()


def _search_hits(
    documents: list[Document],
) -> list[SearchHit]:
    # no url: the agent must read a chunk before it may cite one
    return [SearchHit(
        id=document.id,
        score=document.score,
        source=document.meta.get("source"),
        page=document.meta.get("page_number"),
        headings=document.meta.get("headings", []),
        snippet=(document.meta.get("context") or document.content or "")[:300],
    ) for document in documents]


async def _chunk_contents(
    documents: list[Document],
    minio_store: MinioStore,
) -> list[ChunkContent]:

    urls = await gather(*[
        minio_store.presigned_url(
            document.meta.get("source"), settings.minio_url_expire)
        for document in documents
    ])

    chunks: list[ChunkContent] = []

    for document, url in zip(documents, urls, strict=True):

        page = document.meta.get("page_number")
        chunks.append(ChunkContent(
            id=document.id,
            source=document.meta.get("source"),
            # fragment stays client-side, so the presigned signature is unaffected
            url=f"{url}#page={page}" if page else url,
            page=page,
            content=document.content,
        ))

    return chunks


def search_response(
    documents: list[Document],
) -> SearchResult:
    return SearchResult(
        hint="" if documents else (
            "No matches. Reformulate or broaden the query and search again. Use "
            "`filtered_search` only when you have reliable metadata to narrow by."
        ),
        hits=_search_hits(documents),
    )


async def read_response(
    documents: list[Document],
    minio_store: MinioStore,
) -> ReadResult:
    return ReadResult(
        hint="" if documents else (
            "No chunks found. Run a search first to get valid chunk ids."
        ),
        chunks=await _chunk_contents(documents, minio_store),
    )
