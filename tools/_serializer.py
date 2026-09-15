from typing import cast

from haystack import Document

from schemas.results import ChunkContent, ReadResult, SearchHit, SearchResult
from services.rustfs import RustfsStore


def _search_hits(
    documents: list[Document],
) -> list[SearchHit]:
    # no url: the agent must read a chunk before it may cite one
    return [SearchHit(
        id=document.id,
        # reranked, so never None despite haystack's optional type
        score=cast(float, document.score),
        source=document.meta["source"],
        page=document.meta.get("page_number"),
        headings=document.meta["headings"],
        snippet=(document.meta.get("context") or document.content or "")[:300],
    ) for document in documents]


def _chunk_contents(
    documents: list[Document],
    rustfs_store: RustfsStore,
) -> list[ChunkContent]:

    chunks: list[ChunkContent] = []

    for document in documents:

        source = document.meta["source"]
        url = rustfs_store.presigned_url(source)
        page = document.meta.get("page_number")
        chunks.append(ChunkContent(
            id=document.id,
            source=source,
            # fragment stays client-side, so the presigned signature is unaffected
            url=f"{url}#page={page}" if page else url,
            page=page,
            content=document.content,
        ))

    return chunks


def search_response(
    documents: list[Document],
    no_match_hint: str = (
        "No matches. Reformulate or broaden the query and search again. Use "
        "`filtered_search` only when you have reliable metadata to narrow by."
    ),
) -> SearchResult:
    return SearchResult(
        hint="" if documents else no_match_hint,
        hits=_search_hits(documents),
    )


def read_response(
    documents: list[Document],
    rustfs_store: RustfsStore,
) -> ReadResult:
    return ReadResult(
        hint="" if documents else (
            "No chunks found. Run a search first to get valid chunk ids."
        ),
        chunks=_chunk_contents(documents, rustfs_store),
    )
