from re import compile

from fastmcp.utilities.logging import get_logger
from haystack import Document
from haystack_integrations.components.rankers.vllm import VLLMRanker

from config import get_settings
from models.results import ChunkContent, ReadResult, SearchHit, SearchResult
from services.storage import MinioStore


logger = get_logger(__name__)

settings = get_settings()


_SENTENCE_SPLIT = compile(r"(?<=[.!?])\s+|\n+")


def _split_sentences(content: str | None) -> list[str]:
    return [s for s in map(str.strip, _SENTENCE_SPLIT.split(content or "")) if len(s) > 10]


async def _rerank_snippets(
    documents: list[Document],
    query: str,
    reranker: VLLMRanker,
    max_sentences: int = 2,
) -> dict[str, str]:

    sentences = [
        Document(content=sentence, meta={"chunk_id": document.id, "position": position})
        for document in documents
        for position, sentence in enumerate(_split_sentences(document.content))
    ]

    if not sentences:
        return {}

    try:
        result = await reranker.run_async(query=query, documents=sentences)
        ranked_sentences = result["documents"]
    except Exception as error:
        logger.warning(f"snippet reranking failed, falling back to chunk context/prefix: {error}")
        return {}

    best_snippets: dict[str, list[Document]] = {}

    for sentence in ranked_sentences:
        chosen = best_snippets.setdefault(sentence.meta["chunk_id"], [])
        if len(chosen) < max_sentences:
            chosen.append(sentence)

    return {
        chunk_id: " ... ".join(
            sentence.content
            for sentence in sorted(chosen, key=lambda s: s.meta["position"])
        ) for chunk_id, chosen in best_snippets.items()
    }


def _search_hits(
    documents: list[Document],
    minio_store: MinioStore,
    snippets: dict[str, str],
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
            snippets.get(document.id)
            or document.meta.get("context")
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


async def search_response(
    documents: list[Document],
    minio_store: MinioStore,
    query: str,
    reranker: VLLMRanker,
) -> SearchResult:

    hint = (
        "Open promising hits in full with `read_chunk` before answering. "
        "Use `find_related` to expand via a hit's entities, or "
        "`read_neighbors` for surrounding context."
    ) if documents else (
        "No matches. "
        "Reformulate the query or narrow it with `filtered_search`."
    )

    snippets = await _rerank_snippets(documents, query, reranker)

    return SearchResult(
        hint=hint,
        hits=_search_hits(documents, minio_store, snippets),
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
