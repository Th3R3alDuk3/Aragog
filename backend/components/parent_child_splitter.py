"""
ParentChildSplitter — Haystack 2.x custom component.

Uses Haystack's built-in ``HierarchicalDocumentSplitter`` to create a two-level
hierarchy without relying on markdown structure:

• Level 0 (root): the full document — stored in the parents collection so
  ``AutoMergingRetriever`` can recurse all the way up when the merge
  threshold is met at level 1.

• Level 1 (parents): larger chunks (parent_chunk_size words) — stored in the
  parents collection and returned as context by ``AutoMergingRetriever`` at
  query time.

• Level 2 (children): smaller chunks (child_chunk_size words) — embedded and
  indexed in the children collection for dense + sparse retrieval.

Every child carries ``meta["__parent_id"]`` (set by HierarchicalDocumentSplitter),
which ``AutoMergingRetriever`` uses at query time to fetch the matching parent
from the parents Qdrant collection.
"""

import logging
from bisect import bisect_right

from haystack import Document, component
from haystack.components.preprocessors import HierarchicalDocumentSplitter

from components.docling_converter import PAGE_BREAK_PLACEHOLDER

logger = logging.getLogger(__name__)


@component
class ParentChildSplitter:
    """
    Splits documents into a two-level hierarchy using HierarchicalDocumentSplitter.

    Args:
        parent_chunk_size:   Maximum words per parent chunk (level 1).
        child_chunk_size:    Maximum words per child chunk  (level 2, leaf).
        child_chunk_overlap: Word overlap between adjacent child chunks.
    """

    def __init__(
        self,
        parent_chunk_size: int = 600,
        child_chunk_size: int = 200,
        child_chunk_overlap: int = 20,
    ) -> None:
        self._splitter = HierarchicalDocumentSplitter(
            block_sizes={parent_chunk_size, child_chunk_size},
            split_overlap=child_chunk_overlap,
            split_by="word",
        )

    @component.output_types(children=list[Document], parents=list[Document])
    def run(self, documents: list[Document]) -> dict[str, list[Document]]:
        """Split documents into parent and child chunks.

        Args:
            documents: Cleaned documents from the DocumentCleaner stage.

        Returns:
            ``children``: leaf-level chunks (``__level == 2``) ready for
                          embedding and indexing in the children collection.
                          Each carries ``meta["__parent_id"]`` used by
                          ``AutoMergingRetriever`` at query time.
            ``parents``:  level-0 root + level-1 chunks stored in the parents
                          collection.  Both levels are required so that
                          ``AutoMergingRetriever`` can recurse from level 2 →
                          level 1 → level 0 without a missing-document crash.
        """
        all_docs: list[Document] = []
        for document in documents:
            split_docs = self._splitter.run(documents=[document])["documents"]
            all_docs.extend(_annotate_pages(split_docs, document.content or ""))

        # AutoMergingRetriever recursively walks __parent_id links upward.
        # If a level-1 set merges into level-0, the level-0 root must exist in
        # the parents store or the retriever raises ValueError.
        parents  = [d for d in all_docs if d.meta.get("__level") in (0, 1)]
        children = [d for d in all_docs if d.meta.get("__level") == 2]

        logger.info(
            "ParentChildSplitter: %d doc(s) → %d parent(s), %d child(ren)",
            len(documents),
            len(parents),
            len(children),
        )
        return {"children": children, "parents": parents}


def _annotate_pages(split_docs: list[Document], source_content: str) -> list[Document]:
    page_starts = _page_starts(source_content)
    absolute_starts: dict[str, int] = {}
    annotated: list[Document] = []

    for doc in sorted(split_docs, key=lambda item: int(item.meta.get("__level") or 0)):
        meta = dict(doc.meta)
        parent_id = meta.get("__parent_id")
        relative_start = int(meta.get("split_idx_start") or 0)

        if parent_id and parent_id in absolute_starts:
            absolute_start = absolute_starts[parent_id] + relative_start
        else:
            absolute_start = relative_start if meta.get("__level") else 0

        absolute_starts[doc.id] = absolute_start

        raw_content = doc.content or ""
        absolute_end = absolute_start + max(len(raw_content) - 1, 0)

        meta["page_start"] = _page_number(page_starts, absolute_start)
        meta["page_end"] = _page_number(page_starts, absolute_end)
        meta["page_count"] = len(page_starts)

        if "doc_beginning" in meta:
            meta["doc_beginning"] = _strip_page_breaks(str(meta["doc_beginning"]))

        annotated.append(
            Document(
                content=_strip_page_breaks(raw_content),
                meta=meta,
                id=doc.id,
            )
        )

    return annotated


def _page_starts(content: str) -> list[int]:
    starts = [0]
    offset = 0

    while True:
        marker_index = content.find(PAGE_BREAK_PLACEHOLDER, offset)
        if marker_index < 0:
            return starts

        next_start = marker_index + len(PAGE_BREAK_PLACEHOLDER)
        while next_start < len(content) and content[next_start] in "\r\n\t ":
            next_start += 1

        starts.append(next_start)
        offset = next_start


def _page_number(page_starts: list[int], absolute_offset: int) -> int:
    return max(1, bisect_right(page_starts, absolute_offset))


def _strip_page_breaks(content: str) -> str:
    return content.replace(PAGE_BREAK_PLACEHOLDER, "").strip()
