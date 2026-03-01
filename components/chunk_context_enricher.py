"""
ChunkContextEnricher — Haystack 2.x custom component.

Runs AFTER MarkdownHeaderSplitter + ParentChildSplitter on the list of
chunks.  It groups chunks by their parent document (via ``doc_id``) and
annotates each chunk with:

chunk_index   : 0-based position within its parent document.
chunk_total   : total number of chunks from that document.
section_title : immediate heading text from ``meta["header"]``
                (set by MarkdownHeaderSplitter).
section_path  : breadcrumb built from ``meta["parent_headers"] + [header]``
                (e.g. "Introduction › Background › Overview").
chunk_type    : content-type heuristic — one of:
                  text | table | code | list | figure_caption
"""

import logging
import re
from collections import defaultdict
from typing import Any

from haystack import Document, component

logger = logging.getLogger(__name__)

_TABLE_RE  = re.compile(r"^\|.+\|", re.MULTILINE)
_CODE_RE   = re.compile(r"```[\s\S]*?```|`[^`]+`")
_LIST_RE   = re.compile(r"^(\s*[-*+]|\s*\d+\.)\s", re.MULTILINE)
_FIGURE_RE = re.compile(r"!\[.*?\]\(.*?\)|Figure\s+\d+", re.IGNORECASE)


@component
class ChunkContextEnricher:
    """
    Adds chunk-position and structural context metadata to split chunks.

    Reads ``meta["header"]`` and ``meta["parent_headers"]`` injected by
    ``MarkdownHeaderSplitter`` — no regex scanning required.
    """

    @component.output_types(documents=list[Document])
    def run(self, documents: list[Document]) -> dict[str, list[Document]]:
        """Annotate each chunk with its position and structural context metadata.

        Chunks are grouped by their parent document (``doc_id``) to compute
        per-document counters, then re-sorted to preserve pipeline order.

        Args:
            documents: Chunks from ParentChildSplitter, grouped by source document.

        Returns:
            A dict with key ``"documents"`` containing the annotated chunks in
            their original order.
        """
        # Group chunks by parent doc_id (set by MetadataEnricher)
        groups: dict[str, list[Document]] = defaultdict(list)
        for doc in documents:
            key = doc.meta.get("doc_id", doc.id or "unknown")
            groups[key].append(doc)

        enriched: list[Document] = []
        for doc_id, chunks in groups.items():
            enriched.extend(_enrich_group(doc_id, chunks))

        # Restore original ordering (Haystack pipelines expect stable order)
        id_to_order = {doc.id: i for i, doc in enumerate(documents)}
        enriched.sort(key=lambda d: id_to_order.get(d.id, 0))

        logger.info(
            "ChunkContextEnricher: processed %d chunk(s) across %d document(s)",
            len(enriched),
            len(groups),
        )
        return {"documents": enriched}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _enrich_group(doc_id: str, chunks: list[Document]) -> list[Document]:
    """Annotate all chunks belonging to one source document.

    Args:
        doc_id: Identifier of the parent document (used only for logging).
        chunks: All chunks that originate from this document, in order.

    Returns:
        The same chunks with ``chunk_index``, ``chunk_total``, ``section_title``,
        ``section_path``, and ``chunk_type`` added to their metadata.
    """
    total = len(chunks)

    result: list[Document] = []
    for idx, chunk in enumerate(chunks):
        content = chunk.content or ""
        meta: dict[str, Any] = dict(chunk.meta)

        # ----------------------------------------------------------------
        # Chunk position
        # ----------------------------------------------------------------
        meta["chunk_index"] = idx
        meta["chunk_total"] = total

        # ----------------------------------------------------------------
        # Section title / path — read from MarkdownHeaderSplitter metadata
        # meta["header"]         : immediate heading text (str)
        # meta["parent_headers"] : list of ancestor headings (list[str])
        # ----------------------------------------------------------------
        header         = meta.get("header", "") or ""
        parent_headers = meta.get("parent_headers") or []

        meta["section_title"] = header
        breadcrumb = [h for h in parent_headers if h] + ([header] if header else [])
        meta["section_path"]  = " › ".join(breadcrumb)

        # ----------------------------------------------------------------
        # Chunk type heuristic
        # ----------------------------------------------------------------
        meta["chunk_type"] = _detect_chunk_type(content)

        result.append(Document(content=content, meta=meta, id=chunk.id))

    return result


def _detect_chunk_type(content: str) -> str:
    """Best-effort heuristic for the dominant content type in a chunk."""
    stripped = content.strip()
    if not stripped:
        return "text"

    lines = stripped.splitlines()
    table_lines = sum(1 for l in lines if _TABLE_RE.match(l))
    list_lines  = sum(1 for l in lines if _LIST_RE.match(l))

    if _CODE_RE.search(stripped):
        return "code"
    if table_lines >= max(2, len(lines) * 0.4):
        return "table"
    if _FIGURE_RE.search(stripped):
        return "figure_caption"
    if list_lines >= max(2, len(lines) * 0.4):
        return "list"
    return "text"
