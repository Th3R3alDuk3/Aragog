from collections import defaultdict
from logging import getLogger
from re import IGNORECASE, MULTILINE, compile

from haystack import Document, component

from models.meta import ChunkMetadata

logger = getLogger(__name__)

_TABLE_RE = compile(r"^\|.+\|", MULTILINE)
_CODE_RE = compile(r"```[\s\S]*?```|`[^`]+`")
_LIST_RE = compile(r"^(\s*[-*+]|\s*\d+\.)\s", MULTILINE)
_FIGURE_RE = compile(r"!\[.*?\]\(.*?\)|Figure\s+\d+", IGNORECASE)
_HEADING_RE = compile(r"^(#{1,6})\s+(.*)", MULTILINE)


@component
class ChunkEnricher:
    """
    Adds chunk-position and structural context metadata to split chunks.

    ``HierarchicalDocumentSplitter`` does not inject header metadata, so
    headings are extracted by scanning each chunk's markdown content.
    A heading stack is maintained across chunks so that a heading from a
    previous chunk carries forward to headingless successor chunks.
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
            key = ChunkMetadata.model_validate(doc.meta).doc_id or doc.id or "unknown"
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
    # Heading context carried forward across chunks (index = heading level - 1).
    heading_stack: list[str] = []

    result: list[Document] = []
    for idx, chunk in enumerate(chunks):
        content = chunk.content or ""
        meta = ChunkMetadata.model_validate(chunk.meta)

        meta.chunk_index = idx
        meta.chunk_total = total

        # Update heading context from headings found in this chunk's content.
        # This propagates the section context even into chunks that start mid-section.
        for match in _HEADING_RE.finditer(content):
            level = len(match.group(1))
            title = match.group(2).strip()
            heading_stack = heading_stack[: level - 1] + [title]

        meta.section_title = heading_stack[-1] if heading_stack else ""
        meta.section_path = " › ".join(heading_stack)

        meta.chunk_type = _detect_chunk_type(content)

        dumped = meta.model_dump()
        # Pydantic drops __-prefixed keys (name mangling); restore them from the original meta
        for k, v in chunk.meta.items():
            if k.startswith("__"):
                dumped[k] = v
        result.append(Document(content=content, meta=dumped, id=chunk.id))

    return result


def _detect_chunk_type(content: str) -> str:
    """Best-effort heuristic for the dominant content type in a chunk."""
    stripped = content.strip()
    if not stripped:
        return "text"

    lines = stripped.splitlines()
    table_lines = sum(1 for ln in lines if _TABLE_RE.match(ln))
    list_lines = sum(1 for ln in lines if _LIST_RE.match(ln))

    if _CODE_RE.search(stripped):
        return "code"
    if table_lines >= max(2, len(lines) * 0.4):
        return "table"
    if _FIGURE_RE.search(stripped):
        return "figure_caption"
    if list_lines >= max(2, len(lines) * 0.4):
        return "list"
    return "text"
