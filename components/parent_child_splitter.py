"""
ParentChildSplitter — Haystack 2.x custom component.

Receives section documents from ``MarkdownHeaderSplitter`` (one per heading
section) and implements the parent-child chunking strategy:

• Small sections  (≤ child_chunk_size words):
    Passed through unchanged.  ``meta["parent_content"]`` is set to the
    section's own content — the "parent" and "child" are the same document.

• Large sections  (> child_chunk_size words):
    Split into smaller child chunks via Haystack's built-in
    ``RecursiveDocumentSplitter``.  Every child receives the full original
    section text in ``meta["parent_content"]``.

This is the second half of the Anthropic Contextual Retrieval Option A
splitting strategy:
    1. ``MarkdownHeaderSplitter`` — semantic section boundaries (built-in)
    2. ``ParentChildSplitter``    — size budget + parent-child linking (this file)

The dense embedder embeds the *child* (small, precise signal).
The query pipeline's ``swap_to_parent_content()`` passes the *parent*
(full section text) to the LLM for richer answer context.
"""

import logging
from typing import Any

from haystack import Document, component
from haystack.components.preprocessors import RecursiveDocumentSplitter

logger = logging.getLogger(__name__)


@component
class ParentChildSplitter:
    """
    Adds parent-child linking to section documents.

    Args:
        child_chunk_size:    Maximum words per child chunk.
        child_chunk_overlap: Word overlap carried between adjacent child chunks.
    """

    def __init__(
        self,
        child_chunk_size: int = 200,
        child_chunk_overlap: int = 20,
    ) -> None:
        self.child_chunk_size = child_chunk_size
        # Haystack's RecursiveDocumentSplitter handles the secondary splitting.
        # It tries progressively finer separators (\n\n → sentence → \n → space)
        # ensuring splits happen at natural boundaries.
        self._splitter = RecursiveDocumentSplitter(
            split_length=child_chunk_size,
            split_overlap=child_chunk_overlap,
            split_unit="word",
        )

    @component.output_types(documents=list[Document])
    def run(self, documents: list[Document]) -> dict[str, list[Document]]:
        output: list[Document] = []
        child_count = 0

        for section in documents:
            content    = section.content or ""
            word_count = len(content.split())

            if word_count <= self.child_chunk_size:
                # Section fits in one child — parent and child are identical
                output.append(_with_parent(section, content))
                child_count += 1
            else:
                # Section is too large — split into children
                split_result = self._splitter.run(documents=[section])
                children     = split_result.get("documents", [])
                for child in children:
                    output.append(_with_parent(child, content))
                child_count += len(children)

        logger.info(
            "ParentChildSplitter: %d section(s) → %d chunk(s)",
            len(documents),
            child_count,
        )
        return {"documents": output}


# ---------------------------------------------------------------------------

def _with_parent(child: Document, parent_text: str) -> Document:
    """Return a new Document with parent_content in meta."""
    meta: dict[str, Any] = dict(child.meta)
    meta["parent_content"] = parent_text
    meta["parent_section"] = meta.get("header", "")
    return Document(content=child.content, meta=meta, id=child.id)
