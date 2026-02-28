"""
MetadataEnricher — Haystack 2.x custom component.

Runs BEFORE DocumentSplitter on the full (unsplit) document that DoclingConverter
produced.  Enriches each document with stable, document-level metadata so that
every chunk inherits them after splitting.

Metadata added
--------------
doc_id          : SHA-256 of (file_path + raw content). Stable across re-indexing
                  if the file does not change.  Used to overwrite stale chunks.
title           : First H1 heading found in the markdown, or the filename stem.
word_count      : Approximate word count of the full document.
indexed_at      : ISO-8601 UTC timestamp of this indexing run.
indexed_at_ts   : Unix epoch integer of indexed_at — used for date-range filters
                  (Qdrant range filter on integer fields is natively supported).
language        : ISO 639-1 language code detected by langdetect on the full
                  document text before splitting.  Detecting on the full text is
                  more accurate than detecting per-chunk.  Falls back to "unknown"
                  if detection fails (short or mixed-language documents).
embedding_model : From Settings — recorded once so every chunk knows which model
                  produced its vector.
embedding_provider / embedding_dimension : ditto.
"""

import hashlib
import re
from datetime import datetime, timezone
from typing import Any

import logging

from haystack import Document, component

try:
    from langdetect import detect as _langdetect
    from langdetect import DetectorFactory
    DetectorFactory.seed = 0   # reproducible results
    _LANGDETECT_AVAILABLE = True
except ImportError:
    _LANGDETECT_AVAILABLE = False

logger = logging.getLogger(__name__)

# Matches ATX-style markdown headings: # H1 / ## H2 / ### H3 …
_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+)$", re.MULTILINE)


@component
class MetadataEnricher:
    """
    Enriches full (pre-split) documents with document-level metadata.

    Args:
        embedding_model:     Model identifier recorded in metadata.
        embedding_provider:  Provider name recorded in metadata.
        embedding_dimension: Vector dimension recorded in metadata.
    """

    def __init__(
        self,
        embedding_model: str,
        embedding_provider: str,
        embedding_dimension: int,
        doc_beginning_chars: int = 1500,
    ) -> None:
        self.embedding_model = embedding_model
        self.embedding_provider = embedding_provider
        self.embedding_dimension = embedding_dimension
        self.doc_beginning_chars = doc_beginning_chars

    @component.output_types(documents=list[Document])
    def run(self, documents: list[Document], extra_meta: dict | None = None) -> dict[str, list[Document]]:
        """
        Args:
            documents:  Raw documents from DoclingConverter.
            extra_meta: Optional key/value pairs merged into every document's meta
                        (e.g. ``{"minio_url": "...", "minio_key": "..."}``).
        """
        enriched: list[Document] = []

        for doc in documents:
            content = doc.content or ""
            meta: dict[str, Any] = dict(doc.meta)

            # ----------------------------------------------------------------
            # Stable document identity
            # ----------------------------------------------------------------
            # Use source (original filename) not file_path (temp path) so the
            # doc_id stays the same across re-indexing of the same file.
            raw_key = f"{meta.get('source', '')}{content}"
            meta["doc_id"] = hashlib.sha256(raw_key.encode()).hexdigest()

            # ----------------------------------------------------------------
            # Title — first H1 or filename stem
            # ----------------------------------------------------------------
            if "title" not in meta or not meta["title"]:
                meta["title"] = _extract_title(content, meta.get("source", ""))

            # ----------------------------------------------------------------
            # Word count (rough: split on whitespace)
            # ----------------------------------------------------------------
            meta["word_count"] = len(content.split())

            # ----------------------------------------------------------------
            # Indexing timestamp
            # ----------------------------------------------------------------
            now = datetime.now(timezone.utc)
            meta["indexed_at"]    = now.isoformat()
            meta["indexed_at_ts"] = int(now.timestamp())  # Unix epoch for range filters

            # ----------------------------------------------------------------
            # Language detection (local, no LLM)
            # Detected on the full document before splitting for maximum accuracy.
            # Every chunk inherits this field via split metadata propagation.
            # ----------------------------------------------------------------
            meta["language"] = _detect_language(content)

            # ----------------------------------------------------------------
            # Document beginning — passed to ContentAnalyzer for the contextual
            # prefix LLM call, then stripped before writing to Qdrant.
            # ----------------------------------------------------------------
            meta["doc_beginning"] = content[:self.doc_beginning_chars]

            # ----------------------------------------------------------------
            # Embedding provenance (inherited by every chunk after split)
            # ----------------------------------------------------------------
            meta["embedding_model"] = self.embedding_model
            meta["embedding_provider"] = self.embedding_provider
            meta["embedding_dimension"] = self.embedding_dimension

            # Merge any caller-provided extra fields (e.g. minio_url, minio_key)
            if extra_meta:
                meta.update(extra_meta)

            enriched.append(Document(content=content, meta=meta))
            logger.info(
                "MetadataEnricher: '%s' | title='%s' | lang=%s | words=%d | doc_id=%s…",
                meta.get("source", "?"),
                meta.get("title", ""),
                meta.get("language", "?"),
                meta.get("word_count", 0),
                meta.get("doc_id", "")[:12],
            )

        logger.info("MetadataEnricher: enriched %d document(s)", len(enriched))
        return {"documents": enriched}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _extract_title(content: str, source: str) -> str:
    """Return the first H1 heading in the markdown, or the filename stem."""
    for match in _HEADING_RE.finditer(content):
        if len(match.group(1)) == 1:          # exactly one '#' → H1
            return match.group(2).strip()

    # Fallback: strip extension from source filename
    if source:
        stem = source.rsplit(".", 1)[0]
        return stem.replace("_", " ").replace("-", " ").title()

    return "Untitled"


def _detect_language(content: str) -> str:
    """Detect the document's language using langdetect.

    Uses the first 2000 characters for speed; falls back to 'unknown' on
    any error (e.g. empty documents, mixed-script text, or missing package).
    """
    if not _LANGDETECT_AVAILABLE:
        return "unknown"
    try:
        sample = content[:2000].strip()
        if not sample:
            return "unknown"
        return _langdetect(sample)
    except Exception:
        return "unknown"
