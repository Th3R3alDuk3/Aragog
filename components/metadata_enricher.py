from datetime import datetime, timezone
from hashlib import sha256
from logging import getLogger
from re import compile, MULTILINE

from haystack import Document, component

from models.meta import ChunkMetadata

try:
    from langdetect import detect as _langdetect
    from langdetect import DetectorFactory

    DetectorFactory.seed = 0  # reproducible results
    _LANGDETECT_AVAILABLE = True
except ImportError:
    _LANGDETECT_AVAILABLE = False

logger = getLogger(__name__)

# Matches ATX-style markdown headings: # H1 / ## H2 / ### H3 …
_HEADING_RE = compile(r"^(#{1,6})\s+(.+)$", MULTILINE)


@component
class MetadataEnricher:
    """
    Enriches full (pre-split) documents with document-level metadata.

    Args:
        embedding_provider:  Provider name recorded in metadata.
        embedding_model:     Model identifier recorded in metadata.
        embedding_dimension: Vector dimension recorded in metadata.
    """

    def __init__(
        self,
        embedding_provider: str,
        embedding_model: str,
        embedding_dimension: int,
        doc_beginning_chars: int = 1500,
    ) -> None:

        self.embedding_model = embedding_model
        self.embedding_provider = embedding_provider
        self.embedding_dimension = embedding_dimension
        self.doc_beginning_chars = doc_beginning_chars

    @component.output_types(documents=list[Document])
    def run(
        self, documents: list[Document], extra_meta: dict | None = None
    ) -> dict[str, list[Document]]:
        """Enrich each document with stable, document-level metadata fields.

        Adds doc_id, title, word_count, indexed_at timestamps, detected language,
        doc_beginning (for downstream LLM context), and embedding provenance.
        All fields are inherited by every chunk produced by downstream splitters.

        Args:
            documents:  Raw documents from DoclingConverter (one per source file).
            extra_meta: Optional key/value pairs merged into every document's
                        metadata (e.g. ``{"minio_url": "...", "minio_key": "..."}``).
        """
        enriched: list[Document] = []

        for doc in documents:
            content = doc.content or ""
            meta = ChunkMetadata.model_validate(doc.meta)

            # Use source (original filename) not file_path (temp path) so the
            # doc_id stays the same across re-indexing of the same file.
            meta.doc_id = sha256(f"{meta.source}{content}".encode()).hexdigest()

            if not meta.title:
                meta.title = _extract_title(content, meta.source)

            meta.word_count = len(content.split())

            now = datetime.now(timezone.utc)
            meta.indexed_at = now.isoformat()
            meta.indexed_at_ts = int(now.timestamp())

            meta.language = _detect_language(content)

            # Ephemeral: consumed by ContentAnalyzer, stripped before Qdrant write.
            meta.doc_beginning = content[: self.doc_beginning_chars]

            meta.embedding_model = self.embedding_model
            meta.embedding_provider = self.embedding_provider
            meta.embedding_dimension = self.embedding_dimension

            dumped = meta.model_dump()
            if extra_meta:
                dumped.update(extra_meta)

            enriched.append(Document(content=content, meta=dumped))
            logger.info(
                "MetadataEnricher: '%s' | title='%s' | lang=%s | words=%d | doc_id=%s…",
                meta.source,
                meta.title,
                meta.language,
                meta.word_count,
                meta.doc_id[:12],
            )

        logger.info("MetadataEnricher: enriched %d document(s)", len(enriched))
        return {"documents": enriched}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _extract_title(content: str, source: str) -> str:
    """Return the first H1 heading found in the markdown, or derive a title from the filename.

    Args:
        content: Full markdown text of the document.
        source:  Original filename (used as fallback when no H1 is present).

    Returns:
        Title string — never empty, falls back to ``"Untitled"`` as last resort.
    """
    for match in _HEADING_RE.finditer(content):
        if len(match.group(1)) == 1:  # exactly one '#' → H1
            return match.group(2).strip()

    # Fallback: strip extension from source filename
    if source:
        stem = source.rsplit(".", 1)[0]
        return stem.replace("_", " ").replace("-", " ").title()

    return "Untitled"


def _detect_language(content: str) -> str:
    """Detect the document's primary language using langdetect.

    Samples the first 2 000 characters for speed. Falls back to ``"unknown"``
    on any error (empty document, mixed-script text, or missing package).

    Args:
        content: Raw document text to detect the language of.

    Returns:
        ISO 639-1 language code (e.g. ``"en"``, ``"de"``), or ``"unknown"``.
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
