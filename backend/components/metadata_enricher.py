from dataclasses import dataclass
from datetime import date, datetime, time, timezone
from hashlib import sha256
from logging import getLogger
from pathlib import Path
from string import punctuation

from dateparser.search import search_dates

from haystack import Document, component

from components._markdown_utils import extract_title
from models.meta import ChunkMetadata

try:
    from langdetect import DetectorFactory
    from langdetect import detect as _langdetect

    DetectorFactory.seed = 0  # reproducible results
    _LANGDETECT_AVAILABLE = True
except ImportError:
    _LANGDETECT_AVAILABLE = False

logger = getLogger(__name__)

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
    def run(self, documents: list[Document]) -> dict[str, list[Document]]:
        """Enrich each document with stable, document-level metadata fields.

        Adds doc_id, title, word_count, semantic date fields, detected language,
        doc_beginning (for downstream LLM context), and embedding provenance.
        All fields are inherited by every chunk produced by downstream splitters.

        Args:
            documents: Raw documents from DoclingConverter (one per source file).
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

            meta.language = _detect_language(content)

            # Ephemeral: consumed by ContentAnalyzer, stripped before Qdrant write.
            meta.doc_beginning = content[: self.doc_beginning_chars]

            semantics = infer_document_semantics(
                source=meta.source,
                title=meta.title,
                doc_beginning=meta.doc_beginning,
                language=meta.language,
            )
            meta.document_type = semantics.document_type
            meta.document_date = semantics.document_date
            meta.document_date_ts = semantics.document_date_ts
            meta.period_start = semantics.period_start
            meta.period_start_ts = semantics.period_start_ts
            meta.period_end = semantics.period_end
            meta.period_end_ts = semantics.period_end_ts

            meta.embedding_model = self.embedding_model
            meta.embedding_provider = self.embedding_provider
            meta.embedding_dimension = self.embedding_dimension

            dumped = meta.model_dump()
            enriched.append(Document(content=content, meta=dumped))
            logger.info(
                "MetadataEnricher: '%s' | title='%s' | lang=%s | type=%s | doc_id=%s…",
                meta.source,
                meta.title,
                meta.language,
                meta.document_type,
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
    title = extract_title(content)
    if title:
        return title

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


@dataclass(frozen=True, slots=True)
class DocumentSemantics:
    document_type: str = "general"
    document_date: str = ""
    document_date_ts: int = 0
    period_start: str = ""
    period_start_ts: int = 0
    period_end: str = ""
    period_end_ts: int = 0


def infer_document_semantics(
    source: str,
    title: str,
    doc_beginning: str,
    language: str,
) -> DocumentSemantics:
    document_type = _infer_document_type(source, title, doc_beginning)

    period = (
        _extract_date_range(title, language)
        or _extract_date_range(doc_beginning, language)
        or _extract_quarter_period(title)
        or _extract_quarter_period(doc_beginning)
        or _extract_annual_period(title, document_type)
        or _extract_annual_period(doc_beginning, document_type)
    )

    document_date = (
        _extract_specific_date(title, language)
        or _extract_specific_date(doc_beginning, language)
    )

    if period is None:
        period_start = ""
        period_start_ts = 0
        period_end = ""
        period_end_ts = 0
    else:
        start, end = period
        period_start = start.isoformat()
        period_start_ts = _date_to_ts(start, time.min)
        period_end = end.isoformat()
        period_end_ts = _date_to_ts(end, time.max)

    if document_date is None:
        document_date_iso = ""
        document_date_ts = 0
    else:
        document_date_iso = document_date.isoformat()
        document_date_ts = _date_to_ts(document_date, time.min)

    return DocumentSemantics(
        document_type=document_type,
        document_date=document_date_iso,
        document_date_ts=document_date_ts,
        period_start=period_start,
        period_start_ts=period_start_ts,
        period_end=period_end,
        period_end_ts=period_end_ts,
    )


def _infer_document_type(source: str, title: str, doc_beginning: str) -> str:
    suffix = Path(source).suffix.lower()
    if suffix in _EXTENSION_TO_TYPE:
        return _EXTENSION_TO_TYPE[suffix]

    normalized = " ".join(
        part for part in (source, title, doc_beginning[:800]) if part
    ).casefold()

    scored_types: list[tuple[int, str]] = []
    for doc_type, keywords in _DOC_TYPE_KEYWORDS.items():
        score = sum(1 for keyword in keywords if keyword in normalized)
        if score:
            scored_types.append((score, doc_type))

    if not scored_types:
        return "general"

    scored_types.sort(reverse=True)
    return scored_types[0][1]


def _extract_quarter_period(text: str) -> tuple[date, date] | None:
    tokens = _tokenize(text)
    for idx, token in enumerate(tokens):
        quarter = _QUARTER_TOKENS.get(token)
        year = _year_from_neighbors(tokens, idx)
        if quarter and year:
            return _quarter_bounds(year, quarter)

        if token in {"quarter", "quartal"} and idx + 2 < len(tokens):
            quarter = _parse_quarter_number(tokens[idx + 1])
            year = _parse_year(tokens[idx + 2])
            if quarter and year:
                return _quarter_bounds(year, quarter)
    return None


def _extract_annual_period(text: str, document_type: str) -> tuple[date, date] | None:
    normalized = text.casefold()
    if document_type not in {"report", "article", "general"} and not any(
        hint in normalized for hint in _ANNUAL_HINTS
    ):
        return None

    for token in _tokenize(text):
        year = _parse_year(token)
        if year:
            return date(year, 1, 1), date(year, 12, 31)
    return None


def _extract_date_range(text: str, language: str) -> tuple[date, date] | None:
    if not text:
        return None

    normalized = text.casefold()
    if not any(hint in normalized for hint in _RANGE_HINTS):
        return None

    dates = _search_dates(text, language)
    if len(dates) < 2:
        return None

    ordered = sorted(candidate for _, candidate in dates)
    return ordered[0], ordered[-1]


def _extract_specific_date(text: str, language: str) -> date | None:
    for raw_text, candidate in _search_dates(text, language):
        stripped = raw_text.strip()
        if _is_specific_date_text(stripped):
            return candidate
    return None


def _search_dates(text: str, language: str) -> list[tuple[str, date]]:
    if not text.strip():
        return []

    search_kwargs: dict[str, object] = {
        "settings": {
            "PREFER_DAY_OF_MONTH": "first",
            "RETURN_AS_TIMEZONE_AWARE": False,
        }
    }
    if language and language != "unknown":
        search_kwargs["languages"] = [language]

    matches = search_dates(text, **search_kwargs) or []
    results: list[tuple[str, date]] = []
    for raw_text, parsed in matches:
        if isinstance(parsed, datetime):
            results.append((raw_text, parsed.date()))
    return results


def _is_specific_date_text(text: str) -> bool:
    lowered = text.casefold()
    if len(text) <= 4 and text.isdigit():
        return False
    if any(ch in text for ch in ("/", "-", ".")):
        return True
    has_digits = any(ch.isdigit() for ch in text)
    has_letters = any(ch.isalpha() for ch in text)
    if has_digits and has_letters:
        return True
    return any(month in lowered for month in _MONTH_NAMES)


def _year_from_neighbors(tokens: list[str], idx: int) -> int | None:
    candidates = []
    if idx > 0:
        candidates.append(tokens[idx - 1])
    if idx + 1 < len(tokens):
        candidates.append(tokens[idx + 1])
    for token in candidates:
        year = _parse_year(token)
        if year:
            return year
    return None


def _parse_year(token: str) -> int | None:
    if len(token) != 4 or not token.isdigit():
        return None
    year = int(token)
    if 1900 <= year <= 2100:
        return year
    return None


def _parse_quarter_number(token: str) -> int | None:
    if token.isdigit():
        quarter = int(token)
        if 1 <= quarter <= 4:
            return quarter
    return None


def _quarter_bounds(year: int, quarter: int) -> tuple[date, date]:
    start_month = (quarter - 1) * 3 + 1
    end_month = start_month + 2
    start = date(year, start_month, 1)
    if end_month == 12:
        end = date(year, 12, 31)
    else:
        next_month = date(year, end_month + 1, 1)
        end = date.fromordinal(next_month.toordinal() - 1)
    return start, end


def _tokenize(text: str) -> list[str]:
    translation = str.maketrans({char: " " for char in punctuation})
    return [token.casefold() for token in text.translate(translation).split() if token]


def _date_to_ts(value: date, day_time: time) -> int:
    return int(datetime.combine(value, day_time, tzinfo=timezone.utc).timestamp())


_DOC_TYPE_KEYWORDS: dict[str, tuple[str, ...]] = {
    "report": (
        "report",
        "annual report",
        "quarterly report",
        "jahresbericht",
        "geschaeftsbericht",
        "geschäftsbericht",
        "financial report",
        "lagebericht",
    ),
    "manual": (
        "manual",
        "guide",
        "handbuch",
        "bedienungsanleitung",
        "user guide",
        "instruction",
    ),
    "contract": (
        "contract",
        "agreement",
        "vertrag",
        "vereinbarung",
    ),
    "correspondence": (
        "letter",
        "email",
        "mail",
        "correspondence",
        "brief",
        "e-mail",
    ),
    "policy": (
        "policy",
        "richtlinie",
        "procedure",
        "governance",
        "prozess",
    ),
    "invoice": (
        "invoice",
        "rechnung",
        "bill",
    ),
    "article": (
        "paper",
        "article",
        "study",
        "research",
        "whitepaper",
    ),
    "specification": (
        "specification",
        "spec",
        "requirements",
        "anforderungen",
        "spezifikation",
    ),
    "presentation": (
        "presentation",
        "deck",
        "slides",
        "folien",
    ),
    "spreadsheet": (
        "spreadsheet",
        "worksheet",
        "arbeitsblatt",
    ),
}

_EXTENSION_TO_TYPE = {
    ".ppt": "presentation",
    ".pptx": "presentation",
    ".key": "presentation",
    ".xls": "spreadsheet",
    ".xlsx": "spreadsheet",
    ".csv": "spreadsheet",
}

_ANNUAL_HINTS = (
    "annual",
    "yearly",
    "jahres",
    "geschaeftsjahr",
    "geschäftsjahr",
    "fiscal year",
)
_RANGE_HINTS = ("from", "to", "through", "until", "between", "bis", "von", "zwischen")
_QUARTER_TOKENS = {
    "q1": 1,
    "q2": 2,
    "q3": 3,
    "q4": 4,
    "quarter1": 1,
    "quarter2": 2,
    "quarter3": 3,
    "quarter4": 4,
    "quartal1": 1,
    "quartal2": 2,
    "quartal3": 3,
    "quartal4": 4,
}

_MONTH_NAMES: tuple[str, ...] = tuple(
    name.casefold()
    for name in (
        "january",
        "february",
        "march",
        "april",
        "may",
        "june",
        "july",
        "august",
        "september",
        "october",
        "november",
        "december",
        "januar",
        "februar",
        "maerz",
        "märz",
        "april",
        "mai",
        "juni",
        "juli",
        "august",
        "september",
        "oktober",
        "november",
        "dezember",
    )
)
