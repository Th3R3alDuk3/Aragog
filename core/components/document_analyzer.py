import asyncio
from datetime import date, datetime, time, timezone
from hashlib import sha256
from logging import getLogger

from haystack import Document, component
from openai import AsyncOpenAI
from pydantic import BaseModel, Field

from core.components._markdown_utils import extract_title
from core.models.meta import ChunkMetadata
from core.models.vocabulary import AudienceType, DocumentType, LanguageCode

logger = getLogger(__name__)

_SYSTEM = (
    "You are a document metadata extraction assistant. "
    "Analyze the provided document information and extract structured metadata. "
    "Return only what is clearly indicated by the text. "
    "All date fields must be ISO format (YYYY-MM-DD) or null."
)

_USER_PROMPT = """\
Filename : {source}
Title    : {title}

Document beginning:
\"\"\"
{doc_beginning}
\"\"\"\
"""

class DocumentSemanticAnalysis(BaseModel):
    """LLM response schema for document-level semantic metadata."""

    document_type: DocumentType = Field(
        default="general",
        description="Document type — pick the closest match from the allowed values.",
    )
    language: LanguageCode = Field(
        default="unknown",
        description=(
            "ISO 639-1 language code from the supported top-language set "
            "(e.g. 'en', 'de', 'fr', 'zh'). 'unknown' if unclear."
        ),
    )
    document_date: str | None = Field(
        default=None,
        description="Specific publication or creation date (YYYY-MM-DD), if clearly stated. Null otherwise.",
    )
    period_start: str | None = Field(
        default=None,
        description="Start of the reporting or coverage period (YYYY-MM-DD), if applicable. Null otherwise.",
    )
    period_end: str | None = Field(
        default=None,
        description="End of the reporting or coverage period (YYYY-MM-DD), if applicable. Null otherwise.",
    )
    audience: AudienceType = Field(
        default="general",
        description="Primary intended audience — pick the closest match from the allowed values.",
    )


@component
class DocumentAnalyzer:
    """
    Enriches full (pre-split) documents with document-level metadata.

    Title is extracted heuristically; document type, language, and date
    semantics are inferred via a single LLM call (language-agnostic).

    Args:
        openai_url:          Custom base URL for the LLM endpoint (empty = OpenAI).
        openai_api_key:      API key for the LLM endpoint.
        llm_model:           Model name — use the instruct model (fast, structured tasks).
        embedding_provider:  Provider name recorded in metadata.
        embedding_model:     Model identifier recorded in metadata.
        embedding_dimension: Vector dimension recorded in metadata.
        doc_beginning_chars: Characters of document start passed to the LLM.
        max_concurrency:     Maximum number of concurrent LLM requests.
    """

    def __init__(
        self,
        openai_url: str,
        openai_api_key: str,
        llm_model: str,
        embedding_provider: str,
        embedding_model: str,
        embedding_dimension: int,
        doc_beginning_chars: int = 1500,
        max_concurrency: int = 3,
    ) -> None:
        self._client = AsyncOpenAI(base_url=openai_url or None, api_key=openai_api_key)
        self.llm_model = llm_model
        self.embedding_model = embedding_model
        self.embedding_provider = embedding_provider
        self.embedding_dimension = embedding_dimension
        self.doc_beginning_chars = doc_beginning_chars
        self.max_concurrency = max_concurrency

    @component.output_types(documents=list[Document])
    async def run(self, documents: list[Document]) -> dict[str, list[Document]]:
        """Enrich each document with stable, document-level metadata fields.

        Adds doc_id, title, word_count, language, semantic date fields
        (via LLM), doc_beginning, and embedding provenance.
        """
        if not documents:
            return {"documents": []}
        logger.info("DocumentAnalyzer: analyzing %d document(s) ...", len(documents))
        enriched = await self._run_async(documents)
        logger.info("DocumentAnalyzer: enriched %d document(s)", len(enriched))
        return {"documents": enriched}

    async def _run_async(self, documents: list[Document]) -> list[Document]:
        sem = asyncio.Semaphore(self.max_concurrency)
        tasks = [self._analyze(doc, sem) for doc in documents]
        return list(await asyncio.gather(*tasks))

    async def _analyze(self, doc: Document, sem: asyncio.Semaphore) -> Document:
        content = doc.content or ""
        meta = ChunkMetadata.model_validate(doc.meta)

        meta.doc_id = sha256(f"{meta.source}{content}".encode()).hexdigest()
        if not meta.title:
            meta.title = _extract_title_from(content, meta.source)
        meta.word_count = len(content.split())
        meta.doc_beginning = content[: self.doc_beginning_chars]

        fallback = DocumentSemanticAnalysis()
        try:
            async with sem:
                response = await self._client.beta.chat.completions.parse(
                    model=self.llm_model,
                    messages=[
                        {"role": "system", "content": _SYSTEM},
                        {
                            "role": "user",
                            "content": _USER_PROMPT.format(
                                source=meta.source,
                                title=meta.title,
                                doc_beginning=meta.doc_beginning,
                            ),
                        },
                    ],
                    response_format=DocumentSemanticAnalysis,
                    temperature=0.0,
                )
            analysis = response.choices[0].message.parsed or fallback
        except Exception as exc:
            logger.warning("DocumentAnalyzer: LLM call failed for '%s': %s", meta.source or doc.id, exc)
            analysis = fallback
        return _apply(
            doc,
            meta,
            analysis,
            embedding_model=self.embedding_model,
            embedding_provider=self.embedding_provider,
            embedding_dimension=self.embedding_dimension,
        )


def _apply(
    doc: Document,
    meta: ChunkMetadata,
    analysis: DocumentSemanticAnalysis,
    *,
    embedding_model: str,
    embedding_provider: str,
    embedding_dimension: int,
) -> Document:
    meta.language = analysis.language
    meta.document_type = analysis.document_type
    meta.audience = analysis.audience
    meta.document_date, meta.document_date_ts = _parse_iso_date(analysis.document_date, time.min)
    meta.period_start, meta.period_start_ts = _parse_iso_date(analysis.period_start, time.min)
    meta.period_end, meta.period_end_ts = _parse_iso_date(analysis.period_end, time.max)
    meta.embedding_model = embedding_model
    meta.embedding_provider = embedding_provider
    meta.embedding_dimension = embedding_dimension
    # Full document content flows to ChunkAnalyzer for contextual prefix generation
    # (Anthropic Contextual Retrieval approach). Stripped from Qdrant by ChunkAnalyzer._apply().
    meta.doc_content = doc.content or ""
    return Document(content=doc.content or "", meta=meta.model_dump(), id=doc.id)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _extract_title_from(content: str, source: str) -> str:
    title = extract_title(content)
    if title:
        return title
    if source:
        stem = source.rsplit(".", 1)[0]
        return stem.replace("_", " ").replace("-", " ").title()
    return "Untitled"


def _parse_iso_date(value: str | None, day_time: time) -> tuple[str, int]:
    if not value:
        return "", 0
    try:
        parsed = date.fromisoformat(value)
        ts = int(datetime.combine(parsed, day_time, tzinfo=timezone.utc).timestamp())
        return parsed.isoformat(), ts
    except ValueError:
        return "", 0
