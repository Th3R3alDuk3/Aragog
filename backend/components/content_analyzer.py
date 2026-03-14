import asyncio
from logging import getLogger

from haystack import Document, component
from openai import AsyncOpenAI
from pydantic import BaseModel, Field

from models.meta import ChunkMetadata, Entities

logger = getLogger(__name__)

_SYSTEM = (
    "You are a precise document analysis assistant. "
    "Analyse the given chunk in context of its document and extract structured metadata. "
    "Preserve the original language of the chunk in all text fields."
)

_USER_PROMPT = """\
Document context:
  Title        : {title}
  Beginning    : {doc_beginning}
  Section path : {section_path}

Classification taxonomy: {taxonomy}

Chunk text:
\"\"\"
{chunk_content}
\"\"\"\
"""


class ChunkAnalysis(BaseModel):
    """LLM response format — subset of ChunkMetadata filled by the AI."""

    context_prefix: str = Field(
        description="1-2 sentences situating this chunk within the broader document."
    )
    summary: str = Field(
        description="2-3 sentence abstractive summary in the chunk's language."
    )
    keywords: list[str] = Field(description="5 to 10 key terms or phrases.")
    classification: str = Field(
        description="Exactly one label from the provided taxonomy."
    )
    entities: Entities = Field(default_factory=Entities)


@component
class ContentAnalyzer:
    """
    Combined contextual-prefix generator + semantic metadata extractor.

    Args:
        openai_url:          Custom base URL (empty = official OpenAI API).
        openai_api_key:      API key for the LLM endpoint.
        llm_model:           Model name (any OpenAI-compatible model).
        taxonomy:            Comma-separated classification labels.
        max_chars:           Max chunk characters sent to LLM (longer → truncated).
        doc_beginning_chars: Characters of the document beginning used for context.
    """

    def __init__(
        self,
        openai_url: str,
        openai_api_key: str,
        llm_model: str,
        taxonomy: str = (
            "financial,legal,technical,scientific,hr,"
            "marketing,contract,report,manual,correspondence,general"
        ),
        max_chars: int = 4000,
        doc_beginning_chars: int = 1500,
        max_concurrency: int = 8,
    ) -> None:
        self._client = AsyncOpenAI(base_url=openai_url or None, api_key=openai_api_key)
        self.llm_model = llm_model
        self.taxonomy = taxonomy
        self.max_chars = max_chars
        self.doc_beginning_chars = doc_beginning_chars
        self.max_concurrency = max_concurrency

    @component.output_types(documents=list[Document])
    async def run(self, documents: list[Document]) -> dict[str, list[Document]]:
        if not documents:
            return {"documents": []}
        logger.info("ContentAnalyzer: analyzing %d chunk(s) …", len(documents))
        result = await self._run_async(documents)
        logger.info("ContentAnalyzer: processed %d chunk(s)", len(result))
        return {"documents": result}

    async def _run_async(self, documents: list[Document]) -> list[Document]:
        sem = asyncio.Semaphore(self.max_concurrency)
        tasks = [self._analyze(doc, sem) for doc in documents]
        return list(await asyncio.gather(*tasks))

    async def _analyze(self, doc: Document, sem: asyncio.Semaphore) -> Document:
        meta = ChunkMetadata.model_validate(doc.meta)
        content = (doc.content or "")[: self.max_chars]
        prompt = _USER_PROMPT.format(
            title=meta.title or "Unknown",
            doc_beginning=meta.doc_beginning[: self.doc_beginning_chars],
            section_path=meta.section_path,
            taxonomy=self.taxonomy,
            chunk_content=content,
        )
        fallback = ChunkAnalysis(
            context_prefix="", summary="", keywords=[], classification="general"
        )
        try:
            async with sem:
                response = await self._client.beta.chat.completions.parse(
                    model=self.llm_model,
                    messages=[
                        {"role": "system", "content": _SYSTEM},
                        {"role": "user", "content": prompt},
                    ],
                    response_format=ChunkAnalysis,
                    temperature=0.0,
                )
            analysis = response.choices[0].message.parsed or fallback
        except Exception as exc:
            logger.warning(
                "ContentAnalyzer: LLM call failed for '%s': %s",
                meta.source or doc.id,
                exc,
            )
            analysis = fallback
        return _apply(doc, analysis)


def _apply(doc: Document, analysis: ChunkAnalysis) -> Document:
    meta = ChunkMetadata.model_validate(doc.meta)
    prefix = analysis.context_prefix.strip()

    meta.original_content = doc.content or ""
    meta.context_prefix = prefix
    meta.summary = analysis.summary
    meta.keywords = analysis.keywords
    meta.classification = analysis.classification

    e = analysis.entities
    meta.ent_persons       = e.persons
    meta.ent_organizations = e.organizations
    meta.ent_locations     = e.locations
    meta.ent_dates         = e.dates
    meta.ent_products      = e.products
    meta.ent_laws          = e.laws_and_standards
    meta.ent_events        = e.events
    meta.ent_quantities    = e.quantities

    embedded_content = f"{prefix}\n\n{meta.original_content}" if prefix else meta.original_content
    dumped = meta.model_dump(exclude={"doc_beginning"})
    # Pydantic drops __-prefixed keys (name mangling); restore them from the original meta
    for k, v in doc.meta.items():
        if k.startswith("__"):
            dumped[k] = v
    return Document(content=embedded_content, meta=dumped, id=doc.id)
