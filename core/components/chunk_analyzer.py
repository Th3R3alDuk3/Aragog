import asyncio
from collections import defaultdict
from logging import getLogger

from haystack import Document, component
from pydantic import BaseModel, Field

from core.models.meta import ChunkMetadata, Entities
from core.models.vocabulary import DEFAULT_CHUNK_CLASSIFICATION_TAXONOMY, ChunkClassification

logger = getLogger(__name__)

_SYSTEM = (
    "You are a precise document analysis assistant. "
    "Analyse the given chunk in context of its document and extract structured metadata. "
    "Preserve the original language of the chunk in all text fields."
)

# Prompt follows the Anthropic Contextual Retrieval paper:
# the full document is passed in an XML <document> block so the LLM can situate
# every chunk within its source — producing a retrieval-improving context_prefix.
_USER_PROMPT = """\
<document>
{doc_content}
</document>

Here is the chunk to analyse:
<chunk>
{chunk_content}
</chunk>

Chunk structural context:
  Title        : {title}
  Section path : {section_path}

Classification taxonomy: {taxonomy}

Extract all of the following:
1. context_prefix — 1-2 sentences situating this chunk within the overall document \
for search retrieval purposes.
2. summary — 2-3 sentence abstractive summary in the chunk's language.
3. keywords — 5 to 10 key terms or phrases.
4. classification — exactly one label from the taxonomy.
5. entities — named entities found in the chunk.\
"""

# Tool schema for the Anthropic SDK path (mirrors ChunkAnalysis fields).
_ANTHROPIC_TOOL: dict = {
    "name": "chunk_analysis",
    "description": "Extract structured metadata from a document chunk.",
    "input_schema": {
        "type": "object",
        "properties": {
            "context_prefix": {
                "type": "string",
                "description": "1-2 sentences situating this chunk within the broader document for search retrieval.",
            },
            "summary": {
                "type": "string",
                "description": "2-3 sentence abstractive summary in the chunk's language.",
            },
            "keywords": {
                "type": "array",
                "items": {"type": "string"},
                "description": "5 to 10 key terms or phrases.",
            },
            "classification": {
                "type": "string",
                "description": "Exactly one label from the provided taxonomy.",
            },
            "entities": {
                "type": "object",
                "properties": {
                    "persons":           {"type": "array", "items": {"type": "string"}},
                    "organizations":     {"type": "array", "items": {"type": "string"}},
                    "locations":         {"type": "array", "items": {"type": "string"}},
                    "dates":             {"type": "array", "items": {"type": "string"}},
                    "products":          {"type": "array", "items": {"type": "string"}},
                    "laws_and_standards":{"type": "array", "items": {"type": "string"}},
                    "events":            {"type": "array", "items": {"type": "string"}},
                    "quantities":        {"type": "array", "items": {"type": "string"}},
                },
                "additionalProperties": False,
            },
        },
        "required": ["context_prefix", "summary", "keywords", "classification"],
    },
}


class ChunkAnalysis(BaseModel):
    """LLM response format — subset of ChunkMetadata filled by the AI."""

    context_prefix: str = Field(
        description="1-2 sentences situating this chunk within the broader document."
    )
    summary: str = Field(
        description="2-3 sentence abstractive summary in the chunk's language."
    )
    keywords: list[str] = Field(description="5 to 10 key terms or phrases.")
    classification: ChunkClassification = Field(
        description="Exactly one label from the provided taxonomy."
    )
    entities: Entities = Field(default_factory=Entities)


@component
class ChunkAnalyzer:
    """
    Contextual prefix generator + semantic metadata extractor.

    Implements the Anthropic Contextual Retrieval approach: the **full document**
    is passed as context so the LLM can generate a retrieval-improving
    ``context_prefix`` that situates each chunk within its source document.

    Two execution paths:

    * **Default (OpenAI-compatible)**: uses ``openai.AsyncOpenAI`` with
      ``beta.chat.completions.parse`` for structured output.  Works with any
      OpenAI-compatible backend (Ollama, vLLM, LM Studio, Groq, …).

    * **Anthropic caching** (``anthropic_caching_enabled=True``): uses the
      ``anthropic`` SDK with ``cache_control: ephemeral`` on the document
      content.  Chunks from the same document are grouped so the 5-min KV-cache
      TTL is maximally utilised — reduces token cost by ~50× on large documents.
      Requires the ``anthropic`` package and a valid ``anthropic_api_key``.

    Args:
        openai_url:                Custom base URL (empty = official OpenAI API).
        openai_api_key:            API key for the OpenAI-compatible endpoint.
        llm_model:                 Model name.
        taxonomy:                  Comma-separated classification labels.
        max_chars:                 Max chunk characters sent to the LLM.
        max_doc_chars:             Max document characters used as context
                                   (0 = unlimited, i.e. full document).
        max_concurrency:           Maximum concurrent LLM requests.
        anthropic_caching_enabled: Enable Anthropic SDK + prompt-caching path.
        anthropic_api_key:         Anthropic API key (only for caching path).
    """

    def __init__(
        self,
        openai_url: str,
        openai_api_key: str,
        llm_model: str,
        taxonomy: str = DEFAULT_CHUNK_CLASSIFICATION_TAXONOMY,
        max_chars: int = 4000,
        max_doc_chars: int = 0,
        max_concurrency: int = 3,
        anthropic_caching_enabled: bool = False,
        anthropic_api_key: str = "",
    ) -> None:
        from openai import AsyncOpenAI

        self._client = AsyncOpenAI(base_url=openai_url or None, api_key=openai_api_key)
        self.llm_model = llm_model
        self.taxonomy = taxonomy
        self.max_chars = max_chars
        self.max_doc_chars = max_doc_chars
        self.max_concurrency = max_concurrency
        self.anthropic_caching_enabled = anthropic_caching_enabled

        self._anthropic_client = None
        if anthropic_caching_enabled:
            from anthropic import AsyncAnthropic
            self._anthropic_client = AsyncAnthropic(api_key=anthropic_api_key or openai_api_key)

    # ------------------------------------------------------------------
    # Haystack entry point
    # ------------------------------------------------------------------

    @component.output_types(documents=list[Document])
    async def run(self, documents: list[Document]) -> dict[str, list[Document]]:
        if not documents:
            return {"documents": []}
        logger.info("ChunkAnalyzer: analyzing %d chunk(s) …", len(documents))
        result = await self._run_async(documents)
        logger.info("ChunkAnalyzer: processed %d chunk(s)", len(result))
        return {"documents": result}

    # ------------------------------------------------------------------
    # Dispatch
    # ------------------------------------------------------------------

    async def _run_async(self, documents: list[Document]) -> list[Document]:
        sem = asyncio.Semaphore(self.max_concurrency)

        if self.anthropic_caching_enabled:
            return await self._run_anthropic(documents, sem)
        else:
            tasks = [self._analyze(doc, sem) for doc in documents]
            return list(await asyncio.gather(*tasks))

    async def _run_anthropic(
        self, documents: list[Document], sem: asyncio.Semaphore
    ) -> list[Document]:
        """Group chunks by doc_id so the Anthropic KV-cache for the document
        content is maximally reused within the 5-min TTL window."""
        by_doc: dict[str, list[Document]] = defaultdict(list)
        original_order: dict[str, int] = {doc.id: i for i, doc in enumerate(documents)}

        for doc in documents:
            did = doc.meta.get("doc_id") or doc.id or "unknown"
            by_doc[did].append(doc)

        results: list[tuple[int, Document]] = []
        for doc_chunks in by_doc.values():
            processed = await asyncio.gather(
                *[self._analyze_anthropic(doc, sem) for doc in doc_chunks]
            )
            for orig, proc in zip(doc_chunks, processed):
                results.append((original_order[orig.id], proc))

        results.sort(key=lambda x: x[0])
        return [doc for _, doc in results]

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _get_doc_context(self, meta: ChunkMetadata) -> str:
        """Return the document context string, optionally truncated."""
        content = meta.doc_content or meta.doc_beginning
        if self.max_doc_chars > 0:
            content = content[: self.max_doc_chars]
        return content

    # ------------------------------------------------------------------
    # OpenAI-compatible path
    # ------------------------------------------------------------------

    async def _analyze(self, doc: Document, sem: asyncio.Semaphore) -> Document:
        meta = ChunkMetadata.model_validate(doc.meta)
        chunk_content = (doc.content or "")[: self.max_chars]
        doc_context = self._get_doc_context(meta)

        prompt = _USER_PROMPT.format(
            doc_content=doc_context,
            title=meta.title or "Unknown",
            section_path=meta.section_path,
            taxonomy=self.taxonomy,
            chunk_content=chunk_content,
        )
        fallback = ChunkAnalysis(
            context_prefix="", summary="", keywords=[], classification="other"
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
                "ChunkAnalyzer: LLM call failed for '%s': %s",
                meta.source or doc.id,
                exc,
            )
            analysis = fallback
        return _apply(doc, analysis)

    # ------------------------------------------------------------------
    # Anthropic caching path
    # ------------------------------------------------------------------

    async def _analyze_anthropic(self, doc: Document, sem: asyncio.Semaphore) -> Document:
        """Anthropic SDK path with cache_control on the document content block.

        Structure:
          system[0]  — static instructions (cached automatically by Anthropic)
          system[1]  — full document content with cache_control: ephemeral
                        → Anthropic reuses its KV-cache across all chunks
                        from the same document within the 5-min TTL.
          user       — chunk content + structural metadata
        """
        meta = ChunkMetadata.model_validate(doc.meta)
        chunk_content = (doc.content or "")[: self.max_chars]
        doc_context = self._get_doc_context(meta)

        system_blocks = [
            {"type": "text", "text": _SYSTEM},
            {
                "type": "text",
                "text": f"<document>\n{doc_context}\n</document>",
                "cache_control": {"type": "ephemeral"},
            },
        ]
        user_content = (
            f"<chunk>\n{chunk_content}\n</chunk>\n\n"
            f"Chunk structural context:\n"
            f"  Title        : {meta.title or 'Unknown'}\n"
            f"  Section path : {meta.section_path}\n\n"
            f"Classification taxonomy: {self.taxonomy}"
        )

        fallback = ChunkAnalysis(
            context_prefix="", summary="", keywords=[], classification="other"
        )
        try:
            async with sem:
                response = await self._anthropic_client.messages.create(
                    model=self.llm_model,
                    max_tokens=1024,
                    system=system_blocks,
                    messages=[{"role": "user", "content": user_content}],
                    tools=[_ANTHROPIC_TOOL],
                    tool_choice={"type": "tool", "name": "chunk_analysis"},
                )

            tool_block = next(
                (b for b in response.content if b.type == "tool_use"), None
            )
            if tool_block:
                data = tool_block.input
                ent_data = data.get("entities") or {}
                analysis = ChunkAnalysis(
                    context_prefix=data.get("context_prefix", ""),
                    summary=data.get("summary", ""),
                    keywords=data.get("keywords", []),
                    classification=data.get("classification", "other"),
                    entities=Entities(**ent_data) if ent_data else Entities(),
                )
            else:
                analysis = fallback
        except Exception as exc:
            logger.warning(
                "ChunkAnalyzer (Anthropic): LLM call failed for '%s': %s",
                meta.source or doc.id,
                exc,
            )
            analysis = fallback
        return _apply(doc, analysis)


# ---------------------------------------------------------------------------
# Shared result builder
# ---------------------------------------------------------------------------


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

    # Exclude both ephemeral fields — they must not be stored in Qdrant.
    dumped = meta.model_dump(exclude={"doc_beginning", "doc_content"})
    # Pydantic drops __-prefixed keys (name mangling); restore from original meta.
    for k, v in doc.meta.items():
        if k.startswith("__"):
            dumped[k] = v
    return Document(
        id=doc.id,
        content=meta.original_content,
        blob=doc.blob,
        meta=dumped,
        score=doc.score,
        embedding=doc.embedding,
        sparse_embedding=doc.sparse_embedding,
    )
