"""
ContentAnalyzer — Haystack 2.x custom component.

Makes ONE structured LLM call per chunk (parallelised) that simultaneously:

1. Generates a **contextual prefix** (Anthropic Contextual Retrieval) —
   a 1-2 sentence description that situates this chunk within its parent document.
   ─ The prefix is prepended to the chunk content BEFORE embedding, which
     significantly reduces retrieval failures on ambiguous or short chunks.
   ─ The original chunk text is preserved in ``meta["original_content"]``.

2. Extracts **semantic metadata**:
   ─ summary       : 2-3 sentence abstractive summary
   ─ keywords      : 5-10 key terms / phrases
   ─ classification: single label from the configured taxonomy
   ─ entities      : named entities grouped by type

Note: language detection is handled upstream by ``MetadataEnricher`` using
``langdetect`` (local, no LLM call needed).  The ``language`` field in ``meta``
is already set before this component runs and is preserved unchanged.

Combining both tasks in one LLM call keeps indexing cost low while
delivering state-of-the-art retrieval quality.

LLM requirement
---------------
Any **OpenAI-compatible** endpoint that supports ``json_object`` response
format (or at minimum returns valid JSON text).

Parallel execution
------------------
Chunks are processed concurrently via ``ThreadPoolExecutor``.
``max_workers`` controls parallelism — keep it ≤ your provider's rate limit.

Error handling
--------------
On LLM failure, the chunk passes through unchanged (no analysis fields
added) so the pipeline does not stall.
"""

import json
import logging
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

from haystack import Document, component

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------
_SYSTEM_PROMPT = (
    "You are a precise document analysis assistant. "
    "You always respond with a single valid JSON object — no markdown, no prose."
)

_USER_PROMPT = """\
You are analysing a chunk from a larger document.

Document context:
  Title         : {title}
  Beginning     : {doc_beginning}
  Section path  : {section_path}

Your task — return ONLY a JSON object with these fields:

{{
  "context_prefix":   "<1-2 sentences that situate this chunk within the broader document \
(what topic does the document cover, what aspect does this chunk address?)>",
  "summary":          "<2-3 sentence abstractive summary in the chunk's language>",
  "keywords":         ["<5 to 10 key terms or phrases>"],
  "classification":   "<exactly one label from: {taxonomy}>",
  "entities": {{
    "organizations":    ["<company / authority / institution names>"],
    "persons":          ["<full person names>"],
    "locations":        ["<cities, countries, regions>"],
    "dates":            ["<all temporal expressions>"],
    "technologies":     ["<software, products, systems, standards>"],
    "monetary_amounts": ["<monetary values and quantities with units>"]
  }}
}}

Chunk text:
\"\"\"
{chunk_content}
\"\"\"
"""

# Default entity structure — prevents KeyError downstream
_EMPTY_ENTITIES: dict[str, list] = {
    "organizations": [], "persons": [], "locations": [],
    "dates": [], "technologies": [], "monetary_amounts": [],
}

_DEFAULT_ANALYSIS: dict[str, Any] = {
    "context_prefix": "",
    "summary": "",
    "keywords": [],
    "classification": "general",
    "entities": _EMPTY_ENTITIES,
}


@component
class ContentAnalyzer:
    """
    Combined contextual-prefix generator + semantic metadata extractor.

    Args:
        openai_url:      Custom base URL (empty = official OpenAI API).
        openai_api_key:  API key for the LLM endpoint.
        llm_model:       Model name (any OpenAI-compatible model).
        taxonomy:        Comma-separated classification labels.
        max_workers:     Thread-pool size for parallel chunk processing.
        max_chars:       Max chunk characters sent to LLM (longer → truncated).
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
        max_workers: int = 8,
        max_chars: int = 4000,
        doc_beginning_chars: int = 1500,
    ) -> None:
        self.openai_api_key      = openai_api_key
        self.llm_model           = llm_model
        self.openai_url     = openai_url or None
        self.taxonomy            = taxonomy
        self.max_workers         = max_workers
        self.max_chars           = max_chars
        self.doc_beginning_chars = doc_beginning_chars

        # Lazy-init — OpenAI client is not picklable so we create per-thread
        self._client_kwargs: dict[str, Any] = {
            "api_key": openai_api_key,
        }
        if self.openai_url:
            self._client_kwargs["base_url"] = self.openai_url

    @component.output_types(documents=list[Document])
    def run(self, documents: list[Document]) -> dict[str, list[Document]]:
        """Analyze all chunks in parallel and enrich them with LLM-generated metadata.

        Each chunk is processed in a separate thread. Failed chunks pass through
        unchanged so the pipeline never stalls due to a single LLM error.

        Args:
            documents: Chunks to analyze (produced by ChunkContextEnricher).

        Returns:
            A dict with key ``"documents"`` containing the enriched chunks in
            their original order.
        """
        if not documents:
            return {"documents": []}

        logger.info(
            "ContentAnalyzer: analyzing %d chunk(s) with %d worker(s) …",
            len(documents), self.max_workers,
        )
        results: list[Document | None] = [None] * len(documents)

        with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
            futures = {
                pool.submit(self._analyze, doc): idx
                for idx, doc in enumerate(documents)
            }
            for future in as_completed(futures):
                idx = futures[future]
                try:
                    results[idx] = future.result()
                except Exception as exc:
                    logger.error("ContentAnalyzer error on chunk %d: %s", idx, exc)
                    results[idx] = documents[idx]   # pass through without analysis

        logger.info("ContentAnalyzer: processed %d chunk(s)", len(results))
        return {"documents": [d for d in results if d is not None]}

    # ------------------------------------------------------------------
    def _analyze(self, doc: Document) -> Document:
        """Run a single LLM call for one chunk and return the enriched Document.

        Args:
            doc: The chunk to analyze.

        Returns:
            A new Document with analysis fields merged into its metadata.
        """
        content = (doc.content or "")[: self.max_chars]
        meta    = doc.meta

        prompt = _USER_PROMPT.format(
            title        = meta.get("title", "Unknown"),
            doc_beginning= (meta.get("doc_beginning", "") or "")[:self.doc_beginning_chars],
            section_path = meta.get("section_path", ""),
            taxonomy     = self.taxonomy,
            chunk_content= content,
        )

        try:
            raw      = self._call_llm(prompt)
            analysis = _parse_json(raw)
        except Exception as exc:
            logger.warning(
                "ContentAnalyzer: LLM call failed for '%s': %s",
                meta.get("source", doc.id),
                exc,
            )
            analysis = {**_DEFAULT_ANALYSIS, "entities": {k: [] for k in _EMPTY_ENTITIES}}

        return _apply(doc, analysis)

    def _call_llm(self, prompt: str) -> str:
        """Send the analysis prompt to the LLM and return the raw response text.

        Args:
            prompt: Fully formatted user prompt (chunk content + context).

        Returns:
            Raw response string from the LLM (expected to be valid JSON).
        """
        from openai import OpenAI

        client = OpenAI(**self._client_kwargs)
        response = client.chat.completions.create(
            model    = self.llm_model,
            messages = [
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user",   "content": prompt},
            ],
            response_format = {"type": "json_object"},
            temperature     = 0.0,
        )
        return response.choices[0].message.content or ""


# ---------------------------------------------------------------------------
# Pure helpers (no side effects — easy to unit-test)
# ---------------------------------------------------------------------------

def _parse_json(raw: str) -> dict[str, Any]:
    """Parse and validate the LLM's JSON response into an analysis dict.

    Strips optional markdown code fences, extracts the JSON object, and
    fills in default values for any missing fields.

    Args:
        raw: Raw LLM response text, expected to contain a JSON object.

    Returns:
        A complete analysis dict with all expected keys guaranteed to be present.

    Raises:
        json.JSONDecodeError: If no valid JSON object can be extracted.
    """
    text = raw.strip()

    fence = re.search(r"```(?:json)?\s*([\s\S]+?)\s*```", text)
    if fence:
        text = fence.group(1).strip()

    start, end = text.find("{"), text.rfind("}")
    if 0 <= start < end:
        text = text[start : end + 1]

    parsed: dict[str, Any] = json.loads(text)

    result = dict(_DEFAULT_ANALYSIS)
    result.update(parsed)

    entities = dict(_EMPTY_ENTITIES)
    entities.update(parsed.get("entities", {}))
    result["entities"] = entities

    return result


def _apply(doc: Document, analysis: dict[str, Any]) -> Document:
    """Build an enriched Document from a chunk and its analysis result.

    Prepends the contextual prefix to the embedded text, preserves the original
    chunk text in ``meta["original_content"]``, and merges all analysis fields
    into the document metadata.

    Args:
        doc:      The source chunk document.
        analysis: Parsed analysis dict from ``_parse_json``.

    Returns:
        A new Document with updated content and enriched metadata.
    """
    original_content = doc.content or ""
    prefix           = analysis.get("context_prefix", "").strip()

    # Contextual prefix prepended to the embedded text to improve retrieval on short/ambiguous chunks
    embedded_content = f"{prefix}\n\n{original_content}" if prefix else original_content

    meta = dict(doc.meta)

    # doc_beginning was only needed for the LLM call — strip it before writing
    # to Qdrant to avoid storing ~1500 chars of redundant data per chunk.
    meta.pop("doc_beginning", None)

    meta["original_content"] = original_content   # for display / citation
    meta["context_prefix"]   = prefix
    # language is set by MetadataEnricher (langdetect) — preserve it, don't overwrite
    meta.setdefault("language", "unknown")
    meta["summary"]           = analysis.get("summary",        "")
    meta["keywords"]          = analysis.get("keywords",       [])
    meta["classification"]    = analysis.get("classification", "general")
    meta["entities"]          = analysis.get("entities",       dict(_EMPTY_ENTITIES))

    return Document(content=embedded_content, meta=meta, id=doc.id)
