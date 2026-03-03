from concurrent.futures import ThreadPoolExecutor, as_completed
from hashlib import sha256
from logging import getLogger

from haystack import Document, component

logger = getLogger(__name__)

_SECTION_SYSTEM = (
    "You are a precise document summarisation assistant. "
    "You always respond with plain text — no markdown, no headers."
)

_SECTION_PROMPT = """\
You are given several short summaries of consecutive text chunks that all \
belong to the same section "{section_title}" of document "{title}".

Synthesise a single coherent summary of this section in 3-5 sentences. \
Write in the same language as the input summaries. \
Do NOT add headers, bullet points, or preamble.

Chunk summaries:
{summaries}
"""

_DOC_PROMPT = """\
You are given section summaries from a document titled "{title}".

Write a single high-level overview of the entire document in 5-8 sentences. \
Cover the main topics and key points across all sections. \
Write in the same language as the input summaries. \
Do NOT add headers, bullet points, or preamble.

Section summaries:
{summaries}
"""


def _stable_id(doc_id: str, suffix: str) -> str:
    return sha256(f"{doc_id}::{suffix}".encode()).hexdigest()


@component
class RaptorSummarizer:
    """
    Adds RAPTOR section- and document-level summary chunks to the indexing pipeline.

    Args:
        openai_api_key:  API key for the LLM endpoint.
        llm_model:       OpenAI-compatible model name (use the cheap analyzer model).
        openai_url: Custom base URL.
        max_workers:     Parallel LLM calls for section summarisation.
    """

    def __init__(
        self,
        openai_api_key: str,
        llm_model: str,
        openai_url: str = "",
        max_workers: int = 4,
    ) -> None:
        from openai import OpenAI
        self._llm_model   = llm_model
        self._max_workers = max_workers
        client_kwargs: dict = {"api_key": openai_api_key}
        if openai_url:
            client_kwargs["base_url"] = openai_url
        self._client = OpenAI(**client_kwargs)

    # ------------------------------------------------------------------

    @component.output_types(documents=list[Document])
    def run(self, documents: list[Document]) -> dict[str, list[Document]]:
        if not documents:
            return {"documents": []}

        # Group by doc_id
        by_doc: dict[str, list[Document]] = {}
        for doc in documents:
            did = doc.meta.get("doc_id", doc.id or "unknown")
            by_doc.setdefault(did, []).append(doc)

        raptor_docs: list[Document] = []

        with ThreadPoolExecutor(max_workers=self._max_workers) as pool:
            futures = {
                pool.submit(self._process_document, doc_id, chunks): doc_id
                for doc_id, chunks in by_doc.items()
            }
            for future in as_completed(futures):
                doc_id = futures[future]
                try:
                    raptor_docs.extend(future.result())
                except Exception as exc:
                    logger.warning(
                        "RaptorSummarizer: skipping doc_id=%s due to error: %s", doc_id, exc
                    )

        total = len(raptor_docs)
        logger.info(
            "RaptorSummarizer: added %d RAPTOR chunk(s) for %d document(s)",
            total, len(by_doc),
        )
        return {"documents": documents + raptor_docs}

    # ------------------------------------------------------------------

    def _process_document(self, doc_id: str, chunks: list[Document]) -> list[Document]:
        """Create section-level and doc-level RAPTOR chunks for one document."""
        # Representative meta (title, source, language, timestamps)
        rep_meta = chunks[0].meta

        # Group chunks by section_path (falls back to section_title or empty string)
        by_section: dict[str, list[Document]] = {}
        for chunk in chunks:
            path = chunk.meta.get("section_path") or chunk.meta.get("section_title") or ""
            by_section.setdefault(path, []).append(chunk)

        result: list[Document] = []
        section_summaries: list[str] = []

        for section_path, section_chunks in by_section.items():
            # Collect per-chunk summaries; skip sections with no LLM summaries
            chunk_summaries = [
                c.meta.get("summary", "").strip()
                for c in section_chunks
                if c.meta.get("summary", "").strip()
            ]
            if not chunk_summaries:
                continue

            section_title = (
                section_chunks[0].meta.get("section_title")
                or section_path.split(" › ")[-1]
                or "Section"
            )
            title = rep_meta.get("title", "Document")

            try:
                summary_text = self._call_llm(
                    _SECTION_PROMPT.format(
                        section_title = section_title,
                        title         = title,
                        summaries     = "\n---\n".join(chunk_summaries),
                    )
                )
            except Exception as exc:
                logger.debug("RaptorSummarizer: section LLM call failed: %s", exc)
                summary_text = " ".join(chunk_summaries)   # fallback: join raw summaries

            if not summary_text.strip():
                continue

            section_doc = Document(
                content = summary_text.strip(),
                meta    = {
                    **_base_meta(rep_meta),
                    "section_path":  section_path,
                    "section_title": section_title,
                    "chunk_type":    "raptor_section",
                    "parent_content": summary_text.strip(),
                },
                id = _stable_id(doc_id, f"raptor_section::{section_path}"),
            )
            result.append(section_doc)
            section_summaries.append(summary_text.strip())

        # Document-level RAPTOR chunk
        if section_summaries:
            try:
                doc_summary = self._call_llm(
                    _DOC_PROMPT.format(
                        title     = rep_meta.get("title", "Document"),
                        summaries = "\n---\n".join(section_summaries),
                    )
                )
            except Exception as exc:
                logger.debug("RaptorSummarizer: doc-level LLM call failed: %s", exc)
                doc_summary = " ".join(section_summaries)

            if doc_summary.strip():
                doc_chunk = Document(
                    content = doc_summary.strip(),
                    meta    = {
                        **_base_meta(rep_meta),
                        "section_path":  "DOCUMENT_SUMMARY",
                        "section_title": rep_meta.get("title", "Document"),
                        "chunk_type":    "raptor_doc",
                        "parent_content": doc_summary.strip(),
                    },
                    id = _stable_id(doc_id, "raptor_doc"),
                )
                result.append(doc_chunk)

        return result

    def _call_llm(self, prompt: str) -> str:
        response = self._client.chat.completions.create(
            model    = self._llm_model,
            messages = [
                {"role": "system", "content": _SECTION_SYSTEM},
                {"role": "user",   "content": prompt},
            ],
            temperature = 0.0,
            max_tokens  = 512,
        )
        return response.choices[0].message.content or ""


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _base_meta(source_meta: dict) -> dict:
    """Extract the fields that RAPTOR chunks should inherit from source docs."""
    keep = {
        "doc_id", "title", "source", "language",
        "indexed_at", "indexed_at_ts",
        "embedding_model", "embedding_provider", "embedding_dimension",
    }
    return {k: v for k, v in source_meta.items() if k in keep}
