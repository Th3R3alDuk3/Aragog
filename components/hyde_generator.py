"""
HyDE — Hypothetical Document Embedding.

Before dense retrieval, the LLM generates a short "hypothetical document"
that would ideally answer the user's query.  This text — NOT the raw query —
is embedded and used for the dense retrieval pass.

Why this helps
──────────────
Query vectors and document vectors live in different parts of the embedding
space: queries are short and interrogative; documents are long and declarative.
Embedding a hypothetical document that *looks like* a real answer shifts the
query vector into the document manifold, dramatically improving dense recall.

Sparse retrieval and the cross-encoder reranker always use the original
query (keyword matching and relevance scoring work better with the actual
question text).

Usage
─────
``HyDEGenerator`` is a plain Python class — not a Haystack pipeline component.
It is called at the router layer before each retrieval pass:

    dense_text = hyde.generate(sub_q) if hyde else sub_q
    run_input["dense_embedder"] = {"text": dense_text}
    run_input["sparse_embedder"] = {"text": sub_q}   # always original
    run_input["reranker"]        = {"query": sub_q}  # always original

The generator instance is created once at startup and stored on app.state.

Error handling
──────────────
On any LLM error the original query is returned unchanged — retrieval
continues normally.
"""

import logging
from typing import Any

from openai import OpenAI

logger = logging.getLogger(__name__)

_SYSTEM = (
    "You are a document retrieval assistant. "
    "You write short, factual passages in declarative style."
)

_PROMPT = """\
Write a short factual document passage (3-5 sentences) that would directly \
answer the following question. Write as if excerpted from a knowledge base or \
reference document. Do NOT reference the question. Do NOT add a title or \
preamble — start directly with the content.

Question: {query}
"""


class HyDEGenerator:
    """
    Generates a hypothetical document for improved dense retrieval (HyDE).

    Args:
        openai_api_key:  API key for the LLM endpoint.
        llm_model:       OpenAI-compatible model name.
        openai_base_url: Custom base URL (empty = official OpenAI API).
    """

    def __init__(
        self,
        openai_api_key: str,
        llm_model: str,
        openai_base_url: str = "",
    ) -> None:
        client_kwargs: dict[str, Any] = {"api_key": openai_api_key}
        if openai_base_url:
            client_kwargs["base_url"] = openai_base_url
        self._client    = OpenAI(**client_kwargs)
        self._llm_model = llm_model

    def generate(self, query: str) -> str:
        """
        Return a hypothetical document string for the given query.

        Falls back to the original query on any error so that retrieval
        continues normally without any disruption.
        """
        try:
            result = self._call_llm(query)
            if result.strip():
                logger.debug("HyDE: generated hypothetical doc (%d chars)", len(result))
                return result.strip()
        except Exception as exc:
            logger.warning("HyDE: LLM call failed (%s), falling back to original query", exc)
        return query

    def _call_llm(self, query: str) -> str:
        response = self._client.chat.completions.create(
            model    = self._llm_model,
            messages = [
                {"role": "system", "content": _SYSTEM},
                {"role": "user",   "content": _PROMPT.format(query=query)},
            ],
            temperature = 0.5,   # slight variation helps diversity; too high hurts precision
            max_tokens  = 300,
        )
        return response.choices[0].message.content or ""
