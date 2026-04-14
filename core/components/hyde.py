from logging import getLogger

from haystack import component
from openai import OpenAI

logger = getLogger(__name__)


_PROMPT = """\
Please write a passage to answer the question. \
Try to include as many key details as possible. \
Do NOT add a title or preamble — start directly with the content.

Question: {query}
Passage:"""


@component
class HyDE:
    """
    Haystack component that generates a hypothetical document for improved
    dense retrieval (HyDE — Hypothetical Document Embeddings, Gao et al. 2022).

    At query time the LLM produces a short declarative passage that would
    answer the question.  Its embedding is used for the second dense-retrieval
    branch; the original query text is still used by the sparse retriever and
    the reranker.

    Falls back silently to the original query text on any LLM error so that
    retrieval continues normally without disruption.

    Args:
        openai_url:     Custom base URL (empty = official OpenAI API).
        openai_api_key: API key for the LLM endpoint.
        llm_model:      OpenAI-compatible model name.
    """

    def __init__(
        self,
        openai_url: str,
        openai_api_key: str,
        llm_model: str,
    ) -> None:
        self._client    = OpenAI(
            base_url=openai_url,
            api_key=openai_api_key,
        )
        self._llm_model = llm_model

    @component.output_types(text=str)
    def run(self, query: str) -> dict:
        """Generate a hypothetical passage for *query* and return it as ``text``.

        Falls back to the original query on any LLM error.
        Haystack's AsyncPipeline runs sync components in a thread executor,
        so blocking here is safe.
        """
        try:
            response = self._client.chat.completions.create(
                model=self._llm_model,
                messages=[{"role": "user", "content": _PROMPT.format(query=query)}],
                temperature=0.5,
                max_tokens=250,
            )
            text = (response.choices[0].message.content or "").strip()
            if text:
                logger.info(
                    "  HyDE: hypothetical doc generated (%d chars): %s",
                    len(text), text[:300],
                )
                return {"text": text}
        except Exception as error:
            logger.warning("HyDE: LLM call failed (%s), falling back to original query", error)
        return {"text": query}
