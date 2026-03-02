from logging import getLogger

from openai import AsyncOpenAI

logger = getLogger(__name__)


_SYSTEM = (
    "You are a document retrieval assistant. "
    "You write short, factual passages in declarative style."
)


_PROMPT = """\
Write a short factual document passage (2-3 sentences) that would directly \
answer the following question. Write as if excerpted from a knowledge base or \
reference document. Do NOT reference the question. Do NOT add a title or \
preamble — start directly with the content.

Question: {query}
"""


class HyDEGenerator:
    """
    Generates a hypothetical document for improved dense retrieval (HyDE).

    Args:
        openai_url: Custom base URL (empty = official OpenAI API).
        openai_api_key:  API key for the LLM endpoint.
        llm_model:       OpenAI-compatible model name.
    """

    def __init__(
        self,
        openai_url: str,
        openai_api_key: str,
        llm_model: str,
    ) -> None:

        self._client    = AsyncOpenAI(
            base_url=openai_url, 
            api_key=openai_api_key,
        )

        self._llm_model = llm_model

    async def generate(self, query: str) -> str:
        """Generate a hypothetical document passage for the given query.

        The generated text is intended for dense embedding only; sparse retrieval
        and the reranker always receive the original query unchanged.

        Falls back to the original query on any LLM error so that retrieval
        continues normally without disruption.

        Args:
            query: The user query to generate a hypothetical document for.

        Returns:
            A short declarative passage (3-5 sentences) that would answer the
            query, or the original query string if generation fails.
        """
        try:
            result = await self._call_llm(query)
            if stripped_result := result.strip():
                logger.debug("HyDE: generated hypothetical doc (%d chars)", len(result))
                return stripped_result
        except Exception as error:
            logger.warning("HyDE: LLM call failed (%s), falling back to original query", error)
        return query

    async def _call_llm(self, query: str) -> str:
        """Call the LLM to produce a hypothetical document passage.

        Args:
            query: The user query.

        Returns:
            The raw LLM response text (may be empty).
        """
        response = await self._client.chat.completions.create(
            model=self._llm_model,
            messages=[
                {"role": "system", "content": _SYSTEM},
                {"role": "user",   "content": _PROMPT.format(query=query)},
            ],
            temperature=0.5,
            max_tokens=150,
        )
        return response.choices[0].message.content or ""
