from logging import getLogger

from openai import AsyncOpenAI
from pydantic import BaseModel, Field

logger = getLogger(__name__)


_SYSTEM = (
    "You are a query analysis assistant. "
    "Extract sub-questions and metadata filters from the user query. "
    "Only split on genuinely independent information needs; for a single question set is_compound=false and sub_questions to [original query]. "
    "Only set filter values when clearly stated — never guess. "
    "Preserve the original language of the query in sub_questions."
)


class AnalysisResult(BaseModel):
    is_compound: bool = Field(
        default=False,
        description="True only if multiple independent questions were detected.",
    )
    sub_questions: list[str] = Field(
        min_length=1,
        max_length=5,
        description="Original-language sub-questions; one item for simple queries.",
    )
    date_from: str | None = Field(
        default=None,
        description="Lower date bound as ISO-8601 date (YYYY-MM-DD), e.g. 'since 2023' → '2023-01-01'.",
    )
    date_to: str | None = Field(
        default=None,
        description="Upper date bound as ISO-8601 date (YYYY-MM-DD).",
    )
    classification: str | None = Field(
        default=None,
        description="Classification label from the taxonomy; null if unclear.",
    )
    language: str | None = Field(
        default=None,
        description="ISO 639-1 language code, lowercase (e.g. 'de', 'en').",
    )
    source: str | None = Field(
        default=None,
        description="Exact filename including extension (e.g. report.pdf); only if clearly mentioned.",
    )


class QueryAnalyzer:
    
    def __init__(
        self,
        openai_url: str,
        openai_api_key: str,
        llm_model: str,
        taxonomy: str = (
            "financial,legal,technical,scientific,hr,"
            "marketing,contract,report,manual,correspondence,general"
        ),
    ) -> None:

        self._client = AsyncOpenAI(base_url=openai_url, api_key=openai_api_key)
        self._llm_model = llm_model
        self._taxonomy = taxonomy

    async def analyze(self, query: str) -> AnalysisResult:

        query = query.strip()

        fallback = AnalysisResult(sub_questions=[query], is_compound=False)

        try:
            response = await self._client.beta.chat.completions.parse(
                model=self._llm_model,
                messages=[
                    {"role": "system", "content": _SYSTEM},
                    {
                        "role": "user",
                        "content": f"Classification taxonomy: {self._taxonomy}\n\nQuery: {query}",
                    },
                ],
                response_format=AnalysisResult,
                temperature=0.0,
            )
            return response.choices[0].message.parsed or fallback
        except Exception as exc:
            logger.warning("QueryAnalyzer: analysis failed, fallback (%s)", exc)
            return fallback
