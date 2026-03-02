from logging import getLogger

from openai import AsyncOpenAI
from pydantic import BaseModel, Field

logger = getLogger(__name__)


_SYSTEM = (
    "You are a query analysis assistant. "
    "Extract sub-questions and metadata filters from the user query.\n"
    "Sub-question rules:\n"
    "  - Split whenever the query asks for two or more pieces of information that could realistically "
    "appear in different document sections — even if they concern the same entity.\n"
    "  - Examples that SHOULD be split:\n"
    "      'What are the main risks and what are the revenue figures?' → ['What are the main risks?', 'What are the revenue figures?']\n"
    "      'Who are the main characters and what are their traits?' → ['Who are the main characters?', 'What are the traits of the main characters?']\n"
    "      'When was the company founded and what are its products?' → ['When was the company founded?', 'What are the products of the company?']\n"
    "  - Examples that should NOT be split:\n"
    "      'Who is the CEO of ACME?' → ['Who is the CEO of ACME?']\n"
    "      'What happened in chapter 3?' → ['What happened in chapter 3?']\n"
    "  - Each sub-question MUST be fully self-contained and standalone: replace all pronouns and "
    "references ('their', 'its', 'sie', 'ihre', 'deren', 'es', 'seine') with the explicit noun from the original query.\n"
    "  - Set is_compound=true whenever you produce more than one sub-question.\n"
    "  - Preserve the original language of the query in sub_questions.\n"
    "Filter rules:\n"
    "  - Only set date_from / date_to when a time range is explicitly mentioned.\n"
    "  - Only set classification when a document type is explicitly named.\n"
    "  - Only set language when the user explicitly asks for documents in a specific language "
    "(e.g. 'only German documents'). Never infer language from the language the query is written in.\n"
    "  - Only set source when a specific filename is explicitly mentioned.\n"
    "  - Never guess — leave fields null when uncertain."
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
        description=(
            "ISO 639-1 language code (e.g. 'de', 'en'). "
            "Set ONLY when the user explicitly asks for documents in a specific language. "
            "Never infer from the language the query is written in."
        ),
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
