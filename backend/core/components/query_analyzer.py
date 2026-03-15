from logging import getLogger
from string import punctuation

from openai import AsyncOpenAI
from pydantic import BaseModel, Field

from core.models.vocabulary import (
    DEFAULT_CHUNK_CLASSIFICATION_TAXONOMY,
    ChunkClassification,
    DocumentType,
    LanguageCode,
)

logger = getLogger(__name__)

_COMPOUND_CONNECTORS = {
    "and",
    "und",
    "plus",
    "sowie",
    "also",
    "sowie",
    "compare",
    "versus",
    "vs",
}
_LANGUAGE_HINTS = (
    "german",
    "english",
    "deutsch",
    "englisch",
    "spanish",
    "french",
    "nur deutsche",
    "only german",
)
_SOURCE_HINT_SUFFIXES = (".pdf", ".docx", ".pptx", ".xlsx", ".txt", ".html", ".md")
_DATE_HINT_TOKENS = {
    "q1",
    "q2",
    "q3",
    "q4",
    "quarter",
    "quartal",
    "since",
    "until",
    "before",
    "after",
    "between",
    "from",
    "to",
    "bis",
    "seit",
    "zwischen",
}
_CLASSIFICATION_HINT_TOKENS = {
    "overview",
    "background",
    "definition",
    "requirement",
    "requirements",
    "procedure",
    "procedures",
    "example",
    "examples",
    "warning",
    "warnings",
    "decision",
    "decisions",
    "result",
    "results",
    "reference",
    "references",
    "financial",
    "legal",
    "technical",
    "operational",
}
_DOCUMENT_TYPE_HINTS = {
    "report",
    "manual",
    "contract",
    "agreement",
    "policy",
    "invoice",
    "article",
    "paper",
    "presentation",
    "slides",
    "spreadsheet",
    "bericht",
    "vertrag",
    "rechnung",
    "richtlinie",
    "handbuch",
}


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
    "      'Who is the antagonist and what are his plans?' → ['Who is the antagonist?', 'What are the plans of the antagonist?']\n"
    "      'What is X, where is it located, and why is it important?' → ['What is X?', 'Where is X located?', 'Why is X important?']\n"
    "      'Which magical places exist and what happens there?' → ['Which magical places exist?', 'What happens in the magical places?']\n"
    "  - Examples that should NOT be split:\n"
    "      'Who is the CEO of ACME?' → ['Who is the CEO of ACME?']\n"
    "      'What happened in chapter 3?' → ['What happened in chapter 3?']\n"
    "      'Give a chronological summary of all events.' → ['Give a chronological summary of all events.']\n"
    "  - Each sub-question MUST be fully self-contained and standalone: replace all pronouns and "
    "references ('their', 'its', 'sie', 'ihre', 'deren', 'es', 'seine') with the explicit noun from the original query.\n"
    "  - Set is_compound=true whenever you produce more than one sub-question.\n"
    "  - Preserve the original language of the query in sub_questions.\n"
    "Filter rules:\n"
    "  - Only set date_from / date_to when a time range is explicitly mentioned.\n"
    "  - Only set document_type when a document form is explicitly named (report, contract, manual, policy, invoice, presentation, spreadsheet, article, correspondence, specification).\n"
    "  - Only set classification when the user explicitly asks for a chunk category "
    "such as financial, legal, technical, requirement, procedure, or warning.\n"
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
    classification: ChunkClassification | None = Field(
        default=None,
        description="Classification label from the taxonomy; null if unclear.",
    )
    document_type: DocumentType | None = Field(
        default=None,
        description="Coarse document form such as report, manual, contract, or policy.",
    )
    language: LanguageCode | None = Field(
        default=None,
        description=(
            "ISO 639-1 language code from the supported top-language set "
            "(e.g. 'de', 'en', 'zh'). "
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
        taxonomy: str = DEFAULT_CHUNK_CLASSIFICATION_TAXONOMY,
    ) -> None:

        self._client = AsyncOpenAI(base_url=openai_url, api_key=openai_api_key)
        self._llm_model = llm_model
        self._taxonomy = taxonomy

    async def analyze(self, query: str) -> AnalysisResult:

        query = query.strip()

        fallback = AnalysisResult(sub_questions=[query], is_compound=False)
        if _should_skip_llm(query):
            return fallback

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


def _should_skip_llm(query: str) -> bool:
    tokens = _tokenize(query)
    if not tokens:
        return True

    has_connectors = any(token in _COMPOUND_CONNECTORS for token in tokens)
    has_filter_hints = _has_filter_hints(query, tokens)
    question_marks = query.count("?")

    if not has_connectors and not has_filter_hints and len(tokens) <= 10:
        return True
    if not has_connectors and not has_filter_hints and question_marks <= 1 and len(tokens) <= 14:
        return True
    return False


def _has_filter_hints(query: str, tokens: list[str]) -> bool:
    lowered = query.casefold()
    if any(suffix in lowered for suffix in _SOURCE_HINT_SUFFIXES):
        return True
    if any(hint in lowered for hint in _LANGUAGE_HINTS):
        return True
    if any(token in _DATE_HINT_TOKENS for token in tokens):
        return True
    if any(token in _CLASSIFICATION_HINT_TOKENS for token in tokens):
        return True
    if any(token in _DOCUMENT_TYPE_HINTS for token in tokens):
        return True
    if any(token.isdigit() and len(token) == 4 for token in tokens):
        return True
    return False


def _tokenize(text: str) -> list[str]:
    translation = str.maketrans({char: " " for char in punctuation})
    return [token.casefold() for token in text.translate(translation).split() if token]
