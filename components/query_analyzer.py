"""
QueryAnalyzer — extended query decomposer.

Replaces ``QueryDecomposer`` (which is aliased here for backward compatibility).

In a single LLM call the analyzer:
  1. Detects whether the query contains multiple independent sub-questions
     and splits them (same logic as before).
  2. Extracts structured metadata filter hints from natural language:
       date_from / date_to  — temporal constraints ("since 2024", "Q3 2023")
       classification        — document category ("financial report", "contract")
       language              — document language ("German documents", "auf Deutsch")
       source                — exact filename ("from report.pdf")
     Extracted hints are merged with any explicit ``filters`` the caller
     provides — explicit always win.

Heuristics BEFORE the LLM call
────────────────────────────────
• Short query (≤ 10 words) AND no compound/filter hints → skip LLM entirely.
• Two separate regexes decide whether to call the LLM:
    _COMPOUND_HINTS — multi-question connectors
    _FILTER_HINTS   — date/document-type/language signals
  If either matches, the LLM is called.

Backward compatibility
──────────────────────
``QueryDecomposer = QueryAnalyzer`` alias at the bottom keeps existing
``main.py`` imports working without change.
``decompose(query) → list[str]`` shim is preserved.
"""

import json
import re
from dataclasses import dataclass
from pathlib import PurePath
from typing import Any

from openai import OpenAI

# ---------------------------------------------------------------------------
# Heuristic pre-filters (avoid LLM call for simple, clearly single queries)
# ---------------------------------------------------------------------------

_COMPOUND_HINTS = re.compile(
    r"\b(und|and|sowie|außerdem|additionally|also|furthermore|"
    r"darüber hinaus|wie auch|as well as|moreover|"
    r"\?[^?]+\?)\b",
    re.IGNORECASE,
)

_FILTER_HINTS = re.compile(
    r"\b(since|before|after|until|from|in \d{4}|"
    r"january|february|march|april|may|june|july|august|"
    r"september|october|november|december|"
    r"januar|februar|märz|april|mai|juni|juli|august|"
    r"september|oktober|november|dezember|"
    r"q[1-4]\s*\d{4}|\d{4}\s*q[1-4]|"
    r"financial|legal|technical|scientific|hr|marketing|"
    r"contract|report|manual|correspondence|"
    r"vertrag|bericht|handbuch|"
    r"german|english|french|spanish|deutsch|englisch|"
    r"\.pdf|\.docx|\.pptx)\b",
    re.IGNORECASE,
)

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

_SYSTEM = (
    "You are a query analysis assistant. "
    "You always respond with a single valid JSON object — no markdown, no prose."
)

_PROMPT = """\
Analyse the following user query and return a JSON object with:

1. Whether it contains multiple independent questions.
2. Any metadata filters that can be extracted from natural language.

Return ONLY this JSON structure:
{{
  "is_compound": <true | false>,
  "sub_questions": ["<question 1>", "<question 2>", ...],
  "filters": {{
    "date_from":      "<ISO date YYYY-MM-DD or null>",
    "date_to":        "<ISO date YYYY-MM-DD or null>",
    "classification": "<one of: financial,legal,technical,scientific,hr,marketing,contract,report,manual,correspondence,general — or null>",
    "language":       "<ISO 639-1 code (e.g. 'de', 'en', 'fr') or null>",
    "source":         "<exact filename (e.g. report.pdf) or null>"
  }}
}}

Rules:
- If the query is a single question, set is_compound to false and
  sub_questions to [<the original query unchanged>].
- Only split on genuinely independent information needs.
- Preserve the original language of the query in the sub_questions.
- Maximum 5 sub_questions.
- For filters: only set a value when clearly stated in the query.
  Set null for anything not mentioned. Do NOT guess.
- date_from / date_to: convert expressions like "Q3 2024" to
  "2024-07-01" / "2024-09-30", "since 2023" to "2023-01-01" / null, etc.
- classification: match to the given taxonomy exactly; null if unclear.
- source: only set this when an exact filename is clearly mentioned.
  Return only the filename, never a path or a partial substring.
- classification taxonomy: {taxonomy}

Query: {query}
"""


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class AnalysisResult:
    """Structured result from a single QueryAnalyzer LLM call."""
    sub_questions: list[str]
    is_compound: bool
    # Extracted filter hints — None means "not mentioned in query"
    date_from: str | None = None        # ISO date string "YYYY-MM-DD"
    date_to: str | None = None
    classification: str | None = None   # validated against taxonomy
    language: str | None = None         # ISO 639-1
    source: str | None = None           # exact filename


# ---------------------------------------------------------------------------
# QueryAnalyzer
# ---------------------------------------------------------------------------

class QueryAnalyzer:
    """
    Detects compound queries AND extracts metadata filter hints in one LLM call.

    Args:
        openai_api_key:  API key for the LLM endpoint.
        llm_model:       OpenAI-compatible model name.
        openai_base_url: Custom base URL (empty = official OpenAI API).
        taxonomy:        Comma-separated classification labels.
    """

    def __init__(
        self,
        openai_api_key: str,
        llm_model: str,
        openai_base_url: str = "",
        taxonomy: str = (
            "financial,legal,technical,scientific,hr,"
            "marketing,contract,report,manual,correspondence,general"
        ),
    ) -> None:
        client_kwargs: dict[str, Any] = {"api_key": openai_api_key}
        if openai_base_url:
            client_kwargs["base_url"] = openai_base_url
        self._client    = OpenAI(**client_kwargs)
        self._llm_model = llm_model
        self._taxonomy  = set(t.strip() for t in taxonomy.split(",") if t.strip())
        self._taxonomy_str = taxonomy

    def analyze(self, query: str) -> AnalysisResult:
        """
        Full analysis: decompose + extract filters.

        Fast path: if the query is short and contains no compound/filter hints,
        returns a trivial AnalysisResult with [query] and no filters.
        """
        query = query.strip()
        words = query.split()

        # Fast path — no LLM call
        if len(words) <= 10 and not _COMPOUND_HINTS.search(query) and not _FILTER_HINTS.search(query):
            return AnalysisResult(sub_questions=[query], is_compound=False)

        return self._llm_analyze(query)

    def decompose(self, query: str) -> list[str]:
        """Backward-compatible shim — returns sub_questions from analyze()."""
        return self.analyze(query).sub_questions

    # ------------------------------------------------------------------

    def _llm_analyze(self, query: str) -> AnalysisResult:
        try:
            response = self._client.chat.completions.create(
                model    = self._llm_model,
                messages = [
                    {"role": "system", "content": _SYSTEM},
                    {"role": "user",   "content": _PROMPT.format(
                        query=query, taxonomy=self._taxonomy_str,
                    )},
                ],
                response_format = {"type": "json_object"},
                temperature     = 0.0,
            )
            raw = response.choices[0].message.content or "{}"
            return self._parse(json.loads(raw), query)
        except Exception:
            # Never crash the request over analysis failure
            return AnalysisResult(sub_questions=[query], is_compound=False)

    def _parse(self, data: dict, original_query: str) -> AnalysisResult:
        sub_questions = data.get("sub_questions", [original_query])
        sub_questions = [str(q).strip() for q in sub_questions if str(q).strip()]
        if not sub_questions:
            sub_questions = [original_query]

        is_compound = bool(data.get("is_compound", False)) and len(sub_questions) > 1

        f = data.get("filters") or {}

        # Validate classification against known taxonomy
        cls = f.get("classification")
        if cls and cls.lower() not in self._taxonomy:
            cls = None

        # Basic ISO date format validation
        def _valid_date(s: Any) -> str | None:
            if not s or not isinstance(s, str):
                return None
            s = s.strip()
            if re.fullmatch(r"\d{4}-\d{2}-\d{2}", s):
                return s
            return None

        # Source filter must target a concrete filename to match with "==".
        def _valid_source(s: Any) -> str | None:
            if not s or not isinstance(s, str):
                return None
            text = s.strip()
            if not text:
                return None

            def _extract_name(t: str) -> str | None:
                """Return the filename component of t if it has a clean extension."""
                name = PurePath(t.replace("\\", "/")).name
                ext = PurePath(name).suffix.lstrip(".")   # e.g. "pdf"
                # Extension must be alphanumeric only (catches "pdf file", ".pdf doc", …)
                if ext and ext.isalnum() and 1 <= len(ext) <= 10:
                    return name
                return None

            # 1) Direct filename or path returned by the model
            candidate = _extract_name(text)
            if candidate:
                return candidate

            # 2) Filename inside quotes — handles spaces: "report 2024.pdf"
            for quote in ('"', "'"):
                if quote in text:
                    for part in text.split(quote)[1::2]:   # content between quotes
                        candidate = _extract_name(part)
                        if candidate:
                            return candidate

            return None

        return AnalysisResult(
            sub_questions  = sub_questions,
            is_compound    = is_compound,
            date_from      = _valid_date(f.get("date_from")),
            date_to        = _valid_date(f.get("date_to")),
            classification = cls,
            language       = f.get("language") or None,
            source         = _valid_source(f.get("source")),
        )


# ---------------------------------------------------------------------------
# Backward-compatibility alias
# ---------------------------------------------------------------------------
QueryDecomposer = QueryAnalyzer
