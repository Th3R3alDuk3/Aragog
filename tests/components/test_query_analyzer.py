from components.query_analyzer import QueryAnalyzer


def test_parse_sanitizes_invalid_filter_values() -> None:
    analyzer = QueryAnalyzer.__new__(QueryAnalyzer)
    analyzer._taxonomy = {"financial", "general"}

    parsed = analyzer._parse(
        {
            "is_compound": True,
            "sub_questions": ["", "  "],
            "filters": {
                "date_from": "2024/13/01",
                "date_to": "not-a-date",
                "classification": "not-in-taxonomy",
                "language": "de",
                "source": "report",
            },
        },
        original_query="Originalfrage",
    )

    assert parsed.sub_questions == ["Originalfrage"]
    assert parsed.is_compound is False
    assert parsed.date_from is None
    assert parsed.date_to is None
    assert parsed.classification is None
    assert parsed.language == "de"
    assert parsed.source is None


def test_parse_normalizes_source_to_exact_filename() -> None:
    analyzer = QueryAnalyzer.__new__(QueryAnalyzer)
    analyzer._taxonomy = {"financial", "general"}

    parsed = analyzer._parse(
        {
            "is_compound": False,
            "sub_questions": ["frage"],
            "filters": {
                "source": 'Use path "/tmp/uploads/Quarterly-Report_2025.pdf" please.',
            },
        },
        original_query="frage",
    )

    assert parsed.source == "Quarterly-Report_2025.pdf"


def test_analyze_uses_fast_path_for_short_simple_query() -> None:
    analyzer = QueryAnalyzer.__new__(QueryAnalyzer)

    def fail_if_called(_query: str):
        raise AssertionError("LLM path should not be called for simple short queries")

    analyzer._llm_analyze = fail_if_called

    result = analyzer.analyze("Was ist EBITDA")

    assert result.sub_questions == ["Was ist EBITDA"]
    assert result.is_compound is False
