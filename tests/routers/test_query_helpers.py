from datetime import date

import pytest

from components.query_analyzer import AnalysisResult
from models.schemas import QueryRequest
from routers.query import _analysis_to_filter_dict, _build_filters, _normalize_filter_fields


def test_analysis_to_filter_dict_keeps_only_non_null_fields() -> None:
    analysis = AnalysisResult(
        sub_questions=["q"],
        is_compound=False,
        date_from=None,
        date_to="2024-12-31",
        classification="financial",
        language=None,
        source="report.pdf",
    )

    out = _analysis_to_filter_dict(analysis)

    assert out == {
        "date_to": "2024-12-31",
        "classification": "financial",
        "source": "report.pdf",
    }


def test_build_filters_merges_analysis_and_explicit_request_filters() -> None:
    request = QueryRequest(
        query="frage",
        date_from=date(2025, 1, 1),
        filters={"field": "source", "operator": "==", "value": "explicit.pdf"},
    )
    analysis = AnalysisResult(
        sub_questions=["frage"],
        is_compound=False,
        date_from="2024-01-01",  # ignored — explicit date_from takes priority
        date_to="2024-12-31",
        classification="financial",
        language="de",
        source="analysis.pdf",
    )

    out = _build_filters(request, analysis)

    assert out is not None
    assert out["operator"] == "AND"
    conditions = out["conditions"]

    # Explicit user filter: bare field name gets meta. prefix automatically
    assert {"field": "meta.source", "operator": "==", "value": "explicit.pdf"} in conditions

    # Explicit date_from → meta.indexed_at_ts with >= operator
    assert any(c.get("field") == "meta.indexed_at_ts" and c.get("operator") == ">=" for c in conditions)

    # Analysis date_from must be suppressed (request.date_from wins)
    analysis_date_from_ts = 1704067200  # 2024-01-01T00:00:00Z
    assert not any(
        c.get("field") == "meta.indexed_at_ts"
        and c.get("operator") == ">="
        and c.get("value") == analysis_date_from_ts
        for c in conditions
    )

    # LLM-extracted fields carry the meta. prefix
    assert any(c.get("field") == "meta.classification" for c in conditions)
    assert any(c.get("field") == "meta.language" for c in conditions)
    assert any(c.get("field") == "meta.source" for c in conditions)


def test_normalize_filter_fields_adds_meta_prefix() -> None:
    # Bare field name → meta. prefix added
    f = {"field": "source", "operator": "==", "value": "x.pdf"}
    assert _normalize_filter_fields(f) == {"field": "meta.source", "operator": "==", "value": "x.pdf"}

    # Already prefixed → unchanged
    f2 = {"field": "meta.source", "operator": "==", "value": "x.pdf"}
    assert _normalize_filter_fields(f2) == f2

    # Special top-level fields → unchanged
    assert _normalize_filter_fields({"field": "id", "operator": "==", "value": "abc"})["field"] == "id"
    assert _normalize_filter_fields({"field": "content", "operator": "==", "value": "abc"})["field"] == "content"

    # Nested logical operator → recurses into conditions
    nested = {
        "operator": "AND",
        "conditions": [
            {"field": "classification", "operator": "==", "value": "financial"},
            {"field": "meta.language",  "operator": "==", "value": "de"},
        ],
    }
    result = _normalize_filter_fields(nested)
    assert result["conditions"][0]["field"] == "meta.classification"
    assert result["conditions"][1]["field"] == "meta.language"  # unchanged


def test_build_filters_accepts_shorthand_mapping_filters() -> None:
    request = QueryRequest(
        query="frage",
        filters={"source": "roman.pdf", "classification": "report"},
    )

    out = _build_filters(request, analysis=None)

    assert out is not None
    assert out["operator"] == "AND"
    assert {"field": "meta.source", "operator": "==", "value": "roman.pdf"} in out["conditions"]
    assert {"field": "meta.classification", "operator": "==", "value": "report"} in out["conditions"]


def test_build_filters_raises_for_invalid_filter_expression() -> None:
    request = QueryRequest(
        query="frage",
        filters={"operator": "AND", "conditions": [{"field": "source", "value": "x.pdf"}]},
    )

    with pytest.raises(ValueError, match="Invalid filters format"):
        _build_filters(request, analysis=None)


def test_build_filters_raises_for_object_value_in_expression() -> None:
    request = QueryRequest(
        query="frage",
        filters={"field": "source", "operator": "==", "value": {}},
    )

    with pytest.raises(ValueError, match="Invalid filters format"):
        _build_filters(request, analysis=None)


def test_build_filters_ignores_empty_object_in_shorthand_mapping() -> None:
    request = QueryRequest(
        query="frage",
        filters={"source": {}, "classification": "report"},
    )

    out = _build_filters(request, analysis=None)

    assert out == {"field": "meta.classification", "operator": "==", "value": "report"}
