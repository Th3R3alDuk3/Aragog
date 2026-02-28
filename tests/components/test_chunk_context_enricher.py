from haystack import Document

from components.chunk_context_enricher import _detect_chunk_type, _enrich_group


def test_detect_chunk_type_variants() -> None:
    assert _detect_chunk_type("```python\nprint('x')\n```") == "code"
    assert _detect_chunk_type("| A | B |\n|---|---|\n| 1 | 2 |") == "table"
    assert _detect_chunk_type("- first\n- second\nplain") == "list"
    assert _detect_chunk_type("Figure 3: Revenue chart") == "figure_caption"
    assert _detect_chunk_type("This is normal prose.") == "text"


def test_enrich_group_adds_indices_and_section_path() -> None:
    chunks = [
        Document(id="a", content="one", meta={"header": "Overview", "parent_headers": ["Intro"]}),
        Document(id="b", content="two", meta={"header": "Overview", "parent_headers": ["Intro"]}),
    ]

    enriched = _enrich_group("doc-1", chunks)

    assert len(enriched) == 2
    assert enriched[0].meta["chunk_index"] == 0
    assert enriched[0].meta["chunk_total"] == 2
    assert enriched[0].meta["section_title"] == "Overview"
    assert enriched[0].meta["section_path"] == "Intro \u203a Overview"
