from haystack import Document

from components.content_analyzer import _apply, _parse_json


def test_parse_json_handles_fenced_json_and_entity_defaults() -> None:
    raw = """
```json
{
  "summary": "Kurze Zusammenfassung",
  "entities": {
    "persons": ["Ada Lovelace"]
  }
}
```
"""

    result = _parse_json(raw)

    assert result["summary"] == "Kurze Zusammenfassung"
    assert result["classification"] == "general"
    assert result["context_prefix"] == ""
    assert result["entities"]["persons"] == ["Ada Lovelace"]
    assert result["entities"]["organizations"] == []
    assert result["entities"]["locations"] == []


def test_apply_prepends_prefix_and_preserves_original_content() -> None:
    doc = Document(
        id="doc-1",
        content="Body text",
        meta={"doc_beginning": "Intro", "language": "de", "source": "report.pdf"},
    )
    analysis = _parse_json(
        '{"context_prefix": "Kontext", "summary": "Kurz", "keywords": ["k1"], "classification": "technical"}'
    )

    out = _apply(doc, analysis)

    assert out.content == "Kontext\n\nBody text"
    assert out.meta["original_content"] == "Body text"
    assert out.meta["context_prefix"] == "Kontext"
    assert out.meta["summary"] == "Kurz"
    assert out.meta["keywords"] == ["k1"]
    assert out.meta["classification"] == "technical"
    assert out.meta["language"] == "de"
    assert "doc_beginning" not in out.meta
