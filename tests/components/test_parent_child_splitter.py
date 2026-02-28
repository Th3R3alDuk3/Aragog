from haystack import Document

from components.parent_child_splitter import _with_parent


def test_with_parent_adds_parent_metadata() -> None:
    child = Document(id="child-1", content="child text", meta={"header": "Intro"})

    out = _with_parent(child, "parent section text")

    assert out.content == "child text"
    assert out.meta["parent_content"] == "parent section text"
    assert out.meta["parent_section"] == "Intro"
