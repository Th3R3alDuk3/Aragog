from haystack import Document

from pipelines.retrieval import swap_to_parent_content


def test_swap_to_parent_content_replaces_child_content_and_deduplicates() -> None:
    docs = [
        Document(id="1", content="child-a", meta={"parent_content": "parent-a"}, score=0.9),
        Document(id="2", content="child-b", meta={"parent_content": "parent-a"}, score=0.8),
        Document(id="3", content="child-c", meta={}, score=0.7),
    ]

    result = swap_to_parent_content(docs)

    assert len(result) == 2
    assert result[0].content == "parent-a"
    assert result[0].id == "1"
    assert result[1].content == "child-c"
    assert result[1].id == "3"
