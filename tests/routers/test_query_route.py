from types import SimpleNamespace

from routers.query import _retrieve_simple


class FakePipeline:
    def __init__(self, docs):
        self.docs = docs
        self.last_input = None

    def run(self, run_input):
        self.last_input = run_input
        return {"reranker": {"documents": self.docs}}


class FakeHyDE:
    def __init__(self, generated_text: str) -> None:
        self.generated_text = generated_text
        self.calls = []

    def generate(self, query: str) -> str:
        self.calls.append(query)
        return self.generated_text


def test_retrieve_simple_uses_original_query_when_hyde_disabled() -> None:
    docs = [SimpleNamespace(id="1", score=0.77)]
    pipeline = FakePipeline(docs)

    result = _retrieve_simple(
        pipeline=pipeline,
        sub_q="Was ist EBITDA?",
        filters=None,
        hyde_generator=None,
        use_hyde=False,
    )

    assert result == docs
    assert pipeline.last_input["dense_embedder"]["text"] == "Was ist EBITDA?"
    assert pipeline.last_input["sparse_embedder"]["text"] == "Was ist EBITDA?"
    assert "dense_retriever" not in pipeline.last_input
    assert "sparse_retriever" not in pipeline.last_input


def test_retrieve_simple_uses_hyde_text_for_dense_query_and_applies_filters() -> None:
    docs = [SimpleNamespace(id="1", score=0.82)]
    pipeline = FakePipeline(docs)
    hyde = FakeHyDE("Hypothetical answer passage")
    filters = {"field": "source", "operator": "==", "value": "report.pdf"}

    _retrieve_simple(
        pipeline=pipeline,
        sub_q="Frage",
        filters=filters,
        hyde_generator=hyde,
        use_hyde=True,
    )

    assert hyde.calls == ["Frage"]
    assert pipeline.last_input["dense_embedder"]["text"] == "Hypothetical answer passage"
    assert pipeline.last_input["sparse_embedder"]["text"] == "Frage"
    assert pipeline.last_input["dense_retriever"]["filters"] == filters
    assert pipeline.last_input["sparse_retriever"]["filters"] == filters
