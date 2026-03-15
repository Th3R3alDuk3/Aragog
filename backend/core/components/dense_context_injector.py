from haystack import Document, component

from core.models.meta import ChunkMetadata


@component
class DenseContextInjector:
    """Prepends the LLM-generated context prefix for dense embeddings only."""

    @component.output_types(documents=list[Document])
    def run(self, documents: list[Document]) -> dict[str, list[Document]]:
        return {"documents": [_inject_context(doc) for doc in documents]}


def _inject_context(doc: Document) -> Document:
    meta = ChunkMetadata.model_validate(doc.meta)
    base_content = meta.original_content or doc.content or ""
    prefix = meta.context_prefix.strip()
    content = f"{prefix}\n\n{base_content}" if prefix else base_content
    return Document(
        id=doc.id,
        content=content,
        blob=doc.blob,
        meta=dict(doc.meta),
        score=doc.score,
        embedding=doc.embedding,
        sparse_embedding=doc.sparse_embedding,
    )
