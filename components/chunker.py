from haystack import Document, component
from docling_core.types import DoclingDocument
from docling_core.transforms.chunker.hybrid_chunker import HybridChunker
from fastmcp.utilities.logging import get_logger


logger = get_logger(__name__)


@component
class DoclingHybridChunker:

    def __init__(self,
        tokenizer: str,
        max_tokens: int,
    ) -> None:
        self._chunker = HybridChunker(
            tokenizer=tokenizer,
            max_tokens=max_tokens,
        )

    @component.output_types(documents=list[Document])
    def run(self,
        documents: list[Document],
    ) -> dict[str, list[Document]]:

        all_chunks: list[Document] = []

        for document in documents:

            docling_document = DoclingDocument.model_validate_json(document.content)

            document_chunks = list(self._chunker.chunk(docling_document))
            total_chunks = len(document_chunks)

            for chunk_index, chunk in enumerate(document_chunks):

                page_numbers = sorted({
                    doc_item_prov.page_no
                    for doc_item in chunk.meta.doc_items
                    for doc_item_prov in doc_item.prov
                })

                content_types = sorted({
                    doc_item.label.value for doc_item in chunk.meta.doc_items
                })

                all_chunks.append(Document(
                    content=self._chunker.contextualize(chunk),
                    meta={
                        **document.meta,
                        "headings": chunk.meta.headings or [],
                        "page_number": min(page_numbers, default=None),
                        "page_numbers": page_numbers,
                        "content_types": content_types,
                        "chunk_index": chunk_index,
                        "total_chunks": total_chunks,
                    },
                ))

        logger.info(f"DoclingHybridChunker: {len(documents)} doc(s) → {len(all_chunks)} chunk(s)")

        return {"documents": all_chunks}
