from hashlib import sha256

from fastmcp.utilities.logging import get_logger
from haystack import Document, component
from semchunk import chunkerify
from tokenizers import Tokenizer

logger = get_logger(__name__)


@component
class TextChunker:

    def __init__(self,
        tokenizer: str,
        max_tokens: int,
    ) -> None:
        # splits at paragraphs, then sentences, then words within the token limit
        self._chunk = chunkerify(Tokenizer.from_pretrained(tokenizer), max_tokens)

    @component.output_types(documents=list[Document])
    def run(self,
        documents: list[Document],
    ) -> dict[str, list[Document]]:

        all_chunks: list[Document] = []

        for document in documents:

            chunks = self._chunk(document.content)
            total_chunks = len(chunks)

            for chunk_index, chunk in enumerate(chunks):
                all_chunks.append(Document(
                    id=sha256(
                        f"{document.id}:{chunk_index}:{chunk}".encode()
                    ).hexdigest(),
                    content=chunk,
                    meta={
                        **document.meta,
                        "chunk_index": chunk_index,
                        "total_chunks": total_chunks,
                    },
                ))

        logger.info(f"TextChunker: {len(documents)} doc(s) "
            f"→ {len(all_chunks)} chunk(s)")
        return {"documents": all_chunks}
