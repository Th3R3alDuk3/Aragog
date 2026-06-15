from haystack.utils import Secret
from haystack.components.extractors import LLMMetadataExtractor
from haystack.components.generators.chat import OpenAIChatGenerator
from haystack.components.embedders import (
    OpenAIDocumentEmbedder,
    OpenAITextEmbedder,
)
from haystack_integrations.document_stores.qdrant import QdrantDocumentStore
from haystack_integrations.components.retrievers.qdrant import (
    QdrantSparseEmbeddingRetriever,
    QdrantEmbeddingRetriever,
)
from haystack_integrations.components.embedders.fastembed import (
    FastembedSparseDocumentEmbedder,
    FastembedSparseTextEmbedder,
)
from haystack_integrations.components.converters.docling_serve import DoclingServeConverter, ExportType
from haystack_integrations.components.rankers.vllm import VLLMRanker

from components.chunker import DoclingHybridChunker
from config import get_settings
from models.meta import Meta


settings = get_settings()


def _build_structured_generator(format_model) -> OpenAIChatGenerator:
    return OpenAIChatGenerator(
        api_base_url=settings.openai_url,
        api_key=Secret.from_token(settings.openai_token),
        model=settings.openai_model,
        timeout=settings.openai_timeout,
        generation_kwargs={
            "temperature": 0,
            "response_format": format_model,
            "extra_body": {
                "enable_thinking": False,
                "chat_template_kwargs": {"enable_thinking": False},
            },
        },
    )


#-----------------------------------------------------
# Document Store
#-----------------------------------------------------


def build_document_store() -> QdrantDocumentStore:
    return QdrantDocumentStore(
        url=settings.qdrant_url,
        api_key=Secret.from_token(settings.qdrant_token),
        index=settings.qdrant_collection,
        embedding_dim=settings.qdrant_embedding_dim,
        use_sparse_embeddings=True,
        sparse_idf=True,
        similarity="cosine",
        recreate_index=False,
    )


#-----------------------------------------------------
# Converter
#-----------------------------------------------------


def build_converter() -> DoclingServeConverter:
    return DoclingServeConverter(
        base_url=settings.docling_url,
        mode="async",
        export_type=ExportType.JSON,
        convert_options={
            "image_export_mode": "placeholder",
            "pipeline": "standard",
            "do_ocr": True,
            "force_ocr": False,
            "ocr_engine": "auto",
            "ocr_lang": ["en", "fr", "de", "es"],
            "pdf_backend": "docling_parse",
            "table_mode": "accurate",
            "abort_on_error": False,
            "do_code_enrichment": False,
            "do_formula_enrichment": False,
            "do_picture_classification": False,
            "do_picture_description": False,
        },
        timeout=settings.docling_timeout,
        job_timeout=settings.docling_timeout,
        poll_interval=5,
    )


#-----------------------------------------------------
# Document Chunker
#-----------------------------------------------------


def build_chunker() -> DoclingHybridChunker:
    return DoclingHybridChunker(
        tokenizer=settings.chunker_tokenizer,
        max_tokens=settings.chunker_max_tokens,
    )


#-----------------------------------------------------
# Enricher
#-----------------------------------------------------


_CHUNK_ENRICHER_PROMPT = """\
You are a document metadata extraction assistant.
The text below is one chunk excerpted from a larger document; it begins with its heading path.
Analyze the chunk and extract structured metadata.
Return only what is clearly indicated by the text.

Use the heading path to situate the chunk (field context); extract every other field from the chunk content itself.

<file_content>{{ document.content }}</file_content>"""


def build_chunk_enricher() -> LLMMetadataExtractor:
    return LLMMetadataExtractor(
        prompt=_CHUNK_ENRICHER_PROMPT,
        chat_generator=_build_structured_generator(Meta),
        max_workers=settings.enricher_max_workers,
    )


#-----------------------------------------------------
# Embedders (Dense + Sparse)
#-----------------------------------------------------


def build_dense_document_embedder() -> OpenAIDocumentEmbedder:
    return OpenAIDocumentEmbedder(
        api_base_url=settings.dense_embedding_url,
        api_key=Secret.from_token(settings.dense_embedding_token),
        model=settings.dense_embedding_model,
        meta_fields_to_embed=settings.embedded_meta_fields.split(","),
    )


def build_dense_text_embedder() -> OpenAITextEmbedder:
    return OpenAITextEmbedder(
        api_base_url=settings.dense_embedding_url,
        api_key=Secret.from_token(settings.dense_embedding_token),
        model=settings.dense_embedding_model,
    )


def build_sparse_document_embedder() -> FastembedSparseDocumentEmbedder:
    return FastembedSparseDocumentEmbedder(
        model=settings.sparse_embedding_model,
        model_kwargs={
            "language": settings.sparse_embedding_language,
            "cuda": settings.sparse_embedding_device.startswith("cuda"),
        },
        meta_fields_to_embed=settings.embedded_meta_fields.split(","),
    )


def build_sparse_text_embedder() -> FastembedSparseTextEmbedder:
    return FastembedSparseTextEmbedder(
        model=settings.sparse_embedding_model,
        model_kwargs={
            "language": settings.sparse_embedding_language,
            "cuda": settings.sparse_embedding_device.startswith("cuda"),
        },
    )


#-----------------------------------------------------
# Retriever (Dense + Sparse)
#-----------------------------------------------------


def build_dense_embedding_retriever(
    document_store: QdrantDocumentStore
) -> QdrantEmbeddingRetriever:
    return QdrantEmbeddingRetriever(
        document_store=document_store,
    )


def build_sparse_embedding_retriever(
    document_store: QdrantDocumentStore
) -> QdrantSparseEmbeddingRetriever:
    return QdrantSparseEmbeddingRetriever(
        document_store=document_store,
    )


#-----------------------------------------------------
# Reranker
#-----------------------------------------------------


def build_reranker() -> VLLMRanker:
    return VLLMRanker(
        api_base_url=settings.reranker_url,
        api_key=Secret.from_token(settings.reranker_token),
        model=settings.reranker_model,
    )
