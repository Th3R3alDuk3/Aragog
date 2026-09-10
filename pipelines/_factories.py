from haystack.components.embedders import (
    OpenAIDocumentEmbedder,
    OpenAITextEmbedder,
)
from haystack.components.extractors import LLMMetadataExtractor
from haystack.components.generators.chat import OpenAIChatGenerator
from haystack.utils import Secret
from haystack_integrations.components.converters.docling_serve import (
    DoclingServeConverter,
    ExportType,
)
from haystack_integrations.components.embedders.fastembed import (
    FastembedSparseDocumentEmbedder,
    FastembedSparseTextEmbedder,
)
from haystack_integrations.components.rankers.vllm import VLLMRanker
from haystack_integrations.components.retrievers.qdrant import (
    QdrantEmbeddingRetriever,
    QdrantSparseEmbeddingRetriever,
)
from haystack_integrations.document_stores.qdrant import QdrantDocumentStore

from components.chunker import DoclingHybridChunker
from config import get_settings
from schemas.enrichment import EnrichedMeta
from services.rustfs import RustfsStore

settings = get_settings()


#-----------------------------------------------------
# S3 Storage
#-----------------------------------------------------


def build_rustfs_store() -> RustfsStore:
    return RustfsStore(
        url=settings.rustfs_url,
        public_url=settings.rustfs_public_url,
        access_key=settings.rustfs_access_key,
        secret_key=settings.rustfs_secret_key,
        timeout=settings.rustfs_timeout,
        bucket=settings.rustfs_bucket,
        url_expire=settings.rustfs_url_expire,
    )


#-----------------------------------------------------
# Document Store
#-----------------------------------------------------


def build_document_store() -> QdrantDocumentStore:
    return QdrantDocumentStore(
        url=settings.qdrant_url,
        api_key=Secret.from_token(settings.qdrant_token),
        timeout=settings.qdrant_timeout,
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
            "do_pdf_heading_hierarchy": True,
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
The text below is one chunk excerpted from a larger document titled "{{ document.meta.source }}";
the chunk begins with its heading path within that document.
Analyze the chunk and extract structured metadata.
Return only what is clearly indicated by the text.

Write ALL output fields (context, keywords, hypothetical_questions) in <<LANGUAGE>>, regardless of the
chunk's language — translate where the source text is not <<LANGUAGE>>. Keeping the metadata in one
consistent language makes keyword (BM25) search reliable.

Use the document title and heading path to situate the chunk (field context);
extract every other field from the chunk content itself.

Content rules for each output field:
<<FIELD_REQUIREMENTS>>

<file_content>{{ document.content }}</file_content>"""


def build_chunk_enricher() -> LLMMetadataExtractor:

    generation_kwargs = {
        "temperature": 0,
        "response_format": EnrichedMeta,
        # bounds runaway generations (e.g. greedy repetition loops on local vLLM)
        "max_completion_tokens": 8192,
    }

    # api.openai.com rejects unknown request params with 400
    if "api.openai.com" not in settings.enricher_url:
        generation_kwargs["extra_body"] = {
            "chat_template_kwargs": {"enable_thinking": False},
        }

    # vLLM-class backends never show the json_schema descriptions to the model
    field_requirements = "\n".join(
        f"- {name}: {field.description}"
        for name, field in EnrichedMeta.model_fields.items())
    prompt = (_CHUNK_ENRICHER_PROMPT
        .replace("<<LANGUAGE>>", settings.enricher_language)
        .replace("<<FIELD_REQUIREMENTS>>", field_requirements))

    return LLMMetadataExtractor(
        prompt=prompt,
        chat_generator=OpenAIChatGenerator(
            api_base_url=settings.enricher_url,
            api_key=Secret.from_token(settings.enricher_token),
            model=settings.enricher_model,
            timeout=settings.enricher_timeout,
            max_retries=settings.enricher_max_retries,
            generation_kwargs=generation_kwargs,
        ),
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
        timeout=settings.dense_embedding_timeout,
        # default logs and passes chunks on without a dense vector
        raise_on_failure=True,
    )


def build_dense_text_embedder() -> OpenAITextEmbedder:
    return OpenAITextEmbedder(
        api_base_url=settings.dense_embedding_url,
        api_key=Secret.from_token(settings.dense_embedding_token),
        model=settings.dense_embedding_model,
        timeout=settings.dense_embedding_timeout,
    )


def build_sparse_document_embedder() -> FastembedSparseDocumentEmbedder:
    return FastembedSparseDocumentEmbedder(
        model=settings.sparse_embedding_model,
        meta_fields_to_embed=settings.embedded_meta_fields.split(","),
        model_kwargs={
            "language": settings.sparse_embedding_language,
            "cuda": settings.sparse_embedding_device.startswith("cuda"),
        },
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
    document_store: QdrantDocumentStore,
) -> QdrantEmbeddingRetriever:
    return QdrantEmbeddingRetriever(
        document_store=document_store,
    )


def build_sparse_embedding_retriever(
    document_store: QdrantDocumentStore,
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
        http_client_kwargs={
            "timeout": settings.reranker_timeout
        },
    )
