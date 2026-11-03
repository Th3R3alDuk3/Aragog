from haystack.components.embedders import (
    OpenAIDocumentEmbedder,
    OpenAITextEmbedder,
)
from haystack.components.extractors import LLMMetadataExtractor
from haystack.components.generators.chat import OpenAIChatGenerator
from haystack.utils import Secret
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
from qdrant_client import AsyncQdrantClient
from qdrant_client.http.models import (
    PayloadSchemaType,
    TextIndexParams,
    TextIndexType,
    TokenizerType,
)

from components.chunker import TextChunker
from components.converter import ExportConverter
from config import get_settings
from schemas.enrichment import ENTITY_FIELDS, EnrichedMeta

settings = get_settings()


#-----------------------------------------------------
# Document Store
#-----------------------------------------------------


# every filtered field; Qdrant creates payload indexes with the collection only
PAYLOAD_INDEXES = [
    {"field_name": "id", "field_schema": PayloadSchemaType.KEYWORD},
    {"field_name": "meta.url", "field_schema": PayloadSchemaType.KEYWORD},
    {"field_name": "meta.hash", "field_schema": PayloadSchemaType.KEYWORD},
    {"field_name": "meta.chunk_index", "field_schema": PayloadSchemaType.INTEGER},
    {"field_name": "meta.document_facets", "field_schema": PayloadSchemaType.KEYWORD},
    {"field_name": "meta.keywords", "field_schema": PayloadSchemaType.KEYWORD},
    *({"field_name": f"meta.{field}", "field_schema": PayloadSchemaType.KEYWORD}
      for field in ENTITY_FIELDS),
    {"field_name": "meta.dates", "field_schema": PayloadSchemaType.DATETIME},
    {"field_name": "content", "field_schema": TextIndexParams(
        type=TextIndexType.TEXT,
        # `word` keeps numbers and codes, `multilingual` drops them
        tokenizer=TokenizerType.WORD,
        phrase_matching=True,
    )},
]


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
        payload_fields_to_index=PAYLOAD_INDEXES,
    )


def build_qdrant_client() -> AsyncQdrantClient:
    return AsyncQdrantClient(
        url=settings.qdrant_url,
        api_key=settings.qdrant_token,
        timeout=settings.qdrant_timeout,
    )


#-----------------------------------------------------
# Converter
#-----------------------------------------------------


def build_converter() -> ExportConverter:
    return ExportConverter(
        title_field=settings.document_title_field,
    )


#-----------------------------------------------------
# Document Chunker
#-----------------------------------------------------


def build_chunker() -> TextChunker:
    return TextChunker(
        tokenizer=settings.chunker_tokenizer,
        max_tokens=settings.chunker_max_tokens,
    )


#-----------------------------------------------------
# Enricher
#-----------------------------------------------------


_ENRICHER_PROMPT = """\
You are a document metadata extraction assistant.
The text below is one chunk excerpted from a larger document.
{% if document.meta.document_meta -%}
Metadata of the whole document: {{ document.meta.document_meta }}
{% endif -%}
Analyze the chunk and extract structured metadata.
Return only what is clearly indicated by the text.

Write ALL output fields (context, keywords, hypothetical_questions) in <<LANGUAGE>>, regardless of the
chunk's language — translate where the source text is not <<LANGUAGE>>. Keeping the metadata in one
consistent language makes keyword (BM25) search reliable.

Use the document metadata to situate the chunk (field context);
extract every other field from the chunk content itself.

Content rules for each output field:
<<FIELD_REQUIREMENTS>>

<file_content>{{ document.content }}</file_content>"""


def build_enricher() -> LLMMetadataExtractor:

    generation_kwargs = {
        "temperature": 0,
        "response_format": EnrichedMeta,
        # bounds runaway generations
        "max_completion_tokens": 8192,
    }

    # api.openai.com rejects unknown request params with 400
    if "api.openai.com" not in settings.enricher_url:
        # thinking off: vLLM reads the template switch, OpenRouter the reasoning flag
        generation_kwargs["extra_body"] = {
            "chat_template_kwargs": {"enable_thinking": False},
            "reasoning": {"enabled": False},
        }

    # vLLM-class backends never show schema descriptions to the model
    field_requirements = "\n".join(
        f"- {name}: {field.description}"
        for name, field in EnrichedMeta.model_fields.items())
    prompt = (_ENRICHER_PROMPT
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
        # default passes failed chunks on without a vector
        raise_on_failure=True,
    )


def build_dense_text_embedder() -> OpenAITextEmbedder:
    return OpenAITextEmbedder(
        api_base_url=settings.dense_embedding_url,
        api_key=Secret.from_token(settings.dense_embedding_token),
        model=settings.dense_embedding_model,
        timeout=settings.dense_embedding_timeout,
        # the agent retries, not the server
        max_retries=0,
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
# Retrievers (Dense + Sparse)
#-----------------------------------------------------


def build_dense_embedding_retriever(
    document_store: QdrantDocumentStore,
) -> QdrantEmbeddingRetriever:
    return QdrantEmbeddingRetriever(
        document_store=document_store,
        top_k=settings.search_top_k_before,
    )


def build_sparse_embedding_retriever(
    document_store: QdrantDocumentStore,
) -> QdrantSparseEmbeddingRetriever:
    return QdrantSparseEmbeddingRetriever(
        document_store=document_store,
        top_k=settings.search_top_k_before,
    )


#-----------------------------------------------------
# Reranker
#-----------------------------------------------------


def build_reranker() -> VLLMRanker:
    return VLLMRanker(
        api_base_url=settings.reranker_url,
        api_key=Secret.from_token(settings.reranker_token),
        model=settings.reranker_model,
        top_k=settings.search_top_k_after,
        score_threshold=settings.reranker_score_threshold,
        # retrieved through the document metadata too, so ranked with it
        meta_fields_to_embed=["document_meta"],
        http_client_kwargs={"timeout": settings.reranker_timeout},
    )
