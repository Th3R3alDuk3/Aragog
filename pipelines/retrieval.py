"""
Query pipeline — Advanced Hybrid RAG (Haystack 2.x + Qdrant)

Pipeline flow per sub-question
───────────────────────────────

  Query text
      │
      ├─ SentenceTransformersTextEmbedder ──→ dense vector
      │                                              │
      │                                   QdrantEmbeddingRetriever
      │                                              │
      └─ FastembedSparseTextEmbedder ──→ sparse vector
                                                     │
                                          QdrantSparseEmbeddingRetriever
                                                     │
                                   DocumentJoiner (Reciprocal Rank Fusion)
                                                     │
                              SentenceTransformersSimilarityRanker (cross-encoder)
                                                     │
                                         [swap_to_parent_content]
                                           replace doc.content with
                                           meta["parent_content"]
                                           for richer LLM context
                                                     │
                                           PromptBuilder
                                                     │
                                          OpenAIGenerator
                                                     │
                                           AnswerBuilder

Multi-question handling
───────────────────────
Multi-question decomposition happens at the ROUTER layer (routers/query.py),
not inside this pipeline.  The router calls QueryDecomposer, runs this
pipeline once per sub-question, merges document sets, and calls the
LLM once with the combined context addressing all sub-questions.

Parent-child retrieval
──────────────────────
Child chunks (small, precise) are stored in Qdrant and retrieved.
swap_to_parent_content() replaces each child's content with
meta["parent_content"] (full markdown section) before the LLM
generates its answer — broader context without sacrificing retrieval precision.
"""

from haystack import Pipeline
from haystack.components.builders import AnswerBuilder, PromptBuilder
from haystack.components.joiners import DocumentJoiner
from haystack_integrations.components.retrievers.qdrant import (
    QdrantEmbeddingRetriever,
    QdrantSparseEmbeddingRetriever,
)
from haystack_integrations.document_stores.qdrant import QdrantDocumentStore

from config import Settings
from pipelines._embedders import (
    build_generator,
    build_reranker,
    build_sparse_text_embedder,
    build_text_embedder,
)

# ---------------------------------------------------------------------------
# RAG prompt — multi-question aware
# ---------------------------------------------------------------------------
RAG_PROMPT = """\
You are a precise, helpful assistant. Answer the question(s) below using ONLY
the provided context documents.  If the context does not contain sufficient
information to answer a question, state that clearly instead of guessing.
Do not invent facts.

{% if questions | length > 1 %}
The user asked multiple questions. Address each one separately and clearly.

Questions:
{% for q in questions %}
  {{ loop.index }}. {{ q }}
{% endfor %}
{% else %}
Question: {{ questions[0] }}
{% endif %}

Context documents:
{% for doc in documents %}
────────────────────────────────────────
Source  : {{ doc.meta.get("source", "unknown") }}
Section : {{ doc.meta.get("section_title", "") }}
{% if doc.meta.get("summary") %}Summary : {{ doc.meta.get("summary") }}
{% endif %}
{{ doc.content }}
{% endfor %}
────────────────────────────────────────

Answer:"""


def build_query_pipeline(
    settings: Settings,
    document_store: QdrantDocumentStore,
) -> tuple[Pipeline, Pipeline]:
    """
    Returns (retrieval_pipeline, generation_pipeline).

    Keeping retrieval and generation as separate pipelines avoids Haystack's
    mandatory-input validation: PromptBuilder requires ``questions`` at runtime,
    which is only available after all sub-question retrievals are merged.
    Running one combined pipeline per sub-question would force an LLM call each
    time — defeating the single-LLM-call design.
    """
    # ── Retrieval pipeline ────────────────────────────────────────────────────
    dense_embedder   = build_text_embedder(settings)
    sparse_embedder  = build_sparse_text_embedder(settings)
    dense_retriever  = QdrantEmbeddingRetriever(
        document_store=document_store,
        top_k=settings.dense_retriever_top_k,
    )
    sparse_retriever = QdrantSparseEmbeddingRetriever(
        document_store=document_store,
        top_k=settings.sparse_retriever_top_k,
    )
    joiner = DocumentJoiner(
        join_mode="reciprocal_rank_fusion",
        top_k=max(settings.dense_retriever_top_k, settings.sparse_retriever_top_k),
    )
    reranker = build_reranker(settings)

    retrieval = Pipeline()
    retrieval.add_component("dense_embedder",   dense_embedder)
    retrieval.add_component("sparse_embedder",  sparse_embedder)
    retrieval.add_component("dense_retriever",  dense_retriever)
    retrieval.add_component("sparse_retriever", sparse_retriever)
    retrieval.add_component("joiner",           joiner)
    retrieval.add_component("reranker",         reranker)
    retrieval.connect("dense_embedder.embedding",         "dense_retriever.query_embedding")
    retrieval.connect("dense_retriever.documents",        "joiner.documents")
    retrieval.connect("sparse_embedder.sparse_embedding", "sparse_retriever.query_sparse_embedding")
    retrieval.connect("sparse_retriever.documents",       "joiner.documents")
    retrieval.connect("joiner.documents",                 "reranker.documents")

    # ── Generation pipeline ───────────────────────────────────────────────────
    # documents and questions are runtime inputs — no upstream connections needed.
    prompt_builder = PromptBuilder(template=RAG_PROMPT, required_variables=["questions", "documents"])
    generator      = build_generator(settings)
    answer_builder = AnswerBuilder()

    generation = Pipeline()
    generation.add_component("prompt_builder", prompt_builder)
    generation.add_component("llm",            generator)
    generation.add_component("answer_builder", answer_builder)
    generation.connect("prompt_builder.prompt", "llm.prompt")
    generation.connect("llm.replies",           "answer_builder.replies")
    generation.connect("llm.meta",              "answer_builder.meta")

    return retrieval, generation


# ---------------------------------------------------------------------------
# Parent-content swap  (applied at service layer, not inside the pipeline)
# ---------------------------------------------------------------------------

def swap_to_parent_content(documents: list) -> list:
    """
    Replace each retrieved child chunk's content with its parent section text.

    Called by the router AFTER retrieval, BEFORE passing documents to the
    PromptBuilder.  Gives the LLM full section context while keeping
    retrieval precise (small child chunks for embedding/retrieval).

    If a document has no parent_content the original content is kept.
    Deduplicates by parent content to avoid sending the same section twice.
    """
    from haystack import Document

    seen: set[str] = set()
    result: list[Document] = []

    for doc in documents:
        parent = doc.meta.get("parent_content") or doc.content
        key    = parent[:200]               # dedup key: first 200 chars
        if key in seen:
            continue
        seen.add(key)
        result.append(
            Document(content=parent, meta=doc.meta, id=doc.id, score=doc.score)
        )

    return result
