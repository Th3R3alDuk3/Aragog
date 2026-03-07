# Pipelines

Haystack 2.x pipeline definitions for indexing and retrieval.

---

## Files

| File | Purpose |
|------|---------|
| `indexing.py` | Indexing pipeline (11 stages + optional RAPTOR): document conversion → Qdrant write |
| `retrieval.py` | Hybrid retrieval pipeline: dense + sparse + HyDE → RRF → AutoMerge → ColBERT → cross-encoder |
| `generation.py` | Generation pipeline: PromptBuilder → OpenAIGenerator → AnswerBuilder |
| `_factories.py` | Factory functions for embedder/reranker/generator components |

---

## Indexing Pipeline

Defined in `build_indexing_pipeline(settings)`.
Returns `(pipeline, children_store, parents_store)`.

Two separate Qdrant collections are used:
- `children_store` — dense + sparse vectors, retrieval target
- `parents_store` — no vectors, fetched by `__parent_id` via `AutoMergingRetriever`

### Stage-by-stage flow

```
PDF / DOCX / PPTX
      │
  1. DoclingConverter
      │   converts via docling-serve Gradio API → clean markdown
      │   one Document per file
      │
  2. MetadataEnricher
      │   doc_id (SHA-256), title, word_count
      │   indexed_at / indexed_at_ts (for date filters)
      │   language (langdetect on full doc)
      │   doc_beginning (first N chars for ContentAnalyzer LLM context)
      │   embedding provenance fields
      │
  3. DocumentCleaner  [Haystack built-in]
      │   normalise whitespace, remove empty lines
      │
  4. ParentChildSplitter  ← wraps HierarchicalDocumentSplitter
      │   level 1 (parent chunks, ~PARENT_CHUNK_SIZE words) → parents branch
      │   level 2 (child chunks, ~CHILD_CHUNK_SIZE words)  → children branch
      │   children carry meta["__parent_id"] used by AutoMergingRetriever
      │
      ├─────────────────────────────────────────────────────────
      │  PARENTS BRANCH
      │
      │  P1. DocumentWriter → parents_store (no embedding, fetched by __parent_id)
      │
      ├─────────────────────────────────────────────────────────
      │  CHILDREN BRANCH
      │
  5. ChunkContextEnricher
      │   chunk_index, chunk_total
      │   section_title, section_path (heading breadcrumb from __header/__parent_headers)
      │   chunk_type (text | table | code | list | figure_caption)
      │
  6. ContentAnalyzer  [1 async LLM call per chunk, parallelised via asyncio.gather]
      │   context_prefix → prepended before embedding (Anthropic Contextual Retrieval)
      │   summary, keywords, classification, entities (persons, orgs, locations, …)
      │   original_content stored in meta (for display / citation)
      │   doc_beginning removed from meta (not written to Qdrant)
      │
  7. [RaptorSummarizer]  ← OPTIONAL: RAPTOR_ENABLED=true
      │   groups chunks by doc_id → section_path
      │   LLM synthesises section summary → raptor_section chunk
      │   LLM synthesises document summary → raptor_doc chunk
      │   all levels embedded + stored → enables high-level retrieval
      │
  8. SentenceTransformersDocumentEmbedder  [HuggingFace, local]
      │   embeds: context_prefix prepended to chunk_content (by ContentAnalyzer)
      │   plus meta_fields_to_embed: section_title, title, summary, keywords, entities
      │   1024-dimensional normalised dense vector (BAAI/bge-m3)
      │
  9. FastembedSparseDocumentEmbedder  [local ONNX]
      │   sparse vector over vocabulary (BM42 / SPLADE)
      │   same meta_fields_to_embed as dense embedder
      │
 10. DocumentWriter
          writes children to children_store (QdrantDocumentStore)
          DuplicatePolicy.OVERWRITE (same IDs are overwritten)
```

### RAPTOR conditional wiring

```python
if raptor:
    pipeline.connect("analyzer.documents", "raptor.documents")
    pipeline.connect("raptor.documents",   "dense_embedder.documents")
else:
    pipeline.connect("analyzer.documents", "dense_embedder.documents")
```

### Qdrant collection setup

`build_children_store(settings)` — dense + sparse vectors, retrieval target:
- `use_sparse_embeddings=True` — named-vector collection (dense + sparse per doc)
- `similarity="cosine"`
- `recreate_index=False` — safe for re-indexing existing collection

`build_parents_store(settings)` — parent chunks, no vectors:
- `use_sparse_embeddings=False`
- fetched by `__parent_id` via `AutoMergingRetriever` at query time

Both stores are returned alongside the pipeline so `main.py` can pass the same instances to `build_retrieval_pipeline()`.

---

## Retrieval Pipeline

Defined in `build_retrieval_pipeline(settings, children_store, parents_store)`.
Returns one `AsyncPipeline`.

**Multi-question decomposition, CRAG, and filter building are handled in the service layer
(`services/query.py`). The pipeline handles a single sub-question at a time.**

### Flow

```
Query text (one sub-question)
      │
      ├─ SentenceTransformersTextEmbedder ──→ dense vector (1024-dim)
      │         ↓
      │   QdrantEmbeddingRetriever ──────────→ top-K dense matches
      │
      ├─ FastembedSparseTextEmbedder ──→ sparse vector
      │         ↓
      │   QdrantSparseEmbeddingRetriever ──→ top-K sparse matches
      │
      └─ [HyDE branch, HYDE_ENABLED=true]
                HyDEGenerator (LLM → hypothetical passage)
                    ↓
                SentenceTransformersTextEmbedder (hyde)
                    ↓
                QdrantEmbeddingRetriever (hyde) ──→ top-K dense matches
                         │
              DocumentJoiner (Reciprocal Rank Fusion)
                  merge without score normalisation
                  documents in multiple lists are boosted
                         │
              AutoMergingRetriever
                  if ≥ threshold fraction of a parent's children are retrieved
                  → replaces child set with the parent document (richer context)
                         │
              [ColBERTReranker — COLBERT_ENABLED=true]
                  token-level late-interaction pre-filter
                  reduces to COLBERT_TOP_K (default 20) candidates
                         │
              SentenceTransformersSimilarityRanker
                  cross-encoder (BAAI/bge-reranker-v2-m3)
                  jointly re-scores top candidates
                  → final RERANKER_TOP_K documents (default 5)
```

---

## Generation Pipeline

Defined in `build_generation_pipeline(settings)`.
Returns one `AsyncPipeline`.

```
documents + questions (runtime inputs)
       │
   PromptBuilder (RAG_PROMPT Jinja2 template)
       │
   OpenAIGenerator  (any OpenAI-compatible endpoint)
       │
   AnswerBuilder
       → answer + source documents
```

`RAG_PROMPT` is a module-level Jinja2 template string in `generation.py`.
It is imported directly by `routers/stream.py` for SSE streaming:

```python
from pipelines.generation import RAG_PROMPT
from jinja2 import Environment
prompt_text = Environment().from_string(RAG_PROMPT).render(documents=..., questions=...)
```

---

## `_factories.py` — Factory Functions

| Function | Returns | Notes |
|----------|---------|-------|
| `build_document_embedder(settings)` | `SentenceTransformersDocumentEmbedder` | Dense, for indexing |
| `build_text_embedder(settings)` | `SentenceTransformersTextEmbedder` | Dense, for retrieval |
| `build_sparse_document_embedder(settings)` | `FastembedSparseDocumentEmbedder` | Sparse, for indexing |
| `build_sparse_text_embedder(settings)` | `FastembedSparseTextEmbedder` | Sparse, for retrieval |
| `build_reranker(settings)` | `SentenceTransformersSimilarityRanker` | Cross-encoder |
| `build_generator(settings)` | `OpenAIGenerator` | OpenAI-compatible LLM |

All factories read from `Settings` (populated from `.env`) and require no arguments beyond settings.

---

## Key Configuration (`.env`)

| Variable | Default | Notes |
|----------|---------|-------|
| `EMBEDDING_MODEL` | `BAAI/bge-m3` | Dense embedding model (HuggingFace) |
| `EMBEDDING_DIMENSION` | `1024` | Vector size — must match model |
| `SPARSE_EMBEDDING_MODEL` | `Qdrant/bm42-all-minilm-l6-v2-attentions` | BM42 ONNX model |
| `RERANKER_MODEL` | `BAAI/bge-reranker-v2-m3` | Cross-encoder (HuggingFace) |
| `RERANKER_TOP_K` | `5` | Documents returned by cross-encoder |
| `DENSE_RETRIEVER_TOP_K` | `30` | Dense retriever candidates |
| `SPARSE_RETRIEVER_TOP_K` | `30` | Sparse retriever candidates |
| `PARENT_CHUNK_SIZE` | `600` | Words per parent chunk |
| `CHILD_CHUNK_SIZE` | `200` | Words per child chunk |
| `CHILD_CHUNK_OVERLAP` | `20` | Word overlap between child chunks |
| `ANALYZER_MAX_CONCURRENCY` | `8` | Parallel ContentAnalyzer async tasks |
| `AUTO_MERGE_THRESHOLD` | `0.5` | AutoMergingRetriever threshold |
| `HYDE_ENABLED` | `true` | Enable HyDE second dense branch |
| `COLBERT_ENABLED` | `true` | Enable ColBERT pre-filter |
| `COLBERT_TOP_K` | `20` | Candidates after ColBERT pre-filter |
| `RAPTOR_ENABLED` | `true` | Enable RAPTOR summary chunks |
| `QDRANT_CHILDREN_COLLECTION` | `children` | Children Qdrant collection name |
| `QDRANT_PARENTS_COLLECTION` | `parants` | Parents Qdrant collection name |
