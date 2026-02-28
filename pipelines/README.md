# Pipelines

Haystack 2.x pipeline definitions for indexing and querying.

---

## Files

| File | Purpose |
|------|---------|
| `ingestion_pipeline.py` | Indexing pipeline (10 core stages + optional RAPTOR): document conversion → Qdrant write |
| `retrieval_pipeline.py` | Query pipeline builders: retrieval DAG (dual retrieval → RRF → reranker) + generation DAG (prompt → LLM → answer) |
| `_embedders.py` | Factory functions for embedder/reranker/generator components |

---

## Indexing Pipeline

Defined in `build_indexing_pipeline(settings)`. Returns `(Pipeline, QdrantDocumentStore)`.

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
      │   doc_beginning (first N chars for context LLM)
      │   embedding provenance fields
      │
  3. DocumentCleaner  [Haystack built-in]
      │   normalise whitespace, remove empty lines
      │
  4. MarkdownHeaderSplitter  [Haystack built-in]
      │   split at H1-H6 heading boundaries
      │   each section = one semantically coherent topic
      │   meta["header"] and meta["parent_headers"] set per section
      │
  5. ParentChildSplitter
      │   RecursiveDocumentSplitter (sentence-boundary) → child chunks (200 words)
      │   stores full section as meta["parent_content"] in each child
      │   enables: retrieve small (precise), answer with large (rich context)
      │
  6. ChunkContextEnricher
      │   chunk_index, chunk_total
      │   section_title, section_path (heading breadcrumb)
      │   chunk_type (text | table | code | list | figure_caption)
      │
  7. ContentAnalyzer  [1 LLM call per chunk, parallelised]
      │   context_prefix → prepended before embedding (Contextual Retrieval)
      │   summary, keywords, classification, entities
      │   original_content stored in meta (for display)
      │   doc_beginning removed from meta (not written to Qdrant)
      │
  8. [RaptorSummarizer]  ← OPTIONAL: RAPTOR_ENABLED=true
      │   groups chunks by doc_id → section_path
      │   LLM synthesises section summary → raptor_section chunk
      │   LLM synthesises document summary → raptor_doc chunk
      │   all levels embedded + stored → enables high-level retrieval
      │
  9. SentenceTransformersDocumentEmbedder  [HuggingFace, local]
      │   embeds: context_prefix + section_title + chunk_content
      │   1024-dimensional normalised dense vector (BAAI/bge-m3)
      │
 10. FastembedSparseDocumentEmbedder  [local ONNX]
      │   sparse vector over vocabulary (BM42 / SPLADE)
      │
 11. DocumentWriter
          writes to QdrantDocumentStore
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

`build_document_store(settings)` creates a `QdrantDocumentStore` with:
- `use_sparse_embeddings=True` — named-vector collection (dense + sparse per doc)
- `similarity="cosine"`
- `recreate_index=False` — safe for re-indexing existing collection

The document store is returned alongside the pipeline so `main.py` can pass the
same instance to `build_query_pipeline()` — no duplicate Qdrant connections.

---

## Query Pipeline

Defined in `build_query_pipeline(settings, document_store)`.
Returns `(retrieval_pipeline, generation_pipeline)`.

**Note:** Multi-question decomposition, HyDE, CRAG, and ColBERT reranking are handled
at the **router layer** (`routers/query.py`), not inside this pipeline. The pipeline
handles a single sub-question at a time.

### Flow

```
Retrieval pipeline (one sub-question)
      │
      ├─ SentenceTransformersTextEmbedder ──→ dense vector (1024-dim)
      │         ↓
      │   QdrantEmbeddingRetriever ──────────→ top-K dense matches
      │
      └─ FastembedSparseTextEmbedder ──→ sparse vector
                ↓
        QdrantSparseEmbeddingRetriever ──→ top-K sparse matches
                         │
                DocumentJoiner (RRF)
                    merge without score normalisation
                    documents in both lists are boosted
                         │
          SentenceTransformersSimilarityRanker
                    cross-encoder (BAAI/bge-reranker-v2-m3)
                    re-scores top candidates jointly
                         │
                    retrieved documents

Router layer
  - merge/deduplicate across sub-questions
  - swap_to_parent_content()
  - optional ColBERT rerank
  - final context cut to top_k

Generation pipeline
  documents + questions
       │
   PromptBuilder (Jinja2)
       │
   OpenAIGenerator
       │
   AnswerBuilder
       → answer + source documents
```

### `swap_to_parent_content(documents)`

Utility function applied at the router layer (not inside the pipeline):

```python
from pipelines.retrieval_pipeline import swap_to_parent_content

merged_docs = swap_to_parent_content(list(all_docs_by_id.values()))
```

Replaces each retrieved child chunk's `content` with `meta["parent_content"]`
(the full section markdown). Deduplicates by first 200 chars of parent content
so the same section is only sent to the LLM once.

### RAG Prompt

`RAG_PROMPT` is a module-level Jinja2 template string in `retrieval_pipeline.py`.
It is imported directly by `routers/stream.py` for SSE streaming:

```python
from pipelines.retrieval_pipeline import RAG_PROMPT
from jinja2 import Environment
prompt_text = Environment().from_string(RAG_PROMPT).render(documents=..., questions=...)
```

---

## `_embedders.py` — Factory Functions

| Function | Returns | Notes |
|----------|---------|-------|
| `build_document_embedder(settings)` | `SentenceTransformersDocumentEmbedder` | Dense, for indexing |
| `build_text_embedder(settings)` | `SentenceTransformersTextEmbedder` | Dense, for querying |
| `build_sparse_document_embedder(settings)` | `FastembedSparseDocumentEmbedder` | Sparse, for indexing |
| `build_sparse_text_embedder(settings)` | `FastembedSparseTextEmbedder` | Sparse, for querying |
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
| `RERANKER_TOP_K` | `5` | Documents returned by reranker |
| `DENSE_RETRIEVER_TOP_K` | `20` | Dense retriever candidates |
| `SPARSE_RETRIEVER_TOP_K` | `20` | Sparse retriever candidates |
| `CHILD_CHUNK_SIZE` | `200` | Words per child chunk |
| `CHILD_CHUNK_OVERLAP` | `20` | Word overlap between child chunks |
| `ANALYZER_LLM_MODEL` | `gpt-4o-mini` | Model for indexing LLM calls |
| `ANALYZER_MAX_WORKERS` | `4` | Parallel ContentAnalyzer threads |
| `RAPTOR_ENABLED` | `false` | Enable RAPTOR summary chunks |
| `QDRANT_URL` | `http://localhost:6333` | Qdrant server URL |
| `QDRANT_COLLECTION` | `documents` | Collection name |
