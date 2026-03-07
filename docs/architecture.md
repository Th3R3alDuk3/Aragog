# Architecture — Advanced Hybrid RAG

Deep-dive into every design decision in this system.

---

## 1. Why Qdrant instead of pgvector

### pgvector limitations
- Dense vector search: ✅ (HNSW)
- Keyword search: ⚠️ PostgreSQL `tsvector` (BM25 — statistical, no semantics)
- Sparse vector (SPLADE): ❌ not supported natively
- Payload index for fast metadata filters: ⚠️ relies on standard PG indexes

### Qdrant advantages
- **Named vectors**: stores dense AND sparse vectors per document in one collection
- **Native SPLADE/BM42**: true learned sparse retrieval, semantically aware
- **Payload indexing**: dedicated fast index for metadata fields → sub-millisecond filters
- **Quantization**: 75% RAM reduction with minimal quality loss (scalar/binary/product)
- **gRPC**: up to 10× faster for bulk indexing vs HTTP
- **No separate BM25 engine** needed: one DB for everything

### SPLADE vs BM25

BM25 is a frequency-based term model:
```
score(q, d) = Σ IDF(t) · TF(t,d) · (k+1) / (TF(t,d) + k·(1 - b + b·|d|/avgdl))
```
It does NOT understand that "Auto" and "KFZ" are synonyms.

SPLADE (learned sparse) maps text to a sparse bag-of-weighted-tokens over the full
vocabulary.  The model learns to **expand** both queries and documents with related terms:
- Query "Umsatz Q3" → also activates tokens for "Erlöse", "Quartal", "Ergebnis"
- Combines frequency signal with semantic knowledge from a transformer encoder

Result: SPLADE outperforms BM25 by 5-15% on standard retrieval benchmarks.

For multilingual documents we use **BM42** (`Qdrant/bm42-all-minilm-l6-v2-attentions`),
Qdrant's own multilingual sparse model, instead of English-only SPLADE.

---

## 2. Hybrid Search with Reciprocal Rank Fusion

```
Query → [Dense retriever: top-30] + [Sparse retriever: top-30] + [HyDE dense: top-30]
                                    ↓
                              RRF fusion
                         score(d) = Σ 1 / (k + rank_i(d))
                                    ↓
                       AutoMergingRetriever (parent-context swap)
                                    ↓
                  [ColBERT pre-filter: top-20]  ← COLBERT_ENABLED
                                    ↓
                         Cross-encoder reranker (top-5)
```

**Why RRF?**
Dense and sparse scores are not comparable (different scales, different units).
Normalising them requires tuning a weighting hyperparameter α, which is
document-collection-specific.  RRF uses only the *rank* — no normalisation needed,
robust across different collections, consistently better than weighted sum in practice.

**Why a cross-encoder reranker?**
The bi-encoder (SentenceTransformers) computes query and document embeddings
independently — fast but less accurate.  A cross-encoder reads (query, document)
jointly and produces a relevance score.  Much more accurate, but O(n) forward passes.
We apply it only to the top-20 candidates from RRF, so latency stays low.

---

## 3. Contextual Retrieval — Anthropic Option A

Source: [Anthropic blog, October 2024](https://www.anthropic.com/news/contextual-retrieval)

Option A implements three complementary techniques:

### 3a. Contextual Prefix (most impactful)

The core problem: a chunk like
> "The margin improved to 23 %."

is semantically ambiguous without context.  Which company? Which period?

**Solution**: before embedding each chunk, prepend a 1-2 sentence context:
> "This chunk is from the Annual Report 2024 of Musterfirma GmbH, section
> 'Financial Results Q3 2024', describing the EBITDA margin development."
>
> The margin improved to 23 %.

This is done by `ContentAnalyzer` in a single LLM call per chunk.
The `context_prefix` is prepended to `doc.content` before embedding.
The original text is preserved in `meta["original_content"]` for display.

**Impact (Anthropic)**: -35% retrieval failure rate.  Largest single improvement.

### 3b. Parent-Child Chunking

**Problem**: small chunks → precise retrieval.  Large chunks → rich LLM context.
These goals conflict.

**Solution**: store both, retrieve small, answer with large.
- **Child chunk** (≈200 words, `__level=2`): embedded and stored in the children
  collection for dense + sparse retrieval.  Precise retrieval signal.
- **Parent chunk** (≈600 words, `__level=1`): stored in the parents collection.
  Fetched at query time by `AutoMergingRetriever` when enough siblings are retrieved.

**Implementation**: `ParentChildSplitter` wraps Haystack's `HierarchicalDocumentSplitter`.
Each child carries `meta["__parent_id"]`.  `AutoMergingRetriever` in the retrieval
pipeline uses this to swap the child set for the parent document automatically
(threshold-based merging, `AUTO_MERGE_THRESHOLD`).  Both parent and child documents
are written to separate Qdrant collections at indexing time.

### 3c. Heading-aware Semantic Splitting

Docling produces well-structured markdown with H1/H2/H3 headings.
`HierarchicalDocumentSplitter` (Haystack built-in, used inside `ParentChildSplitter`)
splits documents into word-based chunks and preserves heading structure in
`meta["__header"]` and `meta["__parent_headers"]` per chunk.  `ChunkEnricher` reads
these to populate `section_title` and `section_path` on every child chunk.

---

## 4. Multi-Question Query Handling

### The problem
User sends: *"What are the Q3 revenue figures and how does headcount compare to 2023?"*

A single embedding for this composite query blends two semantic intents.
The retriever optimises for the blend, often missing one of the questions
(whichever is less represented in the top-k results).

### The solution

```
User query
    │
QueryAnalyzer (LLM)
    │ detects independent information needs
    │ and extracts metadata filter hints
    │
    ├─ "What are the Q3 revenue figures?"        → retrieval pipeline → docs_1
    └─ "How does headcount compare to 2023?"     → retrieval pipeline → docs_2
                                                         │
                                              merge + deduplicate
                                                         │
                                   (AutoMergingRetriever handles parent-context swap
                                    inside the retrieval pipeline per sub-question)
                                                         │
                                         single LLM call addressing both questions
```

**Heuristics before the LLM call** avoid unnecessary API calls:
- Queries ≤ 10 words → never compound → skip LLM
- No compound connector words (`und`, `and`, `sowie`, `?…?`) → skip LLM

**Failure resilience**: if `QueryAnalyzer` fails it returns `[original_query]`
and the system continues normally — no request fails due to decomposition errors.

---

## 5. Embedding Strategy

### Dense: BAAI/bge-m3
- Multilingual (100+ languages incl. German)
- 1024 dimensions
- State of the art on MTEB multilingual retrieval benchmark
- Runs locally via `sentence-transformers`

### Sparse: Qdrant/bm42-all-minilm-l6-v2-attentions
- Multilingual (English SPLADE alternatives are English-only)
- Runs locally via FastEmbed ONNX — no GPU needed, very fast
- Produces attention-weighted token activations over the vocabulary

### What gets embedded
The `SentenceTransformersDocumentEmbedder` is configured with:
```python
meta_fields_to_embed=["section_title", "context_prefix"]
```
The embedded text = `section_title + context_prefix + doc.content`

This means the vector carries both the structural location AND the contextual meaning
of the chunk — not just its raw text.

---

## 6. LLM calls during indexing

### ContentAnalyzer: one call per chunk

One structured JSON call generates all metadata in a single round-trip:

```
Input:  document title + beginning + section path + chunk text
Output: context_prefix, summary, keywords, classification, entities
```

Note: `language` is detected locally by `langdetect` in `MetadataEnricher` on
the full document (before splitting) — no LLM call needed and more accurate than
per-chunk detection.

**Cost estimate (gpt-4o-mini)**:
- ~480 input tokens per chunk (context + chunk; no language field saves ~20 tokens)
- ~180 output tokens
- ≈ $0.00013 per chunk
- 1000-chunk document ≈ $0.13

Use a local model (Ollama) to reduce cost to zero.

### QueryAnalyzer: one call per complex query (if needed)

`QueryAnalyzer` combines two tasks in one LLM call:
- decomposition (`sub_questions`)
- filter hint extraction (`date_from`, `date_to`, `classification`, `language`, `source` as exact filename)

The analyzer has a fast path (short/simple query with no hints) and skips the
LLM in those cases.

---

## 7. Performance considerations

| Stage | Latency | Parallelism |
|-------|---------|-------------|
| DoclingConverter | 2-30s / file (depends on size) | per-file |
| ContentAnalyzer (indexing) | ~1s / chunk | `ANALYZER_MAX_WORKERS` threads |
| Dense embed (query) | ~50ms | — |
| Sparse embed (query) | ~20ms | — |
| Qdrant retrieval (dense + sparse) | ~5-20ms | parallel |
| Cross-encoder reranker (top-20) | ~200-500ms | — |
| LLM generation | 1-5s | — |
| **Total query latency** | **~1.5-6s** | |

Reranker is the main query latency bottleneck.
Use `cross-encoder/ms-marco-MiniLM-L-6-v2` for ~3× faster reranking
at a small quality cost (English only).

---

## 8. Known Limitations and Open Improvements

| Feature | Status | Notes |
|---------|--------|-------|
| Hybrid retrieval (dense + sparse + RRF) | ✅ | BAAI/bge-m3 dense + BM42 sparse → RRF fusion |
| Cross-encoder reranking | ✅ | BAAI/bge-reranker-v2-m3, local, applied to top-K RRF candidates |
| ColBERT pre-filter (before cross-encoder) | ✅ | colbert-ir/colbertv2.0 via pylate, optional (`COLBERT_ENABLED`) |
| HyDE (Hypothetical Document Embeddings) | ✅ | Second dense branch in retrieval pipeline (`HYDE_ENABLED`) |
| RAPTOR (multi-level summaries) | ✅ | Section + doc-level summary chunks, ThreadPoolExecutor (`RAPTOR_ENABLED`) |
| CRAG (Corrective RAG) | ✅ | Score-threshold retry loop + LLM query reformulation (`CRAG_ENABLED`) |
| Streaming responses (SSE) | ✅ | `POST /query/stream` — `token`, `sources`, `done` events |
| RAGAS evaluation | ✅ | Faithfulness, answer_relevancy, context_precision (`RAGAS_ENABLED`) |
| MinIO object storage | ✅ | Stores original files; `minio_url` + `minio_key` in chunk metadata |
| Async task tracking | ✅ | HTTP 202 + GET /tasks/{task_id}; BoundedTaskStore evicts done/failed |
| Re-indexing lifecycle | ✅ | Deletes stale chunks by `meta.source` before writing fresh chunks |
| Metadata filter robustness | ✅ | NL filter extraction via QueryAnalyzer; `meta.` prefix normalization; shorthand dict support |
| Named Entity Recognition | ✅ | 8 entity types (persons, orgs, locations, dates, products, laws, events, quantities) via ContentAnalyzer |
| Self-RAG | ❌ | LLM self-evaluation of retrieved docs before generation |
| Graph RAG | ❌ | Knowledge graph for entity-relationship queries |
| Multi-modal (image/table extraction) | ⚠️ | docling extracts tables as markdown; images → placeholder text |
| SPLADE multilingual | ⚠️ | BM42 is an approximation; true multilingual SPLADE pending |
| HierarchicalDocumentSplitter + AutoMergingRetriever | ✅ | ParentChildSplitter uses HierarchicalDocumentSplitter; AutoMergingRetriever handles parent-context swap |
| Async pipeline execution | ⚠️ | Indexing components are sync; run in thread-pool executor |
| Query result caching | ❌ | No Redis/in-memory cache for repeated identical queries |
| End-to-end test coverage | ⚠️ | Integration tests with real Qdrant + LLM backend pending |
