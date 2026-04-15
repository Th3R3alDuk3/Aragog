# Architecture — Advanced Hybrid RAG

Deep-dive into every design decision in this system.

---

## 0. Runtime layout

```text
.
├── app.py          ← Gradio UI + built-in MCP server (entry point)
└── core/
    ├── config.py          ← pydantic-settings (reads .env)
    ├── runtime.py         ← shared RagRuntime (assembly root)
    ├── models/            ← query, retrieval, indexing, task models
    ├── components/        ← Haystack components
    ├── pipelines/         ← indexing, retrieval, generation pipelines
    ├── services/          ← QueryEngine, RetrievalEngine, IndexingService
    └── storage/           ← Qdrant, MinIO, TaskStore
```

`RagRuntime` in `core/runtime.py` is the shared assembly root.  `app.py` initialises
it once (lazy singleton, `asyncio.Lock`) and uses the same `QueryEngine`,
`RetrievalEngine`, and `IndexingService` for all UI and MCP requests.

Indexing runs in-process: `IndexingService.enqueue()` stores the command and
`asyncio.create_task()` starts `IndexingService.run()` in Gradio's event loop.
A `BoundedTaskStore` tracks per-task progress; the streaming progress table in the
UI polls it every 0.75 s.

---

## 1. Qdrant vector database

- **Named vectors**: stores dense AND sparse vectors per document in one collection
- **Native SPLADE/BM42**: learned sparse retrieval, semantically aware
- **Payload indexing**: dedicated fast index for metadata fields → sub-millisecond filters
- **Quantization**: scalar/binary/product quantization available for RAM reduction
- **gRPC**: fast bulk indexing

Two collections: **children** (retrieval) and **parents** (LLM context).

---

## 2. Hybrid Search with Reciprocal Rank Fusion

```
Query → [Dense retriever: top-30] + [Sparse retriever: top-30] + [HyDE dense: top-30]*
                                    ↓
                              RRF fusion
                         score(d) = Σ 1 / (k + rank_i(d))
                                    ↓
                       AutoMergingRetriever (parent-context swap)
                                    ↓
                  [ColBERT pre-filter: top-20]  ← COLBERT_ENABLED
                                    ↓
                         Cross-encoder reranker (top-5)

* HyDE branch only active when use_hyde=True at query time.
```

Dense and sparse scores are not directly comparable.  RRF uses only the *rank* —
no normalisation needed, robust across different collections.

The cross-encoder reranker reads (query, document) jointly and produces an accurate
relevance score.  Applied only to the top-20 ColBERT candidates, so latency stays low.

**HyDE implementation note**: `RagRuntime` builds two retrieval pipeline instances at
startup — one plain (dense + sparse) and one with HyDE components wired in.
`RetrievalEngine._pipeline_for(use_hyde)` selects the correct variant per request.
This is required because Haystack validates all mandatory component inputs at runtime;
a single pipeline containing `hyde_generator` would fail when `use_hyde=False`.

---

## 3. Contextual Retrieval — Anthropic Paper (full implementation)

Source: [Anthropic blog, October 2024](https://www.anthropic.com/news/contextual-retrieval)

### 3a. Contextual Prefix (most impactful)

Before embedding each chunk, a 1-2 sentence context is prepended:
> "This chunk is from the Annual Report 2024 of Musterfirma GmbH, section
> 'Financial Results Q3 2024', describing the EBITDA margin development."
>
> The margin improved to 23 %.

`ChunkAnalyzer` stores `meta["original_content"]` and `meta["context_prefix"]`.
`ContextInjector` prepends the prefix **before both the sparse and dense embedders**,
so both BM42 and the dense embedding encode the contextually enriched text.
This matches the Anthropic paper recommendation ("add context to the BM25 index too").

**Input to ChunkAnalyzer**: the **full document** is passed as a `<document>` XML block
(not just the first N characters).  `DocumentAnalyzer` stores the complete text in the
ephemeral `meta["doc_content"]` field; `ParentChildSplitter` strips it from parent docs
before they reach Qdrant; `ChunkAnalyzer._apply()` strips it from children.

**Optional Anthropic prompt caching** (`ANTHROPIC_CACHING_ENABLED=true`): uses the
`anthropic` SDK with `cache_control: ephemeral` on the document block.  Chunks are
grouped by `doc_id` so the 5-min KV-cache TTL is maximally reused — cuts token cost
by ~50× on large documents.  Requires `ANTHROPIC_API_KEY` and a `claude-*` model.

**Impact (Anthropic)**: -35% retrieval failure rate (context prefix alone),
-49% with contextual prefix + BM25.

### 3b. Parent-Child Chunking

- **Child chunk** (≈200 words, `__level=2`): embedded and stored in the children
  collection for dense + sparse retrieval.
- **Parent chunk** (≈600 words, `__level=1`): stored in the parents collection.
  Fetched at query time by `AutoMergingRetriever` when enough siblings are retrieved.

`ParentChildSplitter` wraps Haystack's `HierarchicalDocumentSplitter`.
Each child carries `meta["__parent_id"]`.  `AutoMergingRetriever` swaps the child
set for the parent document automatically (threshold-based merging,
`AUTO_MERGE_THRESHOLD`).

### 3c. Heading-aware Semantic Splitting

Docling produces well-structured markdown with H1/H2/H3 headings.
`ChunkAnnotator` parses each chunk with `markdown-it-py`, carries a heading stack
across consecutive chunks, and derives `section_title` / `section_path` from the
parsed headings.

---

## 4. Multi-Question Query Handling

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
                                         single LLM call addressing both questions
```

**Heuristics before the LLM call** avoid unnecessary API calls:
- Short queries without compound connectors and without filter hints → skip LLM
- Explicit date / language / file / document-type hints → keep the LLM path enabled

**Failure resilience**: if `QueryAnalyzer` fails it returns `[original_query]`
and the system continues normally.

---

## 5. Embedding Strategy

### Dense: BAAI/bge-m3
- Multilingual (100+ languages incl. German)
- 1024 dimensions
- Runs locally via `sentence-transformers`

### Sparse: Qdrant/bm42-all-minilm-l6-v2-attentions
- Multilingual
- Runs locally via FastEmbed ONNX — no GPU needed
- Produces attention-weighted token activations over the vocabulary

### What gets embedded

Dense embedder (`SentenceTransformersDocumentEmbedder`):
```python
meta_fields_to_embed=[
    "section_title", "title", "document_type", "summary", "keywords",
]
```

Sparse embedder uses a broader set of metadata fields:
```python
meta_fields_to_embed=[
    "section_path", "section_title", "title", "document_type",
    "summary", "keywords", "ent_locations", "ent_dates",
    "ent_events", "ent_quantities", "ent_persons",
    "ent_organizations", "ent_products", "ent_laws",
]
```

`context_prefix` is injected by `ContextInjector` **before both embedders**.
Both dense and sparse vectors encode the contextually enriched text.
Lexical precision for entities, dates, and quantities is preserved because
these terms appear in both the original content and the prepended context.

---

## 6. LLM calls during indexing

### DocumentAnalyzer: one call per document

One structured JSON call on the full document text:
```
Output: doc_id, title, word_count, document_type, document_date/period, language, audience
```

### ChunkAnalyzer: one call per chunk (+ RAPTOR chunks)

One structured JSON call:
```
Input:  full document text (XML-tagged) + section path + chunk text
Output: context_prefix, summary, keywords, classification, entities (8 types)
```

Calls are parallelised up to `ANALYZER_MAX_CONCURRENCY` concurrent requests.

When `RAPTOR_ENABLED=true`, `ChunkAnalyzer` runs a **second pass** on the
RAPTOR-generated summary chunks after `RAPTOR` completes.  RAPTOR inherits
`doc_content` and `doc_beginning` from the source chunks via `_base_meta()` so
the LLM still has full document context for the second pass.

### RAPTOR: two LLM calls per document (when enabled)

```
Input:  per-section chunk summaries → one section-level summary (LLM call)
        all section summaries       → one document-level summary (LLM call)
Output: hier_summary_section + hier_summary_doc chunks, stored as children (level=2)
```

**Cost estimate (gpt-4o-mini)**:
- Input tokens scale with document size (full doc passed each call).
  For a 10 000-token document: ~10 500 input + ~180 output per chunk ≈ $0.0016
- 1000-chunk document ≈ $1.60 (OpenAI) or $0 (Ollama/local).
- With `ANTHROPIC_CACHING_ENABLED`: document tokens cached → costs drop ~50×.

### QueryAnalyzer: one call per complex query (if needed)

Combines two tasks in one LLM call:
- decomposition (`sub_questions`)
- filter hint extraction (`date_from`, `date_to`, `classification`, `document_type`, `language`, `source`)

Short/simple queries without hints skip the LLM call entirely.

---

## 7. Performance considerations

| Stage | Latency | Parallelism |
|-------|---------|-------------|
| Docling | 2-30s / file (depends on size) | per-file |
| ChunkAnalyzer (indexing) | ~1s / chunk | `ANALYZER_MAX_CONCURRENCY` concurrent requests |
| Dense embed (query) | ~50ms | — |
| Sparse embed (query) | ~20ms | — |
| Qdrant retrieval (dense + sparse) | ~5-20ms | parallel |
| Cross-encoder reranker (top-20) | ~200-500ms | — |
| LLM generation | 1-5s | — |
| **Total query latency** | **~1.5-6s** | |

The reranker is the main query latency bottleneck.

---

## 8. Status and open improvements

| Feature | Status | Notes |
|---------|--------|-------|
| Hybrid retrieval (dense + sparse + RRF) | ✅ | BAAI/bge-m3 dense + BM42 sparse → RRF fusion |
| Cross-encoder reranking | ✅ | BAAI/bge-reranker-v2-m3, local, applied to top-K RRF candidates |
| ColBERT pre-filter (before cross-encoder) | ✅ | colbert-ir/colbertv2.0 via pylate, optional (`COLBERT_ENABLED`) |
| HyDE (Hypothetical Document Embeddings) | ✅ | Second dense branch in retrieval pipeline, toggleable per request |
| RAPTOR-inspired summaries | ✅ | **Enabled by default** (`RAPTOR_ENABLED=true` in `.env`); section + doc-level summary chunks enriched by ChunkAnalyzer; no clustering tree |
| CRAG (Corrective RAG) | ✅ | Score-threshold retry loop + LLM query reformulation, toggleable per request |
| RAGAS evaluation | ✅ | Faithfulness, answer_relevancy, context_precision (`RAGAS_ENABLED`) |
| MinIO object storage | ✅ | Stores original files; `minio_key` stored in chunk metadata |
| Async task tracking | ✅ | `asyncio.create_task` + `BoundedTaskStore`; evicts finished tasks when full |
| Re-indexing lifecycle | ✅ | Deletes stale chunks by stable `meta.doc_id` before writing fresh chunks |
| Metadata filter robustness | ✅ | NL filter extraction via QueryAnalyzer; `meta.` prefix normalization |
| Named Entity Recognition | ✅ | 8 entity types (persons, orgs, locations, dates, products, laws, events, quantities) |
| Separate worker process | ❌ | Indexing runs in-process; no dedicated worker runtime or queue yet |
| Self-RAG | ❌ | LLM self-evaluation of retrieved docs before generation |
| Graph RAG | ❌ | Knowledge graph for entity-relationship queries |
| Multi-modal (image/table extraction) | ⚠️ | docling extracts tables as markdown; images → placeholder text |
| Query result caching | ❌ | No cache for repeated identical queries |
| End-to-end test coverage | ⚠️ | Integration tests with real Qdrant + LLM backend pending |
