# Components

Custom Haystack components and standalone helper classes used by the ingestion and retrieval pipelines.

---

## Table of Contents

1. [DoclingConverter](#1-doclingconverter)
2. [MetadataEnricher](#2-metadataenricher)
3. [ParentChildSplitter](#3-parentchildsplitter)
4. [ChunkContextEnricher](#4-chunkcontextenricher)
5. [ContentAnalyzer](#5-contentanalyzer)
6. [QueryAnalyzer](#6-queryanalyzer)
7. [HyDEGenerator](#7-hydegenerator)
8. [RaptorSummarizer](#8-raptorsummarizer)
9. [ColBERTReranker](#9-colbertreranker)
10. [Theory: Dense Embeddings](#10-theory-dense-embeddings--baaibge-m3)
11. [Theory: Sparse Embeddings](#11-theory-sparse-embeddings--splade--bm42)
12. [Theory: Hybrid Search + RRF](#12-theory-hybrid-search--rrf)
13. [Theory: Cross-Encoder Reranking](#13-theory-cross-encoder-reranking)
14. [Theory: Contextual Retrieval](#14-theory-contextual-retrieval--anthropic-option-a)
15. [Theory: Parent-Child Chunking](#15-theory-parent-child-chunking)
16. [Theory: Markdown Splitting](#16-theory-markdown-splitting)
17. [Theory: Multi-Question Decomposition](#17-theory-multi-question-decomposition)
18. [Theory: HyDE](#18-theory-hyde)
19. [Theory: RAPTOR](#19-theory-raptor)
20. [Theory: CRAG](#20-theory-crag)
21. [Theory: ColBERT](#21-theory-colbert-late-interaction)

---

## 1. DoclingConverter

**File:** `docling_converter.py` | **Type:** Haystack `@component`

Converts PDF, DOCX, PPTX, and other document formats to clean markdown via the
[docling-serve](https://github.com/DS4SD/docling-serve) Gradio API.

**Input:** `paths: list[str]` — absolute file paths
**Output:** `documents: list[Document]` — one Document per file, `content` = markdown

**API flow (two-step async):**
```
POST /process_file  →  task_id (str)
POST /wait_task_finish(task_id)  →  9-tuple, index 1 = markdown text
```

**Key settings (via `__init__`):**

| Parameter | Default | Notes |
|-----------|---------|-------|
| `docling_url` | from `.env` | Must include `/ui` suffix (e.g. `http://localhost:5001/ui`) |
| `timeout` | 300 s | Per-file wait time |
| `ocr` | True | Enable OCR for scanned PDFs |
| `force_ocr` | False | Force OCR even if text layer exists |
| `ocr_engine` | `"auto"` | `"auto"` \| `"easyocr"` \| `"tesseract"` |
| `pipeline` | `"standard"` | `"standard"` \| `"simple"` |
| `table_mode` | `"accurate"` | `"accurate"` \| `"fast"` |

Files are processed one at a time for predictable result parsing.

---

## 2. MetadataEnricher

**File:** `metadata_enricher.py` | **Type:** Haystack `@component`

Adds document-level metadata to every document before splitting.

**Input/Output:** `documents: list[Document]`

**Fields added:**

| Field | Source | Notes |
|-------|--------|-------|
| `doc_id` | SHA-256(file_path + content) | Stable re-index key |
| `title` | First H1 heading or filename stem | |
| `word_count` | `len(content.split())` | Approximate |
| `indexed_at` | `datetime.utcnow()` | ISO-8601 UTC string |
| `indexed_at_ts` | Unix epoch (int) | For date-range filters |
| `language` | `langdetect` on full doc | ISO 639-1 (e.g. `"de"`, `"en"`) |
| `doc_beginning` | First N chars of content | Passed to ContentAnalyzer for context prefix |
| `embedding_model` | from settings | e.g. `"BAAI/bge-m3"` |
| `embedding_provider` | `"huggingface"` | Fixed |
| `embedding_dimension` | from settings | e.g. `1024` |

Language detection runs on the full document (most accurate) rather than per-chunk
so the cost is paid once. `doc_beginning` is removed by ContentAnalyzer before writing
to Qdrant.

---

## 3. ParentChildSplitter

**File:** `parent_child_splitter.py` | **Type:** Haystack `@component`

Implements parent-child chunking using Haystack's built-in
`HierarchicalDocumentSplitter`. Creates a two-level hierarchy:

- **Level 1 (parents):** larger chunks (`PARENT_CHUNK_SIZE` words) — written to the
  parents Qdrant collection; fetched at query time by `AutoMergingRetriever`.
- **Level 2 (children):** smaller chunks (`CHILD_CHUNK_SIZE` words) — embedded and
  stored in the children collection for dense + sparse retrieval.

Level 0 (the full document) is discarded.

**Outputs:** `children: list[Document]`, `parents: list[Document]`

**Key metadata fields set by `HierarchicalDocumentSplitter`:**

| Field | Content |
|-------|---------|
| `__level` | 1 for parents, 2 for children |
| `__parent_id` | ID of the parent document (used by `AutoMergingRetriever`) |

**Configuration:**

| Setting | Default | Notes |
|---------|---------|-------|
| `parent_chunk_size` | 600 words | `PARENT_CHUNK_SIZE` in `.env` |
| `child_chunk_size` | 200 words | `CHILD_CHUNK_SIZE` in `.env` |
| `child_chunk_overlap` | 20 words | `CHILD_CHUNK_OVERLAP` in `.env` |

At query time, `AutoMergingRetriever` uses `__parent_id` to replace a child set
with the matching parent document when enough children from the same parent are
retrieved (threshold-based, `AUTO_MERGE_THRESHOLD`).

See [Theory: Parent-Child Chunking](#15-theory-parent-child-chunking) for rationale.

---

## 4. ChunkContextEnricher

**File:** `chunk_enricher.py` | **Type:** Haystack `@component`

Adds structural metadata to each chunk after splitting.

**Input/Output:** `documents: list[Document]`

**Fields added:**

| Field | Content | Example |
|-------|---------|---------|
| `chunk_index` | 0-based position within document | `3` |
| `chunk_total` | Total chunks from this document | `87` |
| `section_title` | Nearest heading (`meta["header"]`) | `"Finanzergebnisse Q3"` |
| `section_path` | Full heading breadcrumb | `"3 › 3.2 › Finanzergebnisse"` |
| `chunk_type` | Content type (heuristic) | `text\|table\|code\|list\|figure_caption` |

`section_path` is built from `meta["parent_headers"]` (set by `MarkdownHeaderSplitter`)
plus the current `header`. Used by RAPTOR to group chunks by section.

---

## 5. ContentAnalyzer

**File:** `content_analyzer.py` | **Type:** Haystack `@component`

Runs one LLM call per chunk (in parallel via `asyncio.gather` + `asyncio.Semaphore`) to generate:

- **`context_prefix`** — 1-2 sentence contextual preamble (Anthropic Contextual Retrieval)
- **`summary`** — 2-3 sentence abstractive summary
- **`keywords`** — 5-10 key terms
- **`classification`** — one label from `CLASSIFICATION_TAXONOMY` in `.env`
- **`entities`** — structured dict with persons, organizations, locations, dates, products, laws_and_standards, events, quantities (OntoNotes 5 / spaCy standard)

**Input/Output:** `documents: list[Document]`

The chunk's `content` is replaced with `context_prefix + "\n\n" + original_text`.
The original text is stored as `original_content` in meta (used for display/citation).
`doc_beginning` is removed from meta before writing to Qdrant (too large, only needed for this LLM call).

See [Theory: Contextual Retrieval](#14-theory-contextual-retrieval--anthropic-option-a).

---

## 6. QueryAnalyzer

**File:** `query_analyzer.py` | **Type:** Standalone class (not a Haystack component)

Replaces the old `query_decomposer.py`. Combines query decomposition and filter
extraction in a single LLM call.

```python
@dataclass
class AnalysisResult:
    sub_questions: list[str]   # always at least [original_query]
    is_compound: bool
    date_from: str | None      # ISO date "YYYY-MM-DD"
    date_to: str | None
    classification: str | None # validated against CLASSIFICATION_TAXONOMY
    language: str | None       # ISO 639-1
    source: str | None         # filename filter
```

**Heuristic pre-filters** (avoid LLM call when not needed):
- Query ≤ 10 words → single question
- No compound connectors (`und`, `and`, `sowie`, `?…?`) and no filter hints → single question, no filters

**Filter validation:**
- `classification` is discarded if not in known taxonomy (prevents hallucinated values)
- Dates must match `YYYY-MM-DD` pattern

See [Theory: Multi-Question Decomposition](#17-theory-multi-question-decomposition).

---

## 7. HyDEGenerator

**File:** `hyde_generator.py` | **Type:** Haystack `@component`

Generates a hypothetical document passage that would answer the query, then uses
that passage for dense embedding instead of the raw query text.

Wired into the retrieval pipeline as a second dense branch when `HYDE_ENABLED=true`.

**Input:** `query: str`
**Output:** `{"text": str}` — hypothetical passage (or original query on error)

**Parameters:** temperature=0.5, max_tokens=250
**Fallback:** Returns the original query on any LLM error — retrieval continues normally.

**Integration:** The HyDE dense embedder/retriever branch receives the hypothetical doc;
the sparse embedder and cross-encoder reranker always use the original query.
Enabled globally via `HYDE_ENABLED=true` in `.env` (pipeline-level, not per-request).

See [Theory: HyDE](#18-theory-hyde).

---

## 8. RaptorSummarizer

**File:** `raptor_summarizer.py` | **Type:** Haystack `@component`

Optional second-level summarization inserted after `ContentAnalyzer` and before
`DenseEmbedder` in the indexing pipeline (when `RAPTOR_ENABLED=true`).

**Input:** `documents: list[Document]` (already analyzed by ContentAnalyzer)
**Output:** `documents: list[Document]` — original docs + RAPTOR summary docs

**Algorithm:**
1. Group chunks by `doc_id` → then by `section_path`
2. **Section level:** combine `meta["summary"]` fields → LLM 3-5 sentence synthesis → `chunk_type="raptor_section"`
3. **Document level:** combine section summaries → LLM 5-8 sentence synthesis → `chunk_type="raptor_doc"`
4. Stable IDs: `sha256((doc_id + "::" + suffix).encode()).hexdigest()`

LLM calls are parallelised via `ThreadPoolExecutor` using `ANALYZER_MAX_CONCURRENCY` workers.
Sections with no usable summaries are skipped gracefully.

See [Theory: RAPTOR](#19-theory-raptor).

---

## 9. ColBERTReranker

**File:** `colbert_reranker.py` | **Type:** Haystack `@component`

Late-interaction pre-filter using ColBERT scoring (via [pylate](https://github.com/lightonai/pylate)).
Applied *before* the cross-encoder as a fast candidate reduction step:
`AutoMergingRetriever → ColBERTReranker (→ COLBERT_TOP_K) → cross-encoder reranker`

**Input:** `query: str`, `documents: list[Document]`
**Output:** `{"documents": list[Document]}` — top `COLBERT_TOP_K` docs by ColBERT score

**Model:** `colbert-ir/colbertv2.0` (~500 MB, downloaded once and cached by HuggingFace hub).
**Fallback:** Any exception → falls back to upstream order silently (never breaks the pipeline).
**Lazy loading:** Model is loaded on first `run()` call, not at startup.

See [Theory: ColBERT Late Interaction](#21-theory-colbert-late-interaction).

---

## 10. Theory: Dense Embeddings — BAAI/bge-m3

Converts text into a 1024-dimensional vector where semantically similar texts
are geometrically close (cosine similarity):

```
"Q3 Umsatz stieg um 12 %"   →  [0.12, -0.04, 0.77, …]  (1024 numbers)
"Revenue grew 12 % in Q3"   →  [0.11, -0.03, 0.76, …]  (very similar!)
```

**Why bge-m3:**
- Multilingual (100+ languages including German)
- Long context: up to 8192 tokens
- Top-ranked on MTEB multilingual retrieval benchmark
- Local inference via `sentence-transformers` — no API needed

**Bi-encoder architecture:** Query and document are encoded independently.
Document vectors are pre-computed at index time; only the query is encoded at query time.
This makes retrieval O(1) per document (dot product against pre-stored vectors).

---

## 11. Theory: Sparse Embeddings — SPLADE & BM42

Sparse vectors have most values at zero — only ~50-200 non-zero entries per vocabulary of ~30,000 tokens:

```
"Umsatz Q3 2024"  →  { "umsatz": 0.82, "q3": 0.71, "2024": 0.65,
                        "quartal": 0.48,   ← model inferred synonym!
                        "erlöse": 0.31,    ← and this!
                        ... (29,990 others = 0.0) }
```

**SPLADE:** Neural model that uses implicit query/document expansion — activates
semantically related tokens beyond exact matches. A query for "Auto" finds documents
containing "KFZ" or "Fahrzeug" without any synonym list.

**BM42 (used here):** Qdrant's multilingual approach combining BM25-style term
frequency with transformer attention weights. Runs via FastEmbed (ONNX, very fast on CPU).
Better than English-only SPLADE for multilingual corpora.

**SPLADE vs BM25:**

| Scenario | BM25 | SPLADE/BM42 |
|----------|------|------------|
| Exact keyword match | ✅ | ✅ |
| Synonym "KFZ" ↔ "Auto" | ❌ | ✅ |
| Abbreviation "Q3" ↔ "drittes Quartal" | ❌ | ✅ |
| Morphology "kaufen" ↔ "Kauf" | partial | ✅ |
| Domain terms "EBITDA" ↔ "Betriebsergebnis" | ❌ | ✅ |

---

## 12. Theory: Hybrid Search + RRF

Dense retrieval excels at semantic similarity; sparse retrieval excels at lexical precision.
Hybrid combines both signals.

**Reciprocal Rank Fusion (RRF)** avoids the problem of incomparable scores
(cosine 0-1 vs. SPLADE dot product 0-∞) by using only document *rank*:

```python
rrf_score(d) = Σᵢ  1 / (k + rankᵢ(d))   # k = 60 (constant)
```

Documents appearing in both dense and sparse results are boosted. RRF needs no tuning
and works consistently across collections.

---

## 13. Theory: Cross-Encoder Reranking

Bi-encoders encode query and document independently — fast but less accurate.
A cross-encoder jointly encodes (query, document) concatenated:

```
[CLS] query [SEP] document [SEP]  →  Transformer  →  relevance score
```

Cross-encoders are 10-20% more accurate on retrieval benchmarks because they can
model fine-grained query-document interactions, negation, and long-range dependencies.

**Strategy:** Apply the cross-encoder only to the top-20 candidates from RRF.
This gives cross-encoder accuracy with manageable latency (~200-500ms for 20 docs).

**Model:** `BAAI/bge-reranker-v2-m3` — multilingual, state of the art on BEIR benchmark,
runs locally via `sentence-transformers`.

---

## 14. Theory: Contextual Retrieval — Anthropic Option A

Source: [Anthropic, October 2024](https://www.anthropic.com/news/contextual-retrieval)

**Problem:** Chunks lose context when a document is split.
`"The margin improved to 23 %."` — which company? which margin? which year?

**Solution:** Prepend an LLM-generated 1-2 sentence context before embedding:

```
CONTEXT: This chunk is from the Annual Report 2024 of Musterfirma GmbH,
section "Financial Results Q3 2024", discussing the EBITDA margin.

The margin improved to 23 %.
```

The combined text is embedded; the original chunk is stored as `original_content` for display.
**Measured result: -35% retrieval failure rate** (Anthropic).

**Cost:** One LLM call per chunk at indexing time — in this system, the context prefix is
generated in the same call as summary, keywords, classification, and entities, so the
marginal cost is zero.

---

## 15. Theory: Parent-Child Chunking

**The dilemma:**
- Small chunks (≈150 words): precise embedding, high retrieval accuracy
- Large chunks (≈800 words): richer context for the LLM

**Solution:** Index small, answer with large.

```
Parent section (800 words):
  "## Financial Results Q3 2024
   Revenue grew 12 % to € 4.2 billion. EBITDA margin improved to 23 %. ..."

Child chunks (≈200 words each):
  Child 1: "Revenue grew 12 % to € 4.2 billion. Operating expenses..."
  Child 2: "The EBITDA margin improved to 23 %. Customer acquisition costs..."
```

- **Indexing:** Embed child chunks (precise retrieval signal). Parents written to a separate Qdrant collection linked by `__parent_id`.
- **Querying:** Retrieve child chunks. `AutoMergingRetriever` replaces a child set with the parent document when enough siblings are retrieved (threshold-based).
- **Deduplication:** Multiple children from the same parent → `AutoMergingRetriever` sends the parent only once.

---

## 16. Theory: Chunk Splitting Strategy

**`ParentChildSplitter` (wraps Haystack `HierarchicalDocumentSplitter`):**
Splits documents into a two-level word-based hierarchy:
- Level 1 (parents): ~`PARENT_CHUNK_SIZE` words
- Level 2 (children): ~`CHILD_CHUNK_SIZE` words, with `CHILD_CHUNK_OVERLAP` overlap

`HierarchicalDocumentSplitter` sets `meta["__level"]`, `meta["__parent_id"]`,
`meta["__header"]`, and `meta["__parent_headers"]` per chunk.
`ChunkEnricher` reads these to populate `section_title` and `section_path`.

**Why not embedding-similarity splitting?** True semantic splitting requires embedding
every sentence during indexing (100s of ms per section). For well-structured markdown
from Docling, heading-based splitting achieves equal semantic coherence at zero extra cost.

---

## 17. Theory: Multi-Question Decomposition

**Problem:** A compound query produces a blended embedding that is mediocre for
all sub-questions instead of optimal for each:

```
"What are the Q3 revenue figures and how does headcount compare to 2023?"
→ embedding ≈ average of two different information needs → diluted retrieval
```

**Solution:** Decompose → retrieve independently → merge → single LLM call

```
QueryAnalyzer detects compound query
→ ["What are the Q3 revenue figures?", "How does headcount compare to 2023?"]

Retrieval 1 → Q3 revenue chunks
Retrieval 2 → headcount chunks

Merge + deduplicate → best evidence for both questions
One LLM call → coherent answer addressing both
```

**Heuristics** avoid the LLM call for simple queries:
- Query ≤ 10 words → always single
- No compound connectors (`und`, `and`, `sowie`) → single

Only ~10-20% of real queries need decomposition.

---

## 18. Theory: HyDE

**Hypothetical Document Embedding** (Gao et al., 2022)

**Problem:** The query "What were the Q3 results?" is phrased as a question.
Documents contain answers written declaratively: "In Q3, revenue grew by...".
These phrasings have different embeddings even if semantically aligned.

**Solution:** Ask the LLM to write a hypothetical document that would answer the query.
Embed that document instead of the raw query for dense retrieval.

```
Query:  "What were the Q3 results?"
HyDE:   "In Q3 2024, the company achieved revenue of 4.2 billion EUR,
         representing a 12 % increase. EBITDA margin improved to 23 %."
→ Embedded text now matches document style → better dense recall
```

The sparse embedder and reranker always use the original query (they are lexical/scoring,
not embedding-based, so they need the real question).

**Enable:** `HYDE_ENABLED=true` in `.env` (global) or `"use_hyde": true` per-request.

---

## 19. Theory: RAPTOR

**Recursive Abstractive Processing for Tree-Organized Retrieval** (Sarthi et al., 2024)

**Problem:** Individual chunks answer specific questions well. But for overview or
summary questions ("Give me a high-level summary of the annual report"), no single chunk
contains the answer — it is spread across many chunks.

**Solution:** Create a hierarchy of summaries at indexing time:

```
Level 0: individual chunks (retrieved for specific questions)
Level 1: section summaries  → raptor_section chunks (1 per heading section)
Level 2: document summaries → raptor_doc chunks     (1 per document)
```

All levels are embedded and stored in Qdrant. Dense retrieval naturally finds the right
level: specific queries match level-0 chunks; overview queries match higher levels.

**Implementation here (simplified RAPTOR):** Uses existing MarkdownHeaderSplitter
section structure instead of embedding-based clustering — zero extra cost for
well-structured markdown documents.

**Enable:** `RAPTOR_ENABLED=true` in `.env`.

---

## 20. Theory: CRAG

**Corrective Retrieval Augmented Generation** (Yan et al., 2024)

**Problem:** Standard RAG retrieves regardless of relevance. If the top result has a
low similarity score, the LLM will hallucinate rather than admit it doesn't know.

**Solution:** Check retrieval confidence → if low, reformulate the query and retry.

```
Retrieval → top score < CRAG_SCORE_THRESHOLD?
  → Yes: _reformulate_query() via LLM → retry (up to CRAG_MAX_RETRIES times)
  → No:  proceed normally

If still low after max retries: set low_confidence=true in response
```

**Implementation (router layer, `routers/query.py`):** Uses the cross-encoder reranker
score as the confidence proxy — no extra LLM call to grade documents. Query reformulation
is one LLM call with prompt: "Rephrase the following question to improve document retrieval."

**Enable:** `CRAG_ENABLED=true` in `.env`.
**Settings:** `CRAG_SCORE_THRESHOLD=0.3`, `CRAG_MAX_RETRIES=2`.

---

## 21. Theory: ColBERT Late Interaction

**ColBERT** (Khattab & Zaharia, 2020)

Unlike bi-encoders (one vector per text), ColBERT encodes every token into a separate
vector. The relevance score uses *MaxSim* aggregation:

```
For each query token qᵢ: find the document token with maximum similarity
Score = Σᵢ max_j( sim(qᵢ, dⱼ) )
```

This catches precise term-level matching that a single dense vector misses, while
remaining more efficient than a full cross-encoder (no joint encoding).

**Position in pipeline:** Applied as a *pre-filter before* the
`SentenceTransformersSimilarityRanker` (cross-encoder).
Pipeline order: `AutoMergingRetriever → ColBERTReranker (→ top-K) → cross-encoder`.
No Qdrant index changes required — scores are computed on-the-fly.

**Model:** `colbert-ir/colbertv2.0` (~500 MB, cached by HuggingFace hub).

**Enable:** `COLBERT_ENABLED=true` in `.env`.
**Setting:** `COLBERT_TOP_K=5`, `COLBERT_MODEL=colbert-ir/colbertv2.0`.

---

## Further Reading

| Topic | Reference |
|-------|-----------|
| Contextual Retrieval | [Anthropic blog, Oct 2024](https://www.anthropic.com/news/contextual-retrieval) |
| SPLADE | [Formal et al. 2021 — arXiv:2109.10086](https://arxiv.org/abs/2109.10086) |
| BM42 | [Qdrant blog](https://qdrant.tech/articles/bm42/) |
| RRF | [Cormack et al. 2009](https://dl.acm.org/doi/10.1145/1571941.1572114) |
| HyDE | [Gao et al. 2022 — arXiv:2212.10496](https://arxiv.org/abs/2212.10496) |
| RAPTOR | [Sarthi et al. 2024 — arXiv:2401.18059](https://arxiv.org/abs/2401.18059) |
| CRAG | [Yan et al. 2024 — arXiv:2401.15884](https://arxiv.org/abs/2401.15884) |
| ColBERT | [Khattab & Zaharia 2020 — arXiv:2004.12832](https://arxiv.org/abs/2004.12832) |
| BAAI/bge-m3 | [HuggingFace model card](https://huggingface.co/BAAI/bge-m3) |
| bge-reranker-v2-m3 | [HuggingFace model card](https://huggingface.co/BAAI/bge-reranker-v2-m3) |
| Haystack 2.x | [docs.haystack.deepset.ai](https://docs.haystack.deepset.ai) |
| pylate (ColBERT) | [github.com/lightonai/pylate](https://github.com/lightonai/pylate) |
| FastEmbed | [github.com/qdrant/fastembed](https://github.com/qdrant/fastembed) |
| HNSW algorithm | [Malkov & Yashunin 2016 — arXiv:1603.09320](https://arxiv.org/abs/1603.09320) |
