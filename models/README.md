# Models

Pydantic schemas for API request/response validation (`schemas.py`) and the
authoritative metadata specification for every document chunk stored in Qdrant.

---

## Table of Contents

1. [Pydantic Schemas](#1-pydantic-schemas)
2. [Chunk Metadata Schema](#2-chunk-metadata-schema)
3. [Field Reference](#3-field-reference)
4. [What is stored vs. embedded](#4-what-is-stored-vs-embedded)
5. [Filtering examples](#5-filtering-examples)

---

## 1. Pydantic Schemas

### Indexing

#### `IndexResponse`

```python
class IndexResponse(BaseModel):
    indexed: int   # number of document chunks written to Qdrant
    source: str    # original filename
```

---

### Query

#### `QueryRequest`

```python
class QueryRequest(BaseModel):
    query: str               # user question (simple or compound), min_length=1
    top_k: int = 5           # max source documents in response (1-50)
    filters: dict | None     # Haystack filter expression (see §5)
    date_from: date | None   # ISO 8601 date, translated to indexed_at_ts >=
    date_to: date | None     # ISO 8601 date, translated to indexed_at_ts <=
```

#### `SourceDocument`

```python
class SourceDocument(BaseModel):
    content: str           # original_content (chunk text without context prefix)
    score: float | None    # cross-encoder score from reranker
    meta: dict             # full chunk metadata (see §2)
```

#### `QueryResponse`

```python
class QueryResponse(BaseModel):
    answer: str
    sources: list[SourceDocument]
    query: str
    sub_questions: list[str] = []   # empty when query was not decomposed
    is_compound: bool = False       # True when automatically decomposed
    low_confidence: bool = False    # True when CRAG threshold not met after retries
    extracted_filters: dict | None  # filters inferred from natural language (debug)
```

---

### Evaluation (RAGAS)

#### `EvaluationSample`

```python
class EvaluationSample(BaseModel):
    question: str      # test question
    ground_truth: str  # expected correct answer for RAGAS scoring
```

#### `EvaluationRequest`

```python
class EvaluationRequest(BaseModel):
    samples: list[EvaluationSample]  # min 1 sample
    top_k: int = 5                   # retrieved documents per question (1-20)
```

#### `EvaluationResponse`

```python
class EvaluationResponse(BaseModel):
    scores: list[dict]          # per-question: question, faithfulness, answer_relevancy, context_precision
    aggregate: dict[str, float] # mean scores across all samples
    num_samples: int
```

---

## 2. Chunk Metadata Schema

Every document chunk stored in Qdrant carries a flat metadata object.
This is the authoritative field specification.

```jsonc
{
  // ── Document Identity ────────────────────────────────────────────────────
  "doc_id":    "sha256:a3f9c2...",   // SHA-256 of (file_path + raw content).
                                     // Stable across re-indexing if file unchanged.
                                     // Used to OVERWRITE stale chunks in Qdrant.
  "source":    "jahresbericht.pdf",  // Original filename
  "file_type": "pdf",                // Extension without dot: pdf | docx | pptx | …
  "file_path": "/uploads/jahresbericht.pdf",

  // ── Document-level Metadata (set by MetadataEnricher, pre-split) ─────────
  "title":          "Jahresbericht 2024",     // First H1 heading or filename stem
  "word_count":     18400,                    // Approximate total word count of source doc
  "indexed_at":     "2026-02-27T10:30:00Z",  // ISO-8601 UTC timestamp of indexing run
  "indexed_at_ts":  1772233800,               // Unix epoch (int). Used for date-range filters.
  "language":       "de",                     // ISO 639-1, detected by langdetect on full doc

  // "doc_beginning": "..."  ← first N chars for ContentAnalyzer context generation.
  //                            NOT stored in Qdrant (removed by ContentAnalyzer).

  // ── Chunk Position (set by ChunkContextEnricher, post-split) ─────────────
  "chunk_index":    3,          // 0-based position within the parent document
  "chunk_total":    87,         // Total chunks from this document
  "section_title":  "Finanzergebnisse Q3",
  "section_path":   "3 › 3.2 › Finanzergebnisse",   // Heading breadcrumb
  "chunk_type":     "text",     // text | table | code | list | figure_caption

  // ── Parent-Child Linking (set by HierarchicalDocumentSplitter) ─────────────
  "__parent_id": "abc123...",   // ID of the parent document in the parents collection.
                                // Used by AutoMergingRetriever to fetch parent context.
  "__level":     2,             // Hierarchy level: 1=parent, 2=child

  // ── Contextual Prefix (set by ContentAnalyzer, Anthropic Option A) ────────
  "context_prefix": "Dieser Abschnitt stammt aus dem Jahresbericht 2024 der…",
                                // 1-2 sentences situating this chunk in the document.
                                // Prepended to the chunk text BEFORE embedding.
  "original_content": "Im Q3 2024 stieg der Umsatz um 12 %…",
                                // The chunk text WITHOUT the context prefix.
                                // Used for display / citation in API responses.

  // ── LLM-extracted Semantic Metadata (set by ContentAnalyzer) ─────────────
  "summary":         "Dieser Abschnitt beschreibt das Umsatzwachstum im Q3 2024…",
  "keywords":        ["Umsatz", "Q3", "Wachstum", "EBITDA", "Vorjahresvergleich"],
  "classification":  "financial",  // One label from CLASSIFICATION_TAXONOMY in .env

  // Entities stored as flat fields (prefix ent_) for meta_fields_to_embed + Qdrant indexing:
  "ent_persons":        ["Max Mustermann"],
  "ent_organizations":  ["Musterfirma GmbH", "Bundesanstalt für…"],
  "ent_locations":      ["München", "Deutschland"],
  "ent_dates":          ["Q3 2024", "September 2024", "FY2023"],
  "ent_products":       ["SAP S/4HANA", "Power BI"],
  "ent_laws":           ["HGB §285", "ISO 9001"],
  "ent_events":         ["Hauptversammlung 2024", "Q3 Earnings Call"],
  "ent_quantities":     ["€ 4,2 Mrd.", "12 %", "ca. 1 500 Mitarbeiter"],

  // ── Embedding Provenance (set by MetadataEnricher) ───────────────────────
  "embedding_model":     "BAAI/bge-m3",
  "embedding_provider":  "huggingface",
  "embedding_dimension": 1024
}
```

---

## 3. Field Reference

### Identity fields

| Field | Type | Set by | Notes |
|-------|------|--------|-------|
| `doc_id` | string | MetadataEnricher | SHA-256(file_path + content). Stable re-index key. |
| `source` | string | DoclingConverter | Original filename |
| `file_type` | string | DoclingConverter | Extension without dot |
| `file_path` | string | DoclingConverter | Absolute path at indexing time |

### Document-level fields

| Field | Type | Set by | Notes |
|-------|------|--------|-------|
| `title` | string | MetadataEnricher | First H1 or filename stem |
| `word_count` | int | MetadataEnricher | Approximate total words in source |
| `indexed_at` | ISO-8601 string | MetadataEnricher | UTC timestamp |
| `indexed_at_ts` | int | MetadataEnricher | Unix epoch — use for date-range filters |
| `language` | string | MetadataEnricher | ISO 639-1, langdetect on full doc |

### Chunk-position fields

| Field | Type | Set by | Notes |
|-------|------|--------|-------|
| `chunk_index` | int | ChunkContextEnricher | 0-based within document |
| `chunk_total` | int | ChunkContextEnricher | Total chunks from document |
| `section_title` | string | ChunkContextEnricher | Nearest heading |
| `section_path` | string | ChunkContextEnricher | Full heading breadcrumb |
| `chunk_type` | string | ChunkContextEnricher | `text\|table\|code\|list\|figure_caption` |

### Parent-child fields (set by HierarchicalDocumentSplitter)

| Field | Type | Set by | Notes |
|-------|------|--------|-------|
| `__parent_id` | string | HierarchicalDocumentSplitter | ID of parent doc in parents collection |
| `__level` | int | HierarchicalDocumentSplitter | 1=parent, 2=child |

### Contextual prefix fields

| Field | Type | Set by | Notes |
|-------|------|--------|-------|
| `context_prefix` | string | ContentAnalyzer | Anthropic contextual retrieval prefix |
| `original_content` | string | ContentAnalyzer | Chunk text without prefix (for display) |

### Semantic metadata fields

| Field | Type | Set by | Notes |
|-------|------|--------|-------|
| `summary` | string | ContentAnalyzer | 2-3 sentence abstractive summary |
| `keywords` | string[] | ContentAnalyzer | 5-10 key terms |
| `classification` | string | ContentAnalyzer | From `CLASSIFICATION_TAXONOMY` |
| `ent_persons` | string[] | ContentAnalyzer | Full person names |
| `ent_organizations` | string[] | ContentAnalyzer | Companies, agencies, institutions |
| `ent_locations` | string[] | ContentAnalyzer | Countries, cities, regions |
| `ent_dates` | string[] | ContentAnalyzer | All temporal expressions |
| `ent_products` | string[] | ContentAnalyzer | Product names, software, brand names |
| `ent_laws` | string[] | ContentAnalyzer | Laws, regulations, norms |
| `ent_events` | string[] | ContentAnalyzer | Named events, projects, incidents |
| `ent_quantities` | string[] | ContentAnalyzer | Monetary values, percentages, measurements |

### Embedding provenance

| Field | Type | Set by | Notes |
|-------|------|--------|-------|
| `embedding_model` | string | MetadataEnricher | HuggingFace model name |
| `embedding_provider` | string | MetadataEnricher | Always `"huggingface"` |
| `embedding_dimension` | int | MetadataEnricher | Vector dimension |

---

## 4. What is stored vs. embedded

| Content | Stored in Qdrant | Used for embedding |
|---------|------------------|--------------------|
| `context_prefix + chunk_content` | as `doc.content` | ✅ dense + sparse |
| `section_title`, `title`, `summary`, `keywords` | in meta | ✅ appended by embedder (`meta_fields_to_embed`) |
| `ent_persons`, `ent_organizations`, `ent_products`, `ent_laws` | in meta | ✅ appended by embedder |
| `original_content` | in meta (children only) | ❌ display / citation only |
| `ent_locations`, `ent_dates`, `ent_events`, `ent_quantities` | in meta | ❌ filterable only |
| parent documents (`__level=1`) | in parents collection | ❌ fetched by AutoMergingRetriever |

---

## 5. Filtering examples

Filters use Haystack's filter syntax and are passed in the `filters` field of `QueryRequest`.
The QueryAnalyzer also extracts filters automatically from natural language queries.
Preferred style is `meta.<field>` (for example `meta.source`); bare metadata fields are
accepted and normalized automatically.

```json
// All chunks from one source file
{ "field": "meta.source", "operator": "==", "value": "jahresbericht.pdf" }

// All financial chunks in German
{ "operator": "AND", "conditions": [
    { "field": "meta.classification", "operator": "==", "value": "financial" },
    { "field": "meta.language",       "operator": "==", "value": "de" }
]}

// Chunks mentioning a specific organisation
{ "field": "meta.entities.organizations", "operator": "in", "value": ["Musterfirma GmbH"] }

// Documents indexed in January 2025 (Unix epoch: 1735689600 – 1738367999)
{ "operator": "AND", "conditions": [
    { "field": "meta.indexed_at_ts", "operator": ">=", "value": 1735689600 },
    { "field": "meta.indexed_at_ts", "operator": "<=", "value": 1738367999 }
]}
```

### Date-range via API fields

Use the convenience fields `date_from` / `date_to` instead of raw epoch filters:

```json
{
  "query": "Was waren die Umsatzziele?",
  "date_from": "2025-01-01",
  "date_to":   "2025-06-30"
}
```

Both fields are optional and can be combined with `filters` — all conditions
are merged with `AND`. QueryAnalyzer date hints are ignored when explicit
`date_from` / `date_to` are provided.

### Automatic filter extraction

The `QueryAnalyzer` can infer filters directly from natural language:

```
Query: "Show me German financial reports from Q1 2025"

Extracted automatically:
  language:       "de"
  classification: "financial"
  date_from:      "2025-01-01"
  date_to:        "2025-03-31"
```

The `extracted_filters` field in `QueryResponse` shows what was inferred (useful for debugging).
