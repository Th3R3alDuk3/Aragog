# Routers

FastAPI route handlers and dependency injection for the Advanced Hybrid RAG API.

---

## Files

| File | Prefix | Purpose |
|------|--------|---------|
| `documents.py` | `/documents` | Index documents (async, HTTP 202) |
| `query.py` | `/query` | RAG query (standard) |
| `stream.py` | `/query` | RAG query (SSE streaming) |
| `evaluation.py` | `/evaluation` | RAGAS evaluation |
| `tasks.py` | `/tasks` | Task status polling |
| `_deps.py` | — | FastAPI dependency providers |

---

## Endpoints

### `POST /documents/index`

Starts async ingestion of a document into Qdrant. Returns immediately with a `task_id`.
Poll **GET /tasks/{task_id}** to track progress.

**Request:** `multipart/form-data` — single file upload (`file` field)

**Response (HTTP 202):**
```json
{ "task_id": "3fa85f64-...", "source": "jahresbericht.pdf", "message": "Ingestion started. Poll /tasks/{task_id} for progress." }
```

**Pipeline stages (visible in `GET /tasks/{task_id}`):**
`deleting_stale → uploading_minio → converting → enriching_metadata → cleaning
→ splitting_chunks → writing_parents → enriching_chunks → analyzing_content
→ [summarizing_raptor] → embedding_dense → embedding_sparse → writing → done`

---

### `GET /tasks/{task_id}`

Poll an indexing task for its current status.

**Response:**
```json
{
  "task_id": "3fa85f64-...",
  "status": "done",
  "step": "done",
  "source": "jahresbericht.pdf",
  "created_at": "2025-01-01T10:00:00Z",
  "updated_at": "2025-01-01T10:02:30Z",
  "result": { "indexed": 42, "source": "jahresbericht.pdf", "minio_url": "http://..." },
  "error": null
}
```

`status` is one of `pending | running | done | failed`.

---

### `POST /query`

Standard RAG query — returns full response after generation completes.

**Request body:**
```json
{
  "query": "What were the Q3 revenue figures and how does headcount compare to 2023?",
  "top_k": 5,
  "date_from": "2024-01-01",
  "date_to": "2024-12-31",
  "filters": {
    "field": "meta.classification",
    "operator": "==",
    "value": "financial"
  }
}
```

| Field | Type | Default | Notes |
|-------|------|---------|-------|
| `query` | string | required | Natural language question (simple or compound) |
| `top_k` | int | 5 | Max source documents in response (1-50) |
| `date_from` | ISO date | null | Filter to docs indexed on or after this date |
| `date_to` | ISO date | null | Filter to docs indexed on or before this date |
| `filters` | Haystack filter | null | Arbitrary metadata filter (see `models/README.md`). `meta.<field>` is preferred; bare metadata fields are normalized automatically. |

**Response body:**
```json
{
  "answer": "In Q3 2024, revenue grew by 12 % to € 4.2 billion...",
  "sources": [
    {
      "content": "Revenue grew 12 % to € 4.2 billion...",
      "score": 0.94,
      "meta": { "source": "jahresbericht.pdf", "section_title": "Financial Results Q3", ... }
    }
  ],
  "query": "What were the Q3 revenue figures...",
  "sub_questions": [
    "What were the Q3 revenue figures?",
    "How does headcount compare to 2023?"
  ],
  "is_compound": true,
  "low_confidence": false,
  "extracted_filters": { "date_from": "2024-01-01", "classification": "financial" }
}
```

| Field | Notes |
|-------|-------|
| `sub_questions` | Empty list if query was not decomposed |
| `is_compound` | True when automatically decomposed |
| `low_confidence` | True when CRAG threshold not met after retries |
| `extracted_filters` | Filters inferred from natural language (debug only) |

**Query processing flow:**

```
1. QueryAnalyzer.analyze(query)
      → sub_questions (decomposition)
      → extracted filters (date, classification, language, source)

2. build_filters(request, analysis)
      merge LLM-extracted + explicit request filters
      explicit request filters always win

3. For each sub-question:
      if CRAG_ENABLED:  retrieve_with_crag()   # retry on low score
      else:             run_retrieval()

   Retrieval pipeline order per sub-question:
      dense + sparse + [HyDE dense] → RRF → AutoMergingRetriever
      → [ColBERT pre-filter] → cross-encoder reranker

4. Merge retrieved docs across sub-questions (deduplicate by doc.id)

5. Generate answer: prompt_builder → llm → answer_builder

6. Build QueryResponse
```

---

### `POST /query/stream`

Same as `POST /query` but streams the LLM answer token-by-token via
Server-Sent Events (SSE).

**Response:** `text/event-stream`

**SSE Event protocol:**

| Event | Data | Notes |
|-------|------|-------|
| `token` | `<text chunk>` | One per LLM output token |
| `sources` | `<JSON array>` | Source documents (after stream ends) |
| `done` | `""` | Stream termination signal |
| `error` | `<error message>` | On LLM streaming failure |

**Client example (curl):**
```bash
curl -X POST http://localhost:8000/query/stream \
  -H "Content-Type: application/json" \
  -d '{"query": "What is the EBITDA?"}' \
  --no-buffer
```

**Client example (JavaScript):**
```javascript
const response = await fetch('/query/stream', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({ query: 'What is the EBITDA?' }),
});
const reader = response.body.getReader();
// read SSE events...
```

**Implementation:** Retrieval runs via `prepare_context()` (same as `/query`).
LLM generation uses `AsyncOpenAI.chat.completions.create(stream=True)`.
The `RAG_PROMPT` Jinja2 template is imported from `pipelines/generation.py`
and rendered with the standard `jinja2` library.

---

### `POST /evaluation/run`

Runs RAGAS evaluation on a test set. Requires `RAGAS_ENABLED=true` in `.env`.
Returns HTTP 403 otherwise.

**Request body:**
```json
{
  "samples": [
    { "question": "What is the EBITDA?", "ground_truth": "The EBITDA is 12M EUR." },
    { "question": "How many employees?", "ground_truth": "15,000 employees." }
  ],
  "top_k": 5
}
```

**Response body:**
```json
{
  "scores": [
    {
      "question": "What is the EBITDA?",
      "faithfulness": 0.92,
      "answer_relevancy": 0.88,
      "context_precision": 0.75
    }
  ],
  "aggregate": {
    "faithfulness": 0.92,
    "answer_relevancy": 0.88,
    "context_precision": 0.75
  },
  "num_samples": 2
}
```

**RAGAS metrics:**

| Metric | Measures |
|--------|---------|
| `faithfulness` | Does the answer contain only claims supported by the retrieved context? |
| `answer_relevancy` | How relevant is the answer to the question? |
| `context_precision` | Are the retrieved chunks actually relevant to the question? |

**Notes:**
- `ragas` and `datasets` are lazily imported inside the handler (never loaded at startup when disabled)
- Samples that fail retrieval/generation are included with empty answer/contexts (degraded score, no crash)

---

## `_deps.py` — Dependency Providers

FastAPI dependency functions that fetch instances from `app.state`
(populated at startup in `main.py`'s `lifespan` context manager).

| Function | Returns | Notes |
|----------|---------|-------|
| `get_settings` | `Settings` | |
| `get_children_store` | `QdrantDocumentStore` | Children collection |
| `get_parents_store` | `QdrantDocumentStore` | Parents collection |
| `get_minio_store` | `MinioStore` | Original file storage |
| `get_indexing_pipeline` | `AsyncPipeline` | |
| `get_indexing_semaphore` | `Semaphore` | Concurrency limiter |
| `get_query_analyzer` | `QueryAnalyzer` | |
| `get_retrieval_pipeline` | `AsyncPipeline` | |
| `get_generation_pipeline` | `AsyncPipeline` | |
| `get_task_store` | `BoundedTaskStore` | Task tracking |

---

## Filter Merging (`build_filters`)

Filters from multiple sources are merged with `AND`:

**Build order (later entries override earlier ones logically):**
1. LLM-extracted date range (`analysis.date_from` / `date_to`) — only when request date fields are not set
2. LLM-extracted metadata (`classification`, `language`, `source`; `source` is exact filename)
3. Explicit `request.filters` (normalized to `meta.<field>` where needed)
4. Explicit `request.date_from` / `request.date_to`

If only one condition: returned as a single filter object.
If multiple: wrapped in `{"operator": "AND", "conditions": [...]}`.

---

## Feature Flags

All features default to `true` and can be disabled independently:

| Flag | Endpoint affected | Notes |
|------|-------------------|-------|
| `HYDE_ENABLED` | `/query`, `/query/stream` | Second dense retrieval branch with hypothetical document |
| `CRAG_ENABLED` | `/query`, `/query/stream` | Retry on low retrieval confidence |
| `COLBERT_ENABLED` | `/query`, `/query/stream` | ColBERT pre-filter before cross-encoder |
| `RAPTOR_ENABLED` | indexing | Summary chunks added at index time |
| `RAGAS_ENABLED` | `/evaluation/run` | Must be `true` or endpoint returns 403 |
