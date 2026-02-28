# Routers

FastAPI route handlers and dependency injection for the Advanced Hybrid RAG API.

---

## Files

| File | Prefix | Purpose |
|------|--------|---------|
| `documents.py` | `/documents` | Index documents |
| `query.py` | `/query` | RAG query (standard) |
| `stream.py` | `/query` | RAG query (SSE streaming) |
| `evaluation.py` | `/evaluation` | RAGAS evaluation |
| `_deps.py` | — | FastAPI dependency providers |

---

## Endpoints

### `POST /documents/index`

Indexes one document into Qdrant.

**Request:** `multipart/form-data` — single file upload (`file` field)

**Response:**
```json
{ "indexed": 87, "source": "jahresbericht.pdf", "minio_url": "http://localhost:9000/rag-docs/abcd1234-jahresbericht.pdf" }
```

**Pipeline:** `DoclingConverter → MetadataEnricher → … → QdrantDocumentStore`

Runs synchronously inside the async route handler. For large PDFs this blocks
the event loop during conversion; wrap in a task queue if needed for production.

---

### `POST /query`

Standard RAG query — returns full response after generation completes.

**Request body:**
```json
{
  "query": "What were the Q3 revenue figures and how does headcount compare to 2023?",
  "top_k": 5,
  "use_hyde": false,
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
| `use_hyde` | bool | false | Enable HyDE for this request only |
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

2. _build_filters(request, analysis)
      merge LLM-extracted + explicit request filters
      explicit request filters always win

3. For each sub-question:
      if CRAG_ENABLED:  _retrieve_with_crag()   # retry on low score
      else:             _retrieve_simple()

4. Merge retrieved docs (deduplicate by doc.id)

5. swap_to_parent_content()  # child → parent section

6. if COLBERT_ENABLED: ColBERTReranker.rerank()

7. _run_generation_only()   # prompt_builder → llm → answer_builder

8. Build QueryResponse
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

**Client example (JavaScript EventSource):**
```javascript
const response = await fetch('/query/stream', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({ query: 'What is the EBITDA?' }),
});
const reader = response.body.getReader();
// read SSE events...
```

**Implementation:** Retrieval runs synchronously via `asyncio.to_thread()`.
LLM generation uses `AsyncOpenAI.chat.completions.create(stream=True)`.
The `RAG_PROMPT` Jinja2 template is imported from `pipelines/retrieval_pipeline.py` and
rendered with the standard `jinja2` library.

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

### `GET /health`

Liveness check. Returns HTTP 200 when the app is running and Qdrant is reachable.

**Response:**
```json
{ "status": "ok", "document_store": "ok (1234 chunks)" }
```

---

## `_deps.py` — Dependency Providers

FastAPI dependency functions that fetch pipeline instances from `app.state`
(populated at startup in `main.py`'s `lifespan` context manager).

| Function | Returns | Notes |
|----------|---------|-------|
| `get_document_store` | `QdrantDocumentStore` | |
| `get_indexing_pipeline` | `Pipeline` | |
| `get_query_pipeline` | `Pipeline` | Retrieval pipeline |
| `get_generation_pipeline` | `Pipeline` | Prompt/LLM/answer pipeline |
| `get_query_analyzer` | `QueryAnalyzer` | |
| `get_hyde_generator` | `HyDEGenerator \| None` | None when `HYDE_ENABLED=false` |
| `get_colbert_reranker` | `ColBERTReranker \| None` | None when `COLBERT_ENABLED=false` |
| `get_minio_store` | `MinioStore \| None` | None when MinIO is not configured |

---

## Filter Merging (`_build_filters`)

Filters from multiple sources are merged with `AND`:

**Build order:**
1. LLM-extracted date range (`analysis.date_from` / `date_to`) — only when request date fields are not set
2. LLM-extracted metadata (`classification`, `language`, `source`; `source` is exact filename)
3. Explicit `request.filters` (normalized to `meta.<field>` where needed)
4. Explicit `request.date_from` / `request.date_to`

If only one condition: returned as a single filter object.
If multiple: wrapped in `{"operator": "AND", "conditions": [...]}`.

---

## Feature Flags

All features default to `false` and can be enabled independently:

| Flag | Endpoint affected | Notes |
|------|-------------------|-------|
| `HYDE_ENABLED` | `/query`, `/query/stream` | Global HyDE; also per-request via `use_hyde` |
| `CRAG_ENABLED` | `/query`, `/query/stream` | Retry on low retrieval confidence |
| `COLBERT_ENABLED` | `/query`, `/query/stream` | Second-pass ColBERT reranking |
| `RAGAS_ENABLED` | `/evaluation/run` | Must be `true` or endpoint returns 403 |
