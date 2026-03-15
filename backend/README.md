# Advanced Hybrid RAG Backend

This directory contains the complete backend. Run the commands in this README from inside `backend/`.

The backend exposes two thin adapters on top of one shared runtime:
- FastAPI for uploads, downloads, task polling, query HTTP endpoints, and evaluation
- FastMCP for `rag_query` and `rag_retrieve`

---

## Architecture overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                         INDEXING PIPELINE                           │
│                                                                     │
│  PDF/DOCX/…                                                         │
│      │                                                              │
│  Docling ──── v1 REST API ──→ docling-serve                │
│      │  (markdown)                                                  │
│  DocumentAnalyzer  (doc_id, title, word_count, document_type,       │
│      │              document_date/period, language via LLM)         │
│  DocumentCleaner                                                    │
│      │                                                              │
│  ParentChildSplitter  ◄── HierarchicalDocumentSplitter               │
│      │  • children (200 words) → children Qdrant collection        │
│      │  • parents  (600 words) → parents  Qdrant collection        │
│      │  • children carry __parent_id (used by AutoMergingRetriever)│
│      │                                                              │
│  ChunkAnnotator  (chunk_index, section_title, section_path)         │
│      │                                                              │
│  ChunkAnalyzer  ◄── 1 LLM call / chunk (async, parallelised)        │
│      │  • context_prefix  (injected into dense-only embedding text) │
│      │  • summary, keywords, classification                         │
│      │  • entities: orgs, persons, locations, dates, …              │
│      │                                                              │
│  [RAPTOR]  ◄── RAPTOR_ENABLED  (section + doc summaries)  │
│      │                                                              │
│  FastembedSparseDocumentEmbedder  (BM42/SPLADE, local ONNX)         │
│      │  sparse vector                                               │
│      │                                                              │
│  DenseContextInjector  (context_prefix + original_content)          │
│      │                                                              │
│  SentenceTransformersDocumentEmbedder  (BAAI/bge-m3, local)         │
│      │  dense vector (1024 dim)                                     │
│      │                                                              │
│  QdrantDocumentStore (children) ────────────────────────────────┐  │
│  QdrantDocumentStore (parents)  ───────────────────────────────┐│  │
│                                                                ││  │
└────────────────────────────────────────────────────────────────┼┼──┘
                                                                 ││
┌────────────────────────────────────────────────────────────────┼┼──┐
│                        RETRIEVAL PIPELINE                      ││  │
│                                                                ││  │
│  User query                                                    ││  │
│      │                                                         ││  │
│  QueryAnalyzer  (fast path or LLM: decompose + extract filters)││  │
│      │  ["What is X?", "How does Y work?"]                     ││  │
│      │                                                         ││  │
│  For each sub-question:                                        ││  │
│      ├─ SentenceTransformersTextEmbedder ──→ dense vector      ││  │
│      │       └──→ QdrantEmbeddingRetriever ◄───────────────────┘│  │
│      │                                        │                  │  │
│      ├─ FastembedSparseTextEmbedder ──→ sparse vector            │  │
│      │       └──→ QdrantSparseEmbeddingRetriever ◄───────────────┘  │
│      │                                        │
│      └─ [HyDE → dense] (HYDE_ENABLED=true)
│                                        │
│  DocumentJoiner  (Reciprocal Rank Fusion)
│      │
│  AutoMergingRetriever  ← parent-context swap (threshold-based)
│      │
│  [ColBERT]  (COLBERT_ENABLED)  ← pre-filter → top 20
│      │
│  SentenceTransformersSimilarityRanker  (BAAI/bge-reranker-v2-m3)  ← final → top 5
│      │
│  top-K grounded documents returned to query service
│
└─────────────────────────────────────────────────────────────────────
```

Generation happens afterwards in a separate Haystack pipeline:

```text
PromptBuilder  (multi-question aware Jinja2 template)
    │
OpenAIGenerator  (any OpenAI-compatible endpoint)
    │
AnswerBuilder
```

---

## Tech stack

| Component | Technology | Why |
|-----------|-----------|-----|
| API adapter | FastAPI | Async HTTP API with OpenAPI docs |
| MCP adapter | FastMCP | MCP tools for query and retrieval |
| RAG framework | Haystack 2.x | Type-safe pipeline DAG, production-ready |
| Vector database | **Qdrant** | Native dense + sparse vectors, fast payload filtering |
| Document extraction | **docling-serve** | Best-in-class PDF/DOCX/PPTX → markdown |
| Dense embeddings | **BAAI/bge-m3** | Multilingual, 1024 dim, state of the art (local) |
| Sparse embeddings | **BM42 / SPLADE** | Learned sparse > BM25 for semantic keyword matching |
| Hybrid fusion | **RRF** | Reciprocal Rank Fusion — no score normalisation needed |
| Reranker (pass 1) | **ColBERT** (colbert-ir/colbertv2.0) | Fast late-interaction pre-filter, top 20 (optional) |
| Reranker (pass 2) | **BAAI/bge-reranker-v2-m3** | Precise cross-encoder final ranking, top 5 (local) |
| LLM | **OpenAI-compatible** | Works with OpenAI, Ollama, vLLM, Groq, … |
| Contextual chunking | **Anthropic Option A** | Contextual prefix + parent-child + word-count-based hierarchical split |
| Multi-question | **QueryAnalyzer** | LLM detects compound queries + extracts metadata filters, retrieves per sub-question |
| HyDE | **HyDE** | Hypothetical document for improved dense retrieval |
| RAPTOR-inspired summaries | **RAPTOR** | Hierarchical section + document summary chunks |
| CRAG | score-threshold retry loop | Corrective re-retrieval + LLM query reformulation |
| Evaluation | **RAGAS** | Faithfulness, answer relevancy, context precision |
| Object storage | **MinIO** | Stores original uploaded documents |
| Task tracking | **TaskStore** | In-memory task state with bounded eviction of finished tasks |

---

## Quickstart

### 1. Start services

```bash
docker compose up -d
# Qdrant Web UI: http://localhost:6333/dashboard
# Docling UI:   http://localhost:5001
```

### 2. Configure

```bash
# Edit .env — required settings:
#   OPENAI_API_KEY     (or configure Ollama: see .env comments)
#   EMBEDDING_MODEL    (default: BAAI/bge-m3)
#   RERANKER_MODEL     (default: BAAI/bge-reranker-v2-m3)
```

### 3. Install with uv & run

```bash
# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Pin Python 3.13 and sync all dependencies from pyproject.toml
uv python pin 3.13
uv sync

# Run the API server
uv run python main_api.py
# or via the packaged entry point
uv run advanced-rag-api
# or with hot-reload:
uv run uvicorn main_api:app --reload

# API docs: http://localhost:8000/docs
```

### 4. Start the MCP server

```bash
# FastMCP runs on stdio by default
uv run python main_mcp.py
# or via the packaged entry point
uv run advanced-rag-mcp
```

`adapters/mcp/server.py` is the composition root for MCP. It mounts the query
server from `adapters/mcp/servers/query.py` and combines its lifespan with
`combine_lifespans(...)`, so the query tools run against the shared runtime
without proxying through the HTTP API.

### Adding / updating packages

```bash
uv add <package>           # add a runtime dependency
uv add --dev <package>     # add a dev dependency
uv remove <package>        # remove a dependency
uv sync                    # reinstall everything from pyproject.toml + uv.lock
```

### 5. Index a document

```bash
curl -X POST http://localhost:8000/documents/index \
  -F "file=@my_document.pdf"

# Response (HTTP 202 — indexing runs async):
# { "task_id": "3fa85f64-...", "source": "my_document.pdf", "message": "Ingestion started. Poll /tasks/{task_id} for progress." }

# Poll for status:
curl http://localhost:8000/tasks/3fa85f64-...
# {
#   "status": "running",
#   "step": "embedding_sparse",
#   "current_step_index": 10,
#   "steps": [
#     {"key": "uploading_minio", "label": "Datei ablegen", "index": 0, "status": "done"},
#     {"key": "converting", "label": "Dokument konvertieren", "index": 1, "status": "done"},
#     {"key": "embedding_sparse", "label": "Sparse Embeddings berechnen", "index": 10, "status": "running"}
#   ]
# }
```

### 6. Query

```bash
# Simple question
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What are the Q3 revenue figures?"}'

# Compound question (automatically decomposed)
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What are the Q3 revenue figures and how does headcount compare to 2023?"}'

# With metadata filter
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is the revenue?",
    "filters": {"field": "meta.source", "operator": "==", "value": "report.pdf"},
    "top_k": 3
  }'

# With date range filter (semantic document date / covered period)
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What changed in Q1?",
    "date_from": "2025-01-01",
    "date_to": "2025-03-31"
  }'
```

---

## LLM backends

The system works with any **OpenAI-compatible** endpoint:

| Backend | `OPENAI_URL` | `OPENAI_API_KEY` | `LLM_MODEL` |
|---------|-------------------|-----------------|-------------|
| OpenAI | *(empty)* | `sk-…` | `gpt-4o-mini` |
| Ollama (local) | `http://localhost:11434/v1` | `ollama` | `llama3.2` |
| vLLM | `http://localhost:8000/v1` | `token-abc` | `mistralai/Mistral-7B-…` |
| Groq | `https://api.groq.com/openai/v1` | `gsk_…` | `llama-3.1-70b-versatile` |
| LM Studio | `http://localhost:1234/v1` | `lm-studio` | `<loaded model>` |

---

## API endpoints

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/documents/index` | Upload & index a document (async, HTTP 202 + task_id) |
| `GET`  | `/tasks/{task_id}` | Poll indexing task progress |
| `POST` | `/query` | Query the RAG system (multi-question, CRAG, ColBERT) |
| `POST` | `/evaluation/run` | RAGAS evaluation on a test set |
| `GET`  | `/docs` | Interactive Swagger UI |
| `GET`  | `/redoc` | ReDoc API documentation |

---

## MCP tools

| Tool | Description |
|------|-------------|
| `rag_query` | Answer a question with grounded sources |
| `rag_retrieve` | Return relevant passages without generation |

---

## Supported document formats

PDF, DOCX, PPTX, XLSX, HTML, TXT, Markdown — all converted via docling-serve.

---

## Configuration reference

See [.env](.env) for all settings with inline documentation.
See [core/config.py](core/config.py) for the grouped settings model.
See [docs/architecture.md](docs/architecture.md) for deep-dive on each design decision.
See [docs/code_style.md](docs/code_style.md) for the backend coding standard.

---

## Project structure

```text
backend/
├── docker-compose.yml     ← Qdrant + MinIO + docling-serve
├── .env                   ← all configuration
├── pyproject.toml         ← dependencies + console entry points
├── main_api.py            ← thin local wrapper for the API adapter
├── main_mcp.py            ← thin local wrapper for the MCP adapter
├── README.md              ← this file
├── docs/
│   ├── architecture.md    ← design decisions and runtime layout
│   └── code_style.md      ← backend coding standard
├── adapters/
│   ├── api/
│   │   ├── app.py         ← FastAPI app + lifespan
│   │   ├── deps.py        ← adapter-local dependency accessors
│   │   ├── models/        ← HTTP request/response models by domain
│   │   └── routes/
│   │       ├── documents.py  ← upload + in-process indexing task launch
│   │       ├── evaluation.py
│   │       ├── query.py
│   │       └── tasks.py
│   └── mcp/
│       ├── server.py      ← main FastMCP server + lifespan composition
│       └── servers/
│           └── query.py   ← mounted query tools server + query lifespan
├── core/
│   ├── config.py          ← grouped settings model
│   ├── runtime.py         ← shared RagRuntime for API and MCP
│   ├── models/
│   │   ├── evaluation.py
│   │   ├── indexing.py
│   │   ├── meta.py
│   │   ├── query.py
│   │   ├── retrieval.py
│   │   ├── tasks.py
│   │   └── vocabulary.py
│   ├── components/        ← Haystack/OpenAI building blocks
│   ├── pipelines/         ← indexing, retrieval, generation
│   ├── services/          ← QueryEngine, RetrievalEngine, IndexingService, EvaluationService
│   └── storage/           ← MinIO, Qdrant, TaskStore
```

Indexing is currently launched from `adapters/api/routes/documents.py` and still runs
in-process. A fully separate worker process would need a persistent queue or task store.
