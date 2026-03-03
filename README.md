# Advanced Hybrid RAG — FastAPI + Haystack + Qdrant

Production-ready Retrieval-Augmented Generation backend with state-of-the-art retrieval quality.

---

## Architecture overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                         INDEXING PIPELINE                           │
│                                                                     │
│  PDF/DOCX/…                                                         │
│      │                                                              │
│  DoclingConverter ──── Gradio API ──→ docling-serve                 │
│      │  (markdown)                                                  │
│  MetadataEnricher  (doc_id, title, word_count, indexed_at,          │
│      │              indexed_at_ts, language via langdetect)         │
│  DocumentCleaner                                                    │
│      │                                                              │
│  MarkdownHeaderSplitter  ◄── Haystack built-in (H1–H6)              │
│      │  • meta["header"], meta["parent_headers"] set per chunk      │
│      │                                                              │
│  ParentChildSplitter  ◄── Anthropic Contextual Retrieval A          │
│      │  • child chunks (200 words) for precise retrieval            │
│      │  • meta["parent_content"] = full section text for LLM        │
│      │                                                              │
│  ChunkContextEnricher  (chunk_index, section_title, section_path)   │
│      │                                                              │
│  ContentAnalyzer  ◄── 1 LLM call / chunk (parallelised)             │
│      │  • context_prefix  (prepended before embedding)              │
│      │  • summary, keywords, classification                         │
│      │  • entities: orgs, persons, locations, dates, …              │
│      │                                                              │
│  SentenceTransformersDocumentEmbedder  (BAAI/bge-m3, local)         │
│      │  dense vector (1024 dim)                                     │
│      │                                                              │
│  FastembedSparseDocumentEmbedder  (BM42/SPLADE, local ONNX)         │
│      │  sparse vector                                               │
│      │                                                              │
│  QdrantDocumentStore  ───────────────────────────────────────────┐  │
│                                                                  │  │
└──────────────────────────────────────────────────────────────────┼──┘
                                                                   │
┌──────────────────────────────────────────────────────────────────┼──┐
│                        RETRIEVAL PIPELINE                        │  │
│                                                                  │  │
│  User query                                                      │  │
│      │                                                           │  │
│  QueryAnalyzer  (LLM: decompose + extract metadata filters)      │  │
│      │  ["What is X?", "How does Y work?"]                       │  │
│      │                                                           │  │
│  For each sub-question:                                          │  │
│      ├─ SentenceTransformersTextEmbedder ──→ dense vector        │  │
│      │       └──→ QdrantEmbeddingRetriever ◄─────────────────────┘  │
│      │                                        │                     │
│      └─ FastembedSparseTextEmbedder ──→ sparse vector               │
│              └──→ QdrantSparseEmbeddingRetriever ◄──────────────────┘
│                                        │
│  DocumentJoiner  (Reciprocal Rank Fusion)
│      │
│  swap_to_parent_content()  ← replace child with full section
│      │
│  SentenceTransformersSimilarityRanker  (BAAI/bge-reranker-v2-m3, local)
│      │
│  PromptBuilder  (multi-question aware Jinja2 template)
│      │
│  OpenAIGenerator  (any OpenAI-compatible endpoint)
│      │
│  AnswerBuilder
│
└─────────────────────────────────────────────────────────────────────
```

---

## Tech stack

| Component | Technology | Why |
|-----------|-----------|-----|
| API framework | FastAPI | Async, OpenAPI docs built-in |
| RAG framework | Haystack 2.x | Type-safe pipeline DAG, production-ready |
| Vector database | **Qdrant** | Native dense + sparse vectors, fast payload filtering |
| Document extraction | **docling-serve** | Best-in-class PDF/DOCX/PPTX → markdown |
| Dense embeddings | **BAAI/bge-m3** | Multilingual, 1024 dim, state of the art (local) |
| Sparse embeddings | **BM42 / SPLADE** | Learned sparse > BM25 for semantic keyword matching |
| Hybrid fusion | **RRF** | Reciprocal Rank Fusion — no score normalisation needed |
| Reranker (pass 1) | **BAAI/bge-reranker-v2-m3** | Multilingual cross-encoder, state of the art (local) |
| Reranker (pass 2) | **ColBERT** (colbert-ir/colbertv2.0) | Late-interaction, token-level precision (optional) |
| LLM | **OpenAI-compatible** | Works with OpenAI, Ollama, vLLM, Groq, … |
| Contextual chunking | **Anthropic Option A** | Contextual prefix + parent-child + heading-aware split |
| Multi-question | **QueryAnalyzer** | LLM detects compound queries + extracts metadata filters, retrieves per sub-question |
| HyDE | **HyDEGenerator** | Hypothetical document for improved dense retrieval |
| RAPTOR | **RaptorSummarizer** | Multi-level section + document summary chunks |
| CRAG | score-threshold retry loop | Corrective re-retrieval + LLM query reformulation |
| Evaluation | **RAGAS** | Faithfulness, answer relevancy, context precision |
| Object storage | **MinIO** | Stores original uploaded documents |
| Task tracking | **BoundedTaskStore** | Async indexing with HTTP 202 + polling |

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
uv run python main.py
# or with hot-reload:
uv run uvicorn main:app --reload

# API docs: http://localhost:8000/docs
```

### Adding / updating packages

```bash
uv add <package>           # add a runtime dependency
uv add --dev <package>     # add a dev dependency
uv remove <package>        # remove a dependency
uv sync                    # reinstall everything from pyproject.toml + uv.lock
```

### 4. Index a document

```bash
curl -X POST http://localhost:8000/documents/index \
  -F "file=@my_document.pdf"

# Response:
# { "indexed": 42, "source": "my_document.pdf", "minio_url": "http://localhost:9000/rag-docs/abcd1234-my_document.pdf" }
# minio_url is null if MinIO is not configured/available
```

### 5. Query

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

# With date range filter (documents indexed between two dates)
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
| `POST` | `/query/stream` | Streaming query via SSE (token / sources / done events) |
| `POST` | `/evaluation/run` | RAGAS evaluation on a test set |
| `GET`  | `/docs` | Interactive Swagger UI |
| `GET`  | `/redoc` | ReDoc API documentation |

---

## Supported document formats

PDF, DOCX, PPTX, XLSX, HTML, TXT, Markdown — all converted via docling-serve.

---

## Configuration reference

See [.env](.env) for all settings with inline documentation.
See [docs/architecture.md](docs/architecture.md) for deep-dive on each design decision.

---

## Project structure

```
RAG/
├── docker-compose.yml     ← Qdrant + MinIO + docling-serve
├── .env                   ← all configuration
├── pyproject.toml         ← dependencies (managed with uv)
├── main.py                ← FastAPI lifespan: pipelines + MinIO + QueryAnalyzer init
├── config.py              ← pydantic-settings (all feature flags)
├── SCHEMA.md              ← metadata field specification
├── README.md              ← this file
├── docs/
│   └── architecture.md   ← design decisions & advanced RAG patterns
├── models/
│   ├── api.py             ← Pydantic request/response models (Query, Task, Evaluation)
│   └── meta.py            ← ChunkMetadata schema (all pipeline stages)
├── components/
│   ├── docling_converter.py     ← PDF/DOCX → markdown (gradio_client)
│   ├── metadata_enricher.py     ← doc-level metadata + language detection
│   ├── parent_child_splitter.py ← child chunks (RecursiveDocumentSplitter) + parent_content
│   ├── chunk_enricher.py        ← chunk position, section_path, chunk_type heuristic
│   ├── content_analyzer.py      ← LLM: context_prefix + summary + keywords + NER
│   ├── raptor_summarizer.py     ← RAPTOR section + doc-level summary chunks
│   ├── hyde_generator.py        ← HyDE hypothetical document for dense retrieval
│   ├── colbert_reranker.py      ← ColBERT late-interaction second-pass reranker
│   └── query_analyzer.py        ← multi-question decomposition + NL filter extraction
├── pipelines/
│   ├── indexing.py        ← 11-stage DAG (converter → … → raptor → dense → sparse → write)
│   ├── retrieval.py       ← dense + sparse + HyDE → RRF → cross-encoder → ColBERT
│   ├── generation.py      ← PromptBuilder (RAG_PROMPT) → OpenAIGenerator → AnswerBuilder
│   └── _factories.py      ← HF dense/sparse embedders, cross-encoder reranker, LLM
├── routers/
│   ├── documents.py       ← POST /documents/index (async, HTTP 202)
│   ├── query.py           ← POST /query (multi-question, CRAG, ColBERT)
│   ├── stream.py          ← POST /query/stream (SSE: token / sources / done)
│   ├── evaluation.py      ← POST /evaluation/run (RAGAS)
│   ├── tasks.py           ← GET /tasks/{task_id}
│   └── _deps.py           ← FastAPI dependency injection from app.state
└── services/
    ├── indexing.py        ← pipeline orchestration, step-by-step, thread executor
    ├── query.py           ← prepare_context, CRAG loop, filter building
    ├── evaluation.py      ← RAGAS with LangChain wrappers (lazy import)
    ├── minio_store.py     ← MinIO upload / URL / delete
    └── tasks.py           ← BoundedTaskStore (evicts done/failed, HTTP 503 on overflow)
```
