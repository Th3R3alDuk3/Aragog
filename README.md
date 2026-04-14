# Advanced Hybrid RAG

A local RAG system with a Gradio web UI and built-in MCP server.
Two tabs — **Indexierung** for uploading documents, **Abfrage** for querying — backed by a fully local hybrid retrieval pipeline.

---

## Architecture overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                         INDEXING PIPELINE                           │
│                                                                     │
│  PDF/DOCX/…                                                         │
│      │                                                              │
│  Docling ──── v1 REST API ──→ docling-serve                         │
│      │  (markdown)                                                  │
│  DocumentAnalyzer  (doc_id, title, word_count, document_type,       │
│      │              document_date/period, language via LLM)         │
│  DocumentCleaner                                                    │
│      │                                                              │
│  ParentChildSplitter  ◄── HierarchicalDocumentSplitter              │
│      │  • children (200 words) → children Qdrant collection         │
│      │  • parents  (600 words) → parents  Qdrant collection         │
│      │  • children carry __parent_id (AutoMergingRetriever)         │
│      │                                                              │
│  ChunkAnnotator  (chunk_index, section_title, section_path)         │
│      │                                                              │
│  ChunkAnalyzer  ◄── 1 LLM call / chunk (async, parallelised)        │
│      │  • context_prefix  (injected into dense embedding text)      │
│      │  • summary, keywords, classification                         │
│      │  • entities: orgs, persons, locations, dates, …              │
│      │                                                              │
│  [RAPTOR]  ◄── RAPTOR_ENABLED  (section + doc summaries)            │
│      │                                                              │
│  FastembedSparseDocumentEmbedder  (BM42/SPLADE, local ONNX)         │
│      │  sparse vector                                               │
│      │                                                              │
│  DenseContextInjector  (context_prefix + original_content)          │
│      │                                                              │
│  SentenceTransformersDocumentEmbedder  (BAAI/bge-m3, local)         │
│      │  dense vector (1024 dim)                                     │
│      │                                                              │
│  QdrantDocumentStore (children + parents)                           │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│                        RETRIEVAL PIPELINE                           │
│                                                                     │
│  User query                                                         │
│      │                                                              │
│  QueryAnalyzer  (LLM: decompose + extract metadata filters)         │
│      │  ["What is X?", "How does Y work?"]                          │
│      │                                                              │
│  For each sub-question:                                             │
│      ├─ SentenceTransformersTextEmbedder → dense vector             │
│      │       └──→ QdrantEmbeddingRetriever                          │
│      ├─ FastembedSparseTextEmbedder → sparse vector                 │
│      │       └──→ QdrantSparseEmbeddingRetriever                    │
│      └─ [HyDE → dense]  (HYDE_ENABLED)                             │
│                                                                     │
│  DocumentJoiner  (Reciprocal Rank Fusion)                           │
│      │                                                              │
│  AutoMergingRetriever  ← parent-context swap (threshold-based)      │
│      │                                                              │
│  [ColBERT]  (COLBERT_ENABLED)  ← pre-filter → top 20               │
│      │                                                              │
│  SentenceTransformersSimilarityRanker  (bge-reranker-v2-m3) → top 5 │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘

Generation (separate Haystack pipeline):

  PromptBuilder  (multi-question Jinja2 template)
      │
  OpenAIGenerator  (any OpenAI-compatible endpoint)
      │
  AnswerBuilder
```

---

## Tech stack

| Component | Technology |
|-----------|------------|
| UI + MCP server | **Gradio** (`gradio[mcp]`) |
| RAG framework | **Haystack 2.x** |
| Vector database | **Qdrant** — native dense + sparse vectors |
| Document extraction | **docling-serve** — PDF/DOCX/PPTX → markdown |
| Dense embeddings | **BAAI/bge-m3** — multilingual, 1024 dim (local) |
| Sparse embeddings | **BM42 / SPLADE** — learned sparse (local ONNX) |
| Hybrid fusion | **RRF** — Reciprocal Rank Fusion |
| Reranker (pass 1) | **ColBERT** colbert-ir/colbertv2.0 — pre-filter, top 20 |
| Reranker (pass 2) | **BAAI/bge-reranker-v2-m3** — cross-encoder, top 5 (local) |
| LLM | **OpenAI-compatible** — OpenAI, Ollama, vLLM, Groq, … |
| Contextual chunking | **Anthropic Option A** — context prefix + parent-child split |
| Multi-question | **QueryAnalyzer** — detects compound queries, retrieves per sub-question |
| HyDE | Hypothetical document for improved dense retrieval |
| RAPTOR | Hierarchical section + document summary chunks |
| CRAG | Score-threshold retry with LLM query reformulation |
| Evaluation | **RAGAS** — faithfulness, answer relevancy, context precision |
| Object storage | **MinIO** — stores original uploaded files |

---

## Runtime-toggleable SOTA features

All advanced features default to the values in `.env` and can be overridden per request in the UI without restarting the server.

### Abfrage tab — Erweiterte Einstellungen

| Setting | `.env` key | What it does |
|---------|-----------|--------------|
| **HyDE** | `HYDE_ENABLED` | Generates a hypothetical answer document before dense retrieval to improve embedding alignment. Toggleable at request time (skips the HyDE branch in the pipeline). |
| **CRAG** | `CRAG_ENABLED` | If the top reranker score is below the threshold, reformulates the query via LLM and retries retrieval. |
| **CRAG Score-Schwelle** | `CRAG_SCORE_THRESHOLD` | Minimum score (0–1) to consider a retrieval result confident enough. |
| **CRAG Max. Wiederholungen** | `CRAG_MAX_RETRIES` | How many reformulation attempts to make before accepting a low-confidence result. |

> **ColBERT** (`COLBERT_ENABLED`) is wired as a fixed graph edge  
> `auto_merging_retriever → colbert_reranker → reranker`  
> and cannot be toggled at runtime. Change `COLBERT_ENABLED` in `.env` and restart.

### Indexierung tab — Erweiterte Einstellungen

| Setting | `.env` key | What it does |
|---------|-----------|--------------|
| **RAPTOR** | `RAPTOR_ENABLED` | Builds additional summary chunks at section and document level for better abstract-query coverage. Can be disabled per upload to speed up indexing. |

---

## Quickstart

### 1. Start services

```bash
docker compose up -d
# Qdrant:  http://localhost:6333/dashboard
# MinIO:   http://localhost:9001
# Docling: http://localhost:5001
```

### 2. Configure

```bash
cp .env.example .env
# Required:
#   OPENAI_API_KEY / OPENAI_URL   (or point to Ollama, see .env)
#   EMBEDDING_MODEL               (default: BAAI/bge-m3)
#   RERANKER_MODEL                (default: BAAI/bge-reranker-v2-m3)
```

### 3. Install & run

```bash
# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

uv sync
uv run python app.py
# → http://localhost:8000
```

### 4. Managing packages

```bash
uv add <package>        # add a runtime dependency
uv add --dev <package>  # add a dev dependency
uv remove <package>     # remove a dependency
uv sync                 # reinstall from pyproject.toml + uv.lock
```

---

## MCP server

The Gradio app starts a built-in MCP server automatically (`mcp_server=True`).
Two tools are exposed:

| Tool | Description |
|------|-------------|
| `rag_query` | Answer a question with generated response + grounded sources |
| `rag_retrieve` | Return relevant passages without generation |

MCP endpoint: `http://localhost:8000/gradio_api/mcp/sse`

---

## LLM backends

| Backend | `OPENAI_URL` | `OPENAI_API_KEY` | `LLM_MODEL` |
|---------|-------------|-----------------|-------------|
| OpenAI | *(empty)* | `sk-…` | `gpt-4o-mini` |
| Ollama | `http://localhost:11434/v1` | `ollama` | `llama3.2` |
| vLLM | `http://localhost:8000/v1` | `token-abc` | `mistralai/Mistral-7B-…` |
| Groq | `https://api.groq.com/openai/v1` | `gsk_…` | `llama-3.1-70b-versatile` |
| LM Studio | `http://localhost:1234/v1` | `lm-studio` | `<loaded model>` |

---

## Supported document formats

PDF, DOCX, PPTX, XLSX, HTML, TXT, Markdown — all converted via docling-serve.

---

## Project structure

```text
.
├── app.py          ← entry point (Gradio UI + MCP server)
├── docker-compose.yml     ← Qdrant + MinIO + docling-serve
├── .env                   ← all configuration
├── pyproject.toml         ← dependencies + console entry point
├── docs/
│   ├── architecture.md    ← design decisions and runtime layout
│   └── code_style.md      ← coding standard
└── core/
    ├── config.py          ← settings model (pydantic-settings)
    ├── runtime.py         ← shared RagRuntime
    ├── models/            ← query, retrieval, indexing, task models
    ├── components/        ← Haystack components (HyDE, ColBERT, RAPTOR, …)
    ├── pipelines/         ← indexing, retrieval, generation pipelines
    ├── services/          ← QueryEngine, RetrievalEngine, IndexingService
    └── storage/           ← Qdrant, MinIO, TaskStore
```

---

## Configuration reference

See [.env.example](.env.example) for all settings with inline documentation.  
See [core/config.py](core/config.py) for the settings model.  
See [docs/architecture.md](docs/architecture.md) for design decisions.
