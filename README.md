<p align="center">
  <img src="logo.png" alt="A-RAG-OG" width="220">
</p>

<h1 align="center">🧬 A-RAG-OG</h1>

<p align="center">
  <b>Agentic RAG, Off Grid</b> — a self-hosted hybrid-RAG <b>MCP server</b> for OpenWebUI:<br>
  structure-aware indexing, hybrid retrieval and cross-encoder reranking, exposed as tools an LLM agent drives.
</p>

---

## ✨ What it is

A-RAG-OG indexes documents into a single hybrid (dense + sparse) Qdrant store and exposes **retrieval as MCP tools**. The agent lives in **OpenWebUI** — its model decides which tool to call, searches in several rounds, reads promising chunks and grounds its answer. This service stays a thin, stateless retrieval layer.

| Part | Description |
|------|-------------|
| 🧩 **HybridChunker** | Docling token- *and* structure-aware chunking, heading path prepended (contextual) |
| 🏷️ **LLM Enrichment** | Per chunk: context, keywords, headings, entities, date |
| 🔄 **Hybrid Retrieval** | Dense (bge-m3) + Sparse (BM25), one Qdrant store |
| 📑 **Reranking** | bge-reranker-v2-m3 cross-encoder, `top_k` before/after per call |
| 🔌 **MCP Server** | FastMCP (streamable-http) + OpenWebUI JWT auth |
| 🤖 **Agent** | OpenWebUI's model orchestrates the tools (A-RAG style) |

---

## 🧰 MCP Tools

| Tool | Purpose |
|------|---------|
| `keyword_and_semantic_search(query, top_k_before, top_k_after)` | **Default** — dense + sparse, fused by reranker |
| `semantic_search(query, top_k_before, top_k_after)` | Dense retrieval (by meaning) + rerank |
| `keyword_search(query, top_k_before, top_k_after)` | Sparse/BM25 retrieval (exact terms) + rerank |
| `filtered_search(query, keywords, entities, content_types, date_from, date_to, …)` | Dense + metadata filter + rerank |
| `find_related(chunk_ids, query, …)` | Associative multi-hop — more chunks sharing the hits' entities |
| `read_chunk(chunk_ids)` | Full content of chunks by id |
| `read_neighbors(chunk_ids, window)` | Full content of the chunks surrounding a hit (document order) |

Each search returns chunk ids + snippets; the agent reads full chunks with `read_chunk`
or pulls surrounding context with `read_neighbors`.

---

## 🚀 Quick Start

```bash
cp .env.example .env

# Backing services (Qdrant, MinIO, Docling converter)
docker compose --profile docling up -d

# Index documents (standalone: uploads to MinIO + builds the store)
uv run python index.py path/to/doc1.pdf path/to/doc2.pdf

# Run the MCP server
uv run python server.py
```

The MCP server listens on `http://HOST:PORT` (default `0.0.0.0:8000`, streamable-http). Point OpenWebUI's MCP integration at it; the seven tools become available to the agent.

> **First run:** the chunker tokenizer (HF) and the sparse BM25 model (FastEmbed) download into `./data` on first use — set `HF_HUB_OFFLINE=0`, then switch back to `1` for offline startups. Dense embeddings and reranking are served by external OpenAI-/`/rerank`-compatible endpoints (configure their URLs in `.env`).

---

## 🏗️ Flow

```
index.py  →  MinIO + Docling HybridChunker → enrich → dense+sparse embed → Qdrant
server.py →  FastMCP tools → (dense|sparse) retrieve → cross-encoder rerank → snippets
OpenWebUI →  agent: search → read → reason → answer (cites chunk ids)
```

---

## ⚙️ Configuration

All settings live in `.env` (see `.env.example`).

### 🖥️ Server
| Variable | Default | Description |
|----------|---------|-------------|
| `HOST` | 0.0.0.0 | MCP bind host |
| `PORT` | 8000 | MCP port (streamable-http) |

### 🔐 Auth (OpenWebUI JWT)
| Variable | Default | Description |
|----------|---------|-------------|
| `JWT_SECRET` | – | OpenWebUI JWT verification key |
| `JWT_ALGORITHM` | HS256 | JWT algorithm |

### 🗂️ S3 Storage (indexing)
| Variable | Default | Description |
|----------|---------|-------------|
| `MINIO_ENDPOINT` | localhost:9000 | S3 endpoint |
| `MINIO_BUCKET` | default | Bucket for indexed files |
| `MINIO_URL_EXPIRE` | 3600 | Lifetime (s) of presigned source URLs in search results |

### 🗄️ Document Store
| Variable | Default | Description |
|----------|---------|-------------|
| `QDRANT_URL` | http://localhost:6333 | Qdrant URL |
| `QDRANT_EMBEDDING_DIM` | 1024 | Dense embedding dimension |
| `QDRANT_COLLECTION` | aragog | Qdrant collection |

### 📑 Converter
| Variable | Default | Description |
|----------|---------|-------------|
| `DOCLING_URL` | http://localhost:5001 | Docling-serve endpoint |
| `DOCLING_TIMEOUT` | 600 | Per-request & job timeout (s) |

### 🧩 Document Chunker
| Variable | Default | Description |
|----------|---------|-------------|
| `CHUNKER_TOKENIZER` | BAAI/bge-m3 | HF tokenizer used to count chunk tokens (match the embedding model) |
| `CHUNKER_MAX_TOKENS` | 1000 | HybridChunker token budget |

### 🏷️ Enricher
| Variable | Default | Description |
|----------|---------|-------------|
| `ENRICHER_MAX_WORKERS` | 3 | Parallel LLM metadata extractions |
| `ENRICHER_MODEL` | gemma4:e4b | Enrichment model |
| `ENRICHER_URL` | http://localhost:11434/v1 | OpenAI-compatible endpoint (enrichment) |
| `ENRICHER_TIMEOUT` | 300 | Enrichment request timeout (s) |

### 🔮 Embedders (Dense + Sparse)
| Variable | Default | Description |
|----------|---------|-------------|
| `DENSE_EMBEDDING_MODEL` | bge-m3:567m | Dense model id on the embedding server (multilingual, 1024-dim) |
| `DENSE_EMBEDDING_URL` | http://localhost:11434/v1 | OpenAI-compatible embeddings endpoint (Ollama by default; also vLLM/TEI/Infinity) |
| `SPARSE_EMBEDDING_MODEL` | Qdrant/bm25 | Sparse model (FastEmbed, runs locally) |
| `SPARSE_EMBEDDING_LANGUAGE` | german | BM25 stop-word / stemming language |
| `SPARSE_EMBEDDING_DEVICE` | cpu | Device for the sparse model (`cpu` or `cuda`) |
| `EMBEDDED_META_FIELDS` | context,keywords,headings,hypothetical_questions | Meta fields embedded with content (dense + sparse) |

### 🔁 Reranker
| Variable | Default | Description |
|----------|---------|-------------|
| `RERANKER_MODEL` | BAAI/bge-reranker-v2-m3 | Reranker model id on the rerank server |
| `RERANKER_URL` | http://localhost:8001/v1 | vLLM base URL (VLLMRanker calls its `/rerank`) |

---

## 🐳 Backing Services

```bash
docker compose up -d                     # MinIO + Qdrant
docker compose --profile docling up -d   # + Docling converter
```

| Service | Port | Profile | Description |
|---------|------|---------|-------------|
| Qdrant | 6333 | – | Vector database (REST + dashboard) |
| MinIO | 9000 / 9001 | – | S3 API / console |
| Docling | 5001 | `docling` | Document converter |

---

## 📄 License

[MIT](LICENSE)
