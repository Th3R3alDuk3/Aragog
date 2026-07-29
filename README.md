<p align="center">
  <img src="logo.png" alt="A-RAG-OG" width="220">
</p>

<h1 align="center">A-RAG-OG</h1>

<p align="center">
  <b>Agentic RAG, Off Grid</b> — a self-hosted retrieval <b>MCP server</b> for OpenWebUI:<br>
  structure-aware indexing into one hybrid store; search, rerank and read exposed as MCP tools for the agent.
</p>

---

## 🚀 Quick Start

One compose file runs the whole self-hosted stack — MCP server, MinIO,
Qdrant, Docling, embedder and reranker.

**Requires:** Docker with the NVIDIA container toolkit (for the bundled
embedder/reranker — not needed with external endpoints); `uv` for development.

```bash
cp .env.example .env        # set your secrets

docker compose up -d --build
```

Running some services elsewhere? Point `.env` at them and start only what
you need, e.g. `docker compose up -d --build server`.

Index documents into the running stack:

```bash
docker compose run --rm -v ./mydocs:/docs server python index.py /docs/doc1.pdf
```

Indexing is idempotent: chunk ids are deterministic (source, position and
content), so re-running `index.py` over the same files updates chunks in
place instead of duplicating them — a failed run can simply be repeated.
Chunk content changes (an edited file, a changed chunker) get new ids, so
rebuild the collection then instead of re-indexing over the old one.

Point OpenWebUI's MCP integration at `http://HOST:8000/mcp` (streamable-http) —
the seven retrieval tools become available to the agent. Give the OpenWebUI
model the system prompt from [PROMPT.md](PROMPT.md); it enforces the
search → read → cite workflow.

> **LLM endpoints:** enrichment (indexing), embeddings and reranking are
> ordinary OpenAI-compatible endpoints configured in `.env` — mix and match
> freely. The bundled embedder + reranker are vLLM containers and need an
> NVIDIA GPU; `DENSE_EMBEDDING_URL` / `RERANKER_URL` can just as well point
> at any external service (OpenAI, vLLM, TEI, …). The enricher
> (`ENRICHER_URL`) works the same and needs a model with structured-output
> (json_schema) support — e.g. OpenRouter (`https://openrouter.ai/api/v1`,
> `deepseek/deepseek-v4-flash`), OpenAI, or Ollama on the docker host via
> `http://host.docker.internal:11434/v1`.

---

## ✨ What it is

A-RAG-OG indexes documents into a single hybrid (dense + sparse) Qdrant store and exposes **retrieval as MCP tools**. The agent lives in **OpenWebUI** — its model decides which tool to call, searches in several rounds, reads promising chunks and grounds its answer. This service stays a thin, stateless retrieval layer.

| Part | Description |
|------|-------------|
| 🧩 **HybridChunker** | Docling token- *and* structure-aware chunking, heading path prepended (contextual) |
| 🏷️ **LLM Enrichment** | Per chunk: context, keywords, hypothetical questions, entities, dates |
| 🔄 **Hybrid Retrieval** | Dense (bge-m3) + Sparse (BM25), one Qdrant store |
| 📑 **Reranking** | bge-reranker-v2-m3 cross-encoder, `top_k` before/after per call |
| 🔌 **MCP Server** | FastMCP (streamable-http) + OpenWebUI JWT auth |
| 🤖 **Agent** | OpenWebUI's model orchestrates the tools (A-RAG style) |

```
index.py  →  MinIO + Docling HybridChunker → enrich → dense+sparse embed → Qdrant
server.py →  FastMCP tools → (dense|sparse) retrieve → cross-encoder rerank → snippets
OpenWebUI →  agent: search → read → reason → answer (cites chunk ids)
```

---

## 🧰 MCP Tools

| Tool | Purpose |
|------|---------|
| `keyword_and_semantic_search(query, top_k_before, top_k_after)` | **Default** — dense + sparse, fused by reranker |
| `semantic_search(query, top_k_before, top_k_after)` | Dense retrieval (by meaning) + rerank |
| `keyword_search(query, top_k_before, top_k_after)` | Sparse/BM25 retrieval (exact terms) + rerank |
| `filtered_search(query, keywords, entities, content_types, date_from, date_to, …)` | Hybrid (dense + sparse) + metadata filter + rerank |
| `find_related(chunk_ids, query, …)` | Associative multi-hop — more chunks sharing the hits' entities |
| `read_chunk(chunk_ids)` | Full content of chunks by id |
| `read_neighbors(chunk_ids, window)` | Full content of the chunks surrounding a hit (document order) |

Each search returns chunk ids + snippets; the agent reads full chunks with `read_chunk`
or pulls surrounding context with `read_neighbors`.

---

## 🛠️ Development

For working on the code, start everything except the containerized server and
run it on the host instead:

```bash
cp .env.example .env

docker compose up -d minio qdrant docling embedder reranker

uv run python index.py path/to/doc1.pdf     # index documents
uv run python server.py                     # run the MCP server
```

The backing services also bind to `127.0.0.1` — point the service URLs in
`.env` at them (`http://localhost:9000`, `http://localhost:6333`,
`http://localhost:5001`, `http://localhost:8001/v1`, `http://localhost:8002/v1`):

| Service | Port | Description |
|---------|------|-------------|
| Qdrant | 6333 / 6334 | Vector database (REST + dashboard) |
| MinIO | 9000 / 9001 | S3 API / console |
| Docling | 5001 | Document converter |
| Embedder | 8001 | vLLM dense embedding server (GPU) |
| Reranker | 8002 | vLLM rerank server (GPU) |

### Retrieval eval

`eval.py` measures retrieval quality against your own indexed documents: it
samples chunks from the live index, generates one question per chunk with the
enricher LLM, then scores each retrieval mode by whether the right chunk
comes back:

```bash
uv run python eval.py generate -n 50    # golden set from the live index
uv run python eval.py run               # Recall / MRR / MAP for dense, sparse, hybrid
```

---

## ⚙️ Configuration

Everything is configured through `.env` — start from
[.env.example](.env.example), which documents the non-obvious constraints
inline. It is the single source of truth; the compose file sets no overrides.
The defaults use the compose-network hostnames (`http://qdrant:6333`,
`http://docling:5001`, …); when running `index.py` / `server.py` on the host,
switch them to their `localhost` ports.

The exception is `MINIO_URL`: presigned source URLs embed this host, so it
must be reachable by the user's browser as well — use an address that works
for both the server and the browser (e.g. the docker host's IP), and publish
MinIO's port `9000` beyond `127.0.0.1` in the compose file if browsers
connect from other machines. `https://` turns on TLS for the connection.

---

## 📄 License

[MIT](LICENSE)
