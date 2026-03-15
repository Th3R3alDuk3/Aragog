<p align="center">
  <img src="logo.png" alt="Advanced Hybrid RAG logo" width="220">
</p>

# Advanced Hybrid RAG

Monorepo fuer ein hybrides RAG-System mit API, MCP-Server und Chainlit-UI.

Der Kern liegt in [`backend/`](backend/README.md):
- Hybrid Retrieval mit Dense + Sparse + Reranking
- FastAPI fuer Upload, Query, Tasks und Evaluation
- FastMCP fuer `rag_query` und `rag_retrieve`
- Qdrant, MinIO und docling-serve als Infrastruktur

Die UI liegt in [`frontend/`](frontend/README.md):
- Chainlit-Oberflaeche fuer Upload, Chat und Quellenansicht

## Repo-Start

```text
RAG/
├── backend/   Python-Backend mit Runtime, API, MCP und Architektur-Doku
├── frontend/  Chainlit-UI fuer das Backend
└── test/      lokale Testdateien und Beispielfiles
```

Wenn du nur einen Einstiegspunkt brauchst:
- Backend-Doku: [backend/README.md](backend/README.md)
- Backend-Architektur: [backend/docs/architecture.md](backend/docs/architecture.md)
- Frontend-Doku: [frontend/README.md](frontend/README.md)

## Schnellstart

### 1. Infrastruktur starten

```bash
cd backend
docker compose up -d
```

### 2. Backend API starten

```bash
cd backend
uv sync
uv run python main_api.py
```

Danach ist die API unter `http://localhost:8000` erreichbar.

### 3. Frontend starten

```bash
cd frontend
uv sync
uv run chainlit run app.py -w --port 8501
```

Danach ist die UI unter `http://localhost:8501` erreichbar.

### 4. Optional: MCP-Server starten

```bash
cd backend
uv sync
uv run python main_mcp.py
```

## Was wo liegt

### Backend

[`backend/`](backend/README.md) ist kein generischer Service-Ordner, sondern der eigentliche Systemkern:
- `core/` enthaelt die Fachlogik, Pipelines, Services und Storage-Anbindung
- `adapters/api/` enthaelt die HTTP-Schicht
- `adapters/mcp/` enthaelt den MCP-Adapter
- `docs/` enthaelt Architektur- und Stil-Doku

### Frontend

[`frontend/`](frontend/README.md) ist eine duenne Chainlit-Anwendung auf dem Backend:
- Dokumente hochladen
- Indexing-Status verfolgen
- Fragen stellen
- Quellen anzeigen

## Arbeitsmodus

Typischer lokaler Ablauf:
1. `backend/docker compose up -d`
2. `backend/main_api.py` starten
3. optional `backend/main_mcp.py` starten
4. `frontend/app.py` starten
5. Dokument hochladen und Queries testen

## Weiterfuehrend

- Backend-Setup und Endpunkte: [backend/README.md](backend/README.md)
- Architekturentscheidungen: [backend/docs/architecture.md](backend/docs/architecture.md)
- Coding-Standard: [backend/docs/code_style.md](backend/docs/code_style.md)
