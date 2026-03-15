# Chainlit Frontend

Schlankes Chainlit-Frontend fuer das RAG-Backend in `../backend`.

## Features

- Dokument-Upload ueber `POST /documents/index`
- Polling des Indexing-Status ueber `GET /tasks/{task_id}`
- Chat gegen das MCP-Tool `rag_query`
- Quellenanzeige in der Sidebar
- Einfache Session-Settings fuer `top_k` und die RAG-Taskliste

Uploads und Task-Polling laufen weiter ueber die HTTP-API. Die eigentliche
Fragebeantwortung startet das Frontend lokal ueber `../backend/main_mcp.py`.

## Setup

```bash
cd frontend
uv sync
```

## Start

```bash
cd frontend
uv run chainlit run app.py -w --port 8501
```

Standardmaessig erwartet das Frontend das Backend unter `http://localhost:8000`.
Fuer Queries wird ausserdem das Backend-MCP-Skript `../backend/main_mcp.py`
mit dem Python aus `../backend/.venv/bin/python` gestartet.

## Wichtige Variablen

- `RAG_API_URL` - Basis-URL des FastAPI-Backends, z. B. `http://localhost:8000`
- `RAG_MCP_SCRIPT` - Pfad zum MCP-Startskript, Standard: `../backend/main_mcp.py`
- `RAG_MCP_PYTHON` - Python fuer den MCP-Subprozess, Standard: `../backend/.venv/bin/python`
- `RAG_MCP_TIMEOUT` - Timeout fuer MCP-Initialisierung und Tool-Calls in Sekunden

## Ablauf

1. Backend starten
2. Frontend starten
3. Im Chat optional zuerst auf `Dokument hochladen` klicken
4. Danach Fragen an das RAG stellen; die Antwort laeuft ueber MCP
