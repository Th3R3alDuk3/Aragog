# Chainlit Frontend

Schlankes Chainlit-Frontend fuer das RAG-Backend in `../backend`.

## Features

- Dokument-Upload ueber `POST /documents/index`
- Polling des Indexing-Status ueber `GET /tasks/{task_id}`
- Chat gegen `POST /query`
- Optionales Antwort-Streaming ueber `POST /query/stream`
- Quellenanzeige in der Sidebar
- Einfache Session-Settings fuer `top_k` und Streaming

## Setup

```bash
cd frontend
cp .env.example .env
uv sync
```

## Start

```bash
cd frontend
uv run chainlit run app.py -w --port 8501
```

Standardmaessig erwartet das Frontend das Backend unter `http://localhost:8000`.

## Wichtige Variablen

- `RAG_API_URL` - Basis-URL des FastAPI-Backends

## Ablauf

1. Backend starten
2. Frontend starten
3. Im Chat optional zuerst auf `Dokument hochladen` klicken
4. Danach Fragen an das RAG stellen
