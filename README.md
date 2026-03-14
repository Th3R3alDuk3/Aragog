# RAG Monorepo

Das bisherige Python-Backend liegt jetzt vollständig in [`backend/`](backend/README.md).  
[`frontend/`](frontend/README.md) ist als eigener Ordner für dein zukünftiges UI angelegt.

## Struktur

- `backend/` enthält die bestehende FastAPI-, Haystack- und Qdrant-Anwendung inklusive Doku, `.env`, `docker-compose.yml` und Python-Projektdateien.
- `frontend/` ist ein leerer Startpunkt für das Frontend.

## Backend starten

```bash
cd backend
docker compose up -d
uv sync
uv run python main.py
```

Weitere Details stehen in [`backend/README.md`](backend/README.md).

## Frontend starten

```bash
cd frontend
cp .env.example .env
uv sync
uv run chainlit run app.py -w --port 8501
```

Die Chainlit-Oberflaeche spricht standardmaessig mit `http://localhost:8000`.
