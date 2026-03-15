# Advanced Hybrid RAG

Willkommen im Chainlit-Frontend fuer dieses RAG-System.

Dieses UI ist bewusst duenn:
- Uploads und Task-Polling laufen ueber die FastAPI unter `http://localhost:8000`
- Fragen laufen ueber den lokalen MCP-Server aus `../backend/main_mcp.py`
- Quellen werden direkt aus den indexierten Dokumenten angezeigt

## Was du hier machen kannst

- Dokumente hochladen und indexieren
- Indexing-Schritte verfolgen
- Fragen an den Dokumentbestand stellen
- Quellen, Seiten und Originaldokumente pruefen

## Typischer Ablauf

1. Backend-Infrastruktur starten
2. API starten
3. Chainlit starten
4. Optional ein Dokument hochladen
5. Danach Fragen stellen

## Gute Einstiegsfragen

- Fasse das zuletzt indexierte Dokument knapp zusammen.
- Welche drei wichtigsten Aussagen stehen im Dokument?
- Welche Risiken, offenen Punkte oder Konflikte nennt das Dokument?

## Hinweise

- Wenn keine passenden Quellen gefunden werden, zuerst ein Dokument hochladen.
- Wenn Query-Aufrufe fehlschlagen, pruefen ob `../backend/.venv/bin/python` und `../backend/main_mcp.py` vorhanden sind.
- Die eigentliche Backend-Doku liegt in `../backend/README.md`.
