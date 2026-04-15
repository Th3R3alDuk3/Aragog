# TODOs — RAG Improvement Backlog

Abgeleitet aus Architektur-Review (2026-04-15). Bewertet für Einsatz mit vielen unterschiedlichen Dokumenttypen.

---

## Priorität 1 — Kritisch für Dokumentenvielfalt

### Vision / Image-Captioning
- Docling liefert bereits Bounding-Boxes für Bilder und Diagramme
- Lokales VLM (z.B. LLaVA, Phi-Vision) während der Indexierung einbinden
- Beschreibungen als eigenen Text-Chunk (`chunk_type: "image"`) indexieren
- Ohne das sind alle scan-lastigen oder grafiklastigen PDFs blind

### Table-QA Pipeline
- Tabellen separat über Docling `TableElement` extrahieren
- Als strukturierten Text (Markdown + row/col-Beschreibung) mit `chunk_type: "table"` speichern
- Bei Queries auf numerische/tabellarische Daten diesen Typ priorisieren
- Aktuell werden Tabellen nur als Markdown-Text behandelt — komplexe Strukturen gehen verloren

### RAPTOR — echtes Clustering aktivieren
- `RAPTOR_ENABLED` ist aktuell `false` und die Impl. macht keine echte Cluster-Hierarchie
- Für 1000+ Dokumente werden Topic-Cluster-Summaries benötigt
- "Überblicksqueries" (Vergleich über viele Docs) funktionieren ohne das schlecht
- Echtes hierarchisches Clustering implementieren (z.B. UMAP + GMM wie im Paper)

---

## Priorität 2 — Wichtig für Produktion

### Document-Type-Routing
- Aktuell laufen alle Dokumenttypen durch dieselbe Indexierungs-Pipeline
- Unterschiedliche Strategien je Typ:
  - **Code-Files**: kürzere Chunks, keine NER, AST-aware Splitting
  - **Legal-Docs**: längere Parents, Gesetzes-NER, Paragraphen-Struktur erhalten
  - **Protokolle**: Zeitstempel-Extraktion, Speaker-Detektion
  - **Tabellenkalkulationen**: zellenbasierte Chunks, Formel-Extraktion
- `DocumentAnalyzer` erkennt `document_type` bereits — darauf aufbauen

### Query Result Caching
- Kein Cache für wiederholte identische Queries (`❌` in architecture.md)
- Einfacher `{query_hash → result}`-Cache mit TTL implementieren
- Bei großem Korpus (10k+ Chunks) erhebliche Latenz- und Kostenersparnis
- Redis oder In-Memory-LRU als Backend

### Self-RAG
- Nach der Generierung bewertet das LLM selbst, ob die Antwort die Frage beantwortet
- Bei unzureichender Antwort → weiterer Retrieval-Pass (ähnlich CRAG, aber post-generation)
- Reduziert Halluzinationen bei schwierigen/mehrdeutigen Queries

---

## Priorität 3 — Nice-to-Have

### Graph RAG
- Entitäts-Beziehungsgraph über indexierte Dokumente aufbauen (z.B. NetworkX + Qdrant oder Neo4j)
- Lohnt sich ab ~500 zusammenhängenden Dokumenten wo Entitäts-Beziehungen wichtig sind
- NER ist bereits implementiert (8 Typen) — Grundlage für Graph-Extraktion liegt vor

### Dedicated Worker Process
- Indexierung läuft aktuell in-process (`asyncio.create_task`) — kein separater Worker (`❌` in architecture.md)
- Für große Korpora: Celery / ARQ + Redis-Queue als saubere Entkopplung
- Verhindert, dass Indexierungs-Last die Query-Latenz beeinflusst

---

## Bereits gut gelöst (kein Handlungsbedarf)

- Hybrid Dense+Sparse+RRF mit HyDE
- Contextual Retrieval (context_prefix in beiden Embeddings)
- Reranking-Kaskade (ColBERT → Cross-Encoder)
- Parent-Child + AutoMerging
- NER + Metadata-Filter + QueryAnalyzer
- CRAG + RAGAS
- Multilingual (DE/EN + 100+ Sprachen via bge-m3 + BM42)
