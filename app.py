"""
Advanced Hybrid RAG – Gradio App

Replaces the FastAPI + FastMCP adapters with a single Gradio application that has:
  - Tab "Abfrage"    : full RAG query (answer + sources) and retrieve-only mode
  - Tab "Indexierung": document upload with streaming step-by-step progress

Gradio's built-in MCP server is enabled via `demo.launch(mcp_server=True)`.
The two MCP tools (rag_query, rag_retrieve) are wired through api_name= in the
event bindings.

Start with:
    uv run python app.py
or (after `uv sync`):
    advanced-rag
"""

from __future__ import annotations

import asyncio
import logging
import sys
from datetime import date
from pathlib import Path

import gradio as gr

from core.models.indexing import IndexCommand
from core.models.query import QueryInput
from core.models.retrieval import RetrievalInput
from core.runtime import RagRuntime
from core.services.query_engine import GenerationError
from core.services.retrieval_engine import NoDocumentsFoundError

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Lazy singleton runtime – initialised once inside Gradio's event loop
# ---------------------------------------------------------------------------

_runtime: RagRuntime | None = None
_lock: asyncio.Lock | None = None


async def ensure_runtime() -> RagRuntime:
    """Return the shared RagRuntime, initialising it on the first call."""
    global _runtime, _lock
    if _runtime is not None:
        return _runtime
    if _lock is None:
        _lock = asyncio.Lock()
    async with _lock:
        if _runtime is None:
            logger.info("Initialisiere RagRuntime …")
            r = RagRuntime()
            await r.startup()
            _runtime = r
            logger.info("RagRuntime bereit.")
    return _runtime


# ---------------------------------------------------------------------------
# Rendering helpers
# ---------------------------------------------------------------------------

_STEP_ICON: dict[str, str] = {
    "pending": "○",
    "running": "►",
    "done": "✓",
    "failed": "✗",
}


def _steps_html(task) -> str:
    rows = "".join(
        f"<tr><td>{_STEP_ICON.get(step.status, '?')}</td><td>{step.label}</td></tr>"
        for step in task.steps
    )
    return f"<table>{rows}</table>"


def _sources_html(sources) -> str:
    if not sources:
        return "<p><em>Keine Quellen gefunden.</em></p>"

    cards: list[str] = []
    for i, src in enumerate(sources, 1):
        meta = src.meta
        source_name = meta.get("source", "Unbekannt")
        title = meta.get("title") or source_name
        page = meta.get("page_start")
        section = meta.get("section_title")
        score = src.score
        snippet = src.content[:700] + ("…" if len(src.content) > 700 else "")

        badge_parts: list[str] = []
        if page is not None:
            badge_parts.append(f"Seite {page}")
        if section:
            badge_parts.append(str(section))
        if score is not None:
            badge_parts.append(f"Score {score:.3f}")
        badges = " · ".join(badge_parts)

        summary = f"<strong>Quelle {i}:</strong> {title}"
        if badges:
            summary += f" <small>({badges})</small>"
        cards.append(f"<details><summary>{summary}</summary><pre>{snippet}</pre></details>")
    return "\n".join(cards)


# ---------------------------------------------------------------------------
# Query handler
# ---------------------------------------------------------------------------


async def query_fn(
    query: str,
    top_k: float,
    date_from_str: str,
    date_to_str: str,
    use_hyde: bool,
    use_crag: bool,
    crag_threshold: float,
    crag_max_retries: float,
) -> tuple[str, str, str]:
    """Answer a question using the full RAG pipeline (retrieval + generation).

    Retrieves relevant document chunks via hybrid dense/sparse search with RRF
    fusion, AutoMergingRetriever parent-context swap, optional ColBERT pre-filter
    and cross-encoder reranking, then generates a grounded answer with an
    OpenAI-compatible LLM. Compound questions are automatically decomposed into
    sub-questions; metadata filters (date, source, classification) are inferred
    from the query text.

    Args:
        query (str): The question to answer.
        top_k (float): Number of source passages to return alongside the answer (1–20).
        date_from_str (str): Optional start date filter in ISO 8601 format (YYYY-MM-DD). Leave empty to skip.
        date_to_str (str): Optional end date filter in ISO 8601 format (YYYY-MM-DD). Leave empty to skip.
        use_hyde (bool): Enable HyDE – generate a hypothetical document to improve dense retrieval.
        use_crag (bool): Enable CRAG – retry retrieval with a reformulated query when confidence is low.
        crag_threshold (float): Minimum reranker score to consider retrieval confident (0.0–1.0).
        crag_max_retries (float): Maximum number of CRAG reformulation attempts (1–5).

    Returns:
        tuple[str, str, str]: Generated answer in Markdown, source passages as HTML, metadata info in Markdown.
    """
    if not query.strip():
        return "", "", ""

    runtime = await ensure_runtime()

    date_from = _parse_date(date_from_str)
    date_to = _parse_date(date_to_str)
    inp = QueryInput(query=query.strip(), top_k=int(top_k), date_from=date_from, date_to=date_to)

    try:
        result = await runtime.query_engine.query(
            inp,
            use_hyde=use_hyde,
            use_crag=use_crag,
            crag_threshold=crag_threshold,
            crag_max_retries=int(crag_max_retries),
        )
    except NoDocumentsFoundError:
        return "**Keine relevanten Dokumente gefunden.**", "", ""
    except GenerationError as exc:
        return f"**Fehler bei der Antwortgenerierung:** {exc}", "", ""
    except Exception as exc:
        logger.exception("Query fehlgeschlagen")
        return f"**Fehler:** {exc}", "", ""

    info_md = _build_info_md(result.sub_questions, result.is_compound,
                             result.extracted_filters, result.low_confidence)
    return result.answer, _sources_html(result.sources), info_md


# ---------------------------------------------------------------------------
# Retrieve-only handler (no generation; also exposed as MCP tool)
# ---------------------------------------------------------------------------


async def retrieve_fn(
    query: str,
    top_k: float,
    date_from_str: str,
    date_to_str: str,
    use_hyde: bool,
    use_crag: bool,
    crag_threshold: float,
    crag_max_retries: float,
) -> tuple[str, str]:
    """Retrieve relevant document passages without generating an answer.

    Runs the full hybrid retrieval pipeline (dense + sparse search, RRF fusion,
    AutoMergingRetriever, optional ColBERT pre-filter, cross-encoder reranker)
    and returns the top-k passages ranked by relevance. No LLM generation step
    is performed. Useful for inspecting index coverage or building custom
    downstream processing on top of the retrieved chunks.

    Args:
        query (str): The search query or question.
        top_k (float): Number of source passages to return (1–20).
        date_from_str (str): Optional start date filter in ISO 8601 format (YYYY-MM-DD). Leave empty to skip.
        date_to_str (str): Optional end date filter in ISO 8601 format (YYYY-MM-DD). Leave empty to skip.
        use_hyde (bool): Enable HyDE – generate a hypothetical document to improve dense retrieval.
        use_crag (bool): Enable CRAG – retry retrieval with a reformulated query when confidence is low.
        crag_threshold (float): Minimum reranker score to consider retrieval confident (0.0–1.0).
        crag_max_retries (float): Maximum number of CRAG reformulation attempts (1–5).

    Returns:
        tuple[str, str]: Source passages as HTML, metadata info in Markdown.
    """
    if not query.strip():
        return "", ""

    runtime = await ensure_runtime()

    date_from = _parse_date(date_from_str)
    date_to = _parse_date(date_to_str)
    inp = RetrievalInput(query=query.strip(), top_k=int(top_k), date_from=date_from, date_to=date_to)

    try:
        result = await runtime.retrieval_engine.retrieve(
            inp,
            use_hyde=use_hyde,
            use_crag=use_crag,
            crag_threshold=crag_threshold,
            crag_max_retries=int(crag_max_retries),
        )
    except NoDocumentsFoundError:
        return "<p><em>Keine relevanten Dokumente gefunden.</em></p>", ""
    except Exception as exc:
        logger.exception("Retrieval fehlgeschlagen")
        return f"<p><em>Fehler: {exc}</em></p>", ""

    info_md = _build_info_md(result.sub_questions, result.is_compound,
                             result.extracted_filters, result.low_confidence)
    return _sources_html(result.sources), info_md


# ---------------------------------------------------------------------------
# Indexing handler – async generator for streaming progress
# ---------------------------------------------------------------------------


async def index_fn(
    file_obj: str | None,
    use_raptor: bool,
    progress: gr.Progress = gr.Progress(track_tqdm=True),
):
    """Index a document into the vector store.

    Uploads the file to MinIO, converts it via docling-serve (PDF/DOCX/PPTX/…
    to Markdown), splits into parent and child chunks, enriches each chunk with
    an LLM-generated context prefix, summary, keywords, classification and named
    entities, computes dense (BAAI/bge-m3) and sparse (BM42) embeddings, and
    writes everything to Qdrant. Progress is streamed step by step.

    Args:
        file_obj (str | None): Local path to the uploaded file provided by Gradio.
        use_raptor (bool): Enable RAPTOR – build hierarchical section and document summary chunks.

    Yields:
        tuple[str, str]: Status message in Markdown and step progress table as HTML.
    """
    if file_obj is None:
        yield "Keine Datei ausgewählt.", ""
        return

    # Gradio 5 passes a str path; older versions may pass a NamedString
    file_path = Path(file_obj) if isinstance(file_obj, str) else Path(
        file_obj.name if hasattr(file_obj, "name") else str(file_obj)
    )

    if not file_path.exists():
        yield "Datei nicht gefunden.", ""
        return

    try:
        file_bytes = file_path.read_bytes()
    except OSError as exc:
        yield f"Lesefehler: {exc}", ""
        return

    runtime = await ensure_runtime()
    command = IndexCommand(file_name=file_path.name, file_bytes=file_bytes)

    try:
        task_info = runtime.indexing_service.enqueue(command, use_raptor=use_raptor)
    except OverflowError:
        yield "⚠️ Indexierungs-Warteschlange ist voll. Bitte später erneut versuchen.", ""
        return

    # Launch the indexing pipeline in the background (same event loop)
    asyncio.create_task(runtime.indexing_service.run(task_info.task_id))

    # Stream progress updates until the task finishes
    while True:
        task = runtime.indexing_service.get_task(task_info.task_id)
        if task is None:
            yield "Fehler: Task nicht gefunden.", ""
            return

        n = len(task.steps)
        cur = task.current_step_index
        if n > 0 and cur >= 0:
            progress((cur + 1) / n, desc=task.steps[cur].label if cur < n else "Fertig")

        steps = _steps_html(task)

        if task.status == "done":
            r = task.result
            yield f"✓ Fertig – {r.indexed} Chunks aus **{r.source}** indiziert.", steps
            return

        if task.status == "failed":
            yield f"✗ Fehler: {task.error}", steps
            return

        current_label = task.steps[cur].label if 0 <= cur < n else "Starte…"
        yield f"► {current_label}", steps
        await asyncio.sleep(0.75)


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------


def _parse_date(s: str) -> date | None:
    s = s.strip()
    if not s:
        return None
    try:
        return date.fromisoformat(s)
    except ValueError:
        return None


def _build_info_md(
    sub_questions: list[str],
    is_compound: bool,
    extracted_filters: dict | None,
    low_confidence: bool,
) -> str:
    parts: list[str] = []
    if is_compound and sub_questions:
        parts.append("**Teilfragen:**\n" + "\n".join(f"- {q}" for q in sub_questions))
    if extracted_filters:
        kv = ", ".join(f"`{k}={v}`" for k, v in extracted_filters.items())
        parts.append(f"**Erkannte Filter:** {kv}")
    if low_confidence:
        parts.append("⚠️ **Niedrige Konfidenz** – die Antwort könnte unvollständig sein.")
    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------


def build_demo() -> gr.Blocks:
    from core.config import get_settings
    s = get_settings()

    with gr.Blocks(title="Advanced Hybrid RAG", theme=gr.themes.Citrus()) as demo:
        gr.Markdown("# Advanced Hybrid RAG")

        with gr.Tabs():
            # ----------------------------------------------------------------
            # Tab 1: Abfrage
            # ----------------------------------------------------------------
            with gr.Tab("Abfrage"):
                query_input = gr.Textbox(
                    label="Frage",
                    placeholder="Was möchten Sie wissen?",
                    lines=3,
                )
                with gr.Row():
                    top_k = gr.Slider(minimum=1, maximum=20, value=5, step=1,
                                      label="Anzahl Quellen (top-k)")
                with gr.Row():
                    date_from = gr.Textbox(label="Von (JJJJ-MM-TT)",
                                           placeholder="2024-01-01", scale=1)
                    date_to = gr.Textbox(label="Bis (JJJJ-MM-TT)",
                                         placeholder="2024-12-31", scale=1)

                with gr.Accordion("Erweiterte Einstellungen", open=False):
                    gr.Markdown("**HyDE – Hypothetical Document Embedding**")
                    use_hyde = gr.Checkbox(
                        label="HyDE aktivieren",
                        value=s.hyde_enabled,
                        info="Generiert ein hypothetisches Dokument zur Verbesserung des Dense-Retrievals. "
                             "Benötigt einen LLM-Aufruf pro Anfrage.",
                    )
                    gr.Markdown("**CRAG – Korrigierendes Retrieval**")
                    use_crag = gr.Checkbox(
                        label="CRAG aktivieren",
                        value=s.crag_enabled,
                        info="Wiederholt das Retrieval mit einer reformulierten Anfrage, "
                             "wenn der Reranker-Score unter dem Schwellenwert liegt.",
                    )
                    with gr.Row():
                        crag_threshold = gr.Slider(
                            minimum=0.0, maximum=1.0, value=s.crag_score_threshold,
                            step=0.05, label="CRAG Score-Schwelle",
                            info="Mindestscore des Rerankers, ab dem das Retrieval als ausreichend gilt.",
                        )
                        crag_max_retries = gr.Slider(
                            minimum=1, maximum=5, value=s.crag_max_retries,
                            step=1, label="CRAG Max. Wiederholungen",
                        )

                with gr.Row():
                    query_btn = gr.Button("Abfragen (mit Antwort)", variant="primary")
                    retrieve_btn = gr.Button("Nur Quellen abrufen")

                answer_out = gr.Markdown(label="Antwort")
                info_out = gr.Markdown(label="Zusatzinfo")
                sources_out = gr.HTML(label="Quellen")

                _query_inputs = [query_input, top_k, date_from, date_to,
                                 use_hyde, use_crag, crag_threshold, crag_max_retries]

                query_btn.click(
                    fn=query_fn,
                    inputs=_query_inputs,
                    outputs=[answer_out, sources_out, info_out],
                    api_name="rag_query",
                )
                retrieve_btn.click(
                    fn=retrieve_fn,
                    inputs=_query_inputs,
                    outputs=[sources_out, info_out],
                    api_name="rag_retrieve",
                )

            # ----------------------------------------------------------------
            # Tab 2: Indexierung
            # ----------------------------------------------------------------
            with gr.Tab("Indexierung"):
                file_upload = gr.File(
                    label="Dokument hochladen",
                    file_types=[
                        ".pdf", ".docx", ".doc", ".pptx", ".xlsx",
                        ".txt", ".md", ".html", ".htm",
                    ],
                )

                with gr.Accordion("Erweiterte Einstellungen", open=False):
                    use_raptor = gr.Checkbox(
                        label="RAPTOR – Hierarchische Zusammenfassungen",
                        value=s.raptor_enabled,
                        info="Erzeugt zusätzliche Abschnitts- und Dokumentzusammenfassungen als Chunks "
                             "(erhöht Indexierungszeit und benötigt LLM-Aufrufe).",
                    )

                index_btn = gr.Button("Indizieren", variant="primary")
                status_out = gr.Markdown(label="Status")
                steps_out = gr.HTML(label="Fortschritt")

                index_btn.click(
                    fn=index_fn,
                    inputs=[file_upload, use_raptor],
                    outputs=[status_out, steps_out],
                )

    return demo


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    from core.config import get_settings

    settings = get_settings()

    demo = build_demo()
    demo.queue()
    demo.launch(
        server_name=settings.app_host,
        server_port=settings.app_port,
        mcp_server=True,
    )


if __name__ == "__main__":
    main()
