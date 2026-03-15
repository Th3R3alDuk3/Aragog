import asyncio
import json
import os
from pathlib import Path
from typing import Any
from urllib.parse import quote

import chainlit as cl
import httpx
from chainlit.input_widget import Slider, Switch
from fastmcp import Client
from fastmcp.client.transports.stdio import PythonStdioTransport


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


API_URL = os.getenv("RAG_API_URL", "http://localhost:8000").rstrip("/")
HTTP_TIMEOUT = httpx.Timeout(120.0, connect=10.0)
QUERY_SETTINGS_KEY = "query_settings"
DEFAULT_QUERY_SETTINGS = {"top_k": 5, "show_rag_tasklist": False}
TASK_LIST_CLOSE_DELAY = 1.5
FRONTEND_DIR = Path(__file__).resolve().parent
MCP_SCRIPT = Path(
    os.getenv("RAG_MCP_SCRIPT", str(FRONTEND_DIR.parent / "backend" / "main_mcp.py"))
).resolve()
MCP_PYTHON = Path(
    os.getenv("RAG_MCP_PYTHON", str(FRONTEND_DIR.parent / "backend" / ".venv" / "bin" / "python"))
).resolve()
MCP_CLIENT_KEY = "mcp_client"
MCP_TIMEOUT = float(os.getenv("RAG_MCP_TIMEOUT", "180"))
UPLOAD_COMMANDS = {"/upload", "upload", "datei hochladen"}
STARTERS = [
    ("Kurzfassung", "Fasse das zuletzt indexierte Dokument knapp zusammen."),
    ("Kernpunkte", "Welche drei wichtigsten Aussagen stehen im Dokument?"),
    ("Risiken", "Welche Risiken, offenen Punkte oder Konflikte nennt das Dokument?"),
]
RAG_TASKS = [
    "Anfrage senden",
    "RAG-Antwort abrufen",
    "Analysehinweise anzeigen",
    "Quellen anzeigen",
]
UPLOAD_ACCEPT = {
    "application/pdf": [".pdf"],
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document": [".docx"],
    "application/vnd.openxmlformats-officedocument.presentationml.presentation": [".pptx"],
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet": [".xlsx"],
    "text/plain": [".txt"],
    "text/markdown": [".md"],
    "text/html": [".html"],
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _query_settings() -> dict[str, Any]:
    settings = cl.user_session.get(QUERY_SETTINGS_KEY)
    return settings or dict(DEFAULT_QUERY_SETTINGS)


def _source_label(index: int, source: dict[str, Any]) -> str:
    meta = source.get("meta") or {}
    name = meta.get("source") or meta.get("title") or f"Quelle {index}"
    section = meta.get("section_title")
    page_start = meta.get("page_start")
    page_end = meta.get("page_end")
    page_label = None

    if isinstance(page_start, int) and page_start > 0:
        if isinstance(page_end, int) and page_end > page_start:
            page_label = f"Seiten {page_start}-{page_end}"
        else:
            page_label = f"Seite {page_start}"

    parts = [f"{index}. {name}"]
    if section:
        parts.append(section)
    if page_label:
        parts.append(page_label)

    return " | ".join(parts)


def _error_text(response: httpx.Response) -> str:
    try:
        payload = response.json()
    except Exception:
        return response.text or f"HTTP {response.status_code}"
    detail = payload.get("detail")
    if isinstance(detail, str):
        return detail
    return json.dumps(payload, ensure_ascii=True)


def _result_payload(result: Any) -> dict[str, Any]:
    if isinstance(result, dict):
        return result
    data = getattr(result, "data", None)
    if isinstance(data, dict):
        return data
    structured = getattr(result, "structured_content", None)
    if isinstance(structured, dict):
        return structured
    if hasattr(result, "model_dump"):
        payload = result.model_dump()
        if isinstance(payload, dict):
            return payload
    raise RuntimeError("MCP tool returned an unexpected payload.")


def _source_download_url(source: dict[str, Any]) -> str | None:
    meta = source.get("meta") or {}
    direct = meta.get("download_url") or meta.get("minio_url")
    if isinstance(direct, str) and direct:
        return direct

    minio_key = meta.get("minio_key")
    if isinstance(minio_key, str) and minio_key:
        return f"{API_URL}/documents/download/{quote(minio_key, safe='')}"

    return None


def _chainlit_task_status(status: str) -> cl.TaskStatus:
    if status == "running":
        return cl.TaskStatus.RUNNING
    if status == "done":
        return cl.TaskStatus.DONE
    if status == "failed":
        return cl.TaskStatus.FAILED
    return cl.TaskStatus.READY


def _current_step_label(task_data: dict[str, Any]) -> str | None:
    steps = task_data.get("steps") or []
    current_step_index = task_data.get("current_step_index", -1)

    if 0 <= current_step_index < len(steps):
        return steps[current_step_index].get("label")

    return None


async def _chat_settings() -> dict[str, Any]:
    return await cl.ChatSettings(
        [
            Slider(
                id="top_k",
                label="Quellen",
                initial=5,
                min=1,
                max=10,
                step=1,
                description="Wie viele Quellen maximal angezeigt werden.",
            ),
            Switch(
                id="show_rag_tasklist",
                label="RAG-Taskliste",
                initial=False,
                description="Zeigt die groben Phasen der RAG-Abfrage als Taskliste.",
            ),
        ]
    ).send()


async def _request_json(method: str, path: str, **kwargs: Any) -> dict[str, Any]:
    async with httpx.AsyncClient(base_url=API_URL, timeout=HTTP_TIMEOUT) as client:
        response = await client.request(method, path, **kwargs)
    if response.is_success:
        return response.json()
    raise RuntimeError(_error_text(response))


async def _mcp_client() -> Client:
    client = cl.user_session.get(MCP_CLIENT_KEY)
    if client is not None and client.is_connected():
        return client

    if not MCP_SCRIPT.is_file():
        raise RuntimeError(f"MCP script not found: {MCP_SCRIPT}")
    if not MCP_PYTHON.is_file():
        raise RuntimeError(f"MCP Python not found: {MCP_PYTHON}. Run 'cd backend && uv sync'.")

    client = Client(
        PythonStdioTransport(
            script_path=MCP_SCRIPT,
            cwd=str(MCP_SCRIPT.parent),
            python_cmd=str(MCP_PYTHON),
            keep_alive=True,
        ),
        timeout=MCP_TIMEOUT,
        init_timeout=MCP_TIMEOUT,
    )
    await client.__aenter__()
    cl.user_session.set(MCP_CLIENT_KEY, client)
    return client


async def _close_mcp_client() -> None:
    client = cl.user_session.get(MCP_CLIENT_KEY)
    if client is None:
        return
    if client.is_connected():
        await client.__aexit__(None, None, None)
    cl.user_session.set(MCP_CLIENT_KEY, None)


async def _new_task_list(status: str, titles: list[str]) -> cl.TaskList:
    task_list = cl.TaskList(status=status)

    for title in titles:
        await task_list.add_task(cl.Task(title=title, status=cl.TaskStatus.READY))

    await task_list.send()
    return task_list


async def _sync_indexing_task_list(task_list: cl.TaskList, task_data: dict[str, Any]) -> None:
    steps = task_data.get("steps") or []
    task_list.tasks = [
        cl.Task(
            title=step.get("label", "Unbekannter Schritt"),
            status=_chainlit_task_status(step.get("status", "pending")),
        )
        for step in steps
    ]

    state = task_data.get("status") or "pending"
    current_label = _current_step_label(task_data)

    if state == "done":
        task_list.status = "Indexierung abgeschlossen"
    elif state == "failed":
        task_list.status = "Indexierung fehlgeschlagen"
    elif current_label:
        task_list.status = f"Indexierung laeuft: {current_label}"
    else:
        task_list.status = "Indexierung wartet"

    await task_list.update()


async def _update_task_list(
    task_list: cl.TaskList,
    *,
    status: str,
    running: int | None = None,
    failed: int | None = None,
    done_through: int = -1,
) -> None:
    for index, task in enumerate(task_list.tasks):
        if failed is not None and index == failed:
            task.status = cl.TaskStatus.FAILED
        elif index <= done_through:
            task.status = cl.TaskStatus.DONE
        elif running is not None and index == running:
            task.status = cl.TaskStatus.RUNNING
        else:
            task.status = cl.TaskStatus.READY

    task_list.status = status
    await task_list.update()


async def _close_task_list(task_list: cl.TaskList | None) -> None:
    if not task_list:
        return
    await asyncio.sleep(TASK_LIST_CLOSE_DELAY)
    await task_list.remove()


# ---------------------------------------------------------------------------
# Indexing
# ---------------------------------------------------------------------------


async def _upload_document() -> None:
    files = await cl.AskFileMessage(
        content="Lade ein Dokument fuer das RAG hoch.",
        accept=UPLOAD_ACCEPT,
        max_size_mb=50,
        max_files=1,
        timeout=180,
    ).send()
    if not files:
        await cl.Message(content="Kein Dokument ausgewaehlt.").send()
        return

    upload = files[0]
    file_path = upload.get("path") if isinstance(upload, dict) else getattr(upload, "path", None)
    file_name = upload.get("name") if isinstance(upload, dict) else getattr(upload, "name", None)
    file_name = file_name or Path(str(file_path)).name

    if not file_path:
        await cl.Message(content="Die hochgeladene Datei konnte nicht gelesen werden.").send()
        return

    status = await cl.Message(content=f"Indexiere `{file_name}` ...").send()
    task_list: cl.TaskList | None = None

    try:
        async with cl.Step(name="Indexing", type="tool", show_input="json") as step:
            step.input = {"file": file_name, "api_url": f"{API_URL}/documents/index"}
            file_bytes = Path(str(file_path)).read_bytes()
            task = await _request_json(
                "POST",
                "/documents/index",
                files={"file": (file_name, file_bytes)},
            )
            step.output = task

        task_data = await _request_json("GET", f"/tasks/{task['task_id']}")
        task_list = await _new_task_list(
            status=f"Indexierung von {file_name}",
            titles=[item.get("label", "Unbekannter Schritt") for item in task_data.get("steps") or []],
        )
        await _sync_indexing_task_list(task_list, task_data)
        await _poll_task(task["task_id"], file_name, status, task_list, initial_task=task_data)
    except Exception as error:
        status.content = f"Upload fehlgeschlagen: {error}"
        await status.update()
        if task_list:
            task_list.status = "Indexierung fehlgeschlagen"
            await task_list.update()


async def _poll_task(
    task_id: str,
    file_name: str,
    status_message: cl.Message,
    task_list: cl.TaskList,
    initial_task: dict[str, Any] | None = None,
) -> None:
    last_step = ""
    task = initial_task

    while True:
        if task is None:
            task = await _request_json("GET", f"/tasks/{task_id}")

        step = task.get("step") or "pending"
        state = task.get("status") or "pending"
        current_label = _current_step_label(task) or step
        await _sync_indexing_task_list(task_list, task)

        if step != last_step or state in {"done", "failed"}:
            status_message.content = (
                f"Indexierung von `{file_name}`\n\n"
                f"- Status: `{state}`\n"
                f"- Schritt: `{current_label}`"
            )
            await status_message.update()
            last_step = step

        if state == "done":
            result = task.get("result") or {}
            indexed = result.get("indexed", 0)
            status_message.content = (
                f"Indexierung abgeschlossen fuer `{file_name}`.\n\n"
                f"- Chunks geschrieben: `{indexed}`"
            )
            await status_message.update()
            await _close_task_list(task_list)
            return

        if state == "failed":
            error = task.get("error") or "Unbekannter Fehler"
            status_message.content = f"Indexierung fehlgeschlagen: {error}"
            await status_message.update()
            await _close_task_list(task_list)
            return

        await asyncio.sleep(1.5)
        task = None


# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------


async def _send_sources(sources: list[dict[str, Any]]) -> None:
    if not sources:
        await cl.ElementSidebar.set_elements([])
        return

    cards: list[str] = []
    elements: list[cl.File | cl.Pdf] = []

    for index, source in enumerate(sources, start=1):
        meta = source.get("meta") or {}
        label = _source_label(index, source)
        name = meta.get("source") or meta.get("title") or f"Quelle {index}"
        section = meta.get("section_title")
        chunk_type = meta.get("chunk_type")
        score = source.get("score")
        page_start = meta.get("page_start")
        page_end = meta.get("page_end")
        download_url = _source_download_url(source)
        excerpt = " ".join((source.get("content") or "").split())

        if len(excerpt) > 360:
            excerpt = excerpt[:357].rstrip() + "..."

        details: list[str] = []
        if section:
            details.append(f"Abschnitt: `{section}`")
        if isinstance(page_start, int) and page_start > 0:
            if isinstance(page_end, int) and page_end > page_start:
                details.append(f"Seiten: `{page_start}-{page_end}`")
            else:
                details.append(f"Seite: `{page_start}`")
        if chunk_type:
            details.append(f"Typ: `{chunk_type}`")
        if isinstance(score, (int, float)):
            details.append(f"Score: `{score:.3f}`")

        card_lines = [f"### {index}. {name}"]
        if details:
            card_lines.append(" | ".join(details))
        if download_url:
            card_lines.append(f"[Original oeffnen]({download_url})")
        if excerpt:
            card_lines.append(f"> {excerpt}")

        cards.append("\n\n".join(card_lines))

        if not download_url:
            continue

        element_name = f"{label} · Original"
        suffix = Path(name).suffix.lower()

        if suffix == ".pdf":
            elements.append(
                cl.Pdf(
                    name=element_name,
                    url=download_url,
                    display="side",
                    size="large",
                    page=page_start if isinstance(page_start, int) and page_start > 0 else None,
                )
            )
        else:
            elements.append(
                cl.File(
                    name=element_name,
                    url=download_url,
                    display="side",
                    size="large",
                )
            )
    await cl.ElementSidebar.set_title("Originaldokumente")
    await cl.ElementSidebar.set_elements(elements)

    await cl.Message(
        author="Sources",
        content="**Verwendete Quellen**\n\n" + "\n\n".join(cards),
    ).send()


async def _run_query(query: str) -> None:
    settings = _query_settings()
    payload = {"query": query, "top_k": int(settings["top_k"])}
    rag_task_list: cl.TaskList | None = None

    if settings.get("show_rag_tasklist", False):
        rag_task_list = await _new_task_list(
            status="RAG-Abfrage wird vorbereitet",
            titles=RAG_TASKS,
        )
        await _update_task_list(
            rag_task_list,
            status="Anfrage wird gesendet",
            running=0,
        )

    try:
        async with cl.Step(name="RAG query", type="tool", show_input="json") as step:
            step.input = payload

            if rag_task_list:
                await _update_task_list(
                    rag_task_list,
                    status="RAG-Antwort wird abgerufen",
                    running=1,
                    done_through=0,
                )

            client = await _mcp_client()
            result = _result_payload(await client.call_tool("rag_query", payload))
            await cl.Message(content=result.get("answer", "")).send()

            step.output = {
                "is_compound": result.get("is_compound", False),
                "sub_questions": result.get("sub_questions", []),
                "low_confidence": result.get("low_confidence", False),
                "sources": len(result.get("sources") or []),
            }

        if rag_task_list:
            await _update_task_list(
                rag_task_list,
                status="Analysehinweise werden angezeigt",
                running=2,
                done_through=1,
            )

        if result.get("sub_questions"):
            questions = "\n".join(f"- {question}" for question in result["sub_questions"])
            await cl.Message(author="Analysis", content=f"**Teilfragen**\n{questions}").send()

        if result.get("extracted_filters"):
            filters = json.dumps(result["extracted_filters"], ensure_ascii=True, indent=2)
            await cl.Message(
                author="Analysis",
                content=f"**Extrahierte Filter**\n```json\n{filters}\n```",
            ).send()

        if result.get("low_confidence"):
            await cl.Message(
                author="Signal",
                content="Die Antwort ist als niedriges Retrieval-Vertrauen markiert. Bitte Quellen pruefen.",
            ).send()

        if rag_task_list:
            await _update_task_list(
                rag_task_list,
                status="Quellen werden angezeigt",
                running=3,
                done_through=2,
            )

        await _send_sources(result.get("sources") or [])

        if rag_task_list:
            await _update_task_list(
                rag_task_list,
                status="RAG-Abfrage abgeschlossen",
                done_through=len(RAG_TASKS) - 1,
            )
            await _close_task_list(rag_task_list)
    except Exception:
        if rag_task_list:
            await _update_task_list(
                rag_task_list,
                status="RAG-Abfrage fehlgeschlagen",
                failed=1,
                done_through=0,
            )
            await _close_task_list(rag_task_list)
        raise


# ---------------------------------------------------------------------------
# Chainlit hooks
# ---------------------------------------------------------------------------


@cl.set_starters
async def set_starters() -> list[cl.Starter]:
    return [cl.Starter(label=label, message=message) for label, message in STARTERS]


@cl.on_chat_start
async def on_chat_start() -> None:
    settings = await _chat_settings()
    cl.user_session.set(QUERY_SETTINGS_KEY, settings)

    await cl.Message(
        content=(
            f"RAG Studio ist mit der API unter `{API_URL}` verbunden.\n\n"
            "Uploads und Task-Polling laufen ueber HTTP. Queries laufen ueber den "
            "MCP-Server aus `../backend/main_mcp.py`."
        ),
        actions=[
            cl.Action(
                name="upload_document",
                label="Dokument hochladen",
                icon="upload",
                payload={},
            )
        ],
    ).send()


@cl.on_settings_update
async def on_settings_update(settings: dict[str, Any]) -> None:
    cl.user_session.set(QUERY_SETTINGS_KEY, settings)


@cl.on_chat_end
async def on_chat_end() -> None:
    await _close_mcp_client()


@cl.action_callback("upload_document")
async def upload_document_action(action: cl.Action) -> None:
    await action.remove()
    await _upload_document()

    await cl.Message(
        content="Du kannst jederzeit wieder ein Dokument hochladen.",
        actions=[
            cl.Action(
                name="upload_document",
                label="Weiteres Dokument hochladen",
                icon="upload",
                payload={},
            )
        ],
    ).send()


@cl.on_message
async def on_message(message: cl.Message) -> None:
    text = message.content.strip()
    if not text:
        return

    if text.lower() in UPLOAD_COMMANDS:
        await _upload_document()
        return

    try:
        await _run_query(text)
    except Exception as error:
        await cl.Message(content=f"RAG-Aufruf fehlgeschlagen: {error}").send()
