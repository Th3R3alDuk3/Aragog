from logging import getLogger
from mimetypes import guess_type
from pathlib import Path

from haystack import Document, component
from httpx import Client, HTTPError, Timeout

logger = getLogger(__name__)
PAGE_BREAK_PLACEHOLDER = "[[DOC_PAGE_BREAK]]"


@component
class Docling:
    """
    Converts local files to markdown via the docling-serve v1 API.

    Each source file becomes one unsplit Haystack Document.
    Downstream pipeline stages handle splitting, chunking, and embedding.

    Args:
        docling_url:    Docling API base URL, e.g. ``http://localhost:5001``.
        timeout:        HTTP timeout in seconds for each API call.
        ocr:            Enable OCR (default True).
        force_ocr:      Force OCR even on text-layer PDFs (default False).
        ocr_engine:     auto | easyocr | tesseract | rapidocr (default auto).
        ocr_lang:       Comma-separated language codes (default en,fr,de,es).
        pipeline:       legacy | standard | vlm | asr (default standard).
        table_mode:     fast | accurate (default accurate).
        abort_on_error: Abort the whole batch on the first error (default False).
        Markdown output always includes ``[[DOC_PAGE_BREAK]]`` between pages so
        later pipeline stages can recover page ranges without changing the
        parent-child splitting strategy.
    """

    def __init__(
        self,
        docling_url: str,
        timeout: int = 300,
        ocr: bool = True,
        force_ocr: bool = False,
        ocr_engine: str = "auto",
        ocr_lang: str = "en,fr,de,es",
        pipeline: str = "standard",
        table_mode: str = "accurate",
        abort_on_error: bool = False,
    ) -> None:
        self.docling_url    = docling_url.rstrip("/")
        self.timeout        = timeout
        self.ocr            = ocr
        self.force_ocr      = force_ocr
        self.ocr_engine     = ocr_engine
        self.ocr_lang       = ocr_lang
        self.pipeline       = pipeline
        self.table_mode     = table_mode
        self.abort_on_error = abort_on_error

    # ------------------------------------------------------------------

    def _api_url(self, path: str) -> str:
        base = self.docling_url
        if base.endswith("/ui"):
            base = base[:-3]
        return f"{base}{path}"

    # ------------------------------------------------------------------

    @component.output_types(documents=list[Document])
    def run(self, paths: list[str]) -> dict[str, list[Document]]:
        """Convert each file path to a Haystack Document containing markdown.

        Failed files are skipped with a warning; the pipeline continues unless
        ``abort_on_error=True`` was set at construction time.

        Args:
            paths: Absolute paths to the files to convert.

        Returns:
            A dict with key ``"documents"`` containing one Document per
            successfully converted file.
        """
        documents: list[Document] = []

        for path in paths:
            try:
                doc = self._convert_file(path)
                if doc is not None:
                    documents.append(doc)
            except Exception as exc:
                logger.warning(
                    "Docling: failed to convert '%s': %s", path, exc
                )
                if self.abort_on_error:
                    raise

        logger.info(
            "Docling: converted %d / %d file(s)",
            len(documents), len(paths),
        )
        return {"documents": documents}

    # ------------------------------------------------------------------

    def _convert_file(self, path: str) -> Document | None:
        """Submit one file to docling-serve and wait for the sync v1 result.

        Args:
            path: Absolute path to the file to convert.

        Returns:
            A Document with the markdown content, or ``None`` if the conversion
            produced no usable output.
        """
        file_name = Path(path).name
        mime_type = guess_type(file_name)[0] or "application/octet-stream"
        data = {
            "to_formats": "md",
            "md_page_break_placeholder": PAGE_BREAK_PLACEHOLDER,
            "image_export_mode": "placeholder",
            "pipeline": self.pipeline,
            "ocr": str(self.ocr).lower(),
            "force_ocr": str(self.force_ocr).lower(),
            "ocr_engine": self.ocr_engine,
            "ocr_lang": self.ocr_lang,
            "pdf_backend": "docling_parse",
            "table_mode": self.table_mode,
            "abort_on_error": str(self.abort_on_error).lower(),
            "do_code_enrichment": "false",
            "do_formula_enrichment": "false",
            "do_picture_classification": "false",
            "do_picture_description": "false",
        }

        try:
            with Path(path).open("rb") as file_obj, Client(timeout=Timeout(self.timeout)) as client:
                response = client.post(
                    self._api_url("/v1/convert/file"),
                    headers={"accept": "application/json"},
                    files={"files": (file_name, file_obj, mime_type)},
                    data=data,
                )
                response.raise_for_status()
        except HTTPError as exc:
            raise RuntimeError(f"Docling v1 request failed for '{file_name}': {exc}") from exc

        payload = response.json()
        document = payload.get("document") or {}
        markdown: str = document.get("md_content") or ""
        if not markdown.strip():
            logger.warning("Docling: empty markdown for '%s'", file_name)
            return None

        logger.info(
            "Docling: '%s' → %d chars of markdown", file_name, len(markdown)
        )

        return Document(
            content = markdown,
            meta    = {"source": file_name, "file_path": path},
        )
