from logging import getLogger
from pathlib import Path

from gradio_client import Client, handle_file
from haystack import Document, component

logger = getLogger(__name__)

# Index of the markdown text in the /wait_task_finish result tuple
_MD_INDEX = 1


@component
class DoclingConverter:
    """
    Converts local files to markdown via docling-serve (Gradio API).

    Each source file becomes one unsplit Haystack Document.
    Downstream pipeline stages handle splitting, chunking, and embedding.

    Args:
        docling_url:    Gradio mount URL, e.g. ``http://localhost:5001/ui``.
        timeout:        HTTP timeout in seconds for each API call.
        ocr:            Enable OCR (default True).
        force_ocr:      Force OCR even on text-layer PDFs (default False).
        ocr_engine:     auto | easyocr | tesseract | rapidocr (default auto).
        ocr_lang:       Comma-separated language codes (default en,fr,de,es).
        pipeline:       legacy | standard | vlm | asr (default standard).
        table_mode:     fast | accurate (default accurate).
        abort_on_error: Abort the whole batch on the first error (default False).
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
        # One persistent client per component instance — thread-safe for reads
        self._client: Client | None = None

    # ------------------------------------------------------------------

    def _get_client(self) -> Client:
        """Return the shared Gradio client, creating it lazily on first access."""
        if self._client is None:
            self._client = Client(
                self.docling_url,
                httpx_kwargs={"timeout": self.timeout},
            )
        return self._client

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
                    "DoclingConverter: failed to convert '%s': %s", path, exc
                )
                if self.abort_on_error:
                    raise

        logger.info(
            "DoclingConverter: converted %d / %d file(s)",
            len(documents), len(paths),
        )
        return {"documents": documents}

    # ------------------------------------------------------------------

    def _convert_file(self, path: str) -> Document | None:
        """Submit one file to docling-serve and block until conversion is done.

        Args:
            path: Absolute path to the file to convert.

        Returns:
            A Document with the markdown content, or ``None`` if the conversion
            produced no usable output.
        """
        client = self._get_client()
        file_name = Path(path).name

        # ── Step 1: submit ────────────────────────────────────────────────
        task_id: str = client.predict(
            auth          = "whatever",
            files         = [handle_file(path)],
            to_formats    = ["md"],
            image_export_mode = "placeholder",
            pipeline      = self.pipeline,
            ocr           = self.ocr,
            force_ocr     = self.force_ocr,
            ocr_engine    = self.ocr_engine,
            ocr_lang      = self.ocr_lang,
            pdf_backend   = "docling_parse",
            table_mode    = self.table_mode,
            abort_on_error= self.abort_on_error,
            return_as_file= False,
            do_code_enrichment         = False,
            do_formula_enrichment      = False,
            do_picture_classification  = False,
            do_picture_description     = False,
            api_name      = "/process_file",
        )

        logger.debug("DoclingConverter: submitted '%s' → task_id=%s", file_name, task_id)

        if not task_id:
            logger.warning("DoclingConverter: empty task_id for '%s'", file_name)
            return None

        # ── Step 2: wait ──────────────────────────────────────────────────
        result: tuple = client.predict(
            auth          = "whatever",
            task_id       = task_id,
            return_as_file= False,
            api_name      = "/wait_task_finish",
        )

        # result is a 9-tuple; index 1 = markdown text
        if not isinstance(result, (list, tuple)) or len(result) <= _MD_INDEX:
            logger.warning(
                "DoclingConverter: unexpected result shape for '%s': %r",
                file_name, type(result),
            )
            return None

        markdown: str = result[_MD_INDEX] or ""
        if not markdown.strip():
            logger.warning("DoclingConverter: empty markdown for '%s'", file_name)
            return None

        logger.info(
            "DoclingConverter: '%s' → %d chars of markdown", file_name, len(markdown)
        )

        return Document(
            content = markdown,
            meta    = {"source": file_name, "file_path": path},
        )
