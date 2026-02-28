import hashlib
import tempfile
from pathlib import Path
from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status
from haystack import Pipeline
from haystack_integrations.document_stores.qdrant import QdrantDocumentStore
from models.schemas import IndexResponse
from routers._deps import get_document_store, get_ingestion_pipeline, get_minio_store

router = APIRouter(prefix="/documents", tags=["documents"])


@router.post(
    "/index",
    response_model=IndexResponse,
    status_code=status.HTTP_200_OK,
    summary="Upload and index a document",
    description=(
        "Upload a single file (PDF, DOCX, PPTX, …). "
        "The file is stored in MinIO, extracted via docling-serve, split into chunks, "
        "embedded and stored in Qdrant. Existing chunks for the same filename are "
        "deleted first to avoid stale results on re-indexing."
    ),
)
async def index_document(
    file: UploadFile = File(...),
    pipeline: Pipeline = Depends(get_ingestion_pipeline),
    document_store: QdrantDocumentStore = Depends(get_document_store),
    minio_store=Depends(get_minio_store),
) -> IndexResponse:

    # Read into memory once — needed for both hash and temp file
    file_bytes = await file.read()
    original_name = Path(file.filename).name if file.filename else "upload"

    # ── 1. Delete stale chunks for this source ────────────────────────────────
    # OVERWRITE policy only replaces chunks with the same ID.  If the document
    # changed, new IDs are generated — old chunks would survive.  Delete first.
    # QdrantDocumentStore.delete_documents() only accepts IDs, so we fetch them
    # first via filter_documents() and then delete by ID.
    try:
        stale = document_store.filter_documents(
            filters={"field": "meta.source", "operator": "==", "value": original_name}
        )
        if stale:
            document_store.delete_documents([doc.id for doc in stale])
    except Exception as exc:
        import logging
        logging.getLogger(__name__).warning(
            "Could not delete stale chunks for '%s': %s", original_name, exc
        )

    # ── 2. Upload original file to MinIO ──────────────────────────────────────
    # Key = <sha256[:16]>-<filename> so the same content always maps to the
    # same key (natural dedup) while different files with the same name differ.
    minio_url: str | None = None
    minio_key: str | None = None
    if minio_store is not None:
        file_hash = hashlib.sha256(file_bytes).hexdigest()[:16]
        minio_key = f"{file_hash}-{original_name}"
        try:
            # Write bytes to a temp file for fput_object (streaming upload)
            with tempfile.NamedTemporaryFile(delete=False, suffix=Path(original_name).suffix) as tmp:
                tmp.write(file_bytes)
                tmp_upload_path = tmp.name
            minio_url = minio_store.upload(tmp_upload_path, minio_key)
            Path(tmp_upload_path).unlink(missing_ok=True)
        except Exception as exc:
            import logging
            logging.getLogger(__name__).warning("MinIO upload failed: %s", exc)
            minio_url = None
            minio_key = None

    # ── 3. Run indexing pipeline ──────────────────────────────────────────────
    # Keep original filename by writing to a temp dir with the correct name.
    # DoclingConverter derives meta["source"] from os.path.basename(path).
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir) / original_name
        tmp_path.write_bytes(file_bytes)

        extra_meta: dict = {}
        if minio_url:
            extra_meta["minio_url"] = minio_url
        if minio_key:
            extra_meta["minio_key"] = minio_key

        try:
            result = pipeline.run({
                "converter":    {"paths": [str(tmp_path)]},
                "meta_enricher": {"extra_meta": extra_meta or None},
            })
        except Exception as exc:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Indexing failed: {exc}",
            ) from exc

    written: int = result.get("writer", {}).get("documents_written", 0)
    return IndexResponse(
        indexed   = written,
        source    = original_name,
        minio_url = minio_url,
    )
