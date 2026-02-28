from io import BytesIO

import pytest
from fastapi import HTTPException, UploadFile
from starlette.datastructures import Headers

from routers.documents import index_document


class DummyPipeline:
    def __init__(self) -> None:
        self.calls = []

    def run(self, payload):
        self.calls.append(payload)
        return {"writer": {"documents_written": 3}}


class DummyDocumentStore:
    """Minimal stub — mirrors the two-step filter+delete pattern."""
    def __init__(self) -> None:
        self.filter_calls = []
        self.delete_calls = []

    def filter_documents(self, filters=None):
        self.filter_calls.append(filters)
        return []  # no stale docs in tests

    def delete_documents(self, document_ids):
        self.delete_calls.append(document_ids)


@pytest.mark.asyncio
async def test_documents_index_success() -> None:
    pipeline = DummyPipeline()
    store = DummyDocumentStore()
    upload = UploadFile(
        filename="report.txt",
        file=BytesIO(b"hello"),
        headers=Headers({"content-type": "text/plain"}),
    )

    response = await index_document(
        file=upload,
        pipeline=pipeline,
        document_store=store,
        minio_store=None,
    )

    assert response.indexed == 3
    assert response.source == "report.txt"
    assert response.minio_url is None
    # filter_documents was called to find stale chunks
    assert len(store.filter_calls) == 1
    assert store.filter_calls[0]["value"] == "report.txt"
    # delete_documents was NOT called (filter returned no stale docs)
    assert len(store.delete_calls) == 0
    # Pipeline received the original filename in the path
    assert len(pipeline.calls) == 1
    assert pipeline.calls[0]["converter"]["paths"][0].endswith("report.txt")


@pytest.mark.asyncio
async def test_documents_index_rejects_unsupported_mime_type() -> None:
    pipeline = DummyPipeline()
    store = DummyDocumentStore()
    upload = UploadFile(
        filename="archive.zip",
        file=BytesIO(b"PK"),
        headers=Headers({"content-type": "application/zip"}),
    )

    with pytest.raises(HTTPException) as exc:
        await index_document(
            file=upload,
            pipeline=pipeline,
            document_store=store,
            minio_store=None,
        )

    assert exc.value.status_code == 415
    assert "Unsupported file type" in str(exc.value.detail)
    # No filter/delete or pipeline call for rejected uploads
    assert len(store.filter_calls) == 0
    assert len(store.delete_calls) == 0
    assert len(pipeline.calls) == 0
