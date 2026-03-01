"""
MinioStore — thin wrapper around the MinIO Python client.

Handles bucket creation, file upload, and URL generation for the RAG pipeline.
Original uploaded documents are stored here so they can be:
  - re-indexed without re-uploading
  - downloaded/previewed by UI clients via minio_url in chunk metadata
  - versioned (same content → same key via content hash prefix)

Configuration (via .env):
  MINIO_ENDPOINT   — host:port, e.g. localhost:9000
  MINIO_ACCESS_KEY — MinIO root user / access key
  MINIO_SECRET_KEY — MinIO root password / secret key
  MINIO_BUCKET     — bucket name (created automatically if missing)
  MINIO_SECURE     — false for HTTP (local), true for HTTPS (production)
"""

import logging
from pathlib import Path
from urllib.parse import quote

logger = logging.getLogger(__name__)


class MinioStore:
    """
    Uploads files to MinIO and returns their public URL.

    Args:
        endpoint:   MinIO host:port (e.g. ``localhost:9000``).
        access_key: MinIO access key.
        secret_key: MinIO secret key.
        bucket:     Target bucket (auto-created if it does not exist).
        secure:     Use HTTPS (default False for local deployments).
    """

    def __init__(
        self,
        endpoint: str,
        access_key: str,
        secret_key: str,
        bucket: str,
        secure: bool = False,
    ) -> None:
        from minio import Minio

        self._client = Minio(
            endpoint,
            access_key=access_key,
            secret_key=secret_key,
            secure=secure,
        )
        self._bucket = bucket
        self._scheme = "https" if secure else "http"
        self._endpoint = endpoint
        self._ensure_bucket()

    def _ensure_bucket(self) -> None:
        if not self._client.bucket_exists(self._bucket):
            self._client.make_bucket(self._bucket)
            logger.info("MinioStore: created bucket '%s'", self._bucket)

    def upload(self, file_path: str, object_name: str) -> str:
        """
        Upload a local file to MinIO.

        Args:
            file_path:   Absolute path to the local file.
            object_name: Key / object name inside the bucket.

        Returns:
            Public URL: ``http(s)://<endpoint>/<bucket>/<object_name>``
        """
        self._client.fput_object(self._bucket, object_name, file_path)
        url = f"{self._scheme}://{self._endpoint}/{self._bucket}/{quote(object_name, safe='/')}"
        logger.info("MinioStore: uploaded '%s' → %s", Path(file_path).name, url)
        return url

    def public_url(self, object_name: str) -> str:
        """Return the public URL for an already-uploaded object."""
        return f"{self._scheme}://{self._endpoint}/{self._bucket}/{quote(object_name, safe='/')}"

    def delete(self, object_name: str) -> None:
        """Delete an object from the bucket (best-effort, ignores errors)."""
        try:
            self._client.remove_object(self._bucket, object_name)
            logger.info("MinioStore: deleted '%s'", object_name)
        except Exception as exc:
            logger.warning("MinioStore: could not delete '%s': %s", object_name, exc)
