"""
MinioStore — thin wrapper around the MinIO Python client.

Handles bucket creation, file upload, and URL generation for the RAG pipeline.
Original uploaded documents are stored here so they can be:
  - re-indexed without re-uploading
  - opened by UI clients via a backend download endpoint that signs access on demand
  - versioned (same content → same key via content hash prefix)

Configuration (via .env):
  MINIO_ENDPOINT   — host:port, e.g. localhost:9000
  MINIO_ACCESS_KEY — MinIO root user / access key
  MINIO_SECRET_KEY — MinIO root password / secret key
  MINIO_BUCKET     — bucket name (created automatically if missing)
  MINIO_SECURE     — false for HTTP (local), true for HTTPS (production)
"""

from datetime import timedelta
from logging import getLogger
from pathlib import Path
from urllib.parse import quote

from minio import Minio

logger = getLogger(__name__)


class MinioStore:
    """
    Uploads files to MinIO and can generate access URLs on demand.

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
            Direct object URL: ``http(s)://<endpoint>/<bucket>/<object_name>``
        """
        self._client.fput_object(self._bucket, object_name, file_path)
        url = f"{self._scheme}://{self._endpoint}/{self._bucket}/{quote(object_name, safe='/')}"
        logger.info("MinioStore: uploaded '%s' → %s", Path(file_path).name, url)
        return url

    def download_url(self, object_name: str, expires_seconds: int = 3600) -> str:
        """Return a presigned GET URL for a private object."""
        return self._client.presigned_get_object(
            self._bucket,
            object_name,
            expires=timedelta(seconds=max(1, expires_seconds)),
        )
