from datetime import timedelta
from logging import getLogger
from pathlib import Path
from urllib.parse import quote

from minio import Minio

logger = getLogger(__name__)


class MinioStore:
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
        self._client.fput_object(self._bucket, object_name, file_path)
        url = f"{self._scheme}://{self._endpoint}/{self._bucket}/{quote(object_name, safe='/')}"
        logger.info("MinioStore: uploaded '%s' → %s", Path(file_path).name, url)
        return url

    def download_url(self, object_name: str, expires_seconds: int = 3600) -> str:
        return self._client.presigned_get_object(
            self._bucket,
            object_name,
            expires=timedelta(seconds=max(1, expires_seconds)),
        )
