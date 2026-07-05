from asyncio import to_thread
from datetime import timedelta
from pathlib import Path
from urllib.parse import urlsplit

from minio import Minio


class MinioStore:

    def __init__(self,
        url: str,
        access_key: str,
        secret_key: str,
        bucket_name: str,
    ) -> None:

        parts = urlsplit(url)

        self._client = Minio(
            endpoint=parts.netloc,
            access_key=access_key,
            secret_key=secret_key,
            secure=parts.scheme == "https",
            # no cert verification: off-grid TLS is typically self-signed
            cert_check=False,
        )

        self._bucket_name = bucket_name

        if not self._client.bucket_exists(self._bucket_name):
            self._client.make_bucket(self._bucket_name)

    async def upload(self,
        file_path: Path | str,
        object_name: str,
    ) -> None:
        await to_thread(
            self._client.fput_object,
            bucket_name=self._bucket_name,
            object_name=object_name,
            file_path=file_path,
        )

    def presigned_url(self,
        object_name: str,
        expires_seconds: int,
    ) -> str:
        return self._client.presigned_get_object(
            bucket_name=self._bucket_name,
            object_name=object_name,
            expires=timedelta(seconds=expires_seconds),
        )
