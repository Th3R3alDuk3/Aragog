from asyncio import to_thread
from datetime import timedelta
from pathlib import Path
from urllib.parse import urlsplit

from minio import Minio


class MinioStore:

    def __init__(self,
        url: str,
        public_url: str,
        access_key: str,
        secret_key: str,
        bucket_name: str,
    ) -> None:

        def client(address: str, region: str | None) -> Minio:
            parts = urlsplit(address)
            return Minio(
                endpoint=parts.netloc,
                access_key=access_key,
                secret_key=secret_key,
                secure=parts.scheme == "https",
                region=region,
                # no cert verification: off-grid TLS is typically self-signed
                cert_check=False,
            )

        self._client = client(url, None)
        # signs links for the browser's address; region pinned so it never connects
        self._signer = client(public_url, "us-east-1")
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

    async def presigned_url(self,
        object_name: str,
        expires_seconds: int,
    ) -> str:
        return await to_thread(
            self._signer.presigned_get_object,
            bucket_name=self._bucket_name,
            object_name=object_name,
            expires=timedelta(seconds=expires_seconds),
        )
