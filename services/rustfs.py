from asyncio import to_thread
from mimetypes import guess_type
from pathlib import Path

from boto3 import client as boto3_client
from botocore.client import BaseClient
from botocore.config import Config
from botocore.exceptions import ClientError


class RustfsStore:

    def __init__(self,
        url: str,
        public_url: str,
        access_key: str,
        secret_key: str,
        timeout: int,
        bucket: str,
        url_expire: int,
    ) -> None:

        def client(endpoint_url: str) -> BaseClient:
            return boto3_client(
                "s3",
                endpoint_url=endpoint_url,
                aws_access_key_id=access_key,
                aws_secret_access_key=secret_key,
                # RustFS default region, part of every signature
                region_name="us-east-1",
                # off-grid TLS is typically self-signed
                verify=False,
                config=Config(
                    # presigning defaults to legacy SigV2
                    signature_version="s3v4",
                    connect_timeout=timeout,
                    read_timeout=timeout,
                    retries={"mode": "standard"},
                ),
            )

        self._client = client(url)
        # signs for the browser's host; never connects
        self._signer = client(public_url)
        self._bucket = bucket
        self._url_expire = url_expire

        try:
            self._client.head_bucket(Bucket=self._bucket)
        except ClientError as error:
            if error.response["Error"]["Code"] != "404":
                raise
            self._client.create_bucket(Bucket=self._bucket)

    async def upload(self,
        file_path: Path,
    ) -> None:
        # served inline, so browsers honour `#page=`
        content_type, _ = guess_type(file_path.name)
        await to_thread(
            self._client.upload_file,
            Filename=file_path,
            Bucket=self._bucket,
            # index.py sets the chunks' `source` to the same name
            Key=file_path.name,
            ExtraArgs={"ContentType": content_type or "application/octet-stream"},
        )

    def presigned_url(self,
        object_name: str,
    ) -> str:
        return self._signer.generate_presigned_url(
            "get_object",
            Params={"Bucket": self._bucket, "Key": object_name},
            ExpiresIn=self._url_expire,
        )
