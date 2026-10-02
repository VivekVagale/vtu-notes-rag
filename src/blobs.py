"""Where uploaded bytes live. Nothing outside this module builds a file path.

LocalBlobStore today; a Supabase Storage driver later implements the same four
methods. Keys are derived from the document id only - never from anything the
uploader typed - so a key can never escape the upload directory.
"""

from __future__ import annotations

import hashlib
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, Iterable

from .config import Settings

PDF_MAGIC = b"%PDF-"
CHUNK = 1 << 20  # 1 MiB


class BlobError(RuntimeError):
    pass


class TooLarge(BlobError):
    def __init__(self, limit: int) -> None:
        super().__init__(f"upload exceeds {limit} bytes")
        self.limit = limit


class NotAPdf(BlobError):
    pass


@dataclass(frozen=True)
class BlobInfo:
    key: str
    bytes: int
    sha256: str


def looks_like_pdf(head: bytes) -> bool:
    """Trust the bytes, never the filename or the client's Content-Type."""
    return head[:5] == PDF_MAGIC


class LocalBlobStore:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, key: str) -> Path:
        # Keys are built here and nowhere else, but verify anyway: a key that
        # resolves outside the root is a bug, and a bug here is a file write
        # anywhere on the disk.
        candidate = (self.root / key).resolve()
        if not str(candidate).startswith(str(self.root.resolve())):
            raise BlobError("blob key escapes the upload directory")
        return candidate

    @staticmethod
    def key_for(doc_id: str) -> str:
        return f"{doc_id[:2]}/{doc_id}.pdf"

    def put_stream(self, *, doc_id: str, stream: Iterable[bytes], max_bytes: int) -> BlobInfo:
        """Stream to disk, hashing as we go, aborting the moment it is too big.

        Never buffers the whole body in memory, and never trusts a declared
        Content-Length - the cap is enforced against bytes actually seen.
        """
        key = self.key_for(doc_id)
        final = self._path(key)
        final.parent.mkdir(parents=True, exist_ok=True)
        temp = final.with_suffix(".part")

        digest = hashlib.sha256()
        total = 0
        head = b""
        try:
            with open(temp, "wb") as handle:
                for block in stream:
                    if not block:
                        continue
                    total += len(block)
                    if total > max_bytes:
                        raise TooLarge(max_bytes)
                    if len(head) < 5:
                        head += block[: 5 - len(head)]
                        if len(head) >= 5 and not looks_like_pdf(head):
                            raise NotAPdf("file does not start with %PDF-")
                    digest.update(block)
                    handle.write(block)
            if total == 0 or not looks_like_pdf(head):
                raise NotAPdf("file does not start with %PDF-")
            os.replace(temp, final)
        except BaseException:
            temp.unlink(missing_ok=True)
            raise
        return BlobInfo(key=key, bytes=total, sha256=digest.hexdigest())

    def open(self, key: str) -> BinaryIO:
        return open(self._path(key), "rb")

    def path_for(self, key: str) -> Path:
        return self._path(key)

    def exists(self, key: str) -> bool:
        return self._path(key).exists()

    def delete(self, key: str) -> None:
        path = self._path(key)
        path.unlink(missing_ok=True)
        parent = path.parent
        if parent != self.root and parent.exists() and not any(parent.iterdir()):
            parent.rmdir()

    def free_mb(self) -> int:
        return int(shutil.disk_usage(self.root).free / (1024 * 1024))


def get_blob_store(settings: Settings) -> LocalBlobStore:
    return LocalBlobStore(settings.upload_dir)
