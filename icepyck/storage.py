"""Storage abstraction for Icechunk repositories.

Provides a protocol for reading/writing files from different backends
(local filesystem, S3, etc.) with support for conditional writes
(atomic updates guarded by version tokens).
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Protocol, runtime_checkable


class VersionMismatchError(Exception):
    """Raised when a conditional write fails due to version mismatch.

    This indicates another writer modified the file since we last read it.
    """


@runtime_checkable
class Storage(Protocol):
    """Protocol for reading and writing files from a storage backend."""

    def read(self, path: str) -> bytes:
        """Read the entire contents of a file."""
        ...

    def read_versioned(self, path: str) -> tuple[bytes, str]:
        """Read file contents and return (data, version_token).

        The version_token is opaque — pass it to conditional_write()
        to ensure the file hasn't changed since this read.
        """
        ...

    def write(self, path: str, data: bytes) -> None:
        """Write data to a file (unconditional)."""
        ...

    def conditional_write(self, path: str, data: bytes, expected_version: str) -> str:
        """Write data only if the file's current version matches expected_version.

        Returns the new version token on success.
        Raises VersionMismatchError if the file was modified by another writer.
        """
        ...

    def exists(self, path: str) -> bool:
        """Check if a file exists."""
        ...

    def list_prefix(self, prefix: str) -> list[str]:
        """List all files under a prefix."""
        ...


@runtime_checkable
class AsyncStorage(Storage, Protocol):
    """Storage that also supports async reads (e.g. S3)."""

    async def aread(self, path: str) -> bytes:
        """Read the entire contents of a file asynchronously."""
        ...


def _file_version(p: Path) -> str:
    """Compute a version token for a local file from its stat."""
    try:
        st = p.stat()
        return f"{st.st_mtime_ns}:{st.st_size}"
    except FileNotFoundError:
        return ""


class LocalStorage:
    """Storage backed by a local filesystem directory.

    Conditional writes use atomic rename + stat-based version tokens.
    """

    def __init__(self, root: str | Path) -> None:
        self._root = Path(root)

    def read(self, path: str) -> bytes:
        return (self._root / path).read_bytes()

    def read_versioned(self, path: str) -> tuple[bytes, str]:
        target = self._root / path
        data = target.read_bytes()
        version = _file_version(target)
        return data, version

    def write(self, path: str, data: bytes) -> None:
        target = self._root / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)

    def conditional_write(self, path: str, data: bytes, expected_version: str) -> str:
        """Atomic conditional write using flock + stat check + atomic rename.

        Acquires an exclusive lock on a .lock file to serialize concurrent
        writers, checks the version hasn't changed, then atomically replaces
        the target file.
        """
        import fcntl

        target = self._root / path
        target.parent.mkdir(parents=True, exist_ok=True)
        lock_path = target.with_suffix(target.suffix + ".lock")

        # Serialize concurrent writers with flock
        lock_fd = os.open(str(lock_path), os.O_CREAT | os.O_RDWR)
        try:
            fcntl.flock(lock_fd, fcntl.LOCK_EX)

            # Re-check version under lock (another writer may have just finished)
            current_version = _file_version(target)
            if current_version != expected_version:
                raise VersionMismatchError(
                    f"File {path!r} was modified: "
                    f"expected version {expected_version!r}, "
                    f"got {current_version!r}"
                )

            # Write to temp file, then atomic rename
            fd, tmp_path = tempfile.mkstemp(dir=target.parent)
            try:
                os.write(fd, data)
                os.close(fd)
                os.replace(tmp_path, target)
            except BaseException:
                import contextlib

                with contextlib.suppress(OSError):
                    os.close(fd)
                with contextlib.suppress(OSError):
                    os.unlink(tmp_path)
                raise

            return _file_version(target)
        finally:
            fcntl.flock(lock_fd, fcntl.LOCK_UN)
            os.close(lock_fd)

    def exists(self, path: str) -> bool:
        return (self._root / path).exists()

    def list_prefix(self, prefix: str) -> list[str]:
        base = self._root / prefix
        if not base.exists():
            return []
        return [str(p.relative_to(self._root)) for p in base.iterdir()]


class S3Storage:
    """Storage backed by an S3 bucket/prefix.

    Conditional writes use S3 ETags (If-Match header).
    """

    def __init__(self, url: str, anon: bool = False, **s3_kwargs: object) -> None:
        import s3fs  # type: ignore[import-not-found]

        s3_kwargs.setdefault(
            "config_kwargs",
            {
                "request_checksum_calculation": "when_required",
                "response_checksum_validation": "when_required",
            },
        )

        self._sync_fs = s3fs.S3FileSystem(anon=anon, **s3_kwargs)
        self._async_fs = s3fs.S3FileSystem(anon=anon, asynchronous=True, **s3_kwargs)
        self._root = url.removeprefix("s3://").rstrip("/")
        self._cache: dict[str, bytes] = {}

    def read(self, path: str) -> bytes:
        if path not in self._cache:
            self._cache[path] = self._sync_fs.cat_file(  # type: ignore[no-any-return]
                f"{self._root}/{path}"
            )
        return self._cache[path]

    def read_versioned(self, path: str) -> tuple[bytes, str]:
        full = f"{self._root}/{path}"
        data = self._sync_fs.cat_file(full)
        # Get ETag from S3 metadata
        info = self._sync_fs.info(full)
        etag = info.get("ETag", "")
        self._cache[path] = data
        return data, etag

    async def aread(self, path: str) -> bytes:
        if path not in self._cache:
            self._cache[path] = await self._async_fs._cat_file(  # type: ignore[no-any-return]
                f"{self._root}/{path}"
            )
        return self._cache[path]

    def exists(self, path: str) -> bool:
        return self._sync_fs.exists(f"{self._root}/{path}")  # type: ignore[no-any-return]

    def write(self, path: str, data: bytes) -> None:
        full = f"{self._root}/{path}"
        self._sync_fs.pipe_file(full, data)
        self._cache.pop(path, None)

    def conditional_write(self, path: str, data: bytes, expected_version: str) -> str:
        """Conditional write using S3 If-Match header.

        S3 returns 412 Precondition Failed if the ETag doesn't match.
        """
        import botocore.exceptions  # type: ignore[import-not-found]

        full = f"{self._root}/{path}"
        s3 = self._sync_fs.s3  # underlying boto3 client

        # Parse bucket and key from full path
        parts = full.split("/", 1)
        bucket = parts[0]
        key = parts[1] if len(parts) > 1 else ""

        try:
            resp = s3.put_object(
                Bucket=bucket,
                Key=key,
                Body=data,
                IfMatch=expected_version,
            )
            new_etag = resp.get("ETag", "")
        except botocore.exceptions.ClientError as e:
            if e.response["Error"]["Code"] == "PreconditionFailed":
                raise VersionMismatchError(
                    f"S3 conditional write failed for {path!r}: "
                    f"ETag mismatch (expected {expected_version!r})"
                ) from e
            raise

        self._cache.pop(path, None)
        return new_etag

    def list_prefix(self, prefix: str) -> list[str]:
        full = f"{self._root}/{prefix}"
        try:
            files: list[str] = self._sync_fs.ls(full, detail=False)
        except FileNotFoundError:
            return []
        root_prefix = f"{self._root}/"
        return [f.removeprefix(root_prefix) for f in files]

    def close(self) -> None:
        """Close the underlying async s3fs HTTP session."""
        try:
            s3creator = self._async_fs._s3creator
        except AttributeError:
            return
        self._async_fs.close_session(self._async_fs.loop, s3creator)

    def __del__(self) -> None:
        self.close()
