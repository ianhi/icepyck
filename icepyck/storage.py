"""Storage abstraction for Icechunk repositories.

Provides a protocol for reading files from different backends (local
filesystem, S3, etc.) and concrete implementations.
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol, runtime_checkable


@runtime_checkable
class Storage(Protocol):
    """Protocol for reading files from a storage backend."""

    def read(self, path: str) -> bytes:
        """Read the entire contents of a file."""
        ...

    def exists(self, path: str) -> bool:
        """Check if a file exists."""
        ...

    def list_prefix(self, prefix: str) -> list[str]:
        """List all files under a prefix."""
        ...


class LocalStorage:
    """Storage backed by a local filesystem directory."""

    def __init__(self, root: str | Path) -> None:
        self._root = Path(root)

    def read(self, path: str) -> bytes:
        return (self._root / path).read_bytes()

    def exists(self, path: str) -> bool:
        return (self._root / path).exists()

    def list_prefix(self, prefix: str) -> list[str]:
        base = self._root / prefix
        if not base.exists():
            return []
        return [str(p.relative_to(self._root)) for p in base.iterdir()]


class S3Storage:
    """Storage backed by an S3 bucket/prefix.

    Uses s3fs for file access. Supports anonymous and credentialed access.
    The ``s3fs`` package must be installed (``pip install icepyck[s3]``).
    """

    def __init__(self, url: str, anon: bool = False, **s3_kwargs: object) -> None:
        import s3fs  # type: ignore[import-not-found]

        # Newer aiobotocore versions send checksum headers that some S3 buckets
        # reject with 400 for anonymous access. Disable them by default.
        s3_kwargs.setdefault(
            "config_kwargs",
            {
                "request_checksum_calculation": "when_required",
                "response_checksum_validation": "when_required",
            },
        )

        # Sync fs for open/session (runs outside any event loop)
        self._sync_fs = s3fs.S3FileSystem(anon=anon, **s3_kwargs)
        # Async fs for zarr store reads (runs inside zarr's event loop)
        self._async_fs = s3fs.S3FileSystem(
            anon=anon, asynchronous=True, **s3_kwargs
        )
        self._root = url.removeprefix("s3://").rstrip("/")
        self._cache: dict[str, bytes] = {}

    def read(self, path: str) -> bytes:
        if path not in self._cache:
            self._cache[path] = self._sync_fs.cat_file(  # type: ignore[no-any-return]
                f"{self._root}/{path}"
            )
        return self._cache[path]

    async def aread(self, path: str) -> bytes:
        """Async read using the async-mode s3fs instance."""
        if path not in self._cache:
            self._cache[path] = await self._async_fs._cat_file(  # type: ignore[no-any-return]
                f"{self._root}/{path}"
            )
        return self._cache[path]

    def exists(self, path: str) -> bool:
        return self._sync_fs.exists(f"{self._root}/{path}")  # type: ignore[no-any-return]

    def list_prefix(self, prefix: str) -> list[str]:
        full = f"{self._root}/{prefix}"
        try:
            files: list[str] = self._sync_fs.ls(full, detail=False)
        except FileNotFoundError:
            return []
        root_prefix = f"{self._root}/"
        return [f.removeprefix(root_prefix) for f in files]

    def close(self) -> None:
        """Close the underlying async s3fs HTTP session.

        Prevents ResourceWarning about an unclosed aiohttp ClientSession
        when the object is garbage-collected.
        """
        try:
            s3creator = self._async_fs._s3creator
        except AttributeError:
            return  # session was never opened
        self._async_fs.close_session(self._async_fs.loop, s3creator)

    def __del__(self) -> None:
        self.close()
