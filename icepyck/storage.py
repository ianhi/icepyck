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

        self._fs = s3fs.S3FileSystem(anon=anon, **s3_kwargs)
        # Normalize: remove s3:// prefix, ensure no trailing slash
        self._root = url.removeprefix("s3://").rstrip("/")

    def read(self, path: str) -> bytes:
        full = f"{self._root}/{path}"
        return self._fs.cat_file(full)  # type: ignore[no-any-return]

    async def aread(self, path: str) -> bytes:
        """Async version of read — uses s3fs's native async I/O."""
        full = f"{self._root}/{path}"
        return await self._fs._cat_file(full)  # type: ignore[no-any-return]

    def exists(self, path: str) -> bool:
        return self._fs.exists(f"{self._root}/{path}")  # type: ignore[no-any-return]

    def list_prefix(self, prefix: str) -> list[str]:
        full = f"{self._root}/{prefix}"
        try:
            files: list[str] = self._fs.ls(full, detail=False)
        except FileNotFoundError:
            return []
        # Return relative paths
        root_prefix = f"{self._root}/"
        return [f.removeprefix(root_prefix) for f in files]
