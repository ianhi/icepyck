"""Chunk data reader for Icechunk repositories.

Reads actual chunk bytes from inline data, native chunk files,
or raises NotImplementedError for virtual chunks.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from icepyck.crockford import encode as crockford_encode
from icepyck.manifest import ChunkRefInfo, ChunkType

if TYPE_CHECKING:
    from pathlib import Path

    from icepyck.storage import Storage


def _slice_chunk(raw: bytes, chunk_ref: ChunkRefInfo) -> bytes:
    """Apply offset/length slicing to raw chunk data."""
    if chunk_ref.length > 0:
        return raw[chunk_ref.offset : chunk_ref.offset + chunk_ref.length]
    return raw[chunk_ref.offset :]


def read_chunk(
    root_path: str | Path | None,
    chunk_ref: ChunkRefInfo,
    *,
    storage: Storage | None = None,
) -> bytes:
    """Read chunk data from inline storage or a native chunk file."""
    if chunk_ref.chunk_type == ChunkType.INLINE:
        if chunk_ref.inline_data is None:
            return b""
        return chunk_ref.inline_data

    elif chunk_ref.chunk_type == ChunkType.NATIVE:
        if chunk_ref.chunk_id is None:
            raise ValueError("Native chunk ref has no chunk_id")
        chunk_name = crockford_encode(chunk_ref.chunk_id)
        if storage is not None:
            raw = storage.read(f"chunks/{chunk_name}")
        elif root_path is not None:
            from pathlib import Path

            raw = (Path(root_path) / "chunks" / chunk_name).read_bytes()
        else:
            raise TypeError("Either root_path or storage must be provided")
        return _slice_chunk(raw, chunk_ref)

    elif chunk_ref.chunk_type == ChunkType.VIRTUAL:
        raise NotImplementedError(
            f"Virtual chunk reading not implemented. Location: {chunk_ref.location}"
        )

    else:
        raise ValueError(f"Unknown chunk type: {chunk_ref.chunk_type}")


async def aread_chunk(
    root_path: str | Path | None,
    chunk_ref: ChunkRefInfo,
    *,
    storage: Storage | None = None,
) -> bytes:
    """Async version of :func:`read_chunk`.

    Uses ``storage.aread()`` for native chunks when the storage backend
    supports it, enabling concurrent S3 fetches via ``asyncio.gather``.
    Falls back to :func:`read_chunk` for inline data and unsupported backends.
    """
    if chunk_ref.chunk_type == ChunkType.INLINE:
        if chunk_ref.inline_data is None:
            return b""
        return chunk_ref.inline_data

    elif chunk_ref.chunk_type == ChunkType.NATIVE:
        if chunk_ref.chunk_id is None:
            raise ValueError("Native chunk ref has no chunk_id")
        chunk_name = crockford_encode(chunk_ref.chunk_id)
        from icepyck.storage import AsyncStorage

        if isinstance(storage, AsyncStorage):
            raw = await storage.aread(f"chunks/{chunk_name}")
        elif storage is not None:
            raw = storage.read(f"chunks/{chunk_name}")
        elif root_path is not None:
            from pathlib import Path

            raw = (Path(root_path) / "chunks" / chunk_name).read_bytes()
        else:
            raise TypeError("Either root_path or storage must be provided")
        return _slice_chunk(raw, chunk_ref)

    elif chunk_ref.chunk_type == ChunkType.VIRTUAL:
        raise NotImplementedError(
            f"Virtual chunk reading not implemented. Location: {chunk_ref.location}"
        )

    else:
        raise ValueError(f"Unknown chunk type: {chunk_ref.chunk_type}")
