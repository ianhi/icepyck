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


def read_chunk(
    root_path: str | Path | None,
    chunk_ref: ChunkRefInfo,
    *,
    storage: Storage | None = None,
) -> bytes:
    """Read chunk data based on the ChunkRefInfo.

    Parameters
    ----------
    root_path : str or Path or None
        Root path of the Icechunk repository.  Ignored when *storage*
        is provided.
    chunk_ref : ChunkRefInfo
        The chunk reference describing where the data is.
    storage : Storage, optional
        Storage backend to read from.

    Returns
    -------
    bytes
        The raw chunk data bytes.

    Raises
    ------
    NotImplementedError
        If the chunk is virtual (remote URL).
    ValueError
        If the chunk type is unknown or data is missing.
    """
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
            from pathlib import Path as _Path

            chunk_path = _Path(root_path) / "chunks" / chunk_name
            raw = chunk_path.read_bytes()
        else:
            raise TypeError("Either root_path or storage must be provided")
        if chunk_ref.length > 0:
            return raw[chunk_ref.offset : chunk_ref.offset + chunk_ref.length]
        else:
            return raw[chunk_ref.offset :]

    elif chunk_ref.chunk_type == ChunkType.VIRTUAL:
        raise NotImplementedError(
            f"Virtual chunk reading not implemented. Location: {chunk_ref.location}"
        )

    else:
        raise ValueError(f"Unknown chunk type: {chunk_ref.chunk_type}")
