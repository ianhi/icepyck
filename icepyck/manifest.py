"""Manifest reader for Icechunk repositories.

Parses a manifest file and extracts chunk references for each array,
supporting inline, native, and virtual chunk types.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from typing import TYPE_CHECKING

from icepyck.crockford import encode as crockford_encode
from icepyck.header import FileType, parse_bytes, parse_file

if TYPE_CHECKING:
    from pathlib import Path

    from icepyck.storage import Storage


class ChunkType(Enum):
    """How the chunk data is stored."""

    INLINE = auto()
    NATIVE = auto()
    VIRTUAL = auto()


@dataclass(frozen=True)
class ChunkRefInfo:
    """Information about a single chunk reference."""

    index: tuple[int, ...]  # chunk coordinates
    chunk_type: ChunkType
    # For INLINE chunks:
    inline_data: bytes | None = None
    # For NATIVE chunks:
    chunk_id: bytes | None = None  # 12-byte ObjectId12
    offset: int = 0
    length: int = 0
    # For VIRTUAL chunks:
    location: str | None = None


def _parse_manifest_payload(
    manifest_id: bytes, header: object, payload: bytes
) -> dict[bytes, list[ChunkRefInfo]]:
    """Parse a manifest flatbuffer payload into a node_id -> ChunkRefInfo list mapping.

    Separated from I/O so both the sync ``__init__`` and async ``afrom_storage``
    can share the same parsing logic.
    """
    from icepyck.generated.Manifest import Manifest

    if header.file_type != FileType.MANIFEST:  # type: ignore[union-attr]
        raise ValueError(
            f"Expected MANIFEST file type ({FileType.MANIFEST}), "
            f"got {header.file_type}"  # type: ignore[union-attr]
        )

    buf = bytearray(payload)
    manifest = Manifest.GetRootAs(buf, 0)

    # Build a lookup from node_id bytes -> list of ChunkRefInfo
    arrays: dict[bytes, list[ChunkRefInfo]] = {}

    for i in range(manifest.ArraysLength()):
        arr = manifest.Arrays(i)

        # Node ID (8 bytes)
        nid_obj = arr.NodeId()
        node_id = bytes(nid_obj.Bytes()) if nid_obj else b""

        refs: list[ChunkRefInfo] = []
        for j in range(arr.RefsLength()):
            cref = arr.Refs(j)

            # Chunk index coordinates
            idx_len = cref.IndexLength()
            index = tuple(cref.Index(k) for k in range(idx_len))

            # Determine chunk type
            inline_len = cref.InlineLength()
            if inline_len > 0:
                # Inline chunk
                inline_data = bytes(
                    cref.Inline(k) for k in range(inline_len)
                )
                refs.append(
                    ChunkRefInfo(
                        index=index,
                        chunk_type=ChunkType.INLINE,
                        inline_data=inline_data,
                    )
                )
            elif cref.ChunkId() is not None:
                # Native chunk
                chunk_id_obj = cref.ChunkId()
                chunk_id = bytes(chunk_id_obj.Bytes())
                refs.append(
                    ChunkRefInfo(
                        index=index,
                        chunk_type=ChunkType.NATIVE,
                        chunk_id=chunk_id,
                        offset=cref.Offset(),
                        length=cref.Length(),
                    )
                )
            elif cref.Location() is not None:
                # Virtual chunk
                loc = cref.Location()
                if isinstance(loc, bytes):
                    loc = loc.decode("utf-8")
                refs.append(
                    ChunkRefInfo(
                        index=index,
                        chunk_type=ChunkType.VIRTUAL,
                        location=loc,
                        offset=cref.Offset(),
                        length=cref.Length(),
                    )
                )
            else:
                # Unknown / empty ref — skip or warn
                refs.append(
                    ChunkRefInfo(
                        index=index,
                        chunk_type=ChunkType.INLINE,
                        inline_data=b"",
                    )
                )

        arrays[node_id] = refs

    return arrays


class ManifestReader:
    """Read and interpret an Icechunk manifest file."""

    def __init__(
        self,
        root_path: str | Path | None = None,
        manifest_id: bytes = b"",
        *,
        storage: Storage | None = None,
    ) -> None:
        self._manifest_id = manifest_id
        manifest_name = crockford_encode(manifest_id)

        if storage is not None:
            raw = storage.read(f"manifests/{manifest_name}")
            header, payload = parse_bytes(raw)
        elif root_path is not None:
            from pathlib import Path

            header, payload = parse_file(Path(root_path) / "manifests" / manifest_name)
        else:
            raise TypeError("Either root_path or storage must be provided")

        self._arrays = _parse_manifest_payload(manifest_id, header, payload)

    @classmethod
    async def afrom_storage(
        cls,
        manifest_id: bytes,
        storage: Storage,
    ) -> ManifestReader:
        """Async constructor: fetch manifest bytes via ``storage.aread()`` then parse.

        Uses the async read path when the storage backend supports it
        (e.g. :class:`~icepyck.storage.S3Storage`), falling back to the
        synchronous ``storage.read()`` otherwise.  This avoids blocking the
        asyncio event loop during manifest fetches from S3.
        """
        manifest_name = crockford_encode(manifest_id)
        path = f"manifests/{manifest_name}"
        if hasattr(storage, "aread"):
            raw: bytes = await storage.aread(path)  # type: ignore[attr-defined]
        else:
            raw = storage.read(path)
        header, payload = parse_bytes(raw)
        instance = cls.__new__(cls)
        instance._manifest_id = manifest_id
        instance._arrays = _parse_manifest_payload(manifest_id, header, payload)
        return instance

    def get_chunk_refs(self, node_id: bytes) -> list[ChunkRefInfo]:
        """Get all chunk refs for an array by its 8-byte node ID."""
        if node_id not in self._arrays:
            raise KeyError(
                f"Node ID {node_id.hex()} not found in manifest "
                f"{crockford_encode(self._manifest_id)}"
            )
        return self._arrays[node_id]

    def list_node_ids(self) -> list[bytes]:
        """List all node IDs present in this manifest."""
        return list(self._arrays.keys())
