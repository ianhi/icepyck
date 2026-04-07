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


class ManifestReader:
    """Read and interpret an Icechunk manifest file.

    Parameters
    ----------
    root_path : str or Path
        Root path of the Icechunk repository.
    manifest_id : bytes
        The 12-byte ObjectId12 identifying the manifest.
    """

    def __init__(
        self,
        root_path: str | Path | None = None,
        manifest_id: bytes = b"",
        *,
        storage: Storage | None = None,
    ) -> None:
        from icepyck.generated.Manifest import Manifest

        self._manifest_id = manifest_id
        manifest_name = crockford_encode(manifest_id)

        if storage is not None:
            raw = storage.read(f"manifests/{manifest_name}")
            header, payload = parse_bytes(raw)
        elif root_path is not None:
            from pathlib import Path as _Path

            self._root_path: Path | None = _Path(root_path)
            manifest_path = self._root_path / "manifests" / manifest_name
            header, payload = parse_file(manifest_path)
        else:
            raise TypeError("Either root_path or storage must be provided")
        if header.file_type != FileType.MANIFEST:
            raise ValueError(
                f"Expected MANIFEST file type ({FileType.MANIFEST}), "
                f"got {header.file_type}"
            )

        buf = bytearray(payload)
        self._manifest = Manifest.GetRootAs(buf, 0)

        # Build a lookup from node_id bytes -> list of ChunkRefInfo
        self._arrays: dict[bytes, list[ChunkRefInfo]] = {}

        for i in range(self._manifest.ArraysLength()):
            arr = self._manifest.Arrays(i)

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

            self._arrays[node_id] = refs

    def get_chunk_refs(self, node_id: bytes) -> list[ChunkRefInfo]:
        """Get all chunk refs for a specific array by its 8-byte node ID.

        Parameters
        ----------
        node_id : bytes
            The 8-byte ObjectId8 of the array node.

        Returns
        -------
        list[ChunkRefInfo]
            The chunk references for this array.

        Raises
        ------
        KeyError
            If no array with this node ID is found in this manifest.
        """
        if node_id not in self._arrays:
            raise KeyError(
                f"Node ID {node_id.hex()} not found in manifest "
                f"{crockford_encode(self._manifest_id)}"
            )
        return self._arrays[node_id]

    def list_node_ids(self) -> list[bytes]:
        """List all node IDs present in this manifest."""
        return list(self._arrays.keys())
