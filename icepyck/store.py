"""Zarr v3 read-only Store backed by icechunk data.

Translates zarr v3 key access (``zarr.json``, chunk keys) to icechunk
snapshot/manifest/chunk lookups.
"""

from __future__ import annotations

import json
import math
from itertools import product
from typing import TYPE_CHECKING

from zarr.abc.store import (
    ByteRequest,
    OffsetByteRequest,
    RangeByteRequest,
    Store,
    SuffixByteRequest,
)

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, Iterable
    from pathlib import Path

    from zarr.core.buffer import Buffer, BufferPrototype

    from icepyck.manifest import ChunkRefInfo, ManifestReader
    from icepyck.snapshot import NodeInfo, SnapshotReader
    from icepyck.storage import Storage


def _parse_key(key: str) -> tuple[str, str | tuple[int, ...]]:
    """Parse a zarr v3 key into (icechunk_node_path, key_kind).

    Returns
    -------
    node_path : str
        Icechunk node path (e.g. ``"/group1/temperatures"``).
    key_kind : str or tuple[int, ...]
        ``"metadata"`` for ``zarr.json`` keys, or a tuple of chunk
        coordinate ints for chunk keys.
    """
    if key == "zarr.json":
        return "/", "metadata"
    if key.endswith("/zarr.json"):
        path = "/" + key[: -len("/zarr.json")]
        return path, "metadata"
    # Chunk key: path/to/array/c/i0/i1/...
    # Find the chunk separator "c" — search from the right since node paths
    # may themselves contain a segment called "c".
    parts = key.split("/")
    c_idx: int | None = None
    for i in range(len(parts) - 1, -1, -1):
        if parts[i] == "c":
            # Everything after must be integer indices
            tail = parts[i + 1 :]
            if all(
                p.isdigit() or (p.startswith("-") and p[1:].isdigit())
                for p in tail
            ):
                c_idx = i
                break
    if c_idx is None:
        return "", "unknown"
    path = "/" + "/".join(parts[:c_idx]) if c_idx > 0 else "/"
    chunk_coords = tuple(int(p) for p in parts[c_idx + 1 :])
    return path, chunk_coords


def _apply_byte_range(data: bytes, byte_range: ByteRequest | None) -> bytes:
    """Slice *data* according to a zarr ByteRequest."""
    if byte_range is None:
        return data
    if isinstance(byte_range, RangeByteRequest):
        return data[byte_range.start : byte_range.end]
    if isinstance(byte_range, OffsetByteRequest):
        return data[byte_range.offset :]
    if isinstance(byte_range, SuffixByteRequest):
        return data[-byte_range.suffix :]
    return data  # pragma: no cover


def _iter_chunk_keys_from_metadata(user_data: bytes) -> list[str]:
    """Derive all chunk key suffixes from zarr metadata without loading manifests.

    Given the zarr.json ``user_data`` for an array node, compute the chunk
    grid dimensions from ``shape`` and ``chunk_grid`` and enumerate every
    possible chunk coordinate string (e.g. ``"0/1/2"``).

    For a scalar (0-d) array the single chunk key is the empty string.
    """
    meta = json.loads(user_data)
    shape = meta.get("shape", [])

    if len(shape) == 0:
        # Scalar array — single chunk with no index components
        return [""]

    chunk_grid = meta.get("chunk_grid", {})
    if chunk_grid.get("name") != "regular":
        return []  # unsupported grid type — fall back to empty
    chunk_shape = chunk_grid.get("configuration", {}).get("chunk_shape", [])
    if not chunk_shape or len(chunk_shape) != len(shape):
        return []

    n_chunks = [
        math.ceil(s / cs) for s, cs in zip(shape, chunk_shape, strict=True)
    ]
    keys: list[str] = []
    for coords in product(*(range(n) for n in n_chunks)):
        keys.append("/".join(str(c) for c in coords))
    return keys


class IcechunkReadStore(Store):
    """A read-only zarr v3 :class:`~zarr.abc.store.Store` over an icechunk snapshot."""

    def __init__(
        self,
        root_path: Path | None = None,
        snapshot: SnapshotReader | None = None,
        *,
        storage: Storage | None = None,
    ) -> None:
        super().__init__(read_only=True)
        self._root_path = root_path
        self._storage = storage
        assert snapshot is not None
        self._snapshot = snapshot

        # Pre-build lookup structures
        self._nodes_by_path: dict[str, NodeInfo] = {}
        for node in self._snapshot.list_nodes():
            self._nodes_by_path[node.path] = node

        # Lazy-loaded manifest cache
        self._manifest_cache: dict[bytes, ManifestReader] = {}

        # Lazy-built chunk index: (node_path, chunk_coords) -> ChunkRefInfo
        # Populated per-array on first chunk access.
        self._chunk_index: dict[tuple[str, tuple[int, ...]], ChunkRefInfo] = {}
        self._chunk_index_built: set[str] = set()  # node paths already indexed

        # Mark as open
        self._is_open = True

    # ------------------------------------------------------------------
    # Abstract property implementations
    # ------------------------------------------------------------------

    @property
    def supports_writes(self) -> bool:  # type: ignore[override]
        return False

    @property
    def supports_deletes(self) -> bool:  # type: ignore[override]
        return False

    @property
    def supports_listing(self) -> bool:  # type: ignore[override]
        return True

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, IcechunkReadStore):
            return NotImplemented
        return (
            self._root_path == other._root_path
            and self._snapshot._snapshot_id == other._snapshot._snapshot_id
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_manifest(self, manifest_id: bytes) -> ManifestReader:
        from icepyck.manifest import ManifestReader as MR

        if manifest_id not in self._manifest_cache:
            self._manifest_cache[manifest_id] = MR(
                self._root_path, manifest_id, storage=self._storage
            )
        return self._manifest_cache[manifest_id]

    def _build_chunk_index(self, node_path: str, node: NodeInfo) -> None:
        """Lazily build the chunk index for *node* (loading its manifests)."""
        if node_path in self._chunk_index_built:
            return
        for mref in node.manifest_refs:
            manifest = self._get_manifest(mref.manifest_id)
            for cref in manifest.get_chunk_refs(node.node_id):
                self._chunk_index[(node_path, cref.index)] = cref
        self._chunk_index_built.add(node_path)

    def _resolve_key(self, key: str) -> bytes | None:
        """Synchronously resolve a zarr key to raw bytes, or *None*."""
        node_path, kind = _parse_key(key)

        if kind == "unknown":
            return None

        node = self._nodes_by_path.get(node_path)
        if node is None:
            return None

        if kind == "metadata":
            return node.user_data if node.user_data else None

        # kind is a chunk coordinate tuple
        assert isinstance(kind, tuple)
        chunk_coords: tuple[int, ...] = kind

        if node.node_type != "array":
            return None

        # Lazily build the chunk index for this array on first access
        self._build_chunk_index(node_path, node)

        cref = self._chunk_index.get((node_path, chunk_coords))
        if cref is None:
            return None

        from icepyck.chunks import read_chunk

        return read_chunk(self._root_path, cref, storage=self._storage)

    async def _aresolve_key(self, key: str) -> bytes | None:
        """Resolve a zarr key, using async I/O for chunk reads."""
        node_path, kind = _parse_key(key)

        if kind == "unknown":
            return None

        node = self._nodes_by_path.get(node_path)
        if node is None:
            return None

        if kind == "metadata":
            return node.user_data if node.user_data else None

        assert isinstance(kind, tuple)
        chunk_coords: tuple[int, ...] = kind

        if node.node_type != "array":
            return None

        self._build_chunk_index(node_path, node)

        cref = self._chunk_index.get((node_path, chunk_coords))
        if cref is None:
            return None

        # Use async read for S3, sync for local
        if (
            self._storage is not None
            and hasattr(self._storage, "aread")
        ):
            return await self._aread_chunk(cref)

        from icepyck.chunks import read_chunk

        return read_chunk(self._root_path, cref, storage=self._storage)

    async def _aread_chunk(self, cref: ChunkRefInfo) -> bytes:
        """Read a chunk using async S3 I/O."""
        from icepyck.manifest import ChunkType

        if cref.chunk_type == ChunkType.INLINE:
            return cref.inline_data or b""
        elif cref.chunk_type == ChunkType.NATIVE:
            if cref.chunk_id is None:
                raise ValueError("Native chunk ref has no chunk_id")
            from icepyck.crockford import encode as crockford_encode

            chunk_name = crockford_encode(cref.chunk_id)
            path = f"chunks/{chunk_name}"
            assert self._storage is not None
            raw = await self._storage.aread(path)
            if cref.length > 0:
                return raw[cref.offset : cref.offset + cref.length]
            return raw[cref.offset :]
        else:
            raise NotImplementedError("Virtual chunk async read")

    # ------------------------------------------------------------------
    # zarr.abc.store.Store interface
    # ------------------------------------------------------------------

    async def get(
        self,
        key: str,
        prototype: BufferPrototype,
        byte_range: ByteRequest | None = None,
    ) -> Buffer | None:
        data = await self._aresolve_key(key)
        if data is None:
            return None
        data = _apply_byte_range(data, byte_range)
        return prototype.buffer.from_bytes(data)

    async def get_partial_values(
        self,
        prototype: BufferPrototype,
        key_ranges: Iterable[tuple[str, ByteRequest | None]],
    ) -> list[Buffer | None]:
        return [
            await self.get(key, prototype, byte_range=br) for key, br in key_ranges
        ]

    async def exists(self, key: str) -> bool:
        return self._resolve_key(key) is not None

    async def set(self, key: str, value: Buffer) -> None:
        raise NotImplementedError("IcechunkReadStore is read-only")

    async def delete(self, key: str) -> None:
        raise NotImplementedError("IcechunkReadStore is read-only")

    async def list(self) -> AsyncGenerator[str, None]:
        for node in self._snapshot.list_nodes():
            if node.user_data:
                if node.path == "/":
                    yield "zarr.json"
                else:
                    yield node.path.lstrip("/") + "/zarr.json"
            if node.node_type == "array" and node.user_data:
                zarr_prefix = "" if node.path == "/" else node.path.lstrip("/")
                # Compute chunk keys from zarr metadata (shape + chunk_grid)
                # instead of loading manifests — avoids N S3 calls.
                for idx_str in _iter_chunk_keys_from_metadata(node.user_data):
                    if zarr_prefix:
                        yield f"{zarr_prefix}/c/{idx_str}"
                    else:
                        yield f"c/{idx_str}"

    async def list_prefix(self, prefix: str) -> AsyncGenerator[str, None]:
        async for key in self.list():
            if key.startswith(prefix):
                yield key

    async def list_dir(self, prefix: str) -> AsyncGenerator[str, None]:
        if prefix and not prefix.endswith("/"):
            prefix = prefix + "/"

        seen: set[str] = set()
        async for key in self.list():
            if prefix and not key.startswith(prefix):
                continue
            remainder = key[len(prefix) :]
            # Immediate child: either a leaf or a directory prefix
            if "/" in remainder:
                child = remainder.split("/")[0]
                entry = child + "/"  # directory marker
            else:
                entry = remainder
            if entry and entry not in seen:
                seen.add(entry)
                yield entry
