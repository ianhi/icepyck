"""Zarr v3 read-only Store backed by icechunk data.

Translates zarr v3 key access (``zarr.json``, chunk keys) to icechunk
snapshot/manifest/chunk lookups.
"""

from __future__ import annotations

import asyncio
import json
import math
from itertools import product
from typing import TYPE_CHECKING

from zarr.abc.store import (
    ByteRequest,
    OffsetByteRequest,
    RangeByteRequest,
    Store,
)

from icepyck.storage import AsyncStorage

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
                p.isdigit() or (p.startswith("-") and p[1:].isdigit()) for p in tail
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
    # SuffixByteRequest — the only remaining variant
    return data[-byte_range.suffix :]


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

    n_chunks = [math.ceil(s / cs) for s, cs in zip(shape, chunk_shape, strict=True)]
    keys: list[str] = []
    for coords in product(*(range(n) for n in n_chunks)):
        keys.append("/".join(str(c) for c in coords))
    return keys


def _extents_contain(extents: list[tuple[int, int]], coords: tuple[int, ...]) -> bool:
    """Check if chunk coords fall within ManifestRef extents.

    Each extent is (from_inclusive, to_exclusive) for one dimension.
    """
    if not extents:
        # Empty extents = scalar array, always matches
        return True
    if len(extents) != len(coords):
        return False
    return all(lo <= c < hi for (lo, hi), c in zip(extents, coords, strict=True))


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
        if snapshot is None:
            raise TypeError("snapshot must be provided")
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

    async def _aget_manifest(self, manifest_id: bytes) -> ManifestReader:
        """Async version of _get_manifest.

        When the storage backend has an ``aread()`` method (e.g. S3), fetches
        the manifest file without blocking the asyncio event loop so that
        multiple manifest fetches can run concurrently.
        """
        if manifest_id in self._manifest_cache:
            return self._manifest_cache[manifest_id]

        from icepyck.manifest import ManifestReader as MR

        if self._storage is not None and isinstance(self._storage, AsyncStorage):
            manifest = await MR.afrom_storage(manifest_id, self._storage)
        else:
            manifest = MR(self._root_path, manifest_id, storage=self._storage)
        self._manifest_cache[manifest_id] = manifest
        return manifest

    async def prefetch_manifests(self) -> None:
        """Concurrently pre-load all manifests referenced by every array node.

        Calling this before zarr reads coordinate arrays eliminates blocking
        S3 GETs from inside the async ``get()`` path: manifests are already
        in ``_manifest_cache`` so ``_afind_chunk_ref`` finds them immediately.

        Safe to call multiple times; already-cached manifests are skipped.
        """
        # Collect every unique manifest_id across all array nodes.
        manifest_ids = {
            mref.manifest_id
            for node in self._snapshot.list_nodes()
            if node.node_type == "array"
            for mref in node.manifest_refs
            if mref.manifest_id not in self._manifest_cache
        }
        if manifest_ids:
            await asyncio.gather(*(self._aget_manifest(mid) for mid in manifest_ids))

    def _find_chunk_ref(
        self,
        node_path: str,
        node: NodeInfo,
        chunk_coords: tuple[int, ...],
    ) -> ChunkRefInfo | None:
        """Find the ChunkRefInfo for specific chunk coords.

        Uses ManifestRef extents to load only the manifest that covers
        the requested chunk, avoiding loading all manifests.
        """
        # Check cache first
        cached = self._chunk_index.get((node_path, chunk_coords))
        if cached is not None:
            return cached

        # Find which manifest covers these coords using extents
        for mref in node.manifest_refs:
            if not _extents_contain(mref.extents, chunk_coords):
                continue
            # This manifest covers our chunk — load it
            manifest = self._get_manifest(mref.manifest_id)
            # Cache all refs from this manifest
            for cref in manifest.get_chunk_refs(node.node_id):
                self._chunk_index[(node_path, cref.index)] = cref
            # Check if our chunk is in this manifest
            result = self._chunk_index.get((node_path, chunk_coords))
            if result is not None:
                return result

        return None

    async def _afind_chunk_ref(
        self,
        node_path: str,
        node: NodeInfo,
        chunk_coords: tuple[int, ...],
    ) -> ChunkRefInfo | None:
        """Async version of _find_chunk_ref.

        Uses ``_aget_manifest`` so manifest fetches don't block the event loop
        on S3 backends.  Once manifests are in the cache the result is
        identical to the sync path.
        """
        # Check chunk index cache first (populated after first manifest load)
        cached = self._chunk_index.get((node_path, chunk_coords))
        if cached is not None:
            return cached

        # Find which manifest covers these coords using extents
        for mref in node.manifest_refs:
            if not _extents_contain(mref.extents, chunk_coords):
                continue
            # Load manifest asynchronously — no-op if already cached
            manifest = await self._aget_manifest(mref.manifest_id)
            # Cache all refs from this manifest
            for cref in manifest.get_chunk_refs(node.node_id):
                self._chunk_index[(node_path, cref.index)] = cref
            # Check if our chunk is in this manifest
            result = self._chunk_index.get((node_path, chunk_coords))
            if result is not None:
                return result

        return None

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
        if not isinstance(kind, tuple):
            return None
        chunk_coords: tuple[int, ...] = kind

        if node.node_type != "array":
            return None

        # Find the chunk ref using extent-aware manifest lookup
        cref = self._find_chunk_ref(node_path, node, chunk_coords)
        if cref is None:
            return None

        from icepyck.chunks import read_chunk

        return read_chunk(self._root_path, cref, storage=self._storage)

    async def _aget(
        self,
        key: str,
        prototype: BufferPrototype,
        byte_range: ByteRequest | None = None,
    ) -> Buffer | None:
        """Async get that uses ``aread_chunk`` for S3 concurrency."""
        node_path, kind = _parse_key(key)

        if kind == "unknown":
            return None

        node = self._nodes_by_path.get(node_path)
        if node is None:
            return None

        if kind == "metadata":
            raw = node.user_data if node.user_data else None
            if raw is None:
                return None
            data = _apply_byte_range(raw, byte_range)
            return prototype.buffer.from_bytes(data)

        # Chunk coordinate tuple
        if not isinstance(kind, tuple):
            return None
        chunk_coords: tuple[int, ...] = kind

        if node.node_type != "array":
            return None

        cref = await self._afind_chunk_ref(node_path, node, chunk_coords)
        if cref is None:
            return None

        from icepyck.chunks import aread_chunk

        data = await aread_chunk(self._root_path, cref, storage=self._storage)
        data = _apply_byte_range(data, byte_range)
        return prototype.buffer.from_bytes(data)

    # ------------------------------------------------------------------
    # zarr.abc.store.Store interface
    # ------------------------------------------------------------------

    async def get(
        self,
        key: str,
        prototype: BufferPrototype,
        byte_range: ByteRequest | None = None,
    ) -> Buffer | None:
        if self._storage is not None and isinstance(self._storage, AsyncStorage):
            return await self._aget(key, prototype, byte_range)
        data = self._resolve_key(key)
        if data is None:
            return None
        data = _apply_byte_range(data, byte_range)
        return prototype.buffer.from_bytes(data)

    async def get_partial_values(
        self,
        prototype: BufferPrototype,
        key_ranges: Iterable[tuple[str, ByteRequest | None]],
    ) -> list[Buffer | None]:
        key_ranges_list = list(key_ranges)
        return list(
            await asyncio.gather(
                *(
                    self.get(key, prototype, byte_range=br)
                    for key, br in key_ranges_list
                )
            )
        )

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
        # If prefix is outside any chunk namespace, serve entirely from node tree.
        # Otherwise, fall back to full list() for the matching chunk keys only.
        chunk_prefix = self._chunk_prefix_for(prefix)
        if chunk_prefix is None:
            # No chunk namespace involved — yield metadata keys from node tree.
            for key in self._node_keys_with_prefix(prefix):
                yield key
        else:
            # prefix touches a chunk subtree; enumerate only the matching chunks.
            node_path, node = chunk_prefix
            zarr_prefix = "" if node_path == "/" else node_path.lstrip("/")
            base = f"{zarr_prefix}/c/" if zarr_prefix else "c/"
            for idx_str in _iter_chunk_keys_from_metadata(node.user_data):  # type: ignore[arg-type]
                key = f"{base}{idx_str}"
                if key.startswith(prefix):
                    yield key

    async def list_dir(self, prefix: str) -> AsyncGenerator[str, None]:
        if prefix and not prefix.endswith("/"):
            prefix = prefix + "/"

        chunk_prefix = self._chunk_prefix_for(prefix)
        if chunk_prefix is None:
            # Fast path: serve from node tree without touching chunk keys.
            for entry in self._node_dir_entries(prefix):
                yield entry
        else:
            # Inside a chunk subtree — compute children from array shape.
            node_path, node = chunk_prefix
            zarr_prefix = "" if node_path == "/" else node_path.lstrip("/")
            chunk_base = f"{zarr_prefix}/c/" if zarr_prefix else "c/"
            seen: set[str] = set()
            for idx_str in _iter_chunk_keys_from_metadata(node.user_data):  # type: ignore[arg-type]
                full_key = f"{chunk_base}{idx_str}"
                if not full_key.startswith(prefix):
                    continue
                remainder = full_key[len(prefix) :]
                entry = remainder.split("/")[0] + "/" if "/" in remainder else remainder
                if entry and entry not in seen:
                    seen.add(entry)
                    yield entry

    # ------------------------------------------------------------------
    # Helpers for fast node-tree traversal (no chunk enumeration)
    # ------------------------------------------------------------------

    def _icechunk_path_to_zarr_prefix(self, ic_path: str) -> str:
        """Convert an icechunk node path like ``/group1`` to zarr prefix ``group1/``."""
        if ic_path == "/":
            return ""
        return ic_path.lstrip("/") + "/"

    def _chunk_prefix_for(self, prefix: str) -> tuple[str, NodeInfo] | None:
        """Return *(node_path, node)* if *prefix* falls inside a chunk subtree.

        A chunk subtree is ``<array_zarr_prefix>c/``.  Returns *None* when the
        prefix does not overlap with any array's chunk namespace.
        """
        for node_path, node in self._nodes_by_path.items():
            if node.node_type != "array" or not node.user_data:
                continue
            zarr_prefix = self._icechunk_path_to_zarr_prefix(node_path)
            chunk_ns = f"{zarr_prefix}c/"
            # Only match when prefix is inside the chunk namespace
            # e.g. prefix="group1/arr/c/" or "group1/arr/c/0/"
            if prefix.startswith(chunk_ns):
                return node_path, node
        return None

    def _node_keys_with_prefix(self, prefix: str) -> list[str]:
        """Return all metadata keys (zarr.json) that start with *prefix*."""
        keys: list[str] = []
        for ic_path, node in self._nodes_by_path.items():
            if not node.user_data:
                continue
            zarr_prefix = self._icechunk_path_to_zarr_prefix(ic_path)
            key = zarr_prefix + "zarr.json" if zarr_prefix else "zarr.json"
            if key.startswith(prefix):
                keys.append(key)
        return keys

    def _node_dir_entries(self, prefix: str) -> list[str]:
        """Return immediate directory children of *prefix* from the node tree only.

        Does not enumerate chunk keys.  The returned entries are either plain
        file names (``zarr.json``) or directory markers (``subgroup/``).
        """
        seen: set[str] = set()
        entries: list[str] = []

        for ic_path, node in self._nodes_by_path.items():
            if not node.user_data:
                continue
            zarr_prefix = self._icechunk_path_to_zarr_prefix(ic_path)
            # The metadata file for this node
            meta_key = zarr_prefix + "zarr.json" if zarr_prefix else "zarr.json"

            if meta_key.startswith(prefix):
                remainder = meta_key[len(prefix) :]
                entry = remainder.split("/")[0] + "/" if "/" in remainder else remainder
                if entry and entry not in seen:
                    seen.add(entry)
                    entries.append(entry)

            # For array nodes, also expose the "c/" directory marker
            if node.node_type == "array":
                chunk_ns_key = zarr_prefix + "c/"  # e.g. "group1/arr/c/"
                if chunk_ns_key.startswith(prefix) or prefix.startswith(chunk_ns_key):
                    remainder = chunk_ns_key[len(prefix) :]
                    entry = remainder.split("/")[0] + "/" if remainder else ""
                    if entry and entry not in seen:
                        seen.add(entry)
                        entries.append(entry)

        return entries


class IcechunkStore(Store):
    """Read-write zarr v3 Store backed by a WritableSession.

    Writes are buffered in-memory. Call :meth:`WritableSession.commit`
    to persist them. Reads check pending changes first, then fall back
    to the base snapshot.
    """

    def __init__(self, session: object) -> None:
        # session is WritableSession but we use object to avoid circular import
        super().__init__(read_only=False)
        self._session = session
        self._is_open = True

        # Pending writes: key -> bytes (zarr.json metadata or chunk data)
        self._pending: dict[str, bytes] = {}
        # Deleted keys
        self._deleted: set[str] = set()

    @property
    def supports_writes(self) -> bool:  # type: ignore[override]
        return True

    @property
    def supports_deletes(self) -> bool:  # type: ignore[override]
        return True

    @property
    def supports_listing(self) -> bool:  # type: ignore[override]
        return True

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, IcechunkStore):
            return NotImplemented
        return self._session is other._session

    async def set(self, key: str, value: Buffer) -> None:
        data = value.to_bytes()
        self._pending[key] = data
        self._deleted.discard(key)

        # Route to session's change tracking
        node_path, kind = _parse_key(key)
        if kind == "metadata":
            self._session.set_metadata(node_path, data)  # type: ignore[attr-defined]
        elif isinstance(kind, tuple):
            self._session.set_chunk(node_path, kind, data)  # type: ignore[attr-defined]

    async def get(
        self,
        key: str,
        prototype: BufferPrototype,
        byte_range: ByteRequest | None = None,
    ) -> Buffer | None:
        # Check deleted
        if key in self._deleted:
            return None

        # Check pending writes first
        if key in self._pending:
            data = _apply_byte_range(self._pending[key], byte_range)
            return prototype.buffer.from_bytes(data)

        # Fall back to base snapshot read via a read store
        read_store = self._get_read_store()
        return await read_store.get(key, prototype, byte_range)

    async def get_partial_values(
        self,
        prototype: BufferPrototype,
        key_ranges: Iterable[tuple[str, ByteRequest | None]],
    ) -> list[Buffer | None]:
        key_ranges_list = list(key_ranges)
        return list(
            await asyncio.gather(
                *(
                    self.get(key, prototype, byte_range=br)
                    for key, br in key_ranges_list
                )
            )
        )

    async def delete(self, key: str) -> None:
        self._deleted.add(key)
        self._pending.pop(key, None)
        node_path, kind = _parse_key(key)
        if kind == "metadata" or isinstance(kind, tuple):
            self._session.delete_node(node_path)  # type: ignore[attr-defined]

    async def exists(self, key: str) -> bool:
        if key in self._deleted:
            return False
        if key in self._pending:
            return True
        read_store = self._get_read_store()
        return await read_store.exists(key)

    async def list(self) -> AsyncGenerator[str, None]:
        read_store = self._get_read_store()
        seen: set[str] = set()
        # Yield from pending writes
        for key in self._pending:
            if key not in self._deleted:
                seen.add(key)
                yield key
        # Yield from base snapshot
        async for key in read_store.list():
            if key not in self._deleted and key not in seen:
                yield key

    async def list_prefix(self, prefix: str) -> AsyncGenerator[str, None]:
        async for key in self.list():
            if key.startswith(prefix):
                yield key

    async def list_dir(self, prefix: str) -> AsyncGenerator[str, None]:
        if prefix and not prefix.endswith("/"):
            prefix = prefix + "/"
        seen: set[str] = set()
        async for key in self.list():
            if not key.startswith(prefix):
                continue
            remainder = key[len(prefix) :]
            entry = remainder.split("/")[0] + "/" if "/" in remainder else remainder
            if entry and entry not in seen:
                seen.add(entry)
                yield entry

    def _get_read_store(self) -> IcechunkReadStore:
        """Lazily build a read store from the session's base snapshot."""
        if not hasattr(self, "_read_store"):
            from icepyck.snapshot import SnapshotReader

            snap = SnapshotReader(
                storage=self._session._storage,  # type: ignore[attr-defined]
                snapshot_id=self._session._base_snapshot_id,  # type: ignore[attr-defined]
            )
            self._read_store = IcechunkReadStore(
                snapshot=snap,
                storage=self._session._storage,  # type: ignore[attr-defined]
            )
        return self._read_store
