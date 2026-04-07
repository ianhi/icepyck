"""Zarr v3 read-only Store backed by icechunk data.

Translates zarr v3 key access (``zarr.json``, chunk keys) to icechunk
snapshot/manifest/chunk lookups.
"""

from __future__ import annotations

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

    from icepyck.manifest import ManifestReader
    from icepyck.snapshot import NodeInfo, SnapshotReader


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


class IcechunkReadStore(Store):
    """A read-only zarr v3 :class:`~zarr.abc.store.Store` over an icechunk snapshot."""

    def __init__(
        self,
        root_path: Path,
        snapshot: SnapshotReader,
    ) -> None:
        super().__init__(read_only=True)
        self._root_path = root_path
        self._snapshot = snapshot

        # Pre-build lookup structures
        self._nodes_by_path: dict[str, NodeInfo] = {}
        for node in self._snapshot.list_nodes():
            self._nodes_by_path[node.path] = node

        # Lazy-loaded manifest cache
        self._manifest_cache: dict[bytes, ManifestReader] = {}

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
            self._manifest_cache[manifest_id] = MR(self._root_path, manifest_id)
        return self._manifest_cache[manifest_id]

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

        from icepyck.chunks import read_chunk

        for mref in node.manifest_refs:
            manifest = self._get_manifest(mref.manifest_id)
            for cref in manifest.get_chunk_refs(node.node_id):
                if cref.index == chunk_coords:
                    return read_chunk(self._root_path, cref)
        return None

    # ------------------------------------------------------------------
    # zarr.abc.store.Store interface
    # ------------------------------------------------------------------

    async def get(
        self,
        key: str,
        prototype: BufferPrototype,
        byte_range: ByteRequest | None = None,
    ) -> Buffer | None:
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
            if node.node_type == "array":
                zarr_prefix = "" if node.path == "/" else node.path.lstrip("/")
                for mref in node.manifest_refs:
                    manifest = self._get_manifest(mref.manifest_id)
                    for cref in manifest.get_chunk_refs(node.node_id):
                        idx_str = "/".join(str(i) for i in cref.index)
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
