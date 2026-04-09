"""Writable session for Icechunk repositories.

Tracks in-memory changes (new nodes, deleted nodes, modified metadata,
pending chunks) and commits them atomically by writing new chunk files,
manifest files, a snapshot file, a transaction log, and updating the
repo file.
"""

from __future__ import annotations

import json
import time
from typing import TYPE_CHECKING

from icepyck.crockford import encode as crockford_encode
from icepyck.ids import content_hash_id12, generate_id8, generate_id12
from icepyck.writers import (
    ArrayManifestData,
    ArrayUpdatedChunksData,
    ChunkRefData,
    ManifestFileData,
    ManifestRefData,
    NodeWriteData,
    build_manifest,
    build_snapshot,
    build_transaction_log,
)

if TYPE_CHECKING:
    from icepyck.repository import Repository
    from icepyck.snapshot import NodeInfo
    from icepyck.storage import Storage


class WritableSession:
    """A writable session that tracks changes and commits them atomically.

    Writes immutable files (chunks, manifests, snapshots, transaction logs)
    directly to storage. Delegates repo file updates to the owning
    Repository via ``repo._apply_commit()``.

    Parameters
    ----------
    storage : Storage
        Storage backend.
    branch : str
        Branch name to commit to.
    base_snapshot_id : bytes
        The 12-byte ObjectId12 of the snapshot this session is based on.
    base_nodes : list[NodeInfo]
        The node list from the base snapshot.
    repo : Repository
        Back-reference to the owning Repository (for repo file updates).
    """

    def __init__(
        self,
        storage: Storage,
        branch: str,
        base_snapshot_id: bytes,
        base_nodes: list[NodeInfo],
        repo: Repository,
    ) -> None:
        self._storage = storage
        self._branch = branch
        self._base_snapshot_id = base_snapshot_id
        self._repo = repo

        # Snapshot of existing nodes, keyed by path
        self._base_nodes: dict[str, NodeInfo] = {n.path: n for n in base_nodes}

        # ----- Change tracking -----
        # New/replaced nodes (path -> NodeWriteData built at commit time)
        self._new_nodes: dict[str, _PendingNode] = {}
        # Deleted paths
        self._deleted_paths: set[str] = set()
        # Pending chunk data: (path, chunk_coords) -> compressed bytes
        self._pending_chunks: dict[tuple[str, tuple[int, ...]], bytes] = {}
        # Modified metadata: path -> new zarr.json bytes
        self._modified_metadata: dict[str, bytes] = {}

    @property
    def snapshot_id(self) -> str:
        """The Crockford Base32 snapshot ID this session is based on."""
        return crockford_encode(self._base_snapshot_id)

    def __repr__(self) -> str:
        from icepyck.crockford import encode as crockford_encode

        sid = crockford_encode(self._base_snapshot_id)
        n_pending = len(self._pending_chunks)
        n_new = len(self._new_nodes)
        n_del = len(self._deleted_paths)
        n_meta = len(self._modified_metadata)
        changes = []
        if n_new:
            changes.append(f"{n_new} new nodes")
        if n_del:
            changes.append(f"{n_del} deleted")
        if n_meta:
            changes.append(f"{n_meta} metadata changes")
        if n_pending:
            changes.append(f"{n_pending} pending chunks")
        status = ", ".join(changes) if changes else "clean"
        return f"WritableSession(branch={self._branch!r}, base={sid!r}, {status})"

    @property
    def store(self) -> object:
        """Return a zarr v3 read-write Store for this session."""
        if not hasattr(self, "_store") or self._store is None:
            from icepyck.store import IcechunkStore

            self._store = IcechunkStore(self)
        return self._store

    # ------------------------------------------------------------------
    # Public mutation API
    # ------------------------------------------------------------------

    def set_metadata(self, path: str, zarr_json: bytes) -> None:
        """Set zarr.json metadata for a node.

        Creates the node if it doesn't exist. For array nodes, chunk data
        is tracked separately via :meth:`set_chunk`.
        """
        meta = json.loads(zarr_json)
        node_type = meta.get("node_type", "group")

        if path in self._new_nodes:
            self._new_nodes[path].user_data = zarr_json
            self._new_nodes[path].node_type = node_type
        elif path in self._base_nodes:
            self._modified_metadata[path] = zarr_json
        else:
            # Brand new node
            self._new_nodes[path] = _PendingNode(
                node_id=generate_id8(),
                path=path,
                user_data=zarr_json,
                node_type=node_type,
            )

    def set_chunk(self, path: str, chunk_coords: tuple[int, ...], data: bytes) -> None:
        """Buffer chunk data for later commit.

        The array node must already exist (via :meth:`set_metadata` or in
        the base snapshot).
        """
        self._pending_chunks[(path, chunk_coords)] = data

    def delete_node(self, path: str) -> None:
        """Mark a node for deletion."""
        self._deleted_paths.add(path)
        self._new_nodes.pop(path, None)
        self._modified_metadata.pop(path, None)
        # Remove pending chunks for this path
        to_remove = [k for k in self._pending_chunks if k[0] == path]
        for k in to_remove:
            del self._pending_chunks[k]

    # ------------------------------------------------------------------
    # Commit
    # ------------------------------------------------------------------

    def commit(self, message: str = "") -> str:
        """Commit all pending changes. Returns the new snapshot ID as a Crockford string.

        Writes:
        1. Chunk files to ``chunks/``
        2. Manifest file(s) to ``manifests/``
        3. Snapshot file to ``snapshots/``
        4. Transaction log to ``transactions/``
        5. Updated repo file
        """
        flushed_at = int(time.time() * 1_000_000)

        # --- Step 1: Write chunks, collect per-array chunk refs ---
        # array_path -> list of ChunkRefData
        array_chunk_refs: dict[str, list[ChunkRefData]] = {}
        for (path, coords), data in self._pending_chunks.items():
            chunk_id = content_hash_id12(data)
            chunk_path = f"chunks/{crockford_encode(chunk_id)}"
            if not self._storage.exists(chunk_path):
                self._storage.write(chunk_path, data)
            cref = ChunkRefData(
                index=coords, chunk_id=chunk_id, offset=0, length=len(data)
            )
            array_chunk_refs.setdefault(path, []).append(cref)

        # --- Step 2: Build manifests ---
        # One manifest per array that has new chunks.
        # key: array_path, val: (manifest_id, ManifestRefData, manifest_bytes_len, num_refs)
        new_manifests: dict[str, tuple[bytes, ManifestRefData, int, int]] = {}
        for array_path, crefs in array_chunk_refs.items():
            node_id = self._get_node_id(array_path)
            # Merge with existing chunks from base snapshot
            merged_refs = self._merge_chunk_refs(array_path, node_id, crefs)

            am = ArrayManifestData(node_id=node_id, refs=merged_refs)
            manifest_id = generate_id12()
            manifest_bytes = build_manifest(manifest_id, [am])
            self._storage.write(
                f"manifests/{crockford_encode(manifest_id)}", manifest_bytes
            )

            # Compute extents from chunk refs
            extents = _compute_extents(merged_refs)
            mref = ManifestRefData(manifest_id=manifest_id, extents=extents)
            new_manifests[array_path] = (
                manifest_id,
                mref,
                len(manifest_bytes),
                len(merged_refs),
            )

        # --- Step 3: Build the new snapshot ---
        snapshot_nodes = self._build_snapshot_nodes(new_manifests)
        manifest_files = [
            ManifestFileData(manifest_id=mid, size_bytes=sz, num_chunk_refs=nrefs)
            for mid, _, sz, nrefs in new_manifests.values()
        ]
        # Also carry forward manifest files from base nodes that weren't replaced
        # (we don't need to — manifest_files_v2 is informational)

        snapshot_id = generate_id12()
        snapshot_bytes = build_snapshot(
            snapshot_id=snapshot_id,
            nodes=snapshot_nodes,
            message=message,
            manifest_files=manifest_files,
            flushed_at=flushed_at,
        )
        self._storage.write(
            f"snapshots/{crockford_encode(snapshot_id)}", snapshot_bytes
        )

        # --- Step 4: Transaction log ---
        new_group_ids = []
        new_array_ids = []
        updated_array_ids = []
        updated_chunks_data = []

        for _path, pnode in self._new_nodes.items():
            if pnode.node_type == "array":
                new_array_ids.append(pnode.node_id)
            else:
                new_group_ids.append(pnode.node_id)

        for path in self._modified_metadata:
            node = self._base_nodes.get(path)
            if node and node.node_type == "array":
                updated_array_ids.append(node.node_id)

        for array_path, crefs in array_chunk_refs.items():
            node_id = self._get_node_id(array_path)
            updated_chunks_data.append(
                ArrayUpdatedChunksData(
                    node_id=node_id,
                    chunk_indices=[c.index for c in crefs],
                )
            )

        txn_bytes = build_transaction_log(
            txn_id=snapshot_id,
            new_groups=new_group_ids,
            new_arrays=new_array_ids,
            updated_arrays=updated_array_ids,
            updated_chunks=updated_chunks_data,
        )
        self._storage.write(f"transactions/{crockford_encode(snapshot_id)}", txn_bytes)

        # --- Step 5: Delegate repo file update to Repository ---
        self._repo._apply_commit(
            branch=self._branch,
            snapshot_id=snapshot_id,
            parent_snapshot_id=self._base_snapshot_id,
            flushed_at=flushed_at,
            message=message,
        )

        # Reset change tracking for potential further commits
        self._base_snapshot_id = snapshot_id
        from icepyck.snapshot import SnapshotReader

        new_snap = SnapshotReader(storage=self._storage, snapshot_id=snapshot_id)
        self._base_nodes = {n.path: n for n in new_snap.list_nodes()}
        self._new_nodes.clear()
        self._deleted_paths.clear()
        self._pending_chunks.clear()
        self._modified_metadata.clear()

        return crockford_encode(snapshot_id)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_node_id(self, path: str) -> bytes:
        """Get the node_id for a path (new or existing)."""
        if path in self._new_nodes:
            return self._new_nodes[path].node_id
        if path in self._base_nodes:
            return self._base_nodes[path].node_id
        raise KeyError(f"Node not found: {path!r}")

    def _merge_chunk_refs(
        self,
        array_path: str,
        node_id: bytes,
        new_crefs: list[ChunkRefData],
    ) -> list[ChunkRefData]:
        """Merge new chunk refs with existing ones from the base snapshot.

        New refs override existing ones at the same index.
        """
        from icepyck.manifest import ManifestReader

        # Start with existing refs from base snapshot
        existing: dict[tuple[int, ...], ChunkRefData] = {}
        base = self._base_nodes.get(array_path)
        if base is not None and base.manifest_refs:
            for mref in base.manifest_refs:
                reader = ManifestReader(None, mref.manifest_id, storage=self._storage)
                for cref in reader.get_chunk_refs(node_id):
                    existing[cref.index] = ChunkRefData(
                        index=cref.index,
                        inline_data=cref.inline_data,
                        chunk_id=cref.chunk_id,
                        offset=cref.offset,
                        length=cref.length,
                    )

        # Override with new refs
        for cref in new_crefs:
            existing[cref.index] = cref

        # Return sorted by index
        return sorted(existing.values(), key=lambda c: c.index)

    def _build_snapshot_nodes(
        self,
        new_manifests: dict[str, tuple[bytes, ManifestRefData, int, int]],
    ) -> list[NodeWriteData]:
        """Build the complete node list for the new snapshot."""
        nodes: list[NodeWriteData] = []

        # Start with base nodes
        for path, base_node in self._base_nodes.items():
            if path in self._deleted_paths:
                continue

            user_data = base_node.user_data
            if path in self._modified_metadata:
                user_data = self._modified_metadata[path]

            # Manifest refs: use new manifest if we have one, else carry forward
            if path in new_manifests:
                _, mref, _, _ = new_manifests[path]
                manifest_refs = [mref]
            else:
                manifest_refs = [
                    ManifestRefData(
                        manifest_id=mr.manifest_id,
                        extents=mr.extents,
                    )
                    for mr in base_node.manifest_refs
                ]

            nodes.append(
                NodeWriteData(
                    node_id=base_node.node_id,
                    path=path,
                    user_data=user_data,
                    node_type=base_node.node_type,
                    manifests=manifest_refs,
                )
            )

        # Add new nodes
        for path, pnode in self._new_nodes.items():
            if path in self._base_nodes:
                continue  # already handled above (metadata update)

            manifest_refs = []
            if path in new_manifests:
                _, mref, _, _ = new_manifests[path]
                manifest_refs = [mref]

            nodes.append(
                NodeWriteData(
                    node_id=pnode.node_id,
                    path=path,
                    user_data=pnode.user_data,
                    node_type=pnode.node_type,
                    manifests=manifest_refs,
                )
            )

        return nodes


def _compute_extents(refs: list[ChunkRefData]) -> list[tuple[int, int]]:
    """Compute per-dimension (min, max+1) extents from chunk refs."""
    if not refs:
        return []
    ndim = len(refs[0].index)
    if ndim == 0:
        return []  # scalar
    extents: list[tuple[int, int]] = []
    for dim in range(ndim):
        vals = [r.index[dim] for r in refs]
        extents.append((min(vals), max(vals) + 1))
    return extents


class _PendingNode:
    """In-memory representation of a node pending commit."""

    __slots__ = ("node_id", "path", "user_data", "node_type")

    def __init__(
        self,
        node_id: bytes,
        path: str,
        user_data: bytes | None,
        node_type: str,
    ) -> None:
        self.node_id = node_id
        self.path = path
        self.user_data = user_data
        self.node_type = node_type
