"""Repository and Session classes for Icechunk repositories.

This module contains the main user-facing API: :func:`open`,
:class:`Repository`, and :class:`Session`.
"""

from __future__ import annotations

import json
from datetime import UTC
from pathlib import Path

from icepyck.chunks import read_chunk
from icepyck.manifest import ChunkRefInfo, ManifestReader
from icepyck.repo_state import RepoState
from icepyck.session import WritableSession
from icepyck.snapshot import NodeInfo, SnapshotReader
from icepyck.storage import LocalStorage, S3Storage, Storage
from icepyck.store import IcechunkReadStore
from icepyck.writers import SnapshotInfoData, UpdateData


class ConflictError(Exception):
    """Raised when a commit conflicts with another writer's changes."""


_INITIAL_SNAPSHOT_ID = bytes.fromhex("0b1cc8d6787580f0e33a6534")
"""Well-known ID for the initial empty snapshot (1CECHNKREP0F1RSTCMT0)."""


def open(
    path: str | Path, *, anon: bool = False, **storage_kwargs: object
) -> Repository:
    """Open an Icechunk repository at the given path or S3 URL.

    Parameters
    ----------
    path : str or Path
        Local filesystem path or an ``s3://`` URL.
    anon : bool
        Use anonymous (unsigned) S3 access.  Only relevant for S3 URLs.
    **storage_kwargs
        Extra keyword arguments forwarded to :class:`S3Storage` (e.g.
        ``endpoint_url``, ``key``, ``secret``).
    """
    path_str = str(path)
    if path_str.startswith("s3://"):
        storage: Storage = S3Storage(path_str, anon=anon, **storage_kwargs)
    else:
        storage = LocalStorage(path_str)

    if not storage.exists("repo"):
        if storage.exists("refs") or len(storage.list_prefix("refs")) > 0:
            raise FileNotFoundError(
                f"No 'repo' file found at {path_str!r}. "
                f"Found a 'refs/' directory — this looks like an Icechunk V1 "
                f"repository. icepyck only supports V2 repositories."
            )
        raise FileNotFoundError(
            f"No 'repo' file found at {path_str!r}. Is this an Icechunk repository?"
        )
    return Repository(storage=storage)


class Session:
    """A read-only session bound to a specific snapshot.

    Parameters
    ----------
    repo : Repository
        The parent repository.
    snapshot_id : bytes
        The 12-byte ObjectId12 identifying the snapshot.
    """

    def __init__(self, repo: Repository, snapshot_id: bytes) -> None:
        self._repo = repo
        self._snapshot_id = snapshot_id
        self._snapshot = repo._get_snapshot_by_id(snapshot_id)
        self._store: IcechunkReadStore | None = None

    @property
    def snapshot_id(self) -> str:
        """The Crockford Base32 snapshot ID this session is bound to."""
        from icepyck.crockford import encode as crockford_encode

        return crockford_encode(self._snapshot_id)

    @property
    def snapshot_id_bytes(self) -> bytes:
        """The raw 12-byte snapshot ID this session is bound to."""
        return self._snapshot_id

    def __repr__(self) -> str:
        from icepyck.crockford import encode as crockford_encode

        sid = crockford_encode(self._snapshot_id)
        nodes = self._snapshot.list_nodes()
        n_arrays = sum(1 for n in nodes if n.node_type == "array")
        n_groups = sum(1 for n in nodes if n.node_type == "group")
        return (
            f"Session(snapshot={sid!r}, "
            f"arrays={n_arrays}, groups={n_groups}, read_only=True)"
        )

    @property
    def store(self) -> IcechunkReadStore:
        """Return a zarr v3 read-only Store for this session."""
        if self._store is None:
            self._store = IcechunkReadStore(
                root_path=None,
                snapshot=self._snapshot,
                storage=self._repo._storage,
            )
        return self._store

    def list_nodes(self) -> list[NodeInfo]:
        """List all nodes in this session's snapshot."""
        return self._snapshot.list_nodes()

    def get_array_metadata(self, path: str) -> dict:  # type: ignore[type-arg]
        """Parse and return the zarr.json metadata for an array."""
        node = self._get_array_node(path)
        if node.user_data is None:
            raise ValueError(f"Array {path!r} has no zarr.json metadata")
        return json.loads(node.user_data)  # type: ignore[no-any-return]

    def _get_array_node(self, array_path: str) -> NodeInfo:
        return self._snapshot.get_array_node(array_path)


class Repository:
    """Interface to an Icechunk repository.

    Parameters
    ----------
    path : str or Path, optional
        Root path of the repository.
    storage : Storage, optional
        Storage backend.
    """

    @staticmethod
    def init(
        path: str | Path,
        *,
        storage: Storage | None = None,
    ) -> Repository:
        """Initialize a new empty Icechunk repository.

        Creates the initial empty snapshot and repo file with a ``main``
        branch pointing to it.
        """
        from icepyck.writers import (
            NodeWriteData,
            build_repo,
            build_snapshot,
            build_transaction_log,
        )

        if storage is None:
            storage = LocalStorage(str(path))

        snapshot_id = _INITIAL_SNAPSHOT_ID
        flushed_at = int(__import__("time").time() * 1_000_000)

        root_meta = json.dumps(
            {
                "zarr_format": 3,
                "node_type": "group",
            }
        ).encode()
        from icepyck.ids import generate_id8

        root_node_id = generate_id8()
        snapshot_bytes = build_snapshot(
            snapshot_id=snapshot_id,
            nodes=[
                NodeWriteData(
                    node_id=root_node_id,
                    path="/",
                    user_data=root_meta,
                    node_type="group",
                )
            ],
            message="Repository initialized",
            flushed_at=flushed_at,
        )
        from icepyck.crockford import encode as crockford_encode

        storage.write(f"snapshots/{crockford_encode(snapshot_id)}", snapshot_bytes)

        # Transaction log for the initial snapshot (root group creation)
        txn_bytes = build_transaction_log(
            txn_id=snapshot_id,
            new_groups=[root_node_id],
        )
        storage.write(f"transactions/{crockford_encode(snapshot_id)}", txn_bytes)

        repo_bytes = build_repo(
            spec_version=2,
            branches={"main": 0},
            tags={},
            snapshots=[
                SnapshotInfoData(
                    snapshot_id=snapshot_id,
                    parent_offset=-1,
                    flushed_at=flushed_at,
                    message="Repository initialized",
                )
            ],
            updates=[
                UpdateData(kind="repo_initialized", updated_at=flushed_at),
            ],
        )
        storage.write("repo", repo_bytes)

        return Repository(storage=storage)

    def __init__(
        self,
        path: str | Path | None = None,
        *,
        storage: Storage | None = None,
    ) -> None:
        if storage is not None:
            self._storage: Storage = storage
        elif path is not None:
            self._storage = LocalStorage(str(path))
        else:
            raise TypeError("Either path or storage must be provided")
        self._state = RepoState.from_storage(self._storage)
        self._snapshot_cache: dict[bytes, SnapshotReader] = {}
        self._manifest_cache: dict[bytes, ManifestReader] = {}

    def __repr__(self) -> str:
        branches = list(self._state.branches.keys())
        tags = list(self._state.tags.keys())
        parts = [f"branches={branches!r}"]
        if tags:
            parts.append(f"tags={tags!r}")
        return f"Repository({', '.join(parts)})"

    # ------------------------------------------------------------------
    # Session-based API
    # ------------------------------------------------------------------

    def readonly_session(
        self,
        *,
        branch: str | None = None,
        tag: str | None = None,
        snapshot: str | None = None,
    ) -> Session:
        """Get a read-only session for a specific branch, tag, or snapshot.

        Exactly one of *branch*, *tag*, or *snapshot* must be provided.
        """
        specified = sum(x is not None for x in (branch, tag, snapshot))
        if specified != 1:
            raise ValueError(
                "Exactly one of branch, tag, or snapshot must be specified"
            )
        if branch is not None:
            ref = branch
        elif tag is not None:
            ref = tag
        else:
            ref = snapshot  # type: ignore[assignment]
        snapshot_id = self._resolve_ref(ref)
        return Session(self, snapshot_id)

    def writable_session(
        self,
        *,
        branch: str = "main",
    ) -> WritableSession:
        """Get a writable session for a branch.

        Changes are buffered in memory until
        :meth:`WritableSession.commit` is called.
        """
        # Re-read to pick up external changes (e.g. another process committed)
        self.refresh()
        snapshot_id = self._resolve_ref(branch)
        snap = self._get_snapshot_by_id(snapshot_id)
        return WritableSession(
            storage=self._storage,
            branch=branch,
            base_snapshot_id=snapshot_id,
            base_nodes=snap.list_nodes(),
            repo=self,
        )

    def log(self, branch: str = "main") -> list[dict[str, object]]:
        """Return the commit ancestry for a branch, newest first.

        Each entry has keys: ``id``, ``parent``, ``message``, ``time``.
        """
        from datetime import datetime

        from icepyck.crockford import encode as crockford_encode

        if branch not in self._state.branches:
            raise KeyError(f"Branch not found: {branch!r}")

        idx = self._state.branches[branch]
        result = []
        visited: set[int] = set()
        while idx >= 0 and idx not in visited:
            visited.add(idx)
            snap = self._state.snapshots[idx]
            ts = datetime.fromtimestamp(snap.flushed_at / 1_000_000, tz=UTC)
            parent_id = (
                crockford_encode(self._state.snapshots[snap.parent_offset].snapshot_id)
                if snap.parent_offset >= 0
                else None
            )
            result.append(
                {
                    "id": crockford_encode(snap.snapshot_id),
                    "parent": parent_id,
                    "message": snap.message,
                    "time": ts.strftime("%Y-%m-%d %H:%M:%S UTC"),
                }
            )
            idx = snap.parent_offset
        return result

    # ------------------------------------------------------------------
    # Branch & tag management
    # ------------------------------------------------------------------

    def create_branch(self, name: str, snapshot: str) -> None:
        """Create a new branch pointing to the given snapshot."""
        if "/" in name:
            raise ValueError("Branch names must not contain '/'")
        if name in self._state.branches:
            raise KeyError(f"Branch already exists: {name!r}")
        snapshot_id = self._resolve_ref(snapshot)
        snap_idx = self._state.find_snapshot_index(snapshot_id)
        if snap_idx < 0:
            raise KeyError(f"Snapshot not found: {snapshot!r}")
        self._state.branches[name] = snap_idx
        self._flush_repo([UpdateData(kind="branch_created", name=name)])

    def delete_branch(self, name: str) -> None:
        """Delete a branch. Cannot delete ``"main"``."""
        if name == "main":
            raise ValueError("Cannot delete the 'main' branch")
        if name not in self._state.branches:
            raise KeyError(f"Branch not found: {name!r}")
        snap_idx = self._state.branches.pop(name)
        prev_snap_id = self._state.snapshots[snap_idx].snapshot_id
        self._flush_repo(
            [
                UpdateData(
                    kind="branch_deleted",
                    name=name,
                    previous_snap_id=prev_snap_id,
                )
            ]
        )

    def create_tag(self, name: str, snapshot: str) -> None:
        """Create an immutable tag pointing to the given snapshot."""
        if "/" in name:
            raise ValueError("Tag names must not contain '/'")
        if name in self._state.tags:
            raise KeyError(f"Tag already exists: {name!r}")
        if name in self._state.deleted_tags:
            raise KeyError(
                f"Tag {name!r} was previously deleted and cannot be recreated"
            )
        snapshot_id = self._resolve_ref(snapshot)
        snap_idx = self._state.find_snapshot_index(snapshot_id)
        if snap_idx < 0:
            raise KeyError(f"Snapshot not found: {snapshot!r}")
        self._state.tags[name] = snap_idx
        self._flush_repo([UpdateData(kind="tag_created", name=name)])

    def delete_tag(self, name: str) -> None:
        """Delete a tag (tombstoned — name cannot be reused)."""
        if name not in self._state.tags:
            raise KeyError(f"Tag not found: {name!r}")
        snap_idx = self._state.tags.pop(name)
        prev_snap_id = self._state.snapshots[snap_idx].snapshot_id
        self._state.deleted_tags.append(name)
        self._flush_repo(
            [
                UpdateData(
                    kind="tag_deleted",
                    name=name,
                    previous_snap_id=prev_snap_id,
                )
            ]
        )

    def list_branches(self) -> list[str]:
        """Return the names of all branches."""
        return sorted(self._state.branches.keys())

    def list_tags(self) -> list[str]:
        """Return the names of all tags."""
        return sorted(self._state.tags.keys())

    # ------------------------------------------------------------------
    # Legacy API (backward-compatible)
    # ------------------------------------------------------------------

    def list_nodes(self, ref: str = "main") -> list[NodeInfo]:
        """List all nodes in the snapshot referenced by *ref*."""
        snapshot = self._get_snapshot(ref)
        return snapshot.list_nodes()

    def read_chunk(
        self, ref: str, array_path: str, chunk_index: tuple[int, ...]
    ) -> bytes:
        """Read a single chunk from an array."""
        chunk_ref = self._find_chunk_ref(ref, array_path, chunk_index)
        return read_chunk(None, chunk_ref, storage=self._storage)

    def read_all_chunks(
        self, ref: str, array_path: str
    ) -> dict[tuple[int, ...], bytes]:
        """Read all chunks for an array."""
        snapshot = self._get_snapshot(ref)
        manifest_refs = snapshot.get_array_manifest_refs(array_path)
        node_info = self._get_array_node(snapshot, array_path)
        node_id = node_info.node_id

        result: dict[tuple[int, ...], bytes] = {}
        for mref in manifest_refs:
            manifest = self._get_manifest(mref.manifest_id)
            for cref in manifest.get_chunk_refs(node_id):
                result[cref.index] = read_chunk(None, cref, storage=self._storage)
        return result

    def get_array_metadata(self, ref: str, array_path: str) -> dict:  # type: ignore[type-arg]
        """Parse and return the zarr.json metadata for an array."""
        snapshot = self._get_snapshot(ref)
        node_info = self._get_array_node(snapshot, array_path)
        if node_info.user_data is None:
            raise ValueError(f"Array {array_path!r} has no zarr.json metadata")
        return json.loads(node_info.user_data)  # type: ignore[no-any-return]

    # ------------------------------------------------------------------
    # Internal: repo file management
    # ------------------------------------------------------------------

    def refresh(self) -> None:
        """Re-read repo state from storage.

        Call this to pick up changes made by external processes.
        Automatically called by :meth:`writable_session`.
        """
        if self._storage is not None:
            self._state = RepoState.from_storage(self._storage)
            self._snapshot_cache.clear()

    def _apply_commit(self, result: object) -> None:
        """Apply a commit: update in-memory state and flush repo file.

        Called by WritableSession.commit(). Accepts a CommitResult.
        Raises ConflictError if the branch has been updated since the
        session was created (another writer committed first).
        """
        from icepyck.session import CommitResult

        if not isinstance(result, CommitResult):
            raise TypeError(f"Expected CommitResult, got {type(result)}")

        # Conflict detection: verify the branch still points to the
        # expected parent snapshot. If it's moved, another writer committed.
        current_branch_idx = self._state.branches.get(result.branch)
        if current_branch_idx is not None:
            current_snap_id = self._state.snapshots[current_branch_idx].snapshot_id
            if current_snap_id != result.parent_snapshot_id:
                from icepyck.crockford import encode as crockford_encode

                raise ConflictError(
                    f"Branch {result.branch!r} was updated by another writer. "
                    f"Expected parent {crockford_encode(result.parent_snapshot_id)}, "
                    f"but branch now points to {crockford_encode(current_snap_id)}. "
                    f"Call repo.refresh() and retry."
                )

        parent_idx = self._state.find_snapshot_index(result.parent_snapshot_id)
        if parent_idx < 0:
            from icepyck.crockford import encode as crockford_encode

            raise RuntimeError(
                f"Parent snapshot {crockford_encode(result.parent_snapshot_id)} "
                f"not found in repository snapshot list"
            )
        new_snap = SnapshotInfoData(
            snapshot_id=result.snapshot_id,
            parent_offset=parent_idx,
            flushed_at=result.flushed_at,
            message=result.message,
        )
        new_idx = self._state.add_snapshot(new_snap)
        self._state.branches[result.branch] = new_idx
        self._flush_repo(
            [
                UpdateData(
                    kind="new_commit",
                    branch=result.branch,
                    snapshot_id=result.snapshot_id,
                    updated_at=result.flushed_at,
                )
            ]
        )

    def _flush_repo(self, updates: list[UpdateData] | None = None) -> None:
        """Serialize self._state to the repo file via conditional write.

        Uses the version token from when we last read the repo file.
        Raises ConflictError if another writer modified it in between.
        """
        from icepyck.storage import VersionMismatchError
        from icepyck.writers import build_repo

        repo_bytes = build_repo(
            spec_version=2,
            branches=self._state.branches,
            tags=self._state.tags,
            snapshots=self._state.snapshots,
            deleted_tags=self._state.deleted_tags,
            updates=updates,
            metadata=self._state.metadata,
            config=self._state.config,
            enabled_feature_flags=self._state.enabled_feature_flags,
            disabled_feature_flags=self._state.disabled_feature_flags,
        )
        try:
            new_version = self._storage.conditional_write(
                "repo", repo_bytes, self._state.version
            )
            self._state.version = new_version
        except VersionMismatchError as e:
            raise ConflictError(f"Repo file was modified by another writer: {e}") from e

    # ------------------------------------------------------------------
    # Internal: ref resolution and caching
    # ------------------------------------------------------------------

    def _resolve_ref(self, ref: str) -> bytes:
        """Resolve a ref string to a 12-byte snapshot ID."""
        from icepyck.crockford import decode as crockford_decode
        from icepyck.crockford import encode as crockford_encode

        try:
            return self._state.get_snapshot_id_by_branch(ref)
        except KeyError:
            pass
        try:
            return self._state.get_snapshot_id_by_tag(ref)
        except KeyError:
            pass
        try:
            raw = crockford_decode(ref)
            if len(raw) == 12:
                return raw
        except (ValueError, KeyError):
            pass
        ref_upper = ref.upper()
        if ref_upper and all(
            c in "0123456789ABCDEFGHJKMNPQRSTVWXYZ" for c in ref_upper
        ):
            matches = [
                s.snapshot_id
                for s in self._state.snapshots
                if crockford_encode(s.snapshot_id).startswith(ref_upper)
            ]
            if len(matches) == 1:
                return matches[0]
            if len(matches) > 1:
                raise KeyError(
                    f"Ambiguous ref {ref!r}: matches {len(matches)} snapshots. "
                    f"Use more characters to disambiguate."
                )
        try:
            raw = bytes.fromhex(ref)
            if len(raw) == 12:
                return raw
        except ValueError:
            pass
        raise KeyError(f"Could not resolve ref: {ref!r}")

    def _get_snapshot(self, ref: str) -> SnapshotReader:
        snapshot_id = self._resolve_ref(ref)
        return self._get_snapshot_by_id(snapshot_id)

    def _get_snapshot_by_id(self, snapshot_id: bytes) -> SnapshotReader:
        if snapshot_id not in self._snapshot_cache:
            self._snapshot_cache[snapshot_id] = SnapshotReader(
                None, snapshot_id, storage=self._storage
            )
        return self._snapshot_cache[snapshot_id]

    def _get_manifest(self, manifest_id: bytes) -> ManifestReader:
        if manifest_id not in self._manifest_cache:
            self._manifest_cache[manifest_id] = ManifestReader(
                None, manifest_id, storage=self._storage
            )
        return self._manifest_cache[manifest_id]

    @staticmethod
    def _get_array_node(snapshot: SnapshotReader, array_path: str) -> NodeInfo:
        return snapshot.get_array_node(array_path)

    def _find_chunk_ref(
        self, ref: str, array_path: str, chunk_index: tuple[int, ...]
    ) -> ChunkRefInfo:
        snapshot = self._get_snapshot(ref)
        manifest_refs = snapshot.get_array_manifest_refs(array_path)
        node_info = self._get_array_node(snapshot, array_path)
        node_id = node_info.node_id

        for mref in manifest_refs:
            manifest = self._get_manifest(mref.manifest_id)
            for cref in manifest.get_chunk_refs(node_id):
                if cref.index == chunk_index:
                    return cref

        raise KeyError(f"Chunk {chunk_index} not found for array {array_path!r}")
