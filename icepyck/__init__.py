"""Icepyck — a Python client for Icechunk repositories."""

from __future__ import annotations

import json
from pathlib import Path

from icepyck.chunks import read_chunk
from icepyck.manifest import ChunkRefInfo, ManifestReader
from icepyck.repo import RepoInfo
from icepyck.session import WritableSession
from icepyck.snapshot import NodeInfo, SnapshotReader
from icepyck.storage import LocalStorage, S3Storage, Storage
from icepyck.store import IcechunkReadStore

__all__ = [
    "LocalStorage",
    "Repository",
    "S3Storage",
    "Session",
    "Storage",
    "WritableSession",
    "open",
]


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
        # Check for V1 layout (refs/ directory instead of repo file)
        if storage.exists("refs") or len(storage.list_prefix("refs")) > 0:
            raise FileNotFoundError(
                f"No 'repo' file found at {path_str!r}. "
                f"Found a 'refs/' directory — this looks like an Icechunk V1 "
                f"repository. icepyck only supports V2 repositories."
            )
        raise FileNotFoundError(
            f"No 'repo' file found at {path_str!r}. "
            f"Is this an Icechunk repository?"
        )
    return Repository(storage=storage)


_INITIAL_SNAPSHOT_ID = bytes.fromhex("0b1cc8d6787580f0e33a6534")
"""Well-known ID for the initial empty snapshot (1CECHNKREP0F1RSTCMT0)."""


class Session:
    """A read-only session bound to a specific snapshot.

    Provides a zarr-compatible :pyclass:`~icepyck.store.IcechunkReadStore`
    and convenience methods for inspecting the snapshot contents.

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
                root_path=self._repo._root,
                snapshot=self._snapshot,
                storage=self._repo._storage,
            )
        return self._store

    def list_nodes(self) -> list[NodeInfo]:
        """List all nodes in this session's snapshot."""
        return self._snapshot.list_nodes()

    def get_array_metadata(self, path: str) -> dict:  # type: ignore[type-arg]
        """Parse and return the zarr.json metadata for an array.

        Parameters
        ----------
        path : str
            Path of the array node (e.g. ``"/group1/temperatures"``).

        Returns
        -------
        dict
            The parsed zarr.json metadata.
        """
        node = self._get_array_node(path)
        if node.user_data is None:
            raise ValueError(f"Array {path!r} has no zarr.json metadata")
        return json.loads(node.user_data)  # type: ignore[no-any-return]

    def _get_array_node(self, array_path: str) -> NodeInfo:
        """Find an array node by path in this session's snapshot."""
        for node in self._snapshot.list_nodes():
            if node.path == array_path and node.node_type == "array":
                return node
        raise KeyError(f"Array node not found: {array_path!r}")


class Repository:
    """Read-only interface to an Icechunk repository.

    Parameters
    ----------
    path : str or Path, optional
        Root path of the repository (the directory containing ``repo``,
        ``snapshots/``, ``manifests/``, ``chunks/``).  Ignored when
        *storage* is provided.
    storage : Storage, optional
        Storage backend.  When provided, all I/O goes through it.
    """

    @staticmethod
    def init(
        path: str | Path,
        *,
        storage: Storage | None = None,
    ) -> "Repository":
        """Initialize a new empty Icechunk repository.

        Creates the initial empty snapshot and repo file with a ``main``
        branch pointing to it.

        Parameters
        ----------
        path : str or Path
            Root path for the repository.
        storage : Storage, optional
            Storage backend. If not provided, uses :class:`LocalStorage`.
        """
        from icepyck.writers import (
            NodeWriteData,
            SnapshotInfoData,
            build_repo,
            build_snapshot,
        )

        if storage is None:
            storage = LocalStorage(str(path))

        snapshot_id = _INITIAL_SNAPSHOT_ID
        flushed_at = int(__import__("time").time() * 1_000_000)

        # Write initial empty snapshot with just a root group
        root_meta = json.dumps({
            "zarr_format": 3,
            "node_type": "group",
        }).encode()
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

        storage.write(
            f"snapshots/{crockford_encode(snapshot_id)}", snapshot_bytes
        )

        # Write repo file
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
            self._storage: Storage | None = storage
            self._root: Path | None = None
            self._repo = RepoInfo(storage=storage)
        elif path is not None:
            self._root = Path(path)
            self._storage = None
            self._repo = RepoInfo(self._root / "repo")
        else:
            raise TypeError("Either path or storage must be provided")
        self._snapshot_cache: dict[bytes, SnapshotReader] = {}
        self._manifest_cache: dict[bytes, ManifestReader] = {}

    def __repr__(self) -> str:
        branches = self._repo.list_branches()
        tags = self._repo.list_tags()
        parts = [f"branches={branches!r}"]
        if tags:
            parts.append(f"tags={tags!r}")
        return f"Repository({', '.join(parts)})"

    # ------------------------------------------------------------------
    # Session-based API (primary)
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

        Parameters
        ----------
        branch : str, optional
            A branch name (e.g. ``"main"``).
        tag : str, optional
            A tag name (e.g. ``"v1"``).
        snapshot : str, optional
            A snapshot ID as a Crockford Base32 string (e.g.
            ``"RF238TWZTXGD49BDPXWG"``), a unique prefix thereof,
            or a hex-encoded ID.

        Returns
        -------
        Session
            A read-only session bound to the resolved snapshot.
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
            # snapshot is not None — guaranteed by the `specified == 1` check above
            ref = snapshot  # type: ignore[assignment]
        snapshot_id = self._resolve_ref(ref)
        return Session(self, snapshot_id)

    def writable_session(
        self,
        *,
        branch: str = "main",
    ) -> WritableSession:
        """Get a writable session for a branch.

        Changes made through the session are buffered in memory until
        :meth:`WritableSession.commit` is called.

        Parameters
        ----------
        branch : str
            Branch name to commit to (default ``"main"``).
        """
        snapshot_id = self._resolve_ref(branch)
        snapshot = self._get_snapshot_by_id(snapshot_id)
        if self._storage is None:
            raise TypeError("Writable sessions require a storage backend")
        return WritableSession(
            storage=self._storage,
            branch=branch,
            base_snapshot_id=snapshot_id,
            base_nodes=snapshot.list_nodes(),
            repo_snapshots=self._repo.get_snapshots_data(),
            repo_branches=self._repo.get_branches_data(),
            repo_tags=self._repo.get_tags_data(),
        )

    def log(self, branch: str = "main") -> list[dict[str, object]]:
        """Return the commit ancestry for a branch, newest first.

        Each entry has keys: ``id``, ``parent``, ``message``, ``flushed_at``.
        """
        from datetime import datetime, timezone

        from icepyck.crockford import encode as crockford_encode

        snapshots = self._repo.get_snapshots_data()
        branches = self._repo.get_branches_data()

        if branch not in branches:
            raise KeyError(f"Branch not found: {branch!r}")

        # Walk the parent chain from the branch tip
        idx = branches[branch]
        result = []
        visited: set[int] = set()
        while idx >= 0 and idx not in visited:
            visited.add(idx)
            sid, parent_offset, flushed_at, message = snapshots[idx]
            ts = datetime.fromtimestamp(
                flushed_at / 1_000_000, tz=timezone.utc
            )
            result.append({
                "id": crockford_encode(sid),
                "parent": crockford_encode(snapshots[parent_offset][0]) if parent_offset >= 0 else None,
                "message": message,
                "time": ts.strftime("%Y-%m-%d %H:%M:%S UTC"),
            })
            idx = parent_offset
        return result

    def list_branches(self) -> list[str]:
        """Return the names of all branches."""
        return self._repo.list_branches()

    def list_tags(self) -> list[str]:
        """Return the names of all tags."""
        return self._repo.list_tags()

    # ------------------------------------------------------------------
    # Legacy API (backward-compatible)
    # ------------------------------------------------------------------

    def list_nodes(self, ref: str = "main") -> list[NodeInfo]:
        """List all nodes in the snapshot referenced by *ref*.

        Parameters
        ----------
        ref : str
            A branch name, tag name, or hex-encoded snapshot ID.
        """
        snapshot = self._get_snapshot(ref)
        return snapshot.list_nodes()

    def read_chunk(
        self, ref: str, array_path: str, chunk_index: tuple[int, ...]
    ) -> bytes:
        """Read a single chunk from an array.

        Parameters
        ----------
        ref : str
            A branch name, tag name, or hex-encoded snapshot ID.
        array_path : str
            Path of the array node (e.g. ``"/group1/temperatures"``).
        chunk_index : tuple[int, ...]
            The chunk coordinates to read.

        Raises
        ------
        KeyError
            If the array, manifest, or chunk index is not found.
        """
        chunk_ref = self._find_chunk_ref(ref, array_path, chunk_index)
        return read_chunk(self._root, chunk_ref, storage=self._storage)

    def read_all_chunks(
        self, ref: str, array_path: str
    ) -> dict[tuple[int, ...], bytes]:
        """Read all chunks for an array.

        Parameters
        ----------
        ref : str
            A branch name, tag name, or hex-encoded snapshot ID.
        array_path : str
            Path of the array node.

        Returns
        -------
        dict[tuple[int, ...], bytes]
            Mapping from chunk index to raw chunk data.
        """
        snapshot = self._get_snapshot(ref)
        manifest_refs = snapshot.get_array_manifest_refs(array_path)

        # Get the node_id for this array
        node_info = self._get_array_node(snapshot, array_path)
        node_id = node_info.node_id

        result: dict[tuple[int, ...], bytes] = {}
        for mref in manifest_refs:
            manifest = self._get_manifest(mref.manifest_id)
            for cref in manifest.get_chunk_refs(node_id):
                result[cref.index] = read_chunk(self._root, cref, storage=self._storage)
        return result

    def get_array_metadata(self, ref: str, array_path: str) -> dict:  # type: ignore[type-arg]
        """Parse and return the zarr.json metadata for an array.

        Parameters
        ----------
        ref : str
            A branch name, tag name, or hex-encoded snapshot ID.
        array_path : str
            Path of the array node.

        Returns
        -------
        dict
            The parsed zarr.json metadata.

        Raises
        ------
        KeyError
            If the array is not found.
        ValueError
            If the array has no user_data (zarr.json).
        """
        snapshot = self._get_snapshot(ref)
        node_info = self._get_array_node(snapshot, array_path)
        if node_info.user_data is None:
            raise ValueError(f"Array {array_path!r} has no zarr.json metadata")
        return json.loads(node_info.user_data)  # type: ignore[no-any-return]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _resolve_ref(self, ref: str) -> bytes:
        """Resolve a ref string to a 12-byte snapshot ID.

        Resolution order:
        1. Branch name
        2. Tag name
        3. Full Crockford Base32 ID (20 chars, e.g. ``"RF238TWZTXGD49BDPXWG"``)
        4. Crockford prefix (unique prefix match against known snapshots)
        5. Full hex-encoded ID (24 hex chars)

        Shortened Crockford prefixes are accepted as long as they
        uniquely identify a single snapshot (like git's short SHAs).
        """
        from icepyck.crockford import decode as crockford_decode
        from icepyck.crockford import encode as crockford_encode

        # Try branch
        try:
            return self._repo.get_snapshot_id(ref)
        except KeyError:
            pass
        # Try tag
        try:
            return self._repo.get_tag_snapshot_id(ref)
        except KeyError:
            pass
        # Try full Crockford Base32 (20-char uppercase string → 12 bytes)
        try:
            raw = crockford_decode(ref)
            if len(raw) == 12:
                return raw
        except (ValueError, KeyError):
            pass
        # Try Crockford prefix match against known snapshot IDs
        ref_upper = ref.upper()
        if ref_upper and all(c in "0123456789ABCDEFGHJKMNPQRSTVWXYZ" for c in ref_upper):
            all_ids = self._repo.get_snapshots_data()
            matches = [
                sid for sid, _, _, _ in all_ids
                if crockford_encode(sid).startswith(ref_upper)
            ]
            if len(matches) == 1:
                return matches[0]
            if len(matches) > 1:
                raise KeyError(
                    f"Ambiguous ref {ref!r}: matches {len(matches)} snapshots. "
                    f"Use more characters to disambiguate."
                )
        # Try hex-encoded snapshot ID
        try:
            raw = bytes.fromhex(ref)
            if len(raw) == 12:
                return raw
        except ValueError:
            pass
        raise KeyError(f"Could not resolve ref: {ref!r}")

    def _get_snapshot(self, ref: str) -> SnapshotReader:
        """Return a (cached) SnapshotReader for the given ref."""
        snapshot_id = self._resolve_ref(ref)
        return self._get_snapshot_by_id(snapshot_id)

    def _get_snapshot_by_id(self, snapshot_id: bytes) -> SnapshotReader:
        """Return a (cached) SnapshotReader for the given snapshot ID."""
        if snapshot_id not in self._snapshot_cache:
            self._snapshot_cache[snapshot_id] = SnapshotReader(
                self._root, snapshot_id, storage=self._storage
            )
        return self._snapshot_cache[snapshot_id]

    def _get_manifest(self, manifest_id: bytes) -> ManifestReader:
        """Return a (cached) ManifestReader for the given manifest ID."""
        if manifest_id not in self._manifest_cache:
            self._manifest_cache[manifest_id] = ManifestReader(
                self._root, manifest_id, storage=self._storage
            )
        return self._manifest_cache[manifest_id]

    @staticmethod
    def _get_array_node(snapshot: SnapshotReader, array_path: str) -> NodeInfo:
        """Find an array node by path in a snapshot."""
        for node in snapshot.list_nodes():
            if node.path == array_path and node.node_type == "array":
                return node
        raise KeyError(f"Array node not found: {array_path!r}")

    def _find_chunk_ref(
        self, ref: str, array_path: str, chunk_index: tuple[int, ...]
    ) -> ChunkRefInfo:
        """Locate the ChunkRefInfo for a specific chunk."""
        snapshot = self._get_snapshot(ref)
        manifest_refs = snapshot.get_array_manifest_refs(array_path)
        node_info = self._get_array_node(snapshot, array_path)
        node_id = node_info.node_id

        for mref in manifest_refs:
            manifest = self._get_manifest(mref.manifest_id)
            for cref in manifest.get_chunk_refs(node_id):
                if cref.index == chunk_index:
                    return cref

        raise KeyError(
            f"Chunk {chunk_index} not found for array {array_path!r}"
        )
