"""Icepyck — a read-only Python client for Icechunk repositories."""

from __future__ import annotations

import json
from pathlib import Path

from icepyck.chunks import read_chunk
from icepyck.manifest import ChunkRefInfo, ManifestReader
from icepyck.repo import RepoInfo
from icepyck.snapshot import NodeInfo, SnapshotReader
from icepyck.storage import LocalStorage, S3Storage, Storage
from icepyck.store import IcechunkReadStore

__all__ = ["LocalStorage", "Repository", "S3Storage", "Session", "Storage", "open"]


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
    return Repository(storage=storage)


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
            A hex-encoded snapshot ID.

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
            assert snapshot is not None
            ref = snapshot
        snapshot_id = self._resolve_ref(ref)
        return Session(self, snapshot_id)

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

        Tries branch names first, then tags, then hex-encoded snapshot IDs.
        """
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
