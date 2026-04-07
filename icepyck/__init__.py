"""Icepyck — a read-only Python client for Icechunk repositories."""

from __future__ import annotations

import json
from pathlib import Path

from icepyck.chunks import read_chunk
from icepyck.manifest import ChunkRefInfo, ManifestReader
from icepyck.repo import RepoInfo
from icepyck.snapshot import NodeInfo, SnapshotReader

__all__ = ["Repository", "open"]


def open(path: str | Path) -> Repository:
    """Open an Icechunk repository at the given path."""
    return Repository(path)


class Repository:
    """Read-only interface to an Icechunk repository.

    Parameters
    ----------
    path : str or Path
        Root path of the repository (the directory containing ``repo``,
        ``snapshots/``, ``manifests/``, ``chunks/``).
    """

    def __init__(self, path: str | Path) -> None:
        self._root = Path(path)
        self._repo = RepoInfo(self._root / "repo")
        self._snapshot_cache: dict[bytes, SnapshotReader] = {}
        self._manifest_cache: dict[bytes, ManifestReader] = {}

    def list_branches(self) -> list[str]:
        """Return the names of all branches."""
        return self._repo.list_branches()

    def list_tags(self) -> list[str]:
        """Return the names of all tags."""
        return self._repo.list_tags()

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
        return read_chunk(self._root, chunk_ref)

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
                result[cref.index] = read_chunk(self._root, cref)
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
        if snapshot_id not in self._snapshot_cache:
            self._snapshot_cache[snapshot_id] = SnapshotReader(
                self._root, snapshot_id
            )
        return self._snapshot_cache[snapshot_id]

    def _get_manifest(self, manifest_id: bytes) -> ManifestReader:
        """Return a (cached) ManifestReader for the given manifest ID."""
        if manifest_id not in self._manifest_cache:
            self._manifest_cache[manifest_id] = ManifestReader(
                self._root, manifest_id
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
