"""Repo info reader for Icechunk repositories.

Parses the ``$ROOT/repo`` file using the header parser and flatbuffers
generated code to expose branch/tag listings and snapshot ID lookups.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from icepyck.crockford import encode as crockford_encode
from icepyck.header import FileType, parse_file

if TYPE_CHECKING:
    from pathlib import Path


class RepoInfo:
    """High-level interface for reading an Icechunk repo file.

    Parameters
    ----------
    path : str or Path
        Path to the ``repo`` file (e.g. ``$ROOT/repo``).
    """

    def __init__(self, path: str | Path) -> None:
        from icepyck.generated.Repo import Repo

        header, payload = parse_file(path)
        if header.file_type != FileType.REPO_INFO:
            raise ValueError(
                f"Expected REPO_INFO file type ({FileType.REPO_INFO}), "
                f"got {header.file_type}"
            )

        self._header = header
        buf = bytearray(payload)
        self._repo = Repo.GetRootAs(buf, 0)

        # Pre-extract snapshot IDs for indexed lookup
        self._snapshot_ids: list[bytes] = []
        for i in range(self._repo.SnapshotsLength()):
            snap = self._repo.Snapshots(i)
            id_obj = snap.Id()
            self._snapshot_ids.append(bytes(id_obj.Bytes()))

    def list_branches(self) -> list[str]:
        """Return the names of all branches."""
        branches = []
        for i in range(self._repo.BranchesLength()):
            ref = self._repo.Branches(i)
            name = ref.Name()
            branches.append(name.decode("utf-8") if isinstance(name, bytes) else name)
        return branches

    def list_tags(self) -> list[str]:
        """Return the names of all tags."""
        tags = []
        for i in range(self._repo.TagsLength()):
            ref = self._repo.Tags(i)
            name = ref.Name()
            tags.append(name.decode("utf-8") if isinstance(name, bytes) else name)
        return tags

    def get_snapshot_id(self, branch_name: str) -> bytes:
        """Get the 12-byte snapshot ID for the given branch.

        Parameters
        ----------
        branch_name : str
            Name of the branch (e.g. ``"main"``).

        Returns
        -------
        bytes
            The 12-byte ObjectId12 for the branch's current snapshot.

        Raises
        ------
        KeyError
            If the branch is not found.
        """
        for i in range(self._repo.BranchesLength()):
            ref = self._repo.Branches(i)
            name = ref.Name()
            if isinstance(name, bytes):
                name = name.decode("utf-8")
            if name == branch_name:
                idx: int = ref.SnapshotIndex()
                return bytes(self._snapshot_ids[idx])
        raise KeyError(f"Branch not found: {branch_name!r}")

    def get_tag_snapshot_id(self, tag_name: str) -> bytes:
        """Get the 12-byte snapshot ID for the given tag.

        Parameters
        ----------
        tag_name : str
            Name of the tag (e.g. ``"v1"``).

        Returns
        -------
        bytes
            The 12-byte ObjectId12 for the tag's snapshot.

        Raises
        ------
        KeyError
            If the tag is not found.
        """
        for i in range(self._repo.TagsLength()):
            ref = self._repo.Tags(i)
            name = ref.Name()
            if isinstance(name, bytes):
                name = name.decode("utf-8")
            if name == tag_name:
                idx: int = ref.SnapshotIndex()
                return bytes(self._snapshot_ids[idx])
        raise KeyError(f"Tag not found: {tag_name!r}")

    @staticmethod
    def snapshot_id_to_path(snapshot_id: bytes) -> str:
        """Convert a 12-byte snapshot ID to its Crockford Base32 filename.

        Parameters
        ----------
        snapshot_id : bytes
            The 12-byte ObjectId12.

        Returns
        -------
        str
            The Crockford Base32 encoded string used as the filename.
        """
        return crockford_encode(snapshot_id)
