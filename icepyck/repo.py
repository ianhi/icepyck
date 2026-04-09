"""Repo info reader for Icechunk repositories.

Parses the ``$ROOT/repo`` file using the header parser and flatbuffers
generated code to expose branch/tag listings and snapshot ID lookups.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from icepyck.crockford import encode as crockford_encode
from icepyck.header import FileType, parse_bytes, parse_file

if TYPE_CHECKING:
    from pathlib import Path

    from icepyck.storage import Storage


def _decode(name: bytes | str) -> str:
    return name.decode("utf-8") if isinstance(name, bytes) else name


class RepoInfo:
    """High-level interface for reading an Icechunk repo file."""

    def __init__(
        self,
        path: str | Path | None = None,
        *,
        storage: Storage | None = None,
    ) -> None:
        from icepyck.generated.Repo import Repo

        if storage is not None:
            raw = storage.read("repo")
            header, payload = parse_bytes(raw)
        elif path is not None:
            header, payload = parse_file(path)
        else:
            raise TypeError("Either path or storage must be provided")
        if header.file_type != FileType.REPO_INFO:
            raise ValueError(
                f"Expected REPO_INFO file type ({FileType.REPO_INFO}), "
                f"got {header.file_type}"
            )

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
        return [
            _decode(self._repo.Branches(i).Name())
            for i in range(self._repo.BranchesLength())
        ]

    def list_tags(self) -> list[str]:
        """Return the names of all tags."""
        return [
            _decode(self._repo.Tags(i).Name())
            for i in range(self._repo.TagsLength())
        ]

    def get_snapshot_id(self, branch_name: str) -> bytes:
        """Get the 12-byte snapshot ID for the given branch."""
        for i in range(self._repo.BranchesLength()):
            ref = self._repo.Branches(i)
            if _decode(ref.Name()) == branch_name:
                return bytes(self._snapshot_ids[ref.SnapshotIndex()])
        raise KeyError(f"Branch not found: {branch_name!r}")

    def get_tag_snapshot_id(self, tag_name: str) -> bytes:
        """Get the 12-byte snapshot ID for the given tag."""
        for i in range(self._repo.TagsLength()):
            ref = self._repo.Tags(i)
            if _decode(ref.Name()) == tag_name:
                return bytes(self._snapshot_ids[ref.SnapshotIndex()])
        raise KeyError(f"Tag not found: {tag_name!r}")

    def get_snapshots_data(self) -> list[tuple[bytes, int, int, str]]:
        """Return (id, parent_offset, flushed_at, message) for all snapshots."""
        result = []
        for i in range(self._repo.SnapshotsLength()):
            snap = self._repo.Snapshots(i)
            sid = bytes(snap.Id().Bytes())
            parent_offset = snap.ParentOffset()
            flushed_at = snap.FlushedAt()
            msg = snap.Message()
            if isinstance(msg, bytes):
                msg = msg.decode("utf-8")
            result.append((sid, parent_offset, flushed_at, msg or ""))
        return result

    def get_branches_data(self) -> dict[str, int]:
        """Return branch name -> snapshot index mapping."""
        return {
            _decode(self._repo.Branches(i).Name()): self._repo.Branches(i).SnapshotIndex()
            for i in range(self._repo.BranchesLength())
        }

    def get_tags_data(self) -> dict[str, int]:
        """Return tag name -> snapshot index mapping."""
        return {
            _decode(self._repo.Tags(i).Name()): self._repo.Tags(i).SnapshotIndex()
            for i in range(self._repo.TagsLength())
        }

    @staticmethod
    def snapshot_id_to_path(snapshot_id: bytes) -> str:
        """Convert a 12-byte snapshot ID to its Crockford Base32 filename."""
        return crockford_encode(snapshot_id)
