"""Mutable in-memory representation of the repo file state.

Repository owns the sole RepoState instance. All mutations go through
Repository, which flushes to storage after each change. This avoids
re-reading from storage on every operation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from icepyck.header import FileType, parse_bytes
from icepyck.writers import SnapshotInfoData

if TYPE_CHECKING:
    from icepyck.repo import RepoInfo
    from icepyck.storage import Storage


@dataclass
class RepoState:
    """Mutable in-memory representation of the repo file contents."""

    snapshots: list[SnapshotInfoData] = field(default_factory=list)
    branches: dict[str, int] = field(default_factory=dict)
    tags: dict[str, int] = field(default_factory=dict)
    deleted_tags: list[str] = field(default_factory=list)
    _snap_index: dict[bytes, int] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        if not self._snap_index:
            self._snap_index = {s.snapshot_id: i for i, s in enumerate(self.snapshots)}

    @staticmethod
    def from_storage(storage: Storage) -> RepoState:
        """Parse the repo file directly from storage into RepoState.

        This is the preferred constructor — avoids the RepoInfo intermediate.
        """
        from icepyck.generated.Repo import Repo

        raw = storage.read("repo")
        header, payload = parse_bytes(raw)
        if header.file_type != FileType.REPO_INFO:
            raise ValueError(f"Expected REPO_INFO, got {header.file_type}")
        repo = Repo.GetRootAs(bytearray(payload), 0)

        # Extract snapshots
        snapshots = []
        for i in range(repo.SnapshotsLength()):
            snap = repo.Snapshots(i)
            sid = bytes(snap.Id().Bytes())
            msg = snap.Message()
            if isinstance(msg, bytes):
                msg = msg.decode("utf-8")
            snapshots.append(
                SnapshotInfoData(
                    snapshot_id=sid,
                    parent_offset=snap.ParentOffset(),
                    flushed_at=snap.FlushedAt(),
                    message=msg or "",
                )
            )

        # Extract branches
        branches = {}
        for i in range(repo.BranchesLength()):
            ref = repo.Branches(i)
            name = ref.Name()
            if isinstance(name, bytes):
                name = name.decode("utf-8")
            branches[name] = ref.SnapshotIndex()

        # Extract tags
        tags = {}
        for i in range(repo.TagsLength()):
            ref = repo.Tags(i)
            name = ref.Name()
            if isinstance(name, bytes):
                name = name.decode("utf-8")
            tags[name] = ref.SnapshotIndex()

        # Extract deleted tags
        deleted_tags = []
        for i in range(repo.DeletedTagsLength()):
            name = repo.DeletedTags(i)
            if isinstance(name, bytes):
                name = name.decode("utf-8")
            if name:
                deleted_tags.append(name)

        return RepoState(
            snapshots=snapshots,
            branches=branches,
            tags=tags,
            deleted_tags=deleted_tags,
        )

    @staticmethod
    def from_repo_info(repo_info: RepoInfo) -> RepoState:
        """Extract mutable state from a read-only RepoInfo parser.

        Used by external consumers that already have a RepoInfo.
        Prefer from_storage() when possible.
        """
        raw_snaps = repo_info.get_snapshots_data()
        return RepoState(
            snapshots=[
                SnapshotInfoData(
                    snapshot_id=sid,
                    parent_offset=poff,
                    flushed_at=fat,
                    message=msg,
                )
                for sid, poff, fat, msg in raw_snaps
            ],
            branches=repo_info.get_branches_data(),
            tags=repo_info.get_tags_data(),
            deleted_tags=repo_info.get_deleted_tags(),
        )

    def find_snapshot_index(self, snapshot_id: bytes) -> int:
        """Return the index of the given snapshot_id, or -1 if not found. O(1)."""
        return self._snap_index.get(snapshot_id, -1)

    def add_snapshot(self, snap: SnapshotInfoData) -> int:
        """Append a snapshot and update the index. Returns the new index."""
        idx = len(self.snapshots)
        self.snapshots.append(snap)
        self._snap_index[snap.snapshot_id] = idx
        return idx

    def get_snapshot_id_by_branch(self, branch: str) -> bytes:
        """Return the snapshot ID that a branch points to."""
        idx = self.branches[branch]
        return self.snapshots[idx].snapshot_id

    def get_snapshot_id_by_tag(self, tag: str) -> bytes:
        """Return the snapshot ID that a tag points to."""
        idx = self.tags[tag]
        return self.snapshots[idx].snapshot_id
