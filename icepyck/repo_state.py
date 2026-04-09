"""Mutable in-memory representation of the repo file state.

Repository owns the sole RepoState instance. All mutations go through
Repository, which flushes to storage after each change. This avoids
re-reading from storage on every operation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from icepyck.writers import SnapshotInfoData

if TYPE_CHECKING:
    from icepyck.repo import RepoInfo


@dataclass
class RepoState:
    """Mutable in-memory representation of the repo file contents."""

    snapshots: list[SnapshotInfoData] = field(default_factory=list)
    branches: dict[str, int] = field(default_factory=dict)
    tags: dict[str, int] = field(default_factory=dict)
    deleted_tags: list[str] = field(default_factory=list)

    @staticmethod
    def from_repo_info(repo_info: RepoInfo) -> RepoState:
        """Extract mutable state from a read-only RepoInfo parser."""
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
        """Return the index of the given snapshot_id, or -1 if not found."""
        for i, snap in enumerate(self.snapshots):
            if snap.snapshot_id == snapshot_id:
                return i
        return -1

    def get_snapshot_id_by_branch(self, branch: str) -> bytes:
        """Return the snapshot ID that a branch points to."""
        idx = self.branches[branch]
        return self.snapshots[idx].snapshot_id

    def get_snapshot_id_by_tag(self, tag: str) -> bytes:
        """Return the snapshot ID that a tag points to."""
        idx = self.tags[tag]
        return self.snapshots[idx].snapshot_id
