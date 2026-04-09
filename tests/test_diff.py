"""Tests for snapshot diff module."""

from pathlib import Path

import pytest

from icepyck.diff import SnapshotDiff, diff_snapshots

BASIC_REPO = Path(__file__).parent.parent / "test-repos" / "basic"


class TestDiffSameSnapshot:
    """Diffing a snapshot against itself should show no changes."""

    pytestmark = pytest.mark.skipif(
        not (BASIC_REPO / "repo").exists(),
        reason="basic test repo not available",
    )

    def test_self_diff_no_added(self, basic_repo: Path) -> None:
        diff = diff_snapshots(basic_repo, "main", "main")
        assert len(diff.added_nodes) == 0

    def test_self_diff_no_removed(self, basic_repo: Path) -> None:
        diff = diff_snapshots(basic_repo, "main", "main")
        assert len(diff.removed_nodes) == 0

    def test_self_diff_no_modified(self, basic_repo: Path) -> None:
        diff = diff_snapshots(basic_repo, "main", "main")
        assert len(diff.modified_nodes) == 0

    def test_self_diff_all_unchanged(self, basic_repo: Path) -> None:
        diff = diff_snapshots(basic_repo, "main", "main")
        assert diff.unchanged_count == 4  # /, /group1, temperatures, timestamps


class TestDiffBetweenCommits:
    """Diff between the two basic repo commits (parent -> main)."""

    pytestmark = pytest.mark.skipif(
        not (BASIC_REPO / "repo").exists(),
        reason="basic test repo not available",
    )

    @pytest.fixture
    def diff(self, basic_repo: Path) -> SnapshotDiff:
        return diff_snapshots(basic_repo, "main~1", "main")

    def test_no_added_nodes(self, diff: SnapshotDiff) -> None:
        assert len(diff.added_nodes) == 0

    def test_no_removed_nodes(self, diff: SnapshotDiff) -> None:
        assert len(diff.removed_nodes) == 0

    def test_temperatures_modified(self, diff: SnapshotDiff) -> None:
        modified_paths = {m.path for m in diff.modified_nodes}
        assert "/group1/temperatures" in modified_paths

    def test_temperatures_chunks_changed(self, diff: SnapshotDiff) -> None:
        temp_change = next(
            m for m in diff.modified_nodes if m.path == "/group1/temperatures"
        )
        assert temp_change.chunks_changed is True
        assert len(temp_change.changed_extents) > 0

    def test_temperatures_extent_covers_chunk_0(self, diff: SnapshotDiff) -> None:
        temp_change = next(
            m for m in diff.modified_nodes if m.path == "/group1/temperatures"
        )
        # The changed extent range should cover flat chunk index 0
        assert any(lo <= 0 for lo, _hi in temp_change.changed_extents)

    def test_unchanged_nodes_present(self, diff: SnapshotDiff) -> None:
        # Root, group1, and timestamps should be unchanged
        assert diff.unchanged_count >= 2

    def test_refs_stored(self, diff: SnapshotDiff) -> None:
        assert diff.old_ref == "main~1"
        assert diff.new_ref == "main"


class TestDiffRelativeRefs:
    """Test that relative refs (main~1) work correctly."""

    pytestmark = pytest.mark.skipif(
        not (BASIC_REPO / "repo").exists(),
        reason="basic test repo not available",
    )

    def test_main_tilde_1_resolves(self, basic_repo: Path) -> None:
        # Should not raise
        diff = diff_snapshots(basic_repo, "main~1", "main")
        assert isinstance(diff, SnapshotDiff)

    def test_invalid_ref_raises(self, basic_repo: Path) -> None:
        with pytest.raises(KeyError):
            diff_snapshots(basic_repo, "nonexistent", "main")


class TestDiffAddedRemoved:
    """Test added/removed node detection via initial vs data commit."""

    pytestmark = pytest.mark.skipif(
        not (BASIC_REPO / "repo").exists(),
        reason="basic test repo not available",
    )

    def test_initial_to_first_commit_has_added_nodes(self, basic_repo: Path) -> None:
        # main~2 is the initial empty commit, main~1 is first data commit
        diff = diff_snapshots(basic_repo, "main~2", "main~1")
        added_paths = {n.path for n in diff.added_nodes}
        # The first data commit added group1, temperatures, timestamps
        assert "/group1" in added_paths or len(diff.added_nodes) > 0

    def test_reverse_diff_shows_removed(self, basic_repo: Path) -> None:
        # Reverse: first data commit -> initial empty
        diff = diff_snapshots(basic_repo, "main~1", "main~2")
        removed_paths = {n.path for n in diff.removed_nodes}
        assert len(removed_paths) > 0
