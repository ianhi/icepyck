"""Tests for RepoInfo: branch/tag listing and snapshot ID resolution."""

import pytest
from pathlib import Path

from icepyck.repo import RepoInfo
from icepyck.crockford import encode as crockford_encode

BASIC_REPO = Path(__file__).parent.parent / "test-repos" / "basic"

pytestmark = pytest.mark.skipif(
    not (BASIC_REPO / "repo").exists(),
    reason="basic test repo not available",
)


@pytest.fixture
def repo_info(basic_repo):
    return RepoInfo(basic_repo / "repo")


class TestBranches:
    def test_list_branches(self, repo_info):
        branches = repo_info.list_branches()
        assert isinstance(branches, list)
        assert "main" in branches

    def test_get_snapshot_id_main(self, repo_info):
        snapshot_id = repo_info.get_snapshot_id("main")
        assert isinstance(snapshot_id, bytes)
        assert len(snapshot_id) == 12

    def test_snapshot_file_exists(self, repo_info, basic_repo):
        snapshot_id = repo_info.get_snapshot_id("main")
        snapshot_name = RepoInfo.snapshot_id_to_path(snapshot_id)
        snapshot_file = basic_repo / "snapshots" / snapshot_name
        assert snapshot_file.exists(), f"Snapshot file not found: {snapshot_file}"

    def test_nonexistent_branch_raises(self, repo_info):
        with pytest.raises(KeyError, match="Branch not found"):
            repo_info.get_snapshot_id("nonexistent-branch")


class TestTags:
    def test_list_tags(self, repo_info):
        tags = repo_info.list_tags()
        assert isinstance(tags, list)
        assert "v1" in tags

    def test_get_tag_snapshot_id(self, repo_info):
        snapshot_id = repo_info.get_tag_snapshot_id("v1")
        assert isinstance(snapshot_id, bytes)
        assert len(snapshot_id) == 12

    def test_tag_snapshot_file_exists(self, repo_info, basic_repo):
        snapshot_id = repo_info.get_tag_snapshot_id("v1")
        snapshot_name = RepoInfo.snapshot_id_to_path(snapshot_id)
        snapshot_file = basic_repo / "snapshots" / snapshot_name
        assert snapshot_file.exists(), f"v1 snapshot file not found: {snapshot_file}"

    def test_nonexistent_tag_raises(self, repo_info):
        with pytest.raises(KeyError, match="Tag not found"):
            repo_info.get_tag_snapshot_id("nonexistent-tag")


class TestSnapshotIdToPath:
    def test_known_encoding(self):
        initial_bytes = bytes.fromhex("0b1cc8d6787580f0e33a6534")
        path = RepoInfo.snapshot_id_to_path(initial_bytes)
        assert path == "1CECHNKREP0F1RSTCMT0"

    def test_returns_string(self):
        path = RepoInfo.snapshot_id_to_path(b"\x00" * 12)
        assert isinstance(path, str)
