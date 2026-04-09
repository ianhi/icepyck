"""Tests for branch and tag management (Phase 7)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

import icepyck
from icepyck.verify import verify_repo


def _make_repo_with_data(tmp_path: Path) -> tuple[Path, str]:
    """Create a repo with one commit and return (path, snapshot_id)."""
    repo_path = tmp_path / "repo"
    repo = icepyck.Repository.init(repo_path)
    ws = repo.writable_session(branch="main")
    meta = json.dumps(
        {
            "zarr_format": 3,
            "node_type": "array",
            "shape": [4],
            "data_type": "float32",
            "chunk_grid": {
                "name": "regular",
                "configuration": {"chunk_shape": [4]},
            },
            "chunk_key_encoding": {
                "name": "default",
                "configuration": {"separator": "/"},
            },
            "fill_value": 0.0,
            "codecs": [
                {"name": "bytes", "configuration": {"endian": "little"}},
            ],
        }
    ).encode()
    ws.set_metadata("/data", meta)
    ws.set_chunk("/data", (0,), np.zeros(4, dtype="<f4").tobytes())
    snap_id = ws.commit("Initial data")
    return repo_path, snap_id


class TestCreateBranch:
    def test_create_branch_from_main(self, tmp_path: Path) -> None:
        repo_path, _ = _make_repo_with_data(tmp_path)
        repo = icepyck.open(repo_path)

        repo.create_branch("dev", "main")
        assert "dev" in repo.list_branches()
        assert "main" in repo.list_branches()

        # Both branches should point to the same snapshot
        main_session = repo.readonly_session(branch="main")
        dev_session = repo.readonly_session(branch="dev")
        assert main_session.snapshot_id == dev_session.snapshot_id

    def test_create_branch_duplicate_raises(self, tmp_path: Path) -> None:
        repo_path, _ = _make_repo_with_data(tmp_path)
        repo = icepyck.open(repo_path)
        with pytest.raises(KeyError, match="already exists"):
            repo.create_branch("main", "main")

    def test_create_branch_slash_raises(self, tmp_path: Path) -> None:
        repo_path, _ = _make_repo_with_data(tmp_path)
        repo = icepyck.open(repo_path)
        with pytest.raises(ValueError, match="must not contain"):
            repo.create_branch("feat/x", "main")

    def test_commit_to_new_branch(self, tmp_path: Path) -> None:
        repo_path, _ = _make_repo_with_data(tmp_path)
        repo = icepyck.open(repo_path)
        repo.create_branch("dev", "main")

        ws = repo.writable_session(branch="dev")
        ws.set_metadata(
            "/extra",
            json.dumps({"zarr_format": 3, "node_type": "group"}).encode(),
        )
        ws.commit("Added group on dev")

        # dev has the new group, main does not
        repo2 = icepyck.open(repo_path)
        dev_nodes = {n.path for n in repo2.readonly_session(branch="dev").list_nodes()}
        main_nodes = {
            n.path for n in repo2.readonly_session(branch="main").list_nodes()
        }
        assert "/extra" in dev_nodes
        assert "/extra" not in main_nodes

    def test_repo_passes_verification(self, tmp_path: Path) -> None:
        repo_path, _ = _make_repo_with_data(tmp_path)
        repo = icepyck.open(repo_path)
        repo.create_branch("dev", "main")
        assert not verify_repo(repo_path)


class TestDeleteBranch:
    def test_delete_branch(self, tmp_path: Path) -> None:
        repo_path, _ = _make_repo_with_data(tmp_path)
        repo = icepyck.open(repo_path)
        repo.create_branch("dev", "main")
        assert "dev" in repo.list_branches()

        repo.delete_branch("dev")
        assert "dev" not in repo.list_branches()
        assert "main" in repo.list_branches()

    def test_delete_main_raises(self, tmp_path: Path) -> None:
        repo_path, _ = _make_repo_with_data(tmp_path)
        repo = icepyck.open(repo_path)
        with pytest.raises(ValueError, match="Cannot delete"):
            repo.delete_branch("main")

    def test_delete_nonexistent_raises(self, tmp_path: Path) -> None:
        repo_path, _ = _make_repo_with_data(tmp_path)
        repo = icepyck.open(repo_path)
        with pytest.raises(KeyError, match="not found"):
            repo.delete_branch("nope")


class TestCreateTag:
    def test_create_tag(self, tmp_path: Path) -> None:
        repo_path, snap_id = _make_repo_with_data(tmp_path)
        repo = icepyck.open(repo_path)

        repo.create_tag("v1", "main")
        assert "v1" in repo.list_tags()

        # Tag should resolve to the same snapshot
        session = repo.readonly_session(tag="v1")
        assert session.snapshot_id == snap_id

    def test_create_tag_duplicate_raises(self, tmp_path: Path) -> None:
        repo_path, _ = _make_repo_with_data(tmp_path)
        repo = icepyck.open(repo_path)
        repo.create_tag("v1", "main")
        with pytest.raises(KeyError, match="already exists"):
            repo.create_tag("v1", "main")

    def test_tag_immutable_after_new_commit(self, tmp_path: Path) -> None:
        """Tag keeps pointing to the original snapshot after new commits."""
        repo_path, snap_id = _make_repo_with_data(tmp_path)
        repo = icepyck.open(repo_path)
        repo.create_tag("v1", "main")

        # Make a new commit on main
        ws = repo.writable_session(branch="main")
        ws.set_metadata(
            "/new_group",
            json.dumps({"zarr_format": 3, "node_type": "group"}).encode(),
        )
        ws.commit("After tag")

        # Re-open and check
        repo2 = icepyck.open(repo_path)
        tag_session = repo2.readonly_session(tag="v1")
        assert tag_session.snapshot_id == snap_id

        main_session = repo2.readonly_session(branch="main")
        assert main_session.snapshot_id != snap_id

    def test_repo_passes_verification(self, tmp_path: Path) -> None:
        repo_path, _ = _make_repo_with_data(tmp_path)
        repo = icepyck.open(repo_path)
        repo.create_tag("v1", "main")
        assert not verify_repo(repo_path)


class TestDeleteTag:
    def test_delete_tag(self, tmp_path: Path) -> None:
        repo_path, _ = _make_repo_with_data(tmp_path)
        repo = icepyck.open(repo_path)
        repo.create_tag("v1", "main")
        assert "v1" in repo.list_tags()

        repo.delete_tag("v1")
        assert "v1" not in repo.list_tags()

    def test_delete_nonexistent_raises(self, tmp_path: Path) -> None:
        repo_path, _ = _make_repo_with_data(tmp_path)
        repo = icepyck.open(repo_path)
        with pytest.raises(KeyError, match="not found"):
            repo.delete_tag("nope")

    def test_cannot_recreate_deleted_tag(self, tmp_path: Path) -> None:
        """Tombstone semantics: deleted tags cannot be recreated."""
        repo_path, _ = _make_repo_with_data(tmp_path)
        repo = icepyck.open(repo_path)
        repo.create_tag("v1", "main")
        repo.delete_tag("v1")

        with pytest.raises(KeyError, match="previously deleted"):
            repo.create_tag("v1", "main")

    def test_repo_passes_verification_after_delete(self, tmp_path: Path) -> None:
        repo_path, _ = _make_repo_with_data(tmp_path)
        repo = icepyck.open(repo_path)
        repo.create_tag("v1", "main")
        repo.delete_tag("v1")
        assert not verify_repo(repo_path)
