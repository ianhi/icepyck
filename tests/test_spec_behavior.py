"""Tests verifying behavioral compliance with the Icechunk V2 spec.

These tests check ALGORITHMS and BEHAVIORS, not just file format.
They verify the spec-defined invariants documented in spec.md.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

import icepyck
from icepyck import ConflictError
from icepyck.verify import verify_repo


def _array_meta(shape: list[int], dtype: str = "float32") -> bytes:
    return json.dumps(
        {
            "zarr_format": 3,
            "node_type": "array",
            "shape": shape,
            "data_type": dtype,
            "chunk_grid": {
                "name": "regular",
                "configuration": {"chunk_shape": shape},
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


# ---------------------------------------------------------------
# Spec: "Initialize New Repository"
# ---------------------------------------------------------------


class TestInitializeRepo:
    """Spec: 'A new repository is initialized by creating a new empty
    snapshot file and then creating the reference for branch main.'"""

    def test_initial_snapshot_has_well_known_id(self, tmp_path: Path) -> None:
        """Spec: 'The first snapshot has a well known id, that encodes to
        a file name: 1CECHNKREP0F1RSTCMT0'"""
        repo_path = tmp_path / "repo"
        icepyck.Repository.init(repo_path)
        assert (repo_path / "snapshots" / "1CECHNKREP0F1RSTCMT0").exists()

    def test_main_branch_exists(self, tmp_path: Path) -> None:
        """Spec: 'Repositories must always have a main branch.'"""
        repo_path = tmp_path / "repo"
        repo = icepyck.Repository.init(repo_path)
        assert "main" in repo.list_branches()

    def test_snapshot_file_written_before_repo(self, tmp_path: Path) -> None:
        """The snapshot file must exist when the repo file references it."""
        repo_path = tmp_path / "repo"
        icepyck.Repository.init(repo_path)
        # If repo file exists, snapshot must too
        assert (repo_path / "repo").exists()
        assert (repo_path / "snapshots" / "1CECHNKREP0F1RSTCMT0").exists()

    def test_initial_repo_has_root_group(self, tmp_path: Path) -> None:
        """Initial snapshot must contain at least a root group node."""
        repo_path = tmp_path / "repo"
        repo = icepyck.Repository.init(repo_path)
        session = repo.readonly_session(branch="main")
        nodes = session.list_nodes()
        assert any(n.path == "/" and n.node_type == "group" for n in nodes)

    def test_passes_spec_verification(self, tmp_path: Path) -> None:
        repo_path = tmp_path / "repo"
        icepyck.Repository.init(repo_path)
        assert not verify_repo(repo_path)


# ---------------------------------------------------------------
# Spec: "Read from Repository"
# ---------------------------------------------------------------


class TestReadFromRepo:
    """Spec sections: 'From Snapshot ID', 'From Branch', 'From Tag'."""

    @pytest.fixture()
    def repo(self, tmp_path: Path) -> icepyck.Repository:
        repo_path = tmp_path / "repo"
        repo = icepyck.Repository.init(repo_path)
        ws = repo.writable_session(branch="main")
        ws.set_metadata("/data", _array_meta([4]))
        ws.set_chunk("/data", (0,), np.zeros(4, dtype="<f4").tobytes())
        ws.commit("Add data")
        return repo

    def test_read_from_branch(self, repo: icepyck.Repository) -> None:
        session = repo.readonly_session(branch="main")
        assert any(n.path == "/data" for n in session.list_nodes())

    def test_read_from_tag(self, repo: icepyck.Repository) -> None:
        repo.create_tag("v1", "main")
        session = repo.readonly_session(tag="v1")
        assert any(n.path == "/data" for n in session.list_nodes())

    def test_read_from_snapshot_id(self, repo: icepyck.Repository) -> None:
        """Spec: 'If the specific snapshot ID is known, a client can open
        it directly in read only mode.'"""
        main_session = repo.readonly_session(branch="main")
        snap_id = main_session.snapshot_id
        session = repo.readonly_session(snapshot=snap_id)
        assert any(n.path == "/data" for n in session.list_nodes())

    def test_read_from_snapshot_prefix(self, repo: icepyck.Repository) -> None:
        """Shortened Crockford prefixes should resolve uniquely."""
        main_session = repo.readonly_session(branch="main")
        snap_id = main_session.snapshot_id
        # Use first 8 chars as prefix — should be unique in a 2-snapshot repo
        prefix = snap_id[:8]
        session = repo.readonly_session(snapshot=prefix)
        assert session.snapshot_id == snap_id


# ---------------------------------------------------------------
# Spec: "Write New Snapshot"
# ---------------------------------------------------------------


class TestWriteNewSnapshot:
    """Spec: 'Write new chunk files, write new chunk manifests, write a
    new transaction log file, write a new snapshot file, do conditional
    update to write the new value of the branch reference file.'"""

    def test_commit_writes_all_files(self, tmp_path: Path) -> None:
        """After commit, all expected files should exist on storage."""
        repo_path = tmp_path / "repo"
        repo = icepyck.Repository.init(repo_path)
        ws = repo.writable_session(branch="main")
        ws.set_metadata("/arr", _array_meta([4]))
        ws.set_chunk("/arr", (0,), np.zeros(4, dtype="<f4").tobytes())
        snap_id = ws.commit("Test commit")

        # Snapshot file
        assert (repo_path / "snapshots" / snap_id).exists()
        # Transaction log
        assert (repo_path / "transactions" / snap_id).exists()
        # At least one manifest
        manifests = list((repo_path / "manifests").iterdir())
        assert len(manifests) >= 1
        # At least one chunk
        chunks = list((repo_path / "chunks").iterdir())
        assert len(chunks) >= 1

    def test_commit_updates_branch_pointer(self, tmp_path: Path) -> None:
        repo_path = tmp_path / "repo"
        repo = icepyck.Repository.init(repo_path)
        old_id = repo.readonly_session(branch="main").snapshot_id

        ws = repo.writable_session(branch="main")
        ws.set_metadata(
            "/g", json.dumps({"zarr_format": 3, "node_type": "group"}).encode()
        )
        new_id = ws.commit("New commit")

        assert new_id != old_id
        assert repo.readonly_session(branch="main").snapshot_id == new_id

    def test_commit_preserves_old_snapshots(self, tmp_path: Path) -> None:
        """Spec: 'Previous snapshots of a repository remain accessible
        after new ones have been written.' (time travel)"""
        repo_path = tmp_path / "repo"
        repo = icepyck.Repository.init(repo_path)
        ws = repo.writable_session(branch="main")
        ws.set_metadata("/arr", _array_meta([4]))
        ws.set_chunk("/arr", (0,), np.array([1, 2, 3, 4], dtype="<f4").tobytes())
        snap1 = ws.commit("Commit 1")

        ws.set_chunk("/arr", (0,), np.array([10, 20, 30, 40], dtype="<f4").tobytes())
        _snap2 = ws.commit("Commit 2")

        # Old snapshot still readable
        old_session = repo.readonly_session(snapshot=snap1)
        assert any(n.path == "/arr" for n in old_session.list_nodes())

    def test_spec_verification_after_commit(self, tmp_path: Path) -> None:
        repo_path = tmp_path / "repo"
        repo = icepyck.Repository.init(repo_path)
        ws = repo.writable_session(branch="main")
        ws.set_metadata("/arr", _array_meta([4]))
        ws.set_chunk("/arr", (0,), np.zeros(4, dtype="<f4").tobytes())
        ws.commit("Test")
        assert not verify_repo(repo_path)


# ---------------------------------------------------------------
# Spec: Branches
# ---------------------------------------------------------------


class TestBranchBehavior:
    """Spec: 'Branches are mutable references to a snapshot.
    Repositories must always have a main branch.'"""

    def test_branch_names_no_slash(self, tmp_path: Path) -> None:
        """Spec: 'Branch names may not contain the / character.'"""
        repo_path = tmp_path / "repo"
        repo = icepyck.Repository.init(repo_path)
        with pytest.raises(ValueError, match="must not contain"):
            repo.create_branch("feat/x", "main")

    def test_main_branch_cannot_be_deleted(self, tmp_path: Path) -> None:
        repo_path = tmp_path / "repo"
        repo = icepyck.Repository.init(repo_path)
        with pytest.raises(ValueError, match="Cannot delete"):
            repo.delete_branch("main")

    def test_branch_points_to_snapshot(self, tmp_path: Path) -> None:
        """Branch must resolve to a valid snapshot."""
        repo_path = tmp_path / "repo"
        repo = icepyck.Repository.init(repo_path)
        ws = repo.writable_session(branch="main")
        ws.set_metadata(
            "/g", json.dumps({"zarr_format": 3, "node_type": "group"}).encode()
        )
        snap_id = ws.commit("Test")

        repo.create_branch("dev", "main")
        dev_session = repo.readonly_session(branch="dev")
        assert dev_session.snapshot_id == snap_id

    def test_branch_delete_removes_from_listing(self, tmp_path: Path) -> None:
        repo_path = tmp_path / "repo"
        repo = icepyck.Repository.init(repo_path)
        repo.create_branch("dev", "main")
        assert "dev" in repo.list_branches()
        repo.delete_branch("dev")
        assert "dev" not in repo.list_branches()


# ---------------------------------------------------------------
# Spec: Tags
# ---------------------------------------------------------------


class TestTagBehavior:
    """Spec: 'Tags are immutable references to a snapshot.
    After creation, tags may never be updated, unlike in Git.'"""

    def test_tag_names_no_slash(self, tmp_path: Path) -> None:
        """Spec: 'Tag names may not contain the / character.'"""
        repo_path = tmp_path / "repo"
        repo = icepyck.Repository.init(repo_path)
        with pytest.raises(ValueError, match="must not contain"):
            repo.create_tag("v1/beta", "main")

    def test_tag_is_immutable(self, tmp_path: Path) -> None:
        """Spec: 'After creation, tags may never be updated.'"""
        repo_path = tmp_path / "repo"
        repo = icepyck.Repository.init(repo_path)
        repo.create_tag("v1", "main")
        # Cannot create same tag again (would be an "update")
        with pytest.raises(KeyError, match="already exists"):
            repo.create_tag("v1", "main")

    def test_tag_create_if_not_exists(self, tmp_path: Path) -> None:
        """Spec: 'When creating a new tag, the client attempts to create
        the tag file using a "create if not exists" operation.'"""
        repo_path = tmp_path / "repo"
        repo = icepyck.Repository.init(repo_path)
        # First creation succeeds
        repo.create_tag("v1", "main")
        assert "v1" in repo.list_tags()

    def test_deleted_tag_cannot_be_recreated(self, tmp_path: Path) -> None:
        """Spec: 'we don't allow recreating tags that were deleted.'"""
        repo_path = tmp_path / "repo"
        repo = icepyck.Repository.init(repo_path)
        repo.create_tag("v1", "main")
        repo.delete_tag("v1")
        with pytest.raises(KeyError, match="previously deleted"):
            repo.create_tag("v1", "main")

    def test_tag_survives_new_commits(self, tmp_path: Path) -> None:
        """Tag still points to original snapshot after new commits."""
        repo_path = tmp_path / "repo"
        repo = icepyck.Repository.init(repo_path)
        ws = repo.writable_session(branch="main")
        ws.set_metadata("/arr", _array_meta([4]))
        ws.set_chunk("/arr", (0,), np.zeros(4, dtype="<f4").tobytes())
        ws.commit("Before tag")

        tagged_id = repo.readonly_session(branch="main").snapshot_id
        repo.create_tag("v1", "main")

        ws2 = repo.writable_session(branch="main")
        ws2.set_metadata(
            "/g", json.dumps({"zarr_format": 3, "node_type": "group"}).encode()
        )
        ws2.commit("After tag")

        assert repo.readonly_session(tag="v1").snapshot_id == tagged_id
        assert repo.readonly_session(branch="main").snapshot_id != tagged_id


# ---------------------------------------------------------------
# Spec: Manifest sorted order
# ---------------------------------------------------------------


class TestManifestSortOrder:
    """Spec: 'A list of ArrayManifest sorted by node id' and
    'a list of ChunkRef sorted by the chunk coordinate.'"""

    def test_multiple_arrays_sorted_by_node_id(self, tmp_path: Path) -> None:
        """Arrays in manifest must be sorted by node_id."""
        repo_path = tmp_path / "repo"
        repo = icepyck.Repository.init(repo_path)
        ws = repo.writable_session(branch="main")

        # Create multiple arrays — node_ids are random, so order varies
        for name in ["zebra", "alpha", "middle"]:
            ws.set_metadata(f"/{name}", _array_meta([3]))
            ws.set_chunk(f"/{name}", (0,), np.zeros(3, dtype="<f4").tobytes())
        ws.commit("Multiple arrays")

        # Verifier checks sorted order
        assert not verify_repo(repo_path)


# ---------------------------------------------------------------
# Spec: Transaction log
# ---------------------------------------------------------------


class TestTransactionLog:
    """Spec: 'Transaction logs keep track of the operations done in a commit.'"""

    def test_transaction_log_written_on_commit(self, tmp_path: Path) -> None:
        repo_path = tmp_path / "repo"
        repo = icepyck.Repository.init(repo_path)
        ws = repo.writable_session(branch="main")
        ws.set_metadata("/arr", _array_meta([4]))
        ws.set_chunk("/arr", (0,), np.zeros(4, dtype="<f4").tobytes())
        snap_id = ws.commit("Test")

        # Transaction log exists with same name as snapshot
        assert (repo_path / "transactions" / snap_id).exists()

    def test_transaction_log_passes_verification(self, tmp_path: Path) -> None:
        repo_path = tmp_path / "repo"
        repo = icepyck.Repository.init(repo_path)
        ws = repo.writable_session(branch="main")
        ws.set_metadata("/arr", _array_meta([4]))
        ws.set_chunk("/arr", (0,), np.zeros(4, dtype="<f4").tobytes())
        ws.commit("Test")
        assert not verify_repo(repo_path)


# ---------------------------------------------------------------
# Spec: Conflict detection
# ---------------------------------------------------------------


class TestConflictBehavior:
    """Spec: 'when updating the branch reference, the client must detect
    whether a different session has updated the branch reference in the
    interim.'"""

    def test_conflict_detected_on_branch_move(self, tmp_path: Path) -> None:
        """Two sessions from same base: second commit must fail."""
        repo_path = tmp_path / "repo"
        repo = icepyck.Repository.init(repo_path)

        ws1 = repo.writable_session(branch="main")
        ws2 = repo.writable_session(branch="main")

        ws1.set_metadata("/a", _array_meta([2]))
        ws1.set_chunk("/a", (0,), np.zeros(2, dtype="<f4").tobytes())
        ws1.commit("First")

        ws2.set_metadata("/b", _array_meta([2]))
        ws2.set_chunk("/b", (0,), np.ones(2, dtype="<f4").tobytes())

        with pytest.raises(ConflictError):
            ws2.commit("Second")
