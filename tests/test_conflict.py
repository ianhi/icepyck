"""Tests for conflict detection when multiple writers target the same branch."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

import icepyck
from icepyck import ConflictError


def _make_array_meta(shape: list[int], dtype: str = "float32") -> bytes:
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


class TestConflictDetection:
    def test_concurrent_commits_conflict(self, tmp_path: Path) -> None:
        """Two sessions from the same base should conflict."""
        repo_path = tmp_path / "repo"
        repo = icepyck.Repository.init(repo_path)

        # Create two sessions from the same base
        ws1 = repo.writable_session(branch="main")
        ws2 = repo.writable_session(branch="main")

        # Session 1 commits first
        ws1.set_metadata("/arr1", _make_array_meta([4]))
        ws1.set_chunk("/arr1", (0,), np.zeros(4, dtype="<f4").tobytes())
        ws1.commit("Writer 1")

        # Session 2 tries to commit — should conflict
        ws2.set_metadata("/arr2", _make_array_meta([4]))
        ws2.set_chunk("/arr2", (0,), np.ones(4, dtype="<f4").tobytes())
        with pytest.raises(ConflictError, match="updated by another writer"):
            ws2.commit("Writer 2")

    def test_sequential_commits_no_conflict(self, tmp_path: Path) -> None:
        """Sequential commits on the same session should not conflict."""
        repo_path = tmp_path / "repo"
        repo = icepyck.Repository.init(repo_path)
        ws = repo.writable_session(branch="main")

        ws.set_metadata("/arr1", _make_array_meta([4]))
        ws.set_chunk("/arr1", (0,), np.zeros(4, dtype="<f4").tobytes())
        ws.commit("Commit 1")

        ws.set_metadata("/arr2", _make_array_meta([4]))
        ws.set_chunk("/arr2", (0,), np.ones(4, dtype="<f4").tobytes())
        ws.commit("Commit 2")  # Should succeed — sequential, no conflict

    def test_different_branches_no_conflict(self, tmp_path: Path) -> None:
        """Commits to different branches should not conflict."""
        repo_path = tmp_path / "repo"
        repo = icepyck.Repository.init(repo_path)
        repo.create_branch("dev", "main")

        ws_main = repo.writable_session(branch="main")
        ws_dev = repo.writable_session(branch="dev")

        ws_main.set_metadata("/main_arr", _make_array_meta([4]))
        ws_main.set_chunk("/main_arr", (0,), np.zeros(4, dtype="<f4").tobytes())
        ws_main.commit("On main")

        ws_dev.set_metadata("/dev_arr", _make_array_meta([4]))
        ws_dev.set_chunk("/dev_arr", (0,), np.ones(4, dtype="<f4").tobytes())
        ws_dev.commit("On dev")  # Should succeed — different branch

    def test_two_repos_two_sessions_conflict(self, tmp_path: Path) -> None:
        """Two independent Repository objects writing to the same branch.

        This simulates two processes (or two machines) both opening the
        same repo from storage and racing to commit.
        """
        repo_path = tmp_path / "repo"
        icepyck.Repository.init(repo_path)

        # Two independent Repository objects (like two processes)
        repo_a = icepyck.open(repo_path)
        repo_b = icepyck.open(repo_path)

        # Both open writable sessions from the same base
        ws_a = repo_a.writable_session(branch="main")
        ws_b = repo_b.writable_session(branch="main")

        # Both prepare changes
        ws_a.set_metadata("/from_a", _make_array_meta([4]))
        ws_a.set_chunk("/from_a", (0,), np.zeros(4, dtype="<f4").tobytes())

        ws_b.set_metadata("/from_b", _make_array_meta([4]))
        ws_b.set_chunk("/from_b", (0,), np.ones(4, dtype="<f4").tobytes())

        # Writer A commits first — succeeds
        ws_a.commit("From A")

        # Writer B tries to commit — repo_b's in-memory state is stale.
        # To detect the conflict, B must refresh first (pick up A's commit).
        # Without refresh, B would silently overwrite A's changes.
        # This is the current behavior — Phase 12 will add storage-level
        # conditional writes to catch this without explicit refresh.
        repo_b.refresh()

        with pytest.raises(ConflictError, match="updated by another writer"):
            ws_b.commit("From B")

        # Verify A's data is intact
        repo_check = icepyck.open(repo_path)
        nodes = {
            n.path for n in repo_check.readonly_session(branch="main").list_nodes()
        }
        assert "/from_a" in nodes
        assert "/from_b" not in nodes

    def test_two_repos_retry_after_conflict(self, tmp_path: Path) -> None:
        """After conflict, writer B can get a new session and retry."""
        repo_path = tmp_path / "repo"
        icepyck.Repository.init(repo_path)

        repo_a = icepyck.open(repo_path)
        repo_b = icepyck.open(repo_path)

        ws_a = repo_a.writable_session(branch="main")
        ws_b = repo_b.writable_session(branch="main")

        ws_a.set_metadata("/from_a", _make_array_meta([4]))
        ws_a.set_chunk("/from_a", (0,), np.zeros(4, dtype="<f4").tobytes())
        ws_a.commit("From A")

        ws_b.set_metadata("/from_b", _make_array_meta([4]))
        ws_b.set_chunk("/from_b", (0,), np.ones(4, dtype="<f4").tobytes())
        repo_b.refresh()

        with pytest.raises(ConflictError):
            ws_b.commit("From B")

        # Retry: get a new session (which refreshes and sees A's commit)
        ws_b2 = repo_b.writable_session(branch="main")
        ws_b2.set_metadata("/from_b", _make_array_meta([4]))
        ws_b2.set_chunk("/from_b", (0,), np.ones(4, dtype="<f4").tobytes())
        ws_b2.commit("From B (retry)")

        # Both arrays should now exist
        repo_check = icepyck.open(repo_path)
        nodes = {
            n.path for n in repo_check.readonly_session(branch="main").list_nodes()
        }
        assert "/from_a" in nodes
        assert "/from_b" in nodes
