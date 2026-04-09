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

    def test_conflict_from_external_writer(self, tmp_path: Path) -> None:
        """Simulate an external writer by using a second Repository instance."""
        repo_path = tmp_path / "repo"
        icepyck.Repository.init(repo_path)

        # Writer A opens a session
        repo_a = icepyck.open(repo_path)
        ws_a = repo_a.writable_session(branch="main")

        # Writer B opens, commits, closes
        repo_b = icepyck.open(repo_path)
        ws_b = repo_b.writable_session(branch="main")
        ws_b.set_metadata("/from_b", _make_array_meta([4]))
        ws_b.set_chunk("/from_b", (0,), np.zeros(4, dtype="<f4").tobytes())
        ws_b.commit("From B")

        # Writer A now tries to commit — should detect conflict
        # (repo_a._state is stale, B already committed)
        # First we need to refresh repo_a to see B's changes
        repo_a.refresh()

        ws_a.set_metadata("/from_a", _make_array_meta([4]))
        ws_a.set_chunk("/from_a", (0,), np.ones(4, dtype="<f4").tobytes())
        with pytest.raises(ConflictError, match="updated by another writer"):
            ws_a.commit("From A")
