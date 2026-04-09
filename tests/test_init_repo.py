"""Tests for Repository.init() — creating new repos from scratch.

Tests the full lifecycle: init → writable session → add data → commit → read.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import zarr

import icepyck


class TestInitRepo:
    def test_init_creates_valid_repo(self, tmp_path: Path) -> None:
        repo_path = tmp_path / "new-repo"
        repo_path.mkdir()
        repo = icepyck.Repository.init(repo_path)

        assert repo.list_branches() == ["main"]
        assert repo.list_tags() == []

        session = repo.readonly_session(branch="main")
        nodes = session.list_nodes()
        assert len(nodes) == 1
        assert nodes[0].path == "/"
        assert nodes[0].node_type == "group"

    def test_init_then_write_and_read(self, tmp_path: Path) -> None:
        repo_path = tmp_path / "new-repo"
        repo_path.mkdir()
        repo = icepyck.Repository.init(repo_path)

        ws = repo.writable_session(branch="main")
        data = np.array([3.14, 2.72, 1.41], dtype="<f8")
        ws.set_metadata("/constants", json.dumps({
            "zarr_format": 3,
            "node_type": "array",
            "shape": [3],
            "data_type": "float64",
            "chunk_grid": {"name": "regular", "configuration": {"chunk_shape": [3]}},
            "chunk_key_encoding": {"name": "default", "configuration": {"separator": "/"}},
            "fill_value": 0.0,
            "codecs": [{"name": "bytes", "configuration": {"endian": "little"}}],
        }).encode())
        ws.set_chunk("/constants", (0,), data.tobytes())
        ws.commit("First real data")

        # Read back through zarr
        repo2 = icepyck.open(repo_path)
        root = zarr.open_group(store=repo2.readonly_session(branch="main").store, mode="r")
        result = root["constants"][:]
        np.testing.assert_array_equal(result, data)

    def test_init_then_multiple_commits(self, tmp_path: Path) -> None:
        repo_path = tmp_path / "multi"
        repo_path.mkdir()
        repo = icepyck.Repository.init(repo_path)

        # Commit 1
        ws = repo.writable_session(branch="main")
        ws.set_metadata("/arr1", json.dumps({
            "zarr_format": 3, "node_type": "array",
            "shape": [2], "data_type": "int32",
            "chunk_grid": {"name": "regular", "configuration": {"chunk_shape": [2]}},
            "chunk_key_encoding": {"name": "default", "configuration": {"separator": "/"}},
            "fill_value": 0, "codecs": [{"name": "bytes", "configuration": {"endian": "little"}}],
        }).encode())
        ws.set_chunk("/arr1", (0,), np.array([1, 2], dtype="<i4").tobytes())
        snap1 = ws.commit("arr1")

        # Commit 2
        ws.set_metadata("/arr2", json.dumps({
            "zarr_format": 3, "node_type": "array",
            "shape": [2], "data_type": "int32",
            "chunk_grid": {"name": "regular", "configuration": {"chunk_shape": [2]}},
            "chunk_key_encoding": {"name": "default", "configuration": {"separator": "/"}},
            "fill_value": 0, "codecs": [{"name": "bytes", "configuration": {"endian": "little"}}],
        }).encode())
        ws.set_chunk("/arr2", (0,), np.array([3, 4], dtype="<i4").tobytes())
        snap2 = ws.commit("arr2")

        # Verify latest has both
        repo2 = icepyck.open(repo_path)
        session = repo2.readonly_session(branch="main")
        paths = {n.path for n in session.list_nodes()}
        assert "/arr1" in paths
        assert "/arr2" in paths

        # Time travel: snap1 has only arr1
        s1 = repo2.readonly_session(snapshot=snap1)
        paths1 = {n.path for n in s1.list_nodes()}
        assert "/arr1" in paths1
        assert "/arr2" not in paths1

    def test_init_then_reopen(self, tmp_path: Path) -> None:
        """Init, close, reopen with icepyck.open() — should work."""
        repo_path = tmp_path / "reopen"
        repo_path.mkdir()
        icepyck.Repository.init(repo_path)

        # Reopen
        repo = icepyck.open(repo_path)
        assert repo.list_branches() == ["main"]
        session = repo.readonly_session(branch="main")
        assert len(session.list_nodes()) == 1
