"""Tests for move/rename operations."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

import icepyck
from icepyck.verify import verify_repo


def _array_meta(shape: list[int]) -> bytes:
    return json.dumps(
        {
            "zarr_format": 3,
            "node_type": "array",
            "shape": shape,
            "data_type": "float32",
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


class TestMoveNode:
    def test_move_array(self, tmp_path: Path) -> None:
        repo_path = tmp_path / "repo"
        repo = icepyck.Repository.init(repo_path)
        ws = repo.writable_session(branch="main")

        ws.set_metadata("/old_name", _array_meta([4]))
        ws.set_chunk("/old_name", (0,), np.zeros(4, dtype="<f4").tobytes())
        ws.commit("Create array")

        ws2 = repo.writable_session(branch="main")
        ws2.move_node("/old_name", "/new_name")
        ws2.commit("Rename array")

        repo2 = icepyck.open(repo_path)
        nodes = {n.path for n in repo2.readonly_session(branch="main").list_nodes()}
        assert "/new_name" in nodes
        assert "/old_name" not in nodes

    def test_move_preserves_data(self, tmp_path: Path) -> None:
        repo_path = tmp_path / "repo"
        repo = icepyck.Repository.init(repo_path)
        ws = repo.writable_session(branch="main")

        data = np.array([1.0, 2.0, 3.0, 4.0], dtype="<f4").tobytes()
        ws.set_metadata("/arr", _array_meta([4]))
        ws.set_chunk("/arr", (0,), data)
        ws.commit("Create")

        ws2 = repo.writable_session(branch="main")
        ws2.move_node("/arr", "/renamed")
        ws2.commit("Move")

        repo2 = icepyck.open(repo_path)
        chunk = repo2.read_chunk("main", "/renamed", (0,))
        np.testing.assert_array_equal(
            np.frombuffer(chunk, dtype="<f4"), [1.0, 2.0, 3.0, 4.0]
        )

    def test_move_new_node(self, tmp_path: Path) -> None:
        """Move a node that was just created in the same session."""
        repo_path = tmp_path / "repo"
        repo = icepyck.Repository.init(repo_path)
        ws = repo.writable_session(branch="main")

        ws.set_metadata("/temp", _array_meta([2]))
        ws.set_chunk("/temp", (0,), np.zeros(2, dtype="<f4").tobytes())
        ws.move_node("/temp", "/final")
        ws.commit("Create and move")

        nodes = {n.path for n in repo.readonly_session(branch="main").list_nodes()}
        assert "/final" in nodes
        assert "/temp" not in nodes

    def test_move_nonexistent_raises(self, tmp_path: Path) -> None:
        repo_path = tmp_path / "repo"
        repo = icepyck.Repository.init(repo_path)
        ws = repo.writable_session(branch="main")

        with pytest.raises(KeyError, match="not found"):
            ws.move_node("/nope", "/somewhere")

    def test_move_group(self, tmp_path: Path) -> None:
        repo_path = tmp_path / "repo"
        repo = icepyck.Repository.init(repo_path)
        ws = repo.writable_session(branch="main")

        ws.set_metadata(
            "/grp",
            json.dumps({"zarr_format": 3, "node_type": "group"}).encode(),
        )
        ws.commit("Create group")

        ws2 = repo.writable_session(branch="main")
        ws2.move_node("/grp", "/renamed_grp")
        ws2.commit("Move group")

        nodes = {n.path for n in repo.readonly_session(branch="main").list_nodes()}
        assert "/renamed_grp" in nodes
        assert "/grp" not in nodes

    def test_passes_verification(self, tmp_path: Path) -> None:
        repo_path = tmp_path / "repo"
        repo = icepyck.Repository.init(repo_path)
        ws = repo.writable_session(branch="main")
        ws.set_metadata("/arr", _array_meta([4]))
        ws.set_chunk("/arr", (0,), np.zeros(4, dtype="<f4").tobytes())
        ws.commit("Create")

        ws2 = repo.writable_session(branch="main")
        ws2.move_node("/arr", "/moved")
        ws2.commit("Move")

        assert not verify_repo(repo_path)
