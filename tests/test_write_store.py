"""End-to-end tests for IcechunkStore (zarr write path).

Write data using zarr's API through IcechunkStore, commit via
WritableSession, then read back through a fresh read-only session.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import zarr

import icepyck

TEST_REPOS = Path(__file__).parent.parent / "test-repos"


def _copy_repo(src_name: str, tmp_path: Path) -> Path:
    dest = tmp_path / src_name
    shutil.copytree(TEST_REPOS / src_name, dest)
    return dest


class TestZarrWriteThenRead:
    """Write through zarr Store, commit, read back through zarr."""

    def test_write_1d_array_then_read_via_zarr(self, tmp_path: Path) -> None:
        repo_path = _copy_repo("basic", tmp_path)
        repo = icepyck.open(repo_path)
        ws = repo.writable_session(branch="main")

        # Write a 1D array
        data = np.array([1.5, 2.5, 3.5, 4.5, 5.5, 6.5], dtype="<f8")
        meta = json.dumps(
            {
                "zarr_format": 3,
                "node_type": "array",
                "shape": [6],
                "data_type": "float64",
                "chunk_grid": {
                    "name": "regular",
                    "configuration": {"chunk_shape": [6]},
                },
                "chunk_key_encoding": {
                    "name": "default",
                    "configuration": {"separator": "/"},
                },
                "fill_value": 0.0,
                "codecs": [{"name": "bytes", "configuration": {"endian": "little"}}],
            }
        ).encode()
        ws.set_metadata("/temps", meta)
        ws.set_chunk("/temps", (0,), data.tobytes())
        ws.commit("Added temps")

        # Read back through zarr
        repo2 = icepyck.open(repo_path)
        session = repo2.readonly_session(branch="main")
        root = zarr.open_group(store=session.store, mode="r")
        result = root["temps"][:]
        np.testing.assert_array_equal(result, data)

    def test_write_2d_array_via_zarr(self, tmp_path: Path) -> None:
        repo_path = _copy_repo("basic", tmp_path)
        repo = icepyck.open(repo_path)
        ws = repo.writable_session(branch="main")

        # Create a 2D array
        data = np.arange(20, dtype="<f4").reshape(4, 5)
        meta = json.dumps(
            {
                "zarr_format": 3,
                "node_type": "array",
                "shape": [4, 5],
                "data_type": "float32",
                "chunk_grid": {
                    "name": "regular",
                    "configuration": {"chunk_shape": [4, 5]},
                },
                "chunk_key_encoding": {
                    "name": "default",
                    "configuration": {"separator": "/"},
                },
                "fill_value": 0.0,
                "codecs": [{"name": "bytes", "configuration": {"endian": "little"}}],
            }
        ).encode()
        ws.set_metadata("/grid", meta)
        ws.set_chunk("/grid", (0, 0), data.tobytes())
        ws.commit("Added 4x5 grid")

        # Read via zarr
        repo2 = icepyck.open(repo_path)
        session = repo2.readonly_session(branch="main")
        root = zarr.open_group(store=session.store, mode="r")
        result = root["grid"][:]
        np.testing.assert_array_equal(result, data)

    def test_existing_data_still_readable_after_write(self, tmp_path: Path) -> None:
        """Ensure that writing new data doesn't break reading existing data."""
        repo_path = _copy_repo("basic", tmp_path)
        repo = icepyck.open(repo_path)

        # Read existing data before write
        old_session = repo.readonly_session(branch="main")
        old_store = old_session.store
        old_root = zarr.open_group(store=old_store, mode="r")
        # basic repo has group1/temperatures and group1/timestamps
        old_temps = old_root["group1/temperatures"][:]

        # Write new data
        ws = repo.writable_session(branch="main")
        ws.set_metadata(
            "/extra",
            json.dumps(
                {
                    "zarr_format": 3,
                    "node_type": "array",
                    "shape": [2],
                    "data_type": "int32",
                    "chunk_grid": {
                        "name": "regular",
                        "configuration": {"chunk_shape": [2]},
                    },
                    "chunk_key_encoding": {
                        "name": "default",
                        "configuration": {"separator": "/"},
                    },
                    "fill_value": 0,
                    "codecs": [
                        {"name": "bytes", "configuration": {"endian": "little"}}
                    ],
                }
            ).encode(),
        )
        ws.set_chunk("/extra", (0,), np.array([42, 43], dtype="<i4").tobytes())
        ws.commit("Added extra")

        # Read new AND old data
        repo2 = icepyck.open(repo_path)
        session = repo2.readonly_session(branch="main")
        root = zarr.open_group(store=session.store, mode="r")

        # Old data should be identical
        new_temps = root["group1/temperatures"][:]
        np.testing.assert_array_equal(new_temps, old_temps)

        # New data should be present
        extra = root["extra"][:]
        np.testing.assert_array_equal(extra, [42, 43])


class TestMultiChunkArray:
    """Test arrays with multiple chunks."""

    def test_write_multiple_chunks(self, tmp_path: Path) -> None:
        repo_path = _copy_repo("basic", tmp_path)
        repo = icepyck.open(repo_path)
        ws = repo.writable_session(branch="main")

        # Array shape [8] with chunk_shape [4] → 2 chunks
        meta = json.dumps(
            {
                "zarr_format": 3,
                "node_type": "array",
                "shape": [8],
                "data_type": "float64",
                "chunk_grid": {
                    "name": "regular",
                    "configuration": {"chunk_shape": [4]},
                },
                "chunk_key_encoding": {
                    "name": "default",
                    "configuration": {"separator": "/"},
                },
                "fill_value": 0.0,
                "codecs": [{"name": "bytes", "configuration": {"endian": "little"}}],
            }
        ).encode()
        ws.set_metadata("/big", meta)

        c0 = np.array([1.0, 2.0, 3.0, 4.0], dtype="<f8")
        c1 = np.array([5.0, 6.0, 7.0, 8.0], dtype="<f8")
        ws.set_chunk("/big", (0,), c0.tobytes())
        ws.set_chunk("/big", (1,), c1.tobytes())
        ws.commit("2 chunks")

        # Read back
        repo2 = icepyck.open(repo_path)
        root = zarr.open_group(
            store=repo2.readonly_session(branch="main").store, mode="r"
        )
        result = root["big"][:]
        np.testing.assert_array_equal(result, np.concatenate([c0, c1]))
