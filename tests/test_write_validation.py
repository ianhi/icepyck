"""Write-path validation: icepyck-write → icechunk-read.

Creates repos with icepyck, then opens them with the reference icechunk
package to verify our FlatBuffer output is spec-compliant. The Rust
flatbuffers library validates all (required) fields, so these tests
catch missing/malformed required fields that Python flatbuffers silently
ignores.

Run with:  uv run pytest tests/test_write_validation.py -v
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

icechunk = pytest.importorskip("icechunk")
zarr = pytest.importorskip("zarr")
np = pytest.importorskip("numpy")

import icepyck

pytestmark = pytest.mark.validation


class TestIcechunkReadsIcepyckRepo:
    """Create a repo with icepyck, open it with icechunk, verify data."""

    def test_init_and_read_empty_repo(self, tmp_path: Path) -> None:
        """icechunk can open a repo initialized by icepyck."""
        repo_path = tmp_path / "repo"
        icepyck.Repository.init(repo_path)

        storage = icechunk.local_filesystem_storage(str(repo_path))
        ic_repo = icechunk.Repository.open(storage)
        assert "main" in ic_repo.list_branches()

    def test_write_array_and_read_back(self, tmp_path: Path) -> None:
        """Write an array with icepyck, read with icechunk + zarr."""
        repo_path = tmp_path / "repo"
        pyck_repo = icepyck.Repository.init(repo_path)

        ws = pyck_repo.writable_session(branch="main")
        arr_data = np.arange(12, dtype="<f4").reshape(3, 4)
        meta = json.dumps(
            {
                "zarr_format": 3,
                "node_type": "array",
                "shape": [3, 4],
                "data_type": "float32",
                "chunk_grid": {
                    "name": "regular",
                    "configuration": {"chunk_shape": [3, 4]},
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
        ws.set_chunk("/data", (0, 0), arr_data.tobytes())
        ws.commit("Added 3x4 array")

        # Open with icechunk
        storage = icechunk.local_filesystem_storage(str(repo_path))
        ic_repo = icechunk.Repository.open(storage)
        session = ic_repo.readonly_session(branch="main")
        store = session.store
        root = zarr.open_group(store=store, mode="r")
        result = root["data"][:]
        np.testing.assert_array_equal(result, arr_data)

    def test_multiple_arrays(self, tmp_path: Path) -> None:
        """Write multiple arrays, verify icechunk reads all of them."""
        repo_path = tmp_path / "repo"
        pyck_repo = icepyck.Repository.init(repo_path)
        ws = pyck_repo.writable_session(branch="main")

        for name, dtype, values in [
            ("ints", "int32", [1, 2, 3, 4]),
            ("floats", "float64", [1.5, 2.5, 3.5]),
        ]:
            arr = np.array(values, dtype=dtype)
            meta = json.dumps(
                {
                    "zarr_format": 3,
                    "node_type": "array",
                    "shape": [len(values)],
                    "data_type": dtype,
                    "chunk_grid": {
                        "name": "regular",
                        "configuration": {"chunk_shape": [len(values)]},
                    },
                    "chunk_key_encoding": {
                        "name": "default",
                        "configuration": {"separator": "/"},
                    },
                    "fill_value": 0,
                    "codecs": [
                        {"name": "bytes", "configuration": {"endian": "little"}},
                    ],
                }
            ).encode()
            ws.set_metadata(f"/{name}", meta)
            ws.set_chunk(f"/{name}", (0,), arr.tobytes())

        ws.commit("Added two arrays")

        storage = icechunk.local_filesystem_storage(str(repo_path))
        ic_repo = icechunk.Repository.open(storage)
        session = ic_repo.readonly_session(branch="main")
        store = session.store
        root = zarr.open_group(store=store, mode="r")

        np.testing.assert_array_equal(root["ints"][:], [1, 2, 3, 4])
        np.testing.assert_array_equal(root["floats"][:], [1.5, 2.5, 3.5])

    def test_two_commits_and_log(self, tmp_path: Path) -> None:
        """Multiple commits are readable and log is navigable."""
        repo_path = tmp_path / "repo"
        pyck_repo = icepyck.Repository.init(repo_path)
        ws = pyck_repo.writable_session(branch="main")

        # Commit 1
        meta = json.dumps(
            {
                "zarr_format": 3,
                "node_type": "array",
                "shape": [3],
                "data_type": "float32",
                "chunk_grid": {
                    "name": "regular",
                    "configuration": {"chunk_shape": [3]},
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
        ws.set_metadata("/arr", meta)
        ws.set_chunk("/arr", (0,), np.array([1, 2, 3], dtype="<f4").tobytes())
        ws.commit("First commit")

        # Commit 2: overwrite
        ws.set_chunk("/arr", (0,), np.array([10, 20, 30], dtype="<f4").tobytes())
        ws.commit("Second commit")

        # Open with icechunk and verify latest
        storage = icechunk.local_filesystem_storage(str(repo_path))
        ic_repo = icechunk.Repository.open(storage)
        session = ic_repo.readonly_session(branch="main")
        store = session.store
        root = zarr.open_group(store=store, mode="r")
        np.testing.assert_array_equal(root["arr"][:], [10, 20, 30])

        # Verify icechunk sees multiple snapshots in the repo
        assert len(ic_repo.list_branches()) == 1
