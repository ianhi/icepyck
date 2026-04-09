"""End-to-end tests for WritableSession.

These tests copy real test repos, open them, mutate via WritableSession,
commit, and read back the data through icepyck's read path to verify
correctness. Also tests time travel — reading old snapshots after commits.
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
    """Copy a test repo to tmp_path and return the copy's path."""
    dest = tmp_path / src_name
    shutil.copytree(TEST_REPOS / src_name, dest)
    return dest


class TestCommitNewArray:
    """Commit a brand-new array with real numpy data, read it back."""

    def test_add_array_to_basic_repo(self, tmp_path: Path) -> None:
        repo_path = _copy_repo("basic", tmp_path)
        repo = icepyck.open(repo_path)

        # Verify initial state — basic repo has main branch
        assert "main" in repo.list_branches()
        old_nodes = repo.readonly_session(branch="main").list_nodes()
        old_paths = {n.path for n in old_nodes}

        # Create a writable session and add a new array
        ws = repo.writable_session(branch="main")

        # Write root group metadata
        root_meta = json.dumps(
            {
                "zarr_format": 3,
                "node_type": "group",
            }
        ).encode()
        ws.set_metadata("/", root_meta)

        # Write array metadata
        arr_data = np.array([10.0, 20.0, 30.0, 40.0], dtype="<f8")
        arr_meta = json.dumps(
            {
                "zarr_format": 3,
                "node_type": "array",
                "shape": [4],
                "data_type": {"name": "float64", "configuration": {"endian": "little"}},
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
        ws.set_metadata("/new_array", arr_meta)

        # Write chunk data (raw little-endian float64 bytes)
        ws.set_chunk("/new_array", (0,), arr_data.tobytes())

        # Commit
        _new_snapshot_id = ws.commit("Added new_array with test data")

        # --- Read back through icepyck ---
        # Re-open the repo (fresh, no caching from previous open)
        repo2 = icepyck.open(repo_path)
        session = repo2.readonly_session(branch="main")
        nodes = session.list_nodes()
        paths = {n.path for n in nodes}

        # The new array should be present
        assert "/new_array" in paths

        # All old paths should still be present
        assert old_paths.issubset(paths)

        # Read the array data through the chunk reader
        chunk_data = repo2.read_chunk("main", "/new_array", (0,))
        recovered = np.frombuffer(chunk_data, dtype="<f8")
        np.testing.assert_array_equal(recovered, arr_data)

    def test_add_array_read_via_zarr_store(self, tmp_path: Path) -> None:
        """Write array, commit, then read through zarr v3 Store."""
        repo_path = _copy_repo("basic", tmp_path)
        repo = icepyck.open(repo_path)
        ws = repo.writable_session(branch="main")

        arr_data = np.arange(12, dtype="<f4").reshape(3, 4)
        arr_meta = json.dumps(
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
        ws.set_metadata("/matrix", arr_meta)
        ws.set_chunk("/matrix", (0, 0), arr_data.tobytes())
        ws.commit("Added 3x4 matrix")

        # Read via zarr store
        repo2 = icepyck.open(repo_path)
        session = repo2.readonly_session(branch="main")
        store = session.store
        root = zarr.open_group(store=store, mode="r")
        result = root["matrix"][:]
        np.testing.assert_array_equal(result, arr_data)


class TestTimeTravelAfterCommit:
    """Verify that old snapshots still return their original data."""

    def test_old_snapshot_unchanged_after_new_commit(self, tmp_path: Path) -> None:
        repo_path = _copy_repo("basic", tmp_path)
        repo = icepyck.open(repo_path)

        # Record what the old snapshot looks like
        old_session = repo.readonly_session(branch="main")
        old_snap_id = old_session.snapshot_id  # Crockford string
        old_node_paths = {n.path for n in old_session.list_nodes()}

        # Read existing data from the basic repo
        old_chunks = {}
        for node in old_session.list_nodes():
            if node.node_type == "array":
                all_chunks = repo.read_all_chunks("main", node.path)
                old_chunks[node.path] = all_chunks

        # Commit a change
        ws = repo.writable_session(branch="main")
        ws.set_metadata(
            "/added_group",
            json.dumps(
                {
                    "zarr_format": 3,
                    "node_type": "group",
                }
            ).encode(),
        )
        new_snap_id = ws.commit("Added a group")
        assert new_snap_id != old_snap_id

        # Re-open and verify old snapshot is still intact
        repo2 = icepyck.open(repo_path)

        # New snapshot (main) should have the added group
        new_session = repo2.readonly_session(branch="main")
        new_paths = {n.path for n in new_session.list_nodes()}
        assert "/added_group" in new_paths

        # Old snapshot should NOT have the added group
        old_session2 = repo2.readonly_session(snapshot=old_snap_id)
        old_paths2 = {n.path for n in old_session2.list_nodes()}
        assert "/added_group" not in old_paths2
        assert old_paths2 == old_node_paths

        # Old snapshot's chunk data should be identical
        for path, chunks in old_chunks.items():
            for idx, expected_data in chunks.items():
                actual = repo2.read_chunk(old_snap_id, path, idx)
                assert actual == expected_data, (
                    f"Chunk data mismatch at {path}[{idx}] after new commit"
                )


class TestMultipleCommits:
    """Test multiple sequential commits on the same session."""

    def test_two_commits_both_readable(self, tmp_path: Path) -> None:
        repo_path = _copy_repo("basic", tmp_path)
        repo = icepyck.open(repo_path)

        ws = repo.writable_session(branch="main")

        # First commit: add array A
        a_data = np.array([1, 2, 3], dtype="<i4")
        ws.set_metadata(
            "/arr_a",
            json.dumps(
                {
                    "zarr_format": 3,
                    "node_type": "array",
                    "shape": [3],
                    "data_type": {
                        "name": "int32",
                        "configuration": {"endian": "little"},
                    },
                    "chunk_grid": {
                        "name": "regular",
                        "configuration": {"chunk_shape": [3]},
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
        ws.set_chunk("/arr_a", (0,), a_data.tobytes())
        snap1 = ws.commit("Added arr_a")

        # Second commit: add array B
        b_data = np.array([100, 200], dtype="<i4")
        ws.set_metadata(
            "/arr_b",
            json.dumps(
                {
                    "zarr_format": 3,
                    "node_type": "array",
                    "shape": [2],
                    "data_type": {
                        "name": "int32",
                        "configuration": {"endian": "little"},
                    },
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
        ws.set_chunk("/arr_b", (0,), b_data.tobytes())
        _snap2 = ws.commit("Added arr_b")

        # Verify: snap2 (main) has both arrays
        repo2 = icepyck.open(repo_path)
        session = repo2.readonly_session(branch="main")
        paths = {n.path for n in session.list_nodes()}
        assert "/arr_a" in paths
        assert "/arr_b" in paths

        # Verify data
        chunk_a = repo2.read_chunk("main", "/arr_a", (0,))
        np.testing.assert_array_equal(np.frombuffer(chunk_a, dtype="<i4"), a_data)
        chunk_b = repo2.read_chunk("main", "/arr_b", (0,))
        np.testing.assert_array_equal(np.frombuffer(chunk_b, dtype="<i4"), b_data)

        # Time travel: snap1 has arr_a but NOT arr_b
        session1 = repo2.readonly_session(snapshot=snap1)
        paths1 = {n.path for n in session1.list_nodes()}
        assert "/arr_a" in paths1
        assert "/arr_b" not in paths1


class TestOverwriteExistingChunks:
    """Overwrite chunk data in an existing array."""

    def test_overwrite_preserves_other_chunks(self, tmp_path: Path) -> None:
        """Write two chunks, commit, overwrite one, commit again.
        Verify the un-overwritten chunk is still correct."""
        repo_path = _copy_repo("basic", tmp_path)
        repo = icepyck.open(repo_path)
        ws = repo.writable_session(branch="main")

        # Create array with two chunks
        meta = json.dumps(
            {
                "zarr_format": 3,
                "node_type": "array",
                "shape": [8],
                "data_type": {"name": "float64", "configuration": {"endian": "little"}},
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
        ws.set_metadata("/multi", meta)

        chunk0 = np.array([1.0, 2.0, 3.0, 4.0], dtype="<f8").tobytes()
        chunk1 = np.array([5.0, 6.0, 7.0, 8.0], dtype="<f8").tobytes()
        ws.set_chunk("/multi", (0,), chunk0)
        ws.set_chunk("/multi", (1,), chunk1)
        snap1 = ws.commit("Two chunks")

        # Overwrite only chunk 0
        new_chunk0 = np.array([10.0, 20.0, 30.0, 40.0], dtype="<f8").tobytes()
        ws.set_chunk("/multi", (0,), new_chunk0)
        _snap2 = ws.commit("Overwrote chunk 0")

        # Verify: chunk 0 has new data, chunk 1 is preserved
        repo2 = icepyck.open(repo_path)
        c0 = repo2.read_chunk("main", "/multi", (0,))
        c1 = repo2.read_chunk("main", "/multi", (1,))
        np.testing.assert_array_equal(
            np.frombuffer(c0, dtype="<f8"),
            [10.0, 20.0, 30.0, 40.0],
        )
        np.testing.assert_array_equal(
            np.frombuffer(c1, dtype="<f8"),
            [5.0, 6.0, 7.0, 8.0],
        )

        # Time travel: snap1 should have original chunk 0
        old_c0 = repo2.read_chunk(snap1, "/multi", (0,))
        np.testing.assert_array_equal(
            np.frombuffer(old_c0, dtype="<f8"),
            [1.0, 2.0, 3.0, 4.0],
        )
