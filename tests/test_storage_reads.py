"""Verify that branch/tag operations don't cause unnecessary storage reads.

Wraps LocalStorage with a counting proxy to measure actual read calls.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import icepyck


class CountingStorage:
    """Proxy that counts read() calls to the underlying storage."""

    def __init__(self, inner: icepyck.Storage) -> None:
        self._inner = inner
        self.read_count = 0
        self.read_paths: list[str] = []

    def read(self, path: str) -> bytes:
        self.read_count += 1
        self.read_paths.append(path)
        return self._inner.read(path)

    def read_versioned(self, path: str) -> tuple[bytes, str]:
        self.read_count += 1
        self.read_paths.append(path)
        return self._inner.read_versioned(path)

    def write(self, path: str, data: bytes) -> None:
        self._inner.write(path, data)

    def conditional_write(self, path: str, data: bytes, expected_version: str) -> str:
        return self._inner.conditional_write(path, data, expected_version)

    def exists(self, path: str) -> bool:
        return self._inner.exists(path)

    def list_prefix(self, prefix: str) -> list[str]:
        return self._inner.list_prefix(prefix)


def _make_repo(tmp_path: Path) -> tuple[Path, CountingStorage]:
    """Create a repo, return (path, counting_storage)."""
    repo_path = tmp_path / "repo"
    icepyck.Repository.init(repo_path)

    # Wrap with counting storage
    inner = icepyck.LocalStorage(str(repo_path))
    counting = CountingStorage(inner)
    return repo_path, counting


class TestBranchTagOpsNoExtraReads:
    """Branch/tag operations should NOT re-read the repo file."""

    def test_create_branch_no_read(self, tmp_path: Path) -> None:
        repo_path, storage = _make_repo(tmp_path)
        repo = icepyck.Repository(storage=storage)

        # Reset counter after construction (which reads the repo file)
        storage.read_count = 0
        storage.read_paths.clear()

        repo.create_branch("dev", "main")

        # Should be zero reads — state is in memory
        repo_reads = [p for p in storage.read_paths if p == "repo"]
        assert len(repo_reads) == 0, (
            f"Expected 0 repo reads, got {len(repo_reads)}: {storage.read_paths}"
        )

    def test_delete_branch_no_read(self, tmp_path: Path) -> None:
        repo_path, storage = _make_repo(tmp_path)
        repo = icepyck.Repository(storage=storage)
        repo.create_branch("dev", "main")

        storage.read_count = 0
        storage.read_paths.clear()

        repo.delete_branch("dev")

        repo_reads = [p for p in storage.read_paths if p == "repo"]
        assert len(repo_reads) == 0, (
            f"Expected 0 repo reads, got {len(repo_reads)}: {storage.read_paths}"
        )

    def test_create_tag_no_read(self, tmp_path: Path) -> None:
        repo_path, storage = _make_repo(tmp_path)
        repo = icepyck.Repository(storage=storage)

        storage.read_count = 0
        storage.read_paths.clear()

        repo.create_tag("v1", "main")

        repo_reads = [p for p in storage.read_paths if p == "repo"]
        assert len(repo_reads) == 0, (
            f"Expected 0 repo reads, got {len(repo_reads)}: {storage.read_paths}"
        )

    def test_delete_tag_no_read(self, tmp_path: Path) -> None:
        repo_path, storage = _make_repo(tmp_path)
        repo = icepyck.Repository(storage=storage)
        repo.create_tag("v1", "main")

        storage.read_count = 0
        storage.read_paths.clear()

        repo.delete_tag("v1")

        repo_reads = [p for p in storage.read_paths if p == "repo"]
        assert len(repo_reads) == 0, (
            f"Expected 0 repo reads, got {len(repo_reads)}: {storage.read_paths}"
        )

    def test_commit_no_repo_read(self, tmp_path: Path) -> None:
        """commit() should not read the repo file (delegates to _apply_commit)."""
        repo_path, storage = _make_repo(tmp_path)
        repo = icepyck.Repository(storage=storage)
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

        # Reset after writable_session (which does a refresh read)
        storage.read_count = 0
        storage.read_paths.clear()

        ws.commit("test")

        # commit reads the snapshot file (to refresh base_nodes), but NOT the repo file
        repo_reads = [p for p in storage.read_paths if p == "repo"]
        assert len(repo_reads) == 0, (
            f"Expected 0 repo reads during commit, got {len(repo_reads)}: {storage.read_paths}"
        )

    def test_writable_session_does_refresh(self, tmp_path: Path) -> None:
        """writable_session() SHOULD read the repo file (to catch external changes)."""
        repo_path, storage = _make_repo(tmp_path)
        repo = icepyck.Repository(storage=storage)

        storage.read_count = 0
        storage.read_paths.clear()

        _ws = repo.writable_session(branch="main")

        # Should read exactly 1 repo file (the refresh) + 1 snapshot file
        repo_reads = [p for p in storage.read_paths if p == "repo"]
        assert len(repo_reads) == 1, (
            f"Expected 1 repo read, got {len(repo_reads)}: {storage.read_paths}"
        )
