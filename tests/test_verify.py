"""Tests for the spec conformance verifier.

Verifies that icepyck-created repos pass all spec checks, and that
the verifier catches known violations.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

import icepyck
from icepyck.verify import Issue, verify_repo


class TestVerifyIcepyckRepos:
    """Repos created by icepyck should pass verification."""

    def test_empty_repo(self, tmp_path: Path) -> None:
        repo_path = tmp_path / "repo"
        icepyck.Repository.init(repo_path)
        issues = verify_repo(repo_path)
        assert not issues, f"Spec violations: {[str(i) for i in issues]}"

    def test_repo_with_array(self, tmp_path: Path) -> None:
        repo_path = tmp_path / "repo"
        repo = icepyck.Repository.init(repo_path)
        ws = repo.writable_session(branch="main")

        meta = json.dumps({
            "zarr_format": 3,
            "node_type": "array",
            "shape": [10],
            "data_type": "float64",
            "chunk_grid": {
                "name": "regular",
                "configuration": {"chunk_shape": [5]},
            },
            "chunk_key_encoding": {
                "name": "default",
                "configuration": {"separator": "/"},
            },
            "fill_value": 0.0,
            "codecs": [
                {"name": "bytes", "configuration": {"endian": "little"}},
            ],
        }).encode()
        ws.set_metadata("/data", meta)
        ws.set_chunk("/data", (0,), np.zeros(5, dtype="<f8").tobytes())
        ws.set_chunk("/data", (1,), np.ones(5, dtype="<f8").tobytes())
        ws.commit("Added array with two chunks")

        issues = verify_repo(repo_path)
        assert not issues, f"Spec violations: {[str(i) for i in issues]}"

    def test_repo_with_multiple_commits(self, tmp_path: Path) -> None:
        repo_path = tmp_path / "repo"
        repo = icepyck.Repository.init(repo_path)
        ws = repo.writable_session(branch="main")

        meta = json.dumps({
            "zarr_format": 3,
            "node_type": "array",
            "shape": [4],
            "data_type": "int32",
            "chunk_grid": {
                "name": "regular",
                "configuration": {"chunk_shape": [4]},
            },
            "chunk_key_encoding": {
                "name": "default",
                "configuration": {"separator": "/"},
            },
            "fill_value": 0,
            "codecs": [
                {"name": "bytes", "configuration": {"endian": "little"}},
            ],
        }).encode()
        ws.set_metadata("/arr", meta)
        ws.set_chunk("/arr", (0,), np.array([1, 2, 3, 4], dtype="<i4").tobytes())
        ws.commit("Commit 1")

        ws.set_chunk("/arr", (0,), np.array([10, 20, 30, 40], dtype="<i4").tobytes())
        ws.commit("Commit 2")

        issues = verify_repo(repo_path)
        assert not issues, f"Spec violations: {[str(i) for i in issues]}"

    def test_repo_with_multiple_arrays(self, tmp_path: Path) -> None:
        repo_path = tmp_path / "repo"
        repo = icepyck.Repository.init(repo_path)
        ws = repo.writable_session(branch="main")

        for name in ["alpha", "beta", "gamma"]:
            meta = json.dumps({
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
            }).encode()
            ws.set_metadata(f"/{name}", meta)
            ws.set_chunk(f"/{name}", (0,), np.zeros(3, dtype="<f4").tobytes())

        ws.commit("Three arrays")
        issues = verify_repo(repo_path)
        assert not issues, f"Spec violations: {[str(i) for i in issues]}"


class TestVerifyExistingTestRepos:
    """Verify the test-repos that were created by reference icechunk."""

    TEST_REPOS = Path(__file__).parent.parent / "test-repos"

    @pytest.fixture(params=["basic", "nested", "scalar", "native-chunks"])
    def repo_name(self, request: pytest.FixtureRequest) -> str:
        return request.param  # type: ignore[return-value]

    def test_reference_repos_pass(self, repo_name: str) -> None:
        repo_path = self.TEST_REPOS / repo_name
        if not (repo_path / "repo").exists():
            pytest.skip(f"Test repo {repo_name!r} not available")
        issues = verify_repo(repo_path)
        assert not issues, f"Spec violations in {repo_name}: {[str(i) for i in issues]}"


class TestVerifyDetectsViolations:
    """Verify that the verifier catches known violations."""

    def test_missing_repo_file(self, tmp_path: Path) -> None:
        issues = verify_repo(tmp_path)
        assert len(issues) == 1
        assert "repo file not found" in str(issues[0])
