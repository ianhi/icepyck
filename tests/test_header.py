"""Tests for the binary header parser."""

from pathlib import Path

import pytest

from icepyck.header import (
    HEADER_SIZE,
    MAGIC,
    Compression,
    FileType,
    parse_file,
)


@pytest.fixture
def repo_file(basic_repo):
    return basic_repo / "repo"


@pytest.fixture
def snapshot_file(basic_repo):
    """Return the path to a snapshot file in the basic repo."""
    snapshots_dir = basic_repo / "snapshots"
    # Get the first snapshot file
    files = list(snapshots_dir.iterdir())
    assert len(files) > 0, "No snapshot files found"
    return files[0]


@pytest.fixture
def manifest_file(basic_repo):
    """Return the path to a manifest file in the basic repo."""
    manifests_dir = basic_repo / "manifests"
    files = list(manifests_dir.iterdir())
    assert len(files) > 0, "No manifest files found"
    return files[0]


class TestMagic:
    def test_magic_bytes(self):
        assert MAGIC == b"ICE\xf0\x9f\xa7\x8aCHUNK"

    def test_magic_length(self):
        assert len(MAGIC) == 12

    def test_header_size(self):
        assert HEADER_SIZE == 39


class TestParseRepoFile:
    @pytest.mark.skipif(
        not (Path(__file__).parent.parent / "test-repos" / "basic" / "repo").exists(),
        reason="basic test repo not available",
    )
    def test_repo_file_type(self, repo_file):
        header, payload = parse_file(repo_file)
        assert header.file_type == FileType.REPO_INFO

    def test_repo_spec_version(self, repo_file):
        header, _ = parse_file(repo_file)
        assert header.spec_version in (1, 2)

    def test_repo_implementation_is_string(self, repo_file):
        header, _ = parse_file(repo_file)
        assert isinstance(header.implementation, str)
        assert len(header.implementation) > 0

    def test_repo_compression(self, repo_file):
        header, _ = parse_file(repo_file)
        assert header.compression in (Compression.NONE, Compression.ZSTD)

    def test_payload_is_bytes(self, repo_file):
        _, payload = parse_file(repo_file)
        assert isinstance(payload, bytes)
        assert len(payload) > 0


class TestParseSnapshotFile:
    @pytest.mark.skipif(
        not (
            Path(__file__).parent.parent / "test-repos" / "basic" / "snapshots"
        ).exists(),
        reason="basic test repo snapshots not available",
    )
    def test_snapshot_file_type(self, snapshot_file):
        header, _ = parse_file(snapshot_file)
        assert header.file_type == FileType.SNAPSHOT


class TestParseManifestFile:
    @pytest.mark.skipif(
        not (
            Path(__file__).parent.parent / "test-repos" / "basic" / "manifests"
        ).exists(),
        reason="basic test repo manifests not available",
    )
    def test_manifest_file_type(self, manifest_file):
        header, _ = parse_file(manifest_file)
        assert header.file_type == FileType.MANIFEST


class TestParseErrors:
    def test_file_too_short(self, tmp_path):
        short_file = tmp_path / "short"
        short_file.write_bytes(b"too short")
        with pytest.raises(ValueError, match="File too short"):
            parse_file(short_file)

    def test_bad_magic(self, tmp_path):
        bad_file = tmp_path / "bad_magic"
        bad_file.write_bytes(b"\x00" * HEADER_SIZE)
        with pytest.raises(ValueError, match="Bad magic"):
            parse_file(bad_file)
