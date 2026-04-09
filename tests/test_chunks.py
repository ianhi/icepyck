"""Tests for read_chunk: inline and native chunk reading."""

from pathlib import Path

import pytest

from icepyck.chunks import read_chunk
from icepyck.manifest import ChunkRefInfo, ChunkType, ManifestReader
from icepyck.repo import RepoInfo
from icepyck.snapshot import SnapshotReader

BASIC_REPO = Path(__file__).parent.parent / "test-repos" / "basic"
NATIVE_REPO = Path(__file__).parent.parent / "test-repos" / "native-chunks"


def _get_chunk_refs(repo_path, array_path):
    """Helper to get all chunk refs for an array node."""
    repo = RepoInfo(repo_path / "repo")
    snapshot_id = repo.get_snapshot_id("main")
    reader = SnapshotReader(repo_path, snapshot_id)
    nodes = {n.path: n for n in reader.list_nodes()}
    node = nodes[array_path]
    all_refs = []
    for mref in node.manifest_refs:
        manifest = ManifestReader(repo_path, mref.manifest_id)
        all_refs.extend(manifest.get_chunk_refs(node.node_id))
    return all_refs


class TestInlineChunks:
    pytestmark = pytest.mark.skipif(
        not (BASIC_REPO / "repo").exists(),
        reason="basic test repo not available",
    )

    @pytest.fixture
    def temperature_chunks(self, basic_repo):
        return _get_chunk_refs(basic_repo, "/group1/temperatures")

    def test_read_inline_chunk_returns_bytes(self, basic_repo, temperature_chunks):
        for cr in temperature_chunks:
            data = read_chunk(basic_repo, cr)
            assert isinstance(data, bytes)

    def test_read_inline_chunk_nonempty(self, basic_repo, temperature_chunks):
        for cr in temperature_chunks:
            data = read_chunk(basic_repo, cr)
            assert len(data) > 0

    def test_read_inline_matches_inline_data(self, basic_repo, temperature_chunks):
        """For inline chunks, read_chunk should return the inline_data directly."""
        for cr in temperature_chunks:
            assert cr.chunk_type == ChunkType.INLINE
            data = read_chunk(basic_repo, cr)
            assert data == cr.inline_data

    def test_timestamps_chunks_readable(self, basic_repo):
        chunks = _get_chunk_refs(basic_repo, "/group1/timestamps")
        for cr in chunks:
            data = read_chunk(basic_repo, cr)
            assert len(data) > 0


class TestNativeChunks:
    pytestmark = pytest.mark.skipif(
        not (NATIVE_REPO / "repo").exists(),
        reason="native-chunks test repo not available",
    )

    def test_read_native_chunks(self, native_chunks_repo):
        repo = RepoInfo(native_chunks_repo / "repo")
        snapshot_id = repo.get_snapshot_id("main")
        reader = SnapshotReader(native_chunks_repo, snapshot_id)

        native_count = 0
        for node in reader.list_nodes():
            if node.node_type != "array":
                continue
            for mref in node.manifest_refs:
                manifest = ManifestReader(native_chunks_repo, mref.manifest_id)
                chunk_refs = manifest.get_chunk_refs(node.node_id)
                for cr in chunk_refs:
                    if cr.chunk_type == ChunkType.NATIVE:
                        data = read_chunk(native_chunks_repo, cr)
                        assert isinstance(data, bytes)
                        assert len(data) > 0
                        native_count += 1

        assert native_count > 0, "Expected at least one native chunk to be read"


class TestVirtualChunks:
    def test_virtual_chunk_file_url(self, tmp_path):
        """Virtual chunk with file:// URL reads from local file."""
        data = b"hello world! extra bytes"
        f = tmp_path / "data.bin"
        f.write_bytes(data)

        ref = ChunkRefInfo(
            index=(0,),
            chunk_type=ChunkType.VIRTUAL,
            location=f"file://{f}",
            offset=6,
            length=5,
        )
        result = read_chunk(None, ref)
        assert result == b"world"

    def test_virtual_chunk_no_location_raises(self):
        ref = ChunkRefInfo(
            index=(0,),
            chunk_type=ChunkType.VIRTUAL,
            location=None,
        )
        with pytest.raises(ValueError, match="no location"):
            read_chunk(None, ref)

    def test_virtual_chunk_unsupported_scheme_raises(self):
        ref = ChunkRefInfo(
            index=(0,),
            chunk_type=ChunkType.VIRTUAL,
            location="ftp://host/path",
        )
        with pytest.raises(ValueError, match="Unsupported"):
            read_chunk(None, ref)


class TestEdgeCases:
    def test_inline_none_returns_empty(self):
        ref = ChunkRefInfo(
            index=(0,),
            chunk_type=ChunkType.INLINE,
            inline_data=None,
        )
        data = read_chunk("/tmp", ref)
        assert data == b""
