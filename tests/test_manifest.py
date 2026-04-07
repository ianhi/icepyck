"""Tests for ManifestReader: chunk refs, inline vs native."""

import pytest
from pathlib import Path

from icepyck.manifest import ManifestReader, ChunkType
from icepyck.repo import RepoInfo
from icepyck.snapshot import SnapshotReader

BASIC_REPO = Path(__file__).parent.parent / "test-repos" / "basic"
NATIVE_REPO = Path(__file__).parent.parent / "test-repos" / "native-chunks"


def _get_array_node(repo_path, array_path):
    """Helper to get an array node and its manifest refs."""
    repo = RepoInfo(repo_path / "repo")
    snapshot_id = repo.get_snapshot_id("main")
    reader = SnapshotReader(repo_path, snapshot_id)
    nodes = {n.path: n for n in reader.list_nodes()}
    return nodes[array_path]


class TestBasicManifest:
    pytestmark = pytest.mark.skipif(
        not (BASIC_REPO / "repo").exists(),
        reason="basic test repo not available",
    )

    @pytest.fixture
    def temperatures_node(self, basic_repo):
        return _get_array_node(basic_repo, "/group1/temperatures")

    @pytest.fixture
    def manifest(self, basic_repo, temperatures_node):
        mref = temperatures_node.manifest_refs[0]
        return ManifestReader(basic_repo, mref.manifest_id)

    def test_manifest_has_chunk_refs(self, manifest, temperatures_node):
        chunk_refs = manifest.get_chunk_refs(temperatures_node.node_id)
        assert len(chunk_refs) > 0

    def test_basic_chunks_are_inline(self, manifest, temperatures_node):
        chunk_refs = manifest.get_chunk_refs(temperatures_node.node_id)
        for cr in chunk_refs:
            assert cr.chunk_type == ChunkType.INLINE

    def test_inline_chunks_have_data(self, manifest, temperatures_node):
        chunk_refs = manifest.get_chunk_refs(temperatures_node.node_id)
        for cr in chunk_refs:
            assert cr.inline_data is not None
            assert len(cr.inline_data) > 0

    def test_chunk_index_is_tuple(self, manifest, temperatures_node):
        chunk_refs = manifest.get_chunk_refs(temperatures_node.node_id)
        for cr in chunk_refs:
            assert isinstance(cr.index, tuple)

    def test_list_node_ids(self, manifest):
        node_ids = manifest.list_node_ids()
        assert len(node_ids) > 0
        for nid in node_ids:
            assert isinstance(nid, bytes)
            assert len(nid) == 8

    def test_missing_node_id_raises(self, manifest):
        with pytest.raises(KeyError):
            manifest.get_chunk_refs(b"\x00" * 8)


class TestNativeManifest:
    pytestmark = pytest.mark.skipif(
        not (NATIVE_REPO / "repo").exists(),
        reason="native-chunks test repo not available",
    )

    def test_has_native_chunks(self, native_chunks_repo):
        repo = RepoInfo(native_chunks_repo / "repo")
        snapshot_id = repo.get_snapshot_id("main")
        reader = SnapshotReader(native_chunks_repo, snapshot_id)

        found_native = False
        for node in reader.list_nodes():
            if node.node_type != "array":
                continue
            for mref in node.manifest_refs:
                manifest = ManifestReader(native_chunks_repo, mref.manifest_id)
                chunk_refs = manifest.get_chunk_refs(node.node_id)
                for cr in chunk_refs:
                    if cr.chunk_type == ChunkType.NATIVE:
                        found_native = True
                        assert cr.chunk_id is not None
                        assert len(cr.chunk_id) == 12

        assert found_native, "Expected at least one NATIVE chunk in native-chunks repo"
