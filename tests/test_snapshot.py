"""Tests for SnapshotReader: nodes, types, and manifest refs."""

import pytest
from pathlib import Path

from icepyck.repo import RepoInfo
from icepyck.snapshot import SnapshotReader, NodeInfo

BASIC_REPO = Path(__file__).parent.parent / "test-repos" / "basic"
NESTED_REPO = Path(__file__).parent.parent / "test-repos" / "nested"
SCALAR_REPO = Path(__file__).parent.parent / "test-repos" / "scalar"


def _get_reader(repo_path):
    """Helper to create a SnapshotReader for the main branch."""
    repo = RepoInfo(repo_path / "repo")
    snapshot_id = repo.get_snapshot_id("main")
    return SnapshotReader(repo_path, snapshot_id)


class TestBasicSnapshot:
    pytestmark = pytest.mark.skipif(
        not (BASIC_REPO / "repo").exists(),
        reason="basic test repo not available",
    )

    @pytest.fixture
    def reader(self, basic_repo):
        return _get_reader(basic_repo)

    def test_node_count(self, reader):
        nodes = reader.list_nodes()
        assert len(nodes) == 4

    def test_expected_paths(self, reader):
        paths = {n.path for n in reader.list_nodes()}
        expected = {"/", "/group1", "/group1/temperatures", "/group1/timestamps"}
        assert paths == expected

    def test_root_is_group(self, reader):
        nodes = {n.path: n for n in reader.list_nodes()}
        assert nodes["/"].node_type == "group"

    def test_group1_is_group(self, reader):
        nodes = {n.path: n for n in reader.list_nodes()}
        assert nodes["/group1"].node_type == "group"

    def test_temperatures_is_array(self, reader):
        nodes = {n.path: n for n in reader.list_nodes()}
        assert nodes["/group1/temperatures"].node_type == "array"

    def test_timestamps_is_array(self, reader):
        nodes = {n.path: n for n in reader.list_nodes()}
        assert nodes["/group1/timestamps"].node_type == "array"

    def test_array_nodes_have_manifest_refs(self, reader):
        nodes = {n.path: n for n in reader.list_nodes()}
        assert len(nodes["/group1/temperatures"].manifest_refs) > 0
        assert len(nodes["/group1/timestamps"].manifest_refs) > 0

    def test_group_nodes_have_no_manifest_refs(self, reader):
        nodes = {n.path: n for n in reader.list_nodes()}
        assert len(nodes["/"].manifest_refs) == 0
        assert len(nodes["/group1"].manifest_refs) == 0

    def test_node_ids_are_8_bytes(self, reader):
        for node in reader.list_nodes():
            assert len(node.node_id) == 8

    def test_get_array_manifest_refs(self, reader):
        refs = reader.get_array_manifest_refs("/group1/temperatures")
        assert len(refs) > 0
        assert len(refs[0].manifest_id) == 12

    def test_get_array_manifest_refs_missing_raises(self, reader):
        with pytest.raises(KeyError):
            reader.get_array_manifest_refs("/nonexistent")


class TestNestedSnapshot:
    pytestmark = pytest.mark.skipif(
        not (NESTED_REPO / "repo").exists(),
        reason="nested test repo not available",
    )

    @pytest.fixture
    def reader(self, nested_repo):
        return _get_reader(nested_repo)

    def test_deep_path_exists(self, reader):
        paths = {n.path for n in reader.list_nodes()}
        assert "/a/b/c/data" in paths

    def test_deep_path_is_array(self, reader):
        nodes = {n.path: n for n in reader.list_nodes()}
        assert nodes["/a/b/c/data"].node_type == "array"

    def test_intermediate_groups_exist(self, reader):
        paths = {n.path for n in reader.list_nodes()}
        assert "/" in paths
        assert "/a" in paths
        assert "/a/b" in paths
        assert "/a/b/c" in paths


class TestScalarSnapshot:
    pytestmark = pytest.mark.skipif(
        not (SCALAR_REPO / "repo").exists(),
        reason="scalar test repo not available",
    )

    @pytest.fixture
    def reader(self, scalar_repo):
        return _get_reader(scalar_repo)

    def test_value_node_exists(self, reader):
        paths = {n.path for n in reader.list_nodes()}
        assert "/value" in paths

    def test_value_is_array(self, reader):
        nodes = {n.path: n for n in reader.list_nodes()}
        assert nodes["/value"].node_type == "array"

    def test_value_has_manifest_refs(self, reader):
        nodes = {n.path: n for n in reader.list_nodes()}
        assert len(nodes["/value"].manifest_refs) > 0
