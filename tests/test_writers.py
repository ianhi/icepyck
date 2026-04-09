"""Tests for Phase 2: FlatBuffer serialization round-trips.

Each builder is tested by building a file and reading it back with
the existing reader classes.
"""

from __future__ import annotations

from pathlib import Path

from icepyck.header import Compression, FileType, parse_bytes
from icepyck.ids import generate_id8, generate_id12
from icepyck.manifest import ManifestReader
from icepyck.repo import RepoInfo
from icepyck.snapshot import SnapshotReader
from icepyck.storage import LocalStorage
from icepyck.writers import (
    ArrayManifestData,
    ArrayUpdatedChunksData,
    ChunkRefData,
    ManifestFileData,
    ManifestRefData,
    NodeWriteData,
    SnapshotInfoData,
    build_manifest,
    build_repo,
    build_snapshot,
    build_transaction_log,
)


class TestManifestRoundTrip:
    def test_inline_chunk(self, tmp_path: Path) -> None:
        manifest_id = generate_id12()
        node_id = generate_id8()
        data = build_manifest(
            manifest_id,
            [
                ArrayManifestData(
                    node_id=node_id,
                    refs=[
                        ChunkRefData(index=(0,), inline_data=b"hello"),
                        ChunkRefData(index=(1,), inline_data=b"world"),
                    ],
                )
            ],
        )
        # Write and read back
        storage = LocalStorage(tmp_path)
        from icepyck.crockford import encode

        path = f"manifests/{encode(manifest_id)}"
        storage.write(path, data)
        reader = ManifestReader(None, manifest_id, storage=storage)

        refs = reader.get_chunk_refs(node_id)
        assert len(refs) == 2
        assert refs[0].index == (0,)
        assert refs[0].inline_data == b"hello"
        assert refs[1].index == (1,)
        assert refs[1].inline_data == b"world"

    def test_native_chunk(self, tmp_path: Path) -> None:
        manifest_id = generate_id12()
        node_id = generate_id8()
        chunk_id = generate_id12()
        data = build_manifest(
            manifest_id,
            [
                ArrayManifestData(
                    node_id=node_id,
                    refs=[
                        ChunkRefData(
                            index=(0, 0),
                            chunk_id=chunk_id,
                            offset=0,
                            length=100,
                        ),
                    ],
                )
            ],
        )
        storage = LocalStorage(tmp_path)
        from icepyck.crockford import encode

        storage.write(f"manifests/{encode(manifest_id)}", data)
        reader = ManifestReader(None, manifest_id, storage=storage)

        refs = reader.get_chunk_refs(node_id)
        assert len(refs) == 1
        assert refs[0].index == (0, 0)
        assert refs[0].chunk_id == chunk_id
        assert refs[0].length == 100

    def test_multiple_arrays(self, tmp_path: Path) -> None:
        manifest_id = generate_id12()
        node_a = generate_id8()
        node_b = generate_id8()
        data = build_manifest(
            manifest_id,
            [
                ArrayManifestData(
                    node_id=node_a,
                    refs=[ChunkRefData(index=(0,), inline_data=b"a")],
                ),
                ArrayManifestData(
                    node_id=node_b,
                    refs=[ChunkRefData(index=(0,), inline_data=b"b")],
                ),
            ],
        )
        storage = LocalStorage(tmp_path)
        from icepyck.crockford import encode

        storage.write(f"manifests/{encode(manifest_id)}", data)
        reader = ManifestReader(None, manifest_id, storage=storage)

        assert len(reader.get_chunk_refs(node_a)) == 1
        assert len(reader.get_chunk_refs(node_b)) == 1
        assert reader.get_chunk_refs(node_a)[0].inline_data == b"a"
        assert reader.get_chunk_refs(node_b)[0].inline_data == b"b"

    def test_header_is_manifest_type(self) -> None:
        data = build_manifest(generate_id12(), [])
        header, _ = parse_bytes(data)
        assert header.file_type == FileType.MANIFEST
        assert header.compression == Compression.ZSTD


class TestSnapshotRoundTrip:
    def test_group_node(self, tmp_path: Path) -> None:
        snapshot_id = generate_id12()
        node_id = generate_id8()
        zarr_json = b'{"zarr_format":3,"node_type":"group"}'
        data = build_snapshot(
            snapshot_id,
            nodes=[
                NodeWriteData(
                    node_id=node_id,
                    path="/",
                    user_data=zarr_json,
                    node_type="group",
                )
            ],
            message="test commit",
            flushed_at=1000000,
        )
        storage = LocalStorage(tmp_path)
        from icepyck.crockford import encode

        storage.write(f"snapshots/{encode(snapshot_id)}", data)
        reader = SnapshotReader(None, snapshot_id, storage=storage)

        nodes = reader.list_nodes()
        assert len(nodes) == 1
        assert nodes[0].path == "/"
        assert nodes[0].node_type == "group"
        assert nodes[0].user_data == zarr_json

    def test_array_node_with_manifest_refs(self, tmp_path: Path) -> None:
        snapshot_id = generate_id12()
        node_id = generate_id8()
        manifest_id = generate_id12()
        zarr_json = b'{"zarr_format":3,"node_type":"array","shape":[10],"chunk_grid":{"name":"regular","configuration":{"chunk_shape":[5]}},"data_type":"float64","codecs":[{"name":"bytes","configuration":{"endian":"little"}}]}'

        data = build_snapshot(
            snapshot_id,
            nodes=[
                NodeWriteData(
                    node_id=node_id,
                    path="/data",
                    user_data=zarr_json,
                    node_type="array",
                    manifests=[
                        ManifestRefData(
                            manifest_id=manifest_id,
                            extents=[(0, 2)],
                        )
                    ],
                )
            ],
        )
        storage = LocalStorage(tmp_path)
        from icepyck.crockford import encode

        storage.write(f"snapshots/{encode(snapshot_id)}", data)
        reader = SnapshotReader(None, snapshot_id, storage=storage)

        nodes = reader.list_nodes()
        assert len(nodes) == 1
        assert nodes[0].node_type == "array"
        assert len(nodes[0].manifest_refs) == 1
        assert nodes[0].manifest_refs[0].manifest_id == manifest_id
        assert nodes[0].manifest_refs[0].extents == [(0, 2)]

    def test_no_parent_id_in_v2(self) -> None:
        """V2 snapshots do NOT write parent_id (tracked via repo parent_offset)."""
        snapshot_id = generate_id12()
        data = build_snapshot(
            snapshot_id,
            nodes=[],
        )
        from icepyck.generated.Snapshot import Snapshot
        from icepyck.header import parse_bytes as hparse

        _, payload = hparse(data)
        snap = Snapshot.GetRootAs(payload)
        assert snap.ParentId() is None

    def test_nodes_sorted_by_path(self, tmp_path: Path) -> None:
        snapshot_id = generate_id12()
        data = build_snapshot(
            snapshot_id,
            nodes=[
                NodeWriteData(
                    node_id=generate_id8(),
                    path="/z",
                    user_data=b'{"zarr_format":3,"node_type":"group"}',
                    node_type="group",
                ),
                NodeWriteData(
                    node_id=generate_id8(),
                    path="/a",
                    user_data=b'{"zarr_format":3,"node_type":"group"}',
                    node_type="group",
                ),
            ],
        )
        storage = LocalStorage(tmp_path)
        from icepyck.crockford import encode

        storage.write(f"snapshots/{encode(snapshot_id)}", data)
        reader = SnapshotReader(None, snapshot_id, storage=storage)
        paths = [n.path for n in reader.list_nodes()]
        assert paths == ["/a", "/z"]

    def test_manifest_files_v2(self, tmp_path: Path) -> None:
        snapshot_id = generate_id12()
        manifest_id = generate_id12()
        data = build_snapshot(
            snapshot_id,
            nodes=[],
            manifest_files=[
                ManifestFileData(
                    manifest_id=manifest_id,
                    size_bytes=512,
                    num_chunk_refs=10,
                )
            ],
        )
        # Just verify it builds without error and parses
        header, _ = parse_bytes(data)
        assert header.file_type == FileType.SNAPSHOT


class TestTransactionLogRoundTrip:
    def test_empty_txn_log(self) -> None:
        txn_id = generate_id12()
        data = build_transaction_log(txn_id)
        header, _ = parse_bytes(data)
        assert header.file_type == FileType.TRANSACTION_LOG

    def test_new_groups_and_arrays(self) -> None:
        txn_id = generate_id12()
        g1 = generate_id8()
        a1 = generate_id8()
        data = build_transaction_log(
            txn_id,
            new_groups=[g1],
            new_arrays=[a1],
        )
        header, payload = parse_bytes(data)
        assert header.file_type == FileType.TRANSACTION_LOG
        # Verify we can parse it with the generated reader
        from icepyck.generated.TransactionLog import TransactionLog

        txn = TransactionLog.GetRootAs(payload)
        assert txn.NewGroupsLength() == 1
        assert txn.NewArraysLength() == 1
        assert bytes(txn.NewGroups(0).Bytes()) == g1
        assert bytes(txn.NewArrays(0).Bytes()) == a1

    def test_updated_chunks(self) -> None:
        txn_id = generate_id12()
        node_id = generate_id8()
        data = build_transaction_log(
            txn_id,
            updated_chunks=[
                ArrayUpdatedChunksData(
                    node_id=node_id,
                    chunk_indices=[(0, 0), (0, 1)],
                )
            ],
        )
        header, payload = parse_bytes(data)
        from icepyck.generated.TransactionLog import TransactionLog

        txn = TransactionLog.GetRootAs(payload)
        assert txn.UpdatedChunksLength() == 1
        uc = txn.UpdatedChunks(0)
        assert bytes(uc.NodeId().Bytes()) == node_id
        assert uc.ChunksLength() == 2


class TestRepoRoundTrip:
    def test_basic_repo(self, tmp_path: Path) -> None:
        snapshot_id = generate_id12()
        data = build_repo(
            spec_version=2,
            branches={"main": 0},
            tags={},
            snapshots=[
                SnapshotInfoData(
                    snapshot_id=snapshot_id,
                    parent_offset=-1,
                    flushed_at=1000000,
                    message="init",
                )
            ],
        )
        storage = LocalStorage(tmp_path)
        storage.write("repo", data)
        repo = RepoInfo(storage=storage)

        assert repo.list_branches() == ["main"]
        assert repo.list_tags() == []
        got_id = repo.get_snapshot_id("main")
        assert got_id == snapshot_id

    def test_multiple_branches_and_tags(self, tmp_path: Path) -> None:
        s1 = generate_id12()
        s2 = generate_id12()
        data = build_repo(
            spec_version=2,
            branches={"main": 0, "dev": 1},
            tags={"v1": 0},
            snapshots=[
                SnapshotInfoData(snapshot_id=s1, parent_offset=-1),
                SnapshotInfoData(snapshot_id=s2, parent_offset=0),
            ],
        )
        storage = LocalStorage(tmp_path)
        storage.write("repo", data)
        repo = RepoInfo(storage=storage)

        assert sorted(repo.list_branches()) == ["dev", "main"]
        assert repo.list_tags() == ["v1"]
        assert repo.get_snapshot_id("main") == s1
        assert repo.get_snapshot_id("dev") == s2
        assert repo.get_tag_snapshot_id("v1") == s1

    def test_header_is_repo_type(self) -> None:
        data = build_repo(
            2, {"main": 0}, {}, [SnapshotInfoData(snapshot_id=generate_id12())]
        )
        header, _ = parse_bytes(data)
        assert header.file_type == FileType.REPO_INFO
