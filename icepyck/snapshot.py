"""Snapshot reader for Icechunk repositories.

Parses a snapshot file and extracts all nodes (groups and arrays)
with their paths, types, zarr metadata, and manifest references.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from icepyck.crockford import encode as crockford_encode
from icepyck.header import FileType, parse_bytes, parse_file

if TYPE_CHECKING:
    from pathlib import Path

    from icepyck.storage import Storage


@dataclass(frozen=True)
class ManifestRefInfo:
    """Reference to a manifest file and the chunk index ranges it covers."""

    manifest_id: bytes  # 12-byte ObjectId12
    extents: list[tuple[int, int]]  # list of (from, to) chunk index ranges


@dataclass(frozen=True)
class NodeInfo:
    """Information about a single node in the snapshot."""

    path: str
    node_type: str  # "array" or "group"
    node_id: bytes  # 8-byte ObjectId8
    user_data: bytes | None  # raw zarr.json bytes, if present
    manifest_refs: list[ManifestRefInfo] = field(default_factory=list)
    dimension_names: list[str] = field(default_factory=list)


class SnapshotReader:
    """Read and interpret an Icechunk snapshot file."""

    def __init__(
        self,
        root_path: str | Path | None = None,
        snapshot_id: bytes = b"",
        *,
        storage: Storage | None = None,
    ) -> None:
        self._snapshot_id = snapshot_id
        snapshot_name = crockford_encode(snapshot_id)

        if storage is not None:
            raw = storage.read(f"snapshots/{snapshot_name}")
            header, payload = parse_bytes(raw)
        elif root_path is not None:
            from pathlib import Path

            header, payload = parse_file(Path(root_path) / "snapshots" / snapshot_name)
        else:
            raise TypeError("Either root_path or storage must be provided")
        self._init_from_payload(snapshot_id, header, payload)

    @classmethod
    async def afrom_storage(
        cls,
        storage: Storage,
        snapshot_id: bytes,
    ) -> SnapshotReader:
        """Async constructor: fetch snapshot bytes in a thread, then parse.

        Uses ``asyncio.to_thread`` so the event loop stays free during the
        blocking S3 read, and the task remains cancellable at the await point.
        """
        import asyncio

        snapshot_name = crockford_encode(snapshot_id)
        path = f"snapshots/{snapshot_name}"
        raw: bytes = await asyncio.to_thread(storage.read, path)
        header, payload = parse_bytes(raw)
        instance = cls.__new__(cls)
        instance._snapshot_id = snapshot_id
        instance._init_from_payload(snapshot_id, header, payload)
        return instance

    def _init_from_payload(
        self,
        snapshot_id: bytes,
        header: object,
        payload: bytes,
    ) -> None:
        from icepyck.generated.ArrayNodeData import ArrayNodeData
        from icepyck.generated.NodeData import NodeData
        from icepyck.generated.Snapshot import Snapshot

        if header.file_type != FileType.SNAPSHOT:
            raise ValueError(
                f"Expected SNAPSHOT file type ({FileType.SNAPSHOT}), "
                f"got {header.file_type}"
            )

        buf = bytearray(payload)
        snapshot = Snapshot.GetRootAs(buf, 0)

        # Parse all nodes upfront
        self._nodes: list[NodeInfo] = []
        self._nodes_by_path: dict[str, NodeInfo] = {}
        self._array_nodes: dict[str, NodeInfo] = {}  # path -> NodeInfo

        for i in range(snapshot.NodesLength()):
            node = snapshot.Nodes(i)

            # Path
            raw_path = node.Path()
            if isinstance(raw_path, bytes):
                raw_path = raw_path.decode("utf-8")
            node_path = raw_path or "/"

            # Node ID (8 bytes)
            id_obj = node.Id()
            node_id = bytes(id_obj.Bytes()) if id_obj is not None else b""

            # Node type
            node_data_type = node.NodeDataType()
            if node_data_type == NodeData.Array:
                node_type = "array"
            elif node_data_type == NodeData.Group:
                node_type = "group"
            else:
                node_type = f"unknown({node_data_type})"

            # User data (zarr.json)
            user_data_len = node.UserDataLength()
            if user_data_len > 0:
                user_data = bytes(node.UserData(j) for j in range(user_data_len))
            else:
                user_data = None

            # Array-specific data (manifest refs, dimension names)
            manifest_refs: list[ManifestRefInfo] = []
            dimension_names: list[str] = []
            if node_data_type == NodeData.Array:
                table = node.NodeData()
                if table is not None:
                    array_data = ArrayNodeData()
                    array_data.Init(table.Bytes, table.Pos)

                    for j in range(array_data.ManifestsLength()):
                        mref = array_data.Manifests(j)
                        obj_id = mref.ObjectId()
                        manifest_id = bytes(obj_id.Bytes()) if obj_id else b""
                        extents = []
                        for k in range(mref.ExtentsLength()):
                            ext = mref.Extents(k)
                            extents.append((ext.From(), ext.To()))
                        manifest_refs.append(
                            ManifestRefInfo(
                                manifest_id=manifest_id,
                                extents=extents,
                            )
                        )

                    for j in range(array_data.DimensionNamesLength()):
                        dn = array_data.DimensionNames(j)
                        if dn is not None:
                            name = dn.Name()
                            if isinstance(name, bytes):
                                name = name.decode("utf-8")
                            if name:
                                dimension_names.append(name)

            info = NodeInfo(
                path=node_path,
                node_type=node_type,
                node_id=node_id,
                user_data=user_data,
                manifest_refs=manifest_refs,
                dimension_names=dimension_names,
            )
            self._nodes.append(info)
            self._nodes_by_path[node_path] = info
            if node_type == "array":
                self._array_nodes[node_path] = info

    def list_nodes(self) -> list[NodeInfo]:
        """List all nodes with path, type, and metadata."""
        return self._nodes

    def get_node(self, path: str) -> NodeInfo:
        """Get a node by path. O(1) dict lookup."""
        try:
            return self._nodes_by_path[path]
        except KeyError:
            raise KeyError(f"Node not found: {path!r}") from None

    def get_array_node(self, path: str) -> NodeInfo:
        """Get an array node by path. O(1) dict lookup."""
        try:
            return self._array_nodes[path]
        except KeyError:
            raise KeyError(f"Array node not found: {path!r}") from None

    def get_array_manifest_refs(self, path: str) -> list[ManifestRefInfo]:
        """Get manifest refs for an array node by its path."""
        return self.get_array_node(path).manifest_refs
