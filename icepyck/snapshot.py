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


class SnapshotReader:
    """Read and interpret an Icechunk snapshot file.

    Parameters
    ----------
    root_path : str or Path
        Root path of the Icechunk repository (the directory containing
        ``snapshots/``, ``manifests/``, etc.).
    snapshot_id : bytes
        The 12-byte ObjectId12 identifying the snapshot.
    """

    def __init__(
        self,
        root_path: str | Path | None = None,
        snapshot_id: bytes = b"",
        *,
        storage: Storage | None = None,
    ) -> None:
        from icepyck.generated.ArrayNodeData import ArrayNodeData
        from icepyck.generated.NodeData import NodeData
        from icepyck.generated.Snapshot import Snapshot

        self._snapshot_id = snapshot_id
        snapshot_name = crockford_encode(snapshot_id)

        if storage is not None:
            raw = storage.read(f"snapshots/{snapshot_name}")
            header, payload = parse_bytes(raw)
        elif root_path is not None:
            from pathlib import Path as _Path

            self._root_path: Path | None = _Path(root_path)
            snapshot_path = self._root_path / "snapshots" / snapshot_name
            header, payload = parse_file(snapshot_path)
        else:
            raise TypeError("Either root_path or storage must be provided")
        if header.file_type != FileType.SNAPSHOT:
            raise ValueError(
                f"Expected SNAPSHOT file type ({FileType.SNAPSHOT}), "
                f"got {header.file_type}"
            )

        buf = bytearray(payload)
        self._snapshot = Snapshot.GetRootAs(buf, 0)

        # Parse all nodes upfront
        self._nodes: list[NodeInfo] = []
        self._array_nodes: dict[str, NodeInfo] = {}  # path -> NodeInfo

        for i in range(self._snapshot.NodesLength()):
            node = self._snapshot.Nodes(i)

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

            # Manifest refs (only for array nodes)
            manifest_refs: list[ManifestRefInfo] = []
            if node_data_type == NodeData.Array:
                table = node.NodeData()
                if table is not None:
                    array_data = ArrayNodeData()
                    array_data.Init(table.Bytes, table.Pos)

                    for j in range(array_data.ManifestsLength()):
                        mref = array_data.Manifests(j)
                        # Manifest ObjectId12
                        obj_id = mref.ObjectId()
                        manifest_id = bytes(obj_id.Bytes()) if obj_id else b""
                        # Extents
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

            info = NodeInfo(
                path=node_path,
                node_type=node_type,
                node_id=node_id,
                user_data=user_data,
                manifest_refs=manifest_refs,
            )
            self._nodes.append(info)
            if node_type == "array":
                self._array_nodes[node_path] = info

    def list_nodes(self) -> list[NodeInfo]:
        """List all nodes with path, type, and metadata."""
        return list(self._nodes)

    def get_array_manifest_refs(self, path: str) -> list[ManifestRefInfo]:
        """Get manifest refs for an array node by its path.

        Parameters
        ----------
        path : str
            The node path (e.g. ``"/group1/temperatures"``).

        Returns
        -------
        list[ManifestRefInfo]
            The manifest references for this array.

        Raises
        ------
        KeyError
            If the path is not found or is not an array node.
        """
        if path not in self._array_nodes:
            raise KeyError(f"Array node not found: {path!r}")
        return self._array_nodes[path].manifest_refs
