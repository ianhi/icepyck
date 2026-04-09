"""FlatBuffer serialization builders for Icechunk files.

Builds complete file bytes (header + flatbuffer payload) for manifests,
snapshots, transaction logs, and repo files.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass, field

import flatbuffers

from icepyck.generated.ArrayManifest import (
    ArrayManifestAddNodeId,
    ArrayManifestAddRefs,
    ArrayManifestEnd,
    ArrayManifestStart,
    ArrayManifestStartRefsVector,
)
from icepyck.generated.ArrayNodeData import (
    ArrayNodeDataAddDimensionNames,
    ArrayNodeDataAddManifests,
    ArrayNodeDataAddShape,
    ArrayNodeDataAddShapeV2,
    ArrayNodeDataEnd,
    ArrayNodeDataStart,
    ArrayNodeDataStartDimensionNamesVector,
    ArrayNodeDataStartManifestsVector,
    ArrayNodeDataStartShapeV2Vector,
    ArrayNodeDataStartShapeVector,
)
from icepyck.generated.ArrayUpdatedChunks import (
    ArrayUpdatedChunksAddChunks,
    ArrayUpdatedChunksAddNodeId,
    ArrayUpdatedChunksEnd,
    ArrayUpdatedChunksStart,
    ArrayUpdatedChunksStartChunksVector,
)
from icepyck.generated.BranchCreatedUpdate import (
    BranchCreatedUpdateAddName,
    BranchCreatedUpdateEnd,
    BranchCreatedUpdateStart,
)
from icepyck.generated.BranchDeletedUpdate import (
    BranchDeletedUpdateAddName,
    BranchDeletedUpdateAddPreviousSnapId,
    BranchDeletedUpdateEnd,
    BranchDeletedUpdateStart,
)
from icepyck.generated.ChunkIndexRange import CreateChunkIndexRange
from icepyck.generated.ChunkIndices import (
    ChunkIndicesAddCoords,
    ChunkIndicesEnd,
    ChunkIndicesStart,
    ChunkIndicesStartCoordsVector,
)
from icepyck.generated.ChunkRef import (
    ChunkRefAddChunkId,
    ChunkRefAddIndex,
    ChunkRefAddInline,
    ChunkRefAddLength,
    ChunkRefAddOffset,
    ChunkRefEnd,
    ChunkRefStart,
    ChunkRefStartIndexVector,
)
from icepyck.generated.DimensionName import (
    DimensionNameAddName,
    DimensionNameEnd,
    DimensionNameStart,
)
from icepyck.generated.DimensionShapeV2 import (
    DimensionShapeV2AddArrayLength,
    DimensionShapeV2AddNumChunks,
    DimensionShapeV2End,
    DimensionShapeV2Start,
)
from icepyck.generated.GroupNodeData import (
    GroupNodeDataEnd,
    GroupNodeDataStart,
)
from icepyck.generated.Manifest import (
    ManifestAddArrays,
    ManifestAddId,
    ManifestEnd,
    ManifestStart,
    ManifestStartArraysVector,
)
from icepyck.generated.ManifestFileInfoV2 import (
    ManifestFileInfoV2AddId,
    ManifestFileInfoV2AddNumChunkRefs,
    ManifestFileInfoV2AddSizeBytes,
    ManifestFileInfoV2End,
    ManifestFileInfoV2Start,
)
from icepyck.generated.ManifestRef import (
    ManifestRefAddExtents,
    ManifestRefAddObjectId,
    ManifestRefEnd,
    ManifestRefStart,
    ManifestRefStartExtentsVector,
)
from icepyck.generated.NewCommitUpdate import (
    NewCommitUpdateAddBranch,
    NewCommitUpdateAddNewSnapId,
    NewCommitUpdateEnd,
    NewCommitUpdateStart,
)
from icepyck.generated.NodeData import NodeData
from icepyck.generated.NodeSnapshot import (
    NodeSnapshotAddId,
    NodeSnapshotAddNodeData,
    NodeSnapshotAddNodeDataType,
    NodeSnapshotAddPath,
    NodeSnapshotAddUserData,
    NodeSnapshotEnd,
    NodeSnapshotStart,
)
from icepyck.generated.ObjectId8 import CreateObjectId8
from icepyck.generated.ObjectId12 import CreateObjectId12
from icepyck.generated.Ref import (
    RefAddName,
    RefAddSnapshotIndex,
    RefEnd,
    RefStart,
)
from icepyck.generated.Repo import (
    RepoAddBranches,
    RepoAddDeletedTags,
    RepoAddLatestUpdates,
    RepoAddSnapshots,
    RepoAddSpecVersion,
    RepoAddStatus,
    RepoAddTags,
    RepoEnd,
    RepoStart,
    RepoStartBranchesVector,
    RepoStartDeletedTagsVector,
    RepoStartLatestUpdatesVector,
    RepoStartSnapshotsVector,
    RepoStartTagsVector,
)
from icepyck.generated.RepoAvailability import RepoAvailability
from icepyck.generated.RepoInitializedUpdate import (
    RepoInitializedUpdateEnd,
    RepoInitializedUpdateStart,
)
from icepyck.generated.RepoStatus import (
    RepoStatusAddAvailability,
    RepoStatusAddSetAt,
    RepoStatusEnd,
    RepoStatusStart,
)
from icepyck.generated.Snapshot import (
    SnapshotAddFlushedAt,
    SnapshotAddId,
    SnapshotAddManifestFiles,
    SnapshotAddManifestFilesV2,
    SnapshotAddMessage,
    SnapshotAddMetadata,
    SnapshotAddNodes,
    SnapshotEnd,
    SnapshotStart,
    SnapshotStartManifestFilesV2Vector,
    SnapshotStartManifestFilesVector,
    SnapshotStartMetadataVector,
    SnapshotStartNodesVector,
)
from icepyck.generated.SnapshotInfo import (
    SnapshotInfoAddFlushedAt,
    SnapshotInfoAddId,
    SnapshotInfoAddMessage,
    SnapshotInfoAddParentOffset,
    SnapshotInfoEnd,
    SnapshotInfoStart,
)
from icepyck.generated.TagCreatedUpdate import (
    TagCreatedUpdateAddName,
    TagCreatedUpdateEnd,
    TagCreatedUpdateStart,
)
from icepyck.generated.TagDeletedUpdate import (
    TagDeletedUpdateAddName,
    TagDeletedUpdateAddPreviousSnapId,
    TagDeletedUpdateEnd,
    TagDeletedUpdateStart,
)
from icepyck.generated.TransactionLog import (
    TransactionLogAddDeletedArrays,
    TransactionLogAddDeletedGroups,
    TransactionLogAddId,
    TransactionLogAddNewArrays,
    TransactionLogAddNewGroups,
    TransactionLogAddUpdatedArrays,
    TransactionLogAddUpdatedChunks,
    TransactionLogAddUpdatedGroups,
    TransactionLogEnd,
    TransactionLogStart,
    TransactionLogStartUpdatedChunksVector,
)
from icepyck.generated.Update import (
    UpdateAddUpdatedAt,
    UpdateAddUpdateType,
    UpdateAddUpdateTypeType,
    UpdateEnd,
    UpdateStart,
)
from icepyck.generated.UpdateType import UpdateType
from icepyck.header import FileType, build_bytes

# ---------------------------------------------------------------------------
# Data classes for builder inputs
# ---------------------------------------------------------------------------


@dataclass
class ChunkRefData:
    """Input data for a single chunk reference."""

    index: tuple[int, ...]
    inline_data: bytes | None = None
    chunk_id: bytes | None = None  # ObjectId12 (12 bytes)
    offset: int = 0
    length: int = 0


@dataclass
class ArrayManifestData:
    """Input data for one array's chunk refs inside a manifest."""

    node_id: bytes  # ObjectId8 (8 bytes)
    refs: list[ChunkRefData] = field(default_factory=list)


@dataclass
class ManifestRefData:
    """Pointer from an array node to a manifest file."""

    manifest_id: bytes  # ObjectId12 (12 bytes)
    extents: list[tuple[int, int]] = field(default_factory=list)  # (from, to)


@dataclass
class NodeWriteData:
    """Input data for a node in a snapshot."""

    node_id: bytes  # ObjectId8 (8 bytes)
    path: str
    user_data: bytes | None  # zarr.json as UTF-8 bytes
    node_type: str  # "array" or "group"
    manifests: list[ManifestRefData] = field(default_factory=list)
    dimension_names: list[str] = field(default_factory=list)


@dataclass
class ManifestFileData:
    """Info about a manifest file for the snapshot's manifest_files_v2."""

    manifest_id: bytes  # ObjectId12 (12 bytes)
    size_bytes: int = 0
    num_chunk_refs: int = 0


@dataclass
class SnapshotInfoData:
    """Info about a snapshot for the repo file."""

    snapshot_id: bytes  # ObjectId12 (12 bytes)
    parent_offset: int = -1  # -1 means no parent
    flushed_at: int = 0  # microseconds since epoch
    message: str = ""


@dataclass
class ArrayUpdatedChunksData:
    """Records which chunks were updated for an array in a transaction."""

    node_id: bytes  # ObjectId8 (8 bytes)
    chunk_indices: list[tuple[int, ...]] = field(default_factory=list)


@dataclass
class UpdateData:
    """Input data for a repo update entry.

    The ``kind`` field selects which union variant to build:
    - ``"repo_initialized"``
    - ``"branch_created"`` (requires ``name``)
    - ``"branch_deleted"`` (requires ``name``, optional ``previous_snap_id``)
    - ``"tag_created"`` (requires ``name``)
    - ``"tag_deleted"`` (requires ``name``, optional ``previous_snap_id``)
    - ``"new_commit"`` (requires ``branch``, ``snapshot_id``)
    """

    kind: str
    updated_at: int = 0  # microseconds since epoch; 0 = auto-fill
    name: str = ""  # branch or tag name
    branch: str = ""  # for new_commit
    snapshot_id: bytes | None = None  # ObjectId12 for new_commit
    previous_snap_id: bytes | None = None  # for delete operations


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_id8_vector(builder: flatbuffers.Builder, ids: list[bytes]) -> int:
    """Build a FlatBuffer vector of ObjectId8 structs."""
    # Structs are written inline — start vector, then create each struct
    # in reverse order.
    builder.StartVector(8, len(ids), 1)
    for id_bytes in reversed(ids):
        CreateObjectId8(builder, list(id_bytes))
    return builder.EndVector()


# ---------------------------------------------------------------------------
# Manifest builder
# ---------------------------------------------------------------------------


def _build_chunk_ref(builder: flatbuffers.Builder, cref: ChunkRefData) -> int:
    """Build a single ChunkRef table. Returns the offset."""
    # Vectors/strings must be created before starting the table.
    # Index vector
    ChunkRefStartIndexVector(builder, len(cref.index))
    for val in reversed(cref.index):
        builder.PrependUint32(val)
    index_vec = builder.EndVector()

    # Inline data vector (if present)
    inline_vec = None
    if cref.inline_data is not None:
        inline_vec = builder.CreateByteVector(cref.inline_data)

    ChunkRefStart(builder)
    ChunkRefAddIndex(builder, index_vec)
    if inline_vec is not None:
        ChunkRefAddInline(builder, inline_vec)
    if cref.chunk_id is not None:
        ChunkRefAddChunkId(builder, CreateObjectId12(builder, list(cref.chunk_id)))
    if cref.offset:
        ChunkRefAddOffset(builder, cref.offset)
    if cref.length:
        ChunkRefAddLength(builder, cref.length)
    return ChunkRefEnd(builder)


def _build_array_manifest(builder: flatbuffers.Builder, am: ArrayManifestData) -> int:
    """Build a single ArrayManifest table. Returns the offset."""
    # Build all ChunkRef tables first
    ref_offsets = [_build_chunk_ref(builder, r) for r in am.refs]

    # Refs vector (vector of offsets to tables)
    ArrayManifestStartRefsVector(builder, len(ref_offsets))
    for off in reversed(ref_offsets):
        builder.PrependUOffsetTRelative(off)
    refs_vec = builder.EndVector()

    ArrayManifestStart(builder)
    ArrayManifestAddNodeId(builder, CreateObjectId8(builder, list(am.node_id)))
    ArrayManifestAddRefs(builder, refs_vec)
    return ArrayManifestEnd(builder)


def build_manifest_payload(
    manifest_id: bytes,
    arrays: list[ArrayManifestData],
) -> bytes:
    """Build a Manifest FlatBuffer payload (without header).

    Returns raw flatbuffer bytes.
    """
    builder = flatbuffers.Builder(1024)

    # Build all ArrayManifest tables (sorted by node_id per spec)
    sorted_arrays = sorted(arrays, key=lambda a: a.node_id)
    array_offsets = [_build_array_manifest(builder, a) for a in sorted_arrays]

    # Arrays vector
    ManifestStartArraysVector(builder, len(array_offsets))
    for off in reversed(array_offsets):
        builder.PrependUOffsetTRelative(off)
    arrays_vec = builder.EndVector()

    ManifestStart(builder)
    ManifestAddId(builder, CreateObjectId12(builder, list(manifest_id)))
    ManifestAddArrays(builder, arrays_vec)
    manifest_off = ManifestEnd(builder)

    builder.Finish(manifest_off)
    return bytes(builder.Output())


def build_manifest(
    manifest_id: bytes,
    arrays: list[ArrayManifestData],
) -> bytes:
    """Build a complete manifest file (header + FlatBuffer payload)."""
    payload = build_manifest_payload(manifest_id, arrays)
    return build_bytes(payload, FileType.MANIFEST)


# ---------------------------------------------------------------------------
# Snapshot builder
# ---------------------------------------------------------------------------


def _build_manifest_ref(builder: flatbuffers.Builder, mref: ManifestRefData) -> int:
    """Build a ManifestRef table."""
    # Extents vector (inline structs)
    ManifestRefStartExtentsVector(builder, len(mref.extents))
    for from_, to in reversed(mref.extents):
        CreateChunkIndexRange(builder, from_, to)
    extents_vec = builder.EndVector()

    ManifestRefStart(builder)
    ManifestRefAddObjectId(builder, CreateObjectId12(builder, list(mref.manifest_id)))
    ManifestRefAddExtents(builder, extents_vec)
    return ManifestRefEnd(builder)


def _build_shape_v2_from_user_data(
    builder: flatbuffers.Builder, user_data: bytes | None
) -> list[int] | None:
    """Parse zarr.json to extract shape/chunk_shape and build DimensionShapeV2 tables.

    Only called for array nodes. Returns a list of FlatBuffer offsets, or
    None if shape info can't be extracted.
    """
    if user_data is None:
        return None
    try:
        meta = json.loads(user_data)
    except (json.JSONDecodeError, UnicodeDecodeError):
        return None

    shape = meta.get("shape")
    chunk_shape = meta.get("chunk_grid", {}).get("configuration", {}).get("chunk_shape")
    if not shape or not chunk_shape:
        return None

    offsets = []
    for array_length, chunk_size in zip(shape, chunk_shape, strict=False):
        num_chunks = math.ceil(array_length / chunk_size) if chunk_size > 0 else 0
        DimensionShapeV2Start(builder)
        DimensionShapeV2AddArrayLength(builder, array_length)
        DimensionShapeV2AddNumChunks(builder, num_chunks)
        offsets.append(DimensionShapeV2End(builder))
    return offsets


def _build_node_snapshot(builder: flatbuffers.Builder, node: NodeWriteData) -> int:
    """Build a NodeSnapshot table."""
    # Pre-build all sub-objects before starting the table.
    path_off = builder.CreateString(node.path)

    # user_data as byte vector
    user_data_off = None
    if node.user_data is not None:
        user_data_off = builder.CreateByteVector(node.user_data)

    # Build node_data union
    if node.node_type == "array":
        # Build ManifestRef tables first
        mref_offsets = [_build_manifest_ref(builder, m) for m in node.manifests]

        # Manifests vector
        ArrayNodeDataStartManifestsVector(builder, len(mref_offsets))
        for off in reversed(mref_offsets):
            builder.PrependUOffsetTRelative(off)
        manifests_vec = builder.EndVector()

        # shape: empty vector for V2 (required field)
        ArrayNodeDataStartShapeVector(builder, 0)
        shape_vec = builder.EndVector()

        # shape_v2: populate from zarr.json if available
        shape_v2_offsets = _build_shape_v2_from_user_data(builder, node.user_data)
        shape_v2_vec = None
        if shape_v2_offsets is not None:
            ArrayNodeDataStartShapeV2Vector(builder, len(shape_v2_offsets))
            for off in reversed(shape_v2_offsets):
                builder.PrependUOffsetTRelative(off)
            shape_v2_vec = builder.EndVector()

        # dimension_names vector
        dn_vec = None
        if node.dimension_names:
            dn_offsets = []
            for dn in node.dimension_names:
                name_off = builder.CreateString(dn)
                DimensionNameStart(builder)
                DimensionNameAddName(builder, name_off)
                dn_offsets.append(DimensionNameEnd(builder))
            ArrayNodeDataStartDimensionNamesVector(builder, len(dn_offsets))
            for off in reversed(dn_offsets):
                builder.PrependUOffsetTRelative(off)
            dn_vec = builder.EndVector()

        ArrayNodeDataStart(builder)
        ArrayNodeDataAddManifests(builder, manifests_vec)
        ArrayNodeDataAddShape(builder, shape_vec)
        if shape_v2_vec is not None:
            ArrayNodeDataAddShapeV2(builder, shape_v2_vec)
        if dn_vec is not None:
            ArrayNodeDataAddDimensionNames(builder, dn_vec)
        node_data_off = ArrayNodeDataEnd(builder)
        node_data_type = NodeData.Array
    else:
        GroupNodeDataStart(builder)
        node_data_off = GroupNodeDataEnd(builder)
        node_data_type = NodeData.Group

    NodeSnapshotStart(builder)
    NodeSnapshotAddId(builder, CreateObjectId8(builder, list(node.node_id)))
    NodeSnapshotAddPath(builder, path_off)
    if user_data_off is not None:
        NodeSnapshotAddUserData(builder, user_data_off)
    NodeSnapshotAddNodeDataType(builder, node_data_type)
    NodeSnapshotAddNodeData(builder, node_data_off)
    return NodeSnapshotEnd(builder)


def _build_manifest_file_info_v2(
    builder: flatbuffers.Builder, mf: ManifestFileData
) -> int:
    """Build a ManifestFileInfoV2 table."""
    ManifestFileInfoV2Start(builder)
    ManifestFileInfoV2AddId(builder, CreateObjectId12(builder, list(mf.manifest_id)))
    ManifestFileInfoV2AddSizeBytes(builder, mf.size_bytes)
    ManifestFileInfoV2AddNumChunkRefs(builder, mf.num_chunk_refs)
    return ManifestFileInfoV2End(builder)


def build_snapshot_payload(
    snapshot_id: bytes,
    nodes: list[NodeWriteData],
    message: str = "",
    manifest_files: list[ManifestFileData] | None = None,
    flushed_at: int | None = None,
) -> bytes:
    """Build a Snapshot FlatBuffer payload (without header).

    V2 snapshots do NOT write ``parent_id`` — the parent relationship
    is tracked via ``repo.snapshots[].parent_offset`` instead.
    """
    builder = flatbuffers.Builder(4096)

    if flushed_at is None:
        flushed_at = int(time.time() * 1_000_000)

    # Pre-build message string (required field — always write it)
    msg_off = builder.CreateString(message)

    # Build all NodeSnapshot tables (must be sorted by path)
    sorted_nodes = sorted(nodes, key=lambda n: n.path)
    node_offsets = [_build_node_snapshot(builder, n) for n in sorted_nodes]

    # Nodes vector
    SnapshotStartNodesVector(builder, len(node_offsets))
    for off in reversed(node_offsets):
        builder.PrependUOffsetTRelative(off)
    nodes_vec = builder.EndVector()

    # ManifestFilesV2 vector (always write, even if empty)
    mf_offsets = [
        _build_manifest_file_info_v2(builder, mf) for mf in (manifest_files or [])
    ]
    SnapshotStartManifestFilesV2Vector(builder, len(mf_offsets))
    for off in reversed(mf_offsets):
        builder.PrependUOffsetTRelative(off)
    mf_v2_vec = builder.EndVector()

    # manifest_files (V1, required) — write empty vector for V2 compliance
    SnapshotStartManifestFilesVector(builder, 0)
    mf_v1_vec = builder.EndVector()

    # metadata (required) — write empty vector
    SnapshotStartMetadataVector(builder, 0)
    metadata_vec = builder.EndVector()

    SnapshotStart(builder)
    SnapshotAddId(builder, CreateObjectId12(builder, list(snapshot_id)))
    # No parent_id in V2 — tracked via repo.snapshots[].parent_offset
    SnapshotAddNodes(builder, nodes_vec)
    SnapshotAddFlushedAt(builder, flushed_at)
    SnapshotAddMessage(builder, msg_off)
    SnapshotAddManifestFilesV2(builder, mf_v2_vec)
    SnapshotAddManifestFiles(builder, mf_v1_vec)
    SnapshotAddMetadata(builder, metadata_vec)
    snapshot_off = SnapshotEnd(builder)

    builder.Finish(snapshot_off)
    return bytes(builder.Output())


def build_snapshot(
    snapshot_id: bytes,
    nodes: list[NodeWriteData],
    message: str = "",
    manifest_files: list[ManifestFileData] | None = None,
    flushed_at: int | None = None,
) -> bytes:
    """Build a complete snapshot file (header + FlatBuffer payload)."""
    payload = build_snapshot_payload(
        snapshot_id, nodes, message, manifest_files, flushed_at
    )
    return build_bytes(payload, FileType.SNAPSHOT)


# ---------------------------------------------------------------------------
# Transaction log builder
# ---------------------------------------------------------------------------


def build_transaction_log_payload(
    txn_id: bytes,
    new_groups: list[bytes] | None = None,
    new_arrays: list[bytes] | None = None,
    deleted_groups: list[bytes] | None = None,
    deleted_arrays: list[bytes] | None = None,
    updated_arrays: list[bytes] | None = None,
    updated_groups: list[bytes] | None = None,
    updated_chunks: list[ArrayUpdatedChunksData] | None = None,
) -> bytes:
    """Build a TransactionLog FlatBuffer payload (without header)."""
    builder = flatbuffers.Builder(1024)

    # Build updated_chunks tables first (they contain sub-objects)
    uc_offsets = []
    if updated_chunks:
        for uc in updated_chunks:
            # Build ChunkIndices tables
            ci_offsets = []
            for idx in uc.chunk_indices:
                ChunkIndicesStartCoordsVector(builder, len(idx))
                for val in reversed(idx):
                    builder.PrependUint32(val)
                vals_vec = builder.EndVector()

                ChunkIndicesStart(builder)
                ChunkIndicesAddCoords(builder, vals_vec)
                ci_offsets.append(ChunkIndicesEnd(builder))

            # Chunks vector
            ArrayUpdatedChunksStartChunksVector(builder, len(ci_offsets))
            for off in reversed(ci_offsets):
                builder.PrependUOffsetTRelative(off)
            chunks_vec = builder.EndVector()

            ArrayUpdatedChunksStart(builder)
            ArrayUpdatedChunksAddNodeId(
                builder, CreateObjectId8(builder, list(uc.node_id))
            )
            ArrayUpdatedChunksAddChunks(builder, chunks_vec)
            uc_offsets.append(ArrayUpdatedChunksEnd(builder))

    # Build ObjectId8 vectors for new/deleted/updated (always write, even empty)
    ng_vec = _build_id8_vector(builder, new_groups or [])
    na_vec = _build_id8_vector(builder, new_arrays or [])
    dg_vec = _build_id8_vector(builder, deleted_groups or [])
    da_vec = _build_id8_vector(builder, deleted_arrays or [])
    ua_vec = _build_id8_vector(builder, updated_arrays or [])
    ug_vec = _build_id8_vector(builder, updated_groups or [])

    # Updated chunks vector (always write, even empty)
    TransactionLogStartUpdatedChunksVector(builder, len(uc_offsets))
    for off in reversed(uc_offsets):
        builder.PrependUOffsetTRelative(off)
    uc_vec = builder.EndVector()

    TransactionLogStart(builder)
    TransactionLogAddId(builder, CreateObjectId12(builder, list(txn_id)))
    TransactionLogAddNewGroups(builder, ng_vec)
    TransactionLogAddNewArrays(builder, na_vec)
    TransactionLogAddDeletedGroups(builder, dg_vec)
    TransactionLogAddDeletedArrays(builder, da_vec)
    TransactionLogAddUpdatedArrays(builder, ua_vec)
    TransactionLogAddUpdatedGroups(builder, ug_vec)
    TransactionLogAddUpdatedChunks(builder, uc_vec)
    txn_off = TransactionLogEnd(builder)

    builder.Finish(txn_off)
    return bytes(builder.Output())


def build_transaction_log(
    txn_id: bytes,
    new_groups: list[bytes] | None = None,
    new_arrays: list[bytes] | None = None,
    deleted_groups: list[bytes] | None = None,
    deleted_arrays: list[bytes] | None = None,
    updated_arrays: list[bytes] | None = None,
    updated_groups: list[bytes] | None = None,
    updated_chunks: list[ArrayUpdatedChunksData] | None = None,
) -> bytes:
    """Build a complete transaction log file (header + FlatBuffer payload)."""
    payload = build_transaction_log_payload(
        txn_id,
        new_groups,
        new_arrays,
        deleted_groups,
        deleted_arrays,
        updated_arrays,
        updated_groups,
        updated_chunks,
    )
    return build_bytes(payload, FileType.TRANSACTION_LOG)


# ---------------------------------------------------------------------------
# Repo file builder
# ---------------------------------------------------------------------------


def _build_ref(builder: flatbuffers.Builder, name: str, snapshot_index: int) -> int:
    """Build a Ref table."""
    name_off = builder.CreateString(name)
    RefStart(builder)
    RefAddName(builder, name_off)
    RefAddSnapshotIndex(builder, snapshot_index)
    return RefEnd(builder)


def _build_snapshot_info(builder: flatbuffers.Builder, info: SnapshotInfoData) -> int:
    """Build a SnapshotInfo table."""
    msg_off = builder.CreateString(info.message) if info.message else None

    SnapshotInfoStart(builder)
    SnapshotInfoAddId(builder, CreateObjectId12(builder, list(info.snapshot_id)))
    SnapshotInfoAddParentOffset(builder, info.parent_offset)
    SnapshotInfoAddFlushedAt(builder, info.flushed_at)
    if msg_off is not None:
        SnapshotInfoAddMessage(builder, msg_off)
    return SnapshotInfoEnd(builder)


def _build_update(builder: flatbuffers.Builder, upd: UpdateData) -> int:
    """Build an Update table (union wrapper around a specific update type)."""
    now = upd.updated_at or int(time.time() * 1_000_000)

    if upd.kind == "repo_initialized":
        RepoInitializedUpdateStart(builder)
        inner_off = RepoInitializedUpdateEnd(builder)
        type_id = UpdateType.RepoInitializedUpdate

    elif upd.kind == "branch_created":
        name_off = builder.CreateString(upd.name)
        BranchCreatedUpdateStart(builder)
        BranchCreatedUpdateAddName(builder, name_off)
        inner_off = BranchCreatedUpdateEnd(builder)
        type_id = UpdateType.BranchCreatedUpdate

    elif upd.kind == "branch_deleted":
        name_off = builder.CreateString(upd.name)
        BranchDeletedUpdateStart(builder)
        BranchDeletedUpdateAddName(builder, name_off)
        if upd.previous_snap_id is not None:
            BranchDeletedUpdateAddPreviousSnapId(
                builder, CreateObjectId12(builder, list(upd.previous_snap_id))
            )
        inner_off = BranchDeletedUpdateEnd(builder)
        type_id = UpdateType.BranchDeletedUpdate

    elif upd.kind == "tag_created":
        name_off = builder.CreateString(upd.name)
        TagCreatedUpdateStart(builder)
        TagCreatedUpdateAddName(builder, name_off)
        inner_off = TagCreatedUpdateEnd(builder)
        type_id = UpdateType.TagCreatedUpdate

    elif upd.kind == "tag_deleted":
        name_off = builder.CreateString(upd.name)
        TagDeletedUpdateStart(builder)
        TagDeletedUpdateAddName(builder, name_off)
        if upd.previous_snap_id is not None:
            TagDeletedUpdateAddPreviousSnapId(
                builder, CreateObjectId12(builder, list(upd.previous_snap_id))
            )
        inner_off = TagDeletedUpdateEnd(builder)
        type_id = UpdateType.TagDeletedUpdate

    elif upd.kind == "new_commit":
        branch_off = builder.CreateString(upd.branch)
        NewCommitUpdateStart(builder)
        NewCommitUpdateAddBranch(builder, branch_off)
        if upd.snapshot_id is not None:
            NewCommitUpdateAddNewSnapId(
                builder, CreateObjectId12(builder, list(upd.snapshot_id))
            )
        inner_off = NewCommitUpdateEnd(builder)
        type_id = UpdateType.NewCommitUpdate

    else:
        raise ValueError(f"Unknown update kind: {upd.kind!r}")

    UpdateStart(builder)
    UpdateAddUpdateTypeType(builder, type_id)
    UpdateAddUpdateType(builder, inner_off)
    UpdateAddUpdatedAt(builder, now)
    return UpdateEnd(builder)


def build_repo_payload(
    spec_version: int,
    branches: dict[str, int],
    tags: dict[str, int],
    snapshots: list[SnapshotInfoData],
    deleted_tags: list[str] | None = None,
    updates: list[UpdateData] | None = None,
) -> bytes:
    """Build a Repo FlatBuffer payload (without header)."""
    builder = flatbuffers.Builder(2048)

    # Build Ref tables for branches and tags (sorted by name)
    branch_offsets = [
        _build_ref(builder, name, idx) for name, idx in sorted(branches.items())
    ]
    tag_offsets = [_build_ref(builder, name, idx) for name, idx in sorted(tags.items())]

    # Build SnapshotInfo tables
    snap_offsets = [_build_snapshot_info(builder, s) for s in snapshots]

    # Build Update tables
    update_offsets = [_build_update(builder, u) for u in (updates or [])]

    # Branches vector
    RepoStartBranchesVector(builder, len(branch_offsets))
    for off in reversed(branch_offsets):
        builder.PrependUOffsetTRelative(off)
    branches_vec = builder.EndVector()

    # Tags vector (required — always write, even if empty)
    RepoStartTagsVector(builder, len(tag_offsets))
    for off in reversed(tag_offsets):
        builder.PrependUOffsetTRelative(off)
    tags_vec = builder.EndVector()

    # Snapshots vector
    RepoStartSnapshotsVector(builder, len(snap_offsets))
    for off in reversed(snap_offsets):
        builder.PrependUOffsetTRelative(off)
    snaps_vec = builder.EndVector()

    # deleted_tags (required)
    dt_list = sorted(deleted_tags or [])
    dt_string_offsets = [builder.CreateString(t) for t in dt_list]
    RepoStartDeletedTagsVector(builder, len(dt_string_offsets))
    for off in reversed(dt_string_offsets):
        builder.PrependUOffsetTRelative(off)
    deleted_tags_vec = builder.EndVector()

    # latest_updates (required)
    RepoStartLatestUpdatesVector(builder, len(update_offsets))
    for off in reversed(update_offsets):
        builder.PrependUOffsetTRelative(off)
    latest_updates_vec = builder.EndVector()

    # status (required) — Online
    flushed_at = int(time.time() * 1_000_000)
    RepoStatusStart(builder)
    RepoStatusAddAvailability(builder, RepoAvailability.Online)
    RepoStatusAddSetAt(builder, flushed_at)
    status_off = RepoStatusEnd(builder)

    RepoStart(builder)
    RepoAddSpecVersion(builder, spec_version)
    RepoAddBranches(builder, branches_vec)
    RepoAddTags(builder, tags_vec)
    RepoAddSnapshots(builder, snaps_vec)
    RepoAddDeletedTags(builder, deleted_tags_vec)
    RepoAddLatestUpdates(builder, latest_updates_vec)
    RepoAddStatus(builder, status_off)
    repo_off = RepoEnd(builder)

    builder.Finish(repo_off)
    return bytes(builder.Output())


def build_repo(
    spec_version: int,
    branches: dict[str, int],
    tags: dict[str, int],
    snapshots: list[SnapshotInfoData],
    deleted_tags: list[str] | None = None,
    updates: list[UpdateData] | None = None,
) -> bytes:
    """Build a complete repo file (header + FlatBuffer payload)."""
    payload = build_repo_payload(
        spec_version, branches, tags, snapshots, deleted_tags, updates
    )
    return build_bytes(payload, FileType.REPO_INFO)
