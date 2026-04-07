"""Snapshot diff module for Icechunk repositories.

Compares two snapshots and reports added, removed, modified, and
unchanged nodes, including per-chunk change details for arrays.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from icepyck.manifest import ChunkRefInfo, ChunkType, ManifestReader
from icepyck.repo import RepoInfo
from icepyck.snapshot import NodeInfo, SnapshotReader


@dataclass
class NodeChange:
    """A node that exists in both snapshots but changed."""

    path: str
    node_type: str
    metadata_changed: bool
    chunks_changed: bool
    old_chunk_count: int
    new_chunk_count: int
    changed_chunk_indices: list[tuple[int, ...]] = field(default_factory=list)


@dataclass
class SnapshotDiff:
    """Differences between two snapshots."""

    old_ref: str
    new_ref: str
    added_nodes: list[NodeInfo] = field(default_factory=list)
    removed_nodes: list[NodeInfo] = field(default_factory=list)
    modified_nodes: list[NodeChange] = field(default_factory=list)
    unchanged_count: int = 0


def _chunk_ref_key(cref: ChunkRefInfo) -> tuple[object, ...]:
    """Return a comparable key for a chunk reference.

    Two chunks are "same" if they have the same type and content identity:
    - INLINE: same bytes
    - NATIVE: same chunk_id + offset + length
    - VIRTUAL: same location + offset + length
    """
    if cref.chunk_type == ChunkType.INLINE:
        return (ChunkType.INLINE, cref.inline_data)
    elif cref.chunk_type == ChunkType.NATIVE:
        return (ChunkType.NATIVE, cref.chunk_id, cref.offset, cref.length)
    else:
        return (ChunkType.VIRTUAL, cref.location, cref.offset, cref.length)


def _load_chunk_refs(
    root: Path,
    node: NodeInfo,
) -> dict[tuple[int, ...], ChunkRefInfo]:
    """Load all chunk refs for an array node, keyed by chunk index."""
    result: dict[tuple[int, ...], ChunkRefInfo] = {}
    for mref in node.manifest_refs:
        manifest = ManifestReader(root, mref.manifest_id)
        for cref in manifest.get_chunk_refs(node.node_id):
            result[cref.index] = cref
    return result


def diff_snapshots(
    repo_path: str | Path,
    old_ref: str,
    new_ref: str,
) -> SnapshotDiff:
    """Compare two snapshots of a repository.

    Parameters
    ----------
    repo_path : str or Path
        Root path of the Icechunk repository.
    old_ref : str
        Reference for the old snapshot (branch, tag, or hex snapshot ID).
    new_ref : str
        Reference for the new snapshot (branch, tag, or hex snapshot ID).

    Returns
    -------
    SnapshotDiff
        The differences between the two snapshots.
    """
    root = Path(repo_path)
    repo = RepoInfo(root / "repo")

    old_snapshot_id = _resolve_ref(repo, old_ref)
    new_snapshot_id = _resolve_ref(repo, new_ref)

    old_snapshot = SnapshotReader(root, old_snapshot_id)
    new_snapshot = SnapshotReader(root, new_snapshot_id)

    old_nodes = {n.path: n for n in old_snapshot.list_nodes()}
    new_nodes = {n.path: n for n in new_snapshot.list_nodes()}

    old_paths = set(old_nodes.keys())
    new_paths = set(new_nodes.keys())

    diff = SnapshotDiff(old_ref=old_ref, new_ref=new_ref)

    # Added nodes
    for path in sorted(new_paths - old_paths):
        diff.added_nodes.append(new_nodes[path])

    # Removed nodes
    for path in sorted(old_paths - new_paths):
        diff.removed_nodes.append(old_nodes[path])

    # Common nodes: compare
    for path in sorted(old_paths & new_paths):
        old_node = old_nodes[path]
        new_node = new_nodes[path]

        metadata_changed = old_node.user_data != new_node.user_data
        chunks_changed = False
        changed_indices: list[tuple[int, ...]] = []
        old_chunk_count = 0
        new_chunk_count = 0

        if old_node.node_type == "array" and new_node.node_type == "array":
            old_chunks = _load_chunk_refs(root, old_node)
            new_chunks = _load_chunk_refs(root, new_node)
            old_chunk_count = len(old_chunks)
            new_chunk_count = len(new_chunks)

            all_indices = sorted(set(old_chunks.keys()) | set(new_chunks.keys()))
            for idx in all_indices:
                old_cref = old_chunks.get(idx)
                new_cref = new_chunks.get(idx)
                if old_cref is None or new_cref is None:
                    # Chunk added or removed
                    changed_indices.append(idx)
                elif _chunk_ref_key(old_cref) != _chunk_ref_key(new_cref):
                    changed_indices.append(idx)

            chunks_changed = len(changed_indices) > 0

        if metadata_changed or chunks_changed:
            diff.modified_nodes.append(
                NodeChange(
                    path=path,
                    node_type=old_node.node_type,
                    metadata_changed=metadata_changed,
                    chunks_changed=chunks_changed,
                    old_chunk_count=old_chunk_count,
                    new_chunk_count=new_chunk_count,
                    changed_chunk_indices=changed_indices,
                )
            )
        else:
            diff.unchanged_count += 1

    return diff


def _resolve_ref(repo: RepoInfo, ref: str) -> bytes:
    """Resolve a ref string to a snapshot ID.

    Supports branch names, tag names, hex snapshot IDs, and
    relative refs like ``main~1`` (parent of main).
    """
    # Handle relative refs like "main~1", "main~2"
    if "~" in ref:
        base, offset_str = ref.rsplit("~", 1)
        try:
            offset = int(offset_str)
        except ValueError:
            raise KeyError(f"Invalid relative ref: {ref!r}") from None
        snapshot_id = _resolve_ref(repo, base)
        return _get_ancestor(repo, snapshot_id, offset)

    # Try branch
    try:
        return repo.get_snapshot_id(ref)
    except KeyError:
        pass
    # Try tag
    try:
        return repo.get_tag_snapshot_id(ref)
    except KeyError:
        pass
    # Try hex ID
    try:
        raw = bytes.fromhex(ref)
        if len(raw) == 12:
            return raw
    except ValueError:
        pass
    raise KeyError(f"Could not resolve ref: {ref!r}")


def _get_ancestor(repo: RepoInfo, snapshot_id: bytes, generations: int) -> bytes:
    """Walk back through parent offsets to find an ancestor snapshot.

    The ``ParentOffset`` field in the flatbuffers snapshot entry is an
    absolute index into the repo's snapshot list (not a relative offset).
    A value of -1 means no parent (initial commit).
    """
    # Build index from snapshot_id -> list index
    id_to_idx: dict[bytes, int] = {}
    for i, sid in enumerate(repo._snapshot_ids):
        id_to_idx[sid] = i

    current_id = snapshot_id
    for _ in range(generations):
        idx = id_to_idx.get(current_id)
        if idx is None:
            raise KeyError(
                f"Snapshot {current_id.hex()} not found in repo snapshot list"
            )
        snap = repo._repo.Snapshots(idx)
        parent_idx = snap.ParentOffset()
        if parent_idx < 0:
            raise KeyError(
                f"Snapshot {current_id.hex()} has no parent (initial commit)"
            )
        if parent_idx >= len(repo._snapshot_ids):
            raise KeyError(
                f"Parent index {parent_idx} is out of range"
            )
        current_id = repo._snapshot_ids[parent_idx]

    return current_id
