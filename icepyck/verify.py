"""Spec conformance verifier for Icechunk V2 repositories.

Reads a repo's FlatBuffer files and checks all (required) fields,
sorted order, V2 semantics, and structural invariants. Reports
violations as a list of issues.

Library usage::

    from icepyck.verify import verify_repo
    issues = verify_repo("/path/to/repo")
    assert not issues, f"Spec violations found: {issues}"

CLI usage::

    uv run python -m icepyck.verify /path/to/repo
    icepyck-verify /path/to/repo          # if installed as console_script
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from icepyck.crockford import encode as crockford_encode
from icepyck.header import FileType, parse_bytes


def _is_sorted(items: list) -> bool:  # type: ignore[type-arg]
    """Check if a list is sorted in O(n) without allocating a sorted copy."""
    return all(a <= b for a, b in zip(items, items[1:], strict=False))


@dataclass
class Issue:
    """A single spec violation."""

    file: str
    field: str
    message: str

    def __str__(self) -> str:
        return f"[{self.file}] {self.field}: {self.message}"


def verify_repo(
    path: str | Path,
    *,
    storage: object | None = None,
) -> list[Issue]:
    """Verify that a repository conforms to the Icechunk V2 spec.

    Parameters
    ----------
    path : str or Path
        Root path of the repository.
    storage : Storage, optional
        Storage backend. If not provided, uses :class:`LocalStorage`.

    Returns
    -------
    list[Issue]
        List of spec violations. Empty means the repo is conformant.
    """
    from icepyck.storage import LocalStorage
    from icepyck.storage import Storage as StorageType

    if storage is None:
        store: StorageType = LocalStorage(str(path))
    else:
        store = storage  # type: ignore[assignment]

    issues: list[Issue] = []

    # --- Verify repo file ---
    if not store.exists("repo"):
        issues.append(Issue("repo", "", "repo file not found"))
        return issues

    repo_data = store.read("repo")
    issues.extend(_verify_repo_file(repo_data))

    # --- Extract snapshot IDs from repo file to verify each snapshot ---
    try:
        _, repo_payload = parse_bytes(repo_data)
        from icepyck.generated.Repo import Repo

        repo = Repo.GetRootAs(repo_payload)
        snapshot_ids = _get_snapshot_ids_from_repo(repo)
    except Exception as e:
        issues.append(Issue("repo", "", f"failed to parse: {e}"))
        return issues

    # --- Verify each snapshot ---
    for snap_id in snapshot_ids:
        snap_name = crockford_encode(snap_id)
        snap_path = f"snapshots/{snap_name}"
        if not store.exists(snap_path):
            issues.append(Issue(snap_path, "", "snapshot file not found"))
            continue
        snap_data = store.read(snap_path)
        issues.extend(_verify_snapshot_file(snap_data, snap_name))

        # Extract manifest IDs from snapshot and verify each
        try:
            _, snap_payload = parse_bytes(snap_data)
            from icepyck.generated.Snapshot import Snapshot

            snap = Snapshot.GetRootAs(snap_payload)
            manifest_ids = _get_manifest_ids_from_snapshot(snap)
        except Exception:
            continue

        for mid in manifest_ids:
            m_name = crockford_encode(mid)
            m_path = f"manifests/{m_name}"
            if not store.exists(m_path):
                issues.append(Issue(m_path, "", "manifest file not found"))
                continue
            m_data = store.read(m_path)
            issues.extend(_verify_manifest_file(m_data, m_name))

    # --- Verify transaction logs ---
    for snap_id in snapshot_ids:
        txn_name = crockford_encode(snap_id)
        txn_path = f"transactions/{txn_name}"
        if store.exists(txn_path):
            txn_data = store.read(txn_path)
            issues.extend(_verify_transaction_log_file(txn_data, txn_name))

    return issues


# ---------------------------------------------------------------------------
# Repo file verification
# ---------------------------------------------------------------------------


def _verify_repo_file(data: bytes) -> list[Issue]:
    issues: list[Issue] = []
    fname = "repo"

    try:
        header, payload = parse_bytes(data)
    except Exception as e:
        issues.append(Issue(fname, "header", f"failed to parse: {e}"))
        return issues

    if header.file_type != FileType.REPO_INFO:
        issues.append(
            Issue(fname, "file_type", f"expected REPO_INFO, got {header.file_type}")
        )

    from icepyck.generated.Repo import Repo

    try:
        repo = Repo.GetRootAs(payload)
    except Exception as e:
        issues.append(Issue(fname, "", f"FlatBuffer parse failed: {e}"))
        return issues

    # spec_version
    if repo.SpecVersion() != 2:
        issues.append(
            Issue(fname, "spec_version", f"expected 2, got {repo.SpecVersion()}")
        )

    # branches (required)
    if repo.BranchesIsNone():
        issues.append(Issue(fname, "branches", "required field is absent"))
    else:
        # Check sorted order
        names = [repo.Branches(i).Name().decode() for i in range(repo.BranchesLength())]
        if not _is_sorted(names):
            issues.append(Issue(fname, "branches", f"not sorted: {names}"))

    # tags (required)
    if repo.TagsIsNone():
        issues.append(Issue(fname, "tags", "required field is absent"))
    else:
        names = [repo.Tags(i).Name().decode() for i in range(repo.TagsLength())]
        if not _is_sorted(names):
            issues.append(Issue(fname, "tags", f"not sorted: {names}"))

    # deleted_tags (required)
    if repo.DeletedTagsIsNone():
        issues.append(Issue(fname, "deleted_tags", "required field is absent"))

    # snapshots (required)
    if repo.SnapshotsIsNone():
        issues.append(Issue(fname, "snapshots", "required field is absent"))

    # status (required)
    if repo.Status() is None:
        issues.append(Issue(fname, "status", "required field is absent"))

    # latest_updates (required)
    if repo.LatestUpdatesIsNone():
        issues.append(Issue(fname, "latest_updates", "required field is absent"))

    return issues


# ---------------------------------------------------------------------------
# Snapshot file verification
# ---------------------------------------------------------------------------


def _verify_snapshot_file(data: bytes, name: str) -> list[Issue]:
    issues: list[Issue] = []
    fname = f"snapshots/{name}"

    try:
        header, payload = parse_bytes(data)
    except Exception as e:
        issues.append(Issue(fname, "header", f"failed to parse: {e}"))
        return issues

    if header.file_type != FileType.SNAPSHOT:
        issues.append(
            Issue(fname, "file_type", f"expected SNAPSHOT, got {header.file_type}")
        )

    from icepyck.generated.Snapshot import Snapshot

    try:
        snap = Snapshot.GetRootAs(payload)
    except Exception as e:
        issues.append(Issue(fname, "", f"FlatBuffer parse failed: {e}"))
        return issues

    # id (required)
    if snap.Id() is None:
        issues.append(Issue(fname, "id", "required field is absent"))

    # message (required)
    if snap.Message() is None:
        issues.append(Issue(fname, "message", "required field is absent"))

    # metadata (required)
    if snap.MetadataIsNone():
        issues.append(Issue(fname, "metadata", "required field is absent"))

    # manifest_files (required for V2)
    if snap.ManifestFilesIsNone():
        issues.append(Issue(fname, "manifest_files", "required field is absent (V2)"))

    # nodes (required)
    if snap.NodesIsNone():
        issues.append(Issue(fname, "nodes", "required field is absent"))
    else:
        # Check sorted order by path
        paths = []
        for i in range(snap.NodesLength()):
            node = snap.Nodes(i)
            path = node.Path()
            paths.append(path.decode() if path else "")
        if not _is_sorted(paths):
            issues.append(Issue(fname, "nodes", f"not sorted by path: {paths}"))

        # Check each node
        for i in range(snap.NodesLength()):
            node = snap.Nodes(i)
            node_path = node.Path().decode() if node.Path() else f"[index {i}]"
            issues.extend(_verify_node_snapshot(fname, node_path, node))

    # V2: parent_id should NOT be present
    if snap.ParentId() is not None:
        issues.append(
            Issue(fname, "parent_id", "V2 snapshots should not write parent_id")
        )

    return issues


def _verify_node_snapshot(fname: str, node_path: str, node: object) -> list[Issue]:
    """Verify a single NodeSnapshot within a snapshot."""
    issues: list[Issue] = []
    prefix = f"{node_path}"

    from icepyck.generated.NodeData import NodeData

    data_type = node.NodeDataType()  # type: ignore[attr-defined]

    if data_type == NodeData.Array:
        # Array nodes must have shape (required, empty [] for V2)
        array_data = node.NodeData()  # type: ignore[attr-defined]
        if array_data is None:
            issues.append(Issue(fname, f"{prefix}.node_data", "absent for array node"))
            return issues

        from icepyck.generated.ArrayNodeData import ArrayNodeData

        arr = ArrayNodeData()
        arr.Init(array_data.Bytes, array_data.Pos)

        if arr.ShapeIsNone():
            issues.append(Issue(fname, f"{prefix}.shape", "required field is absent"))

    return issues


# ---------------------------------------------------------------------------
# Manifest file verification
# ---------------------------------------------------------------------------


def _verify_manifest_file(data: bytes, name: str) -> list[Issue]:
    issues: list[Issue] = []
    fname = f"manifests/{name}"

    try:
        header, payload = parse_bytes(data)
    except Exception as e:
        issues.append(Issue(fname, "header", f"failed to parse: {e}"))
        return issues

    if header.file_type != FileType.MANIFEST:
        issues.append(
            Issue(fname, "file_type", f"expected MANIFEST, got {header.file_type}")
        )

    from icepyck.generated.Manifest import Manifest

    try:
        manifest = Manifest.GetRootAs(payload)
    except Exception as e:
        issues.append(Issue(fname, "", f"FlatBuffer parse failed: {e}"))
        return issues

    # id (required)
    if manifest.Id() is None:
        issues.append(Issue(fname, "id", "required field is absent"))

    # arrays: check sorted by node_id
    if not manifest.ArraysIsNone() and manifest.ArraysLength() > 1:
        node_ids = []
        for i in range(manifest.ArraysLength()):
            arr = manifest.Arrays(i)
            nid = arr.NodeId()
            if nid is not None:
                node_ids.append(bytes(nid.Bytes()))
        if not _is_sorted(node_ids):
            issues.append(Issue(fname, "arrays", "not sorted by node_id"))

    return issues


# ---------------------------------------------------------------------------
# Transaction log verification
# ---------------------------------------------------------------------------


def _verify_transaction_log_file(data: bytes, name: str) -> list[Issue]:
    issues: list[Issue] = []
    fname = f"transactions/{name}"

    try:
        header, payload = parse_bytes(data)
    except Exception as e:
        issues.append(Issue(fname, "header", f"failed to parse: {e}"))
        return issues

    if header.file_type != FileType.TRANSACTION_LOG:
        issues.append(
            Issue(
                fname,
                "file_type",
                f"expected TRANSACTION_LOG, got {header.file_type}",
            )
        )

    from icepyck.generated.TransactionLog import TransactionLog

    try:
        txn = TransactionLog.GetRootAs(payload)
    except Exception as e:
        issues.append(Issue(fname, "", f"FlatBuffer parse failed: {e}"))
        return issues

    # All vector fields are required
    required_vectors = [
        ("new_groups", txn.NewGroupsIsNone),
        ("new_arrays", txn.NewArraysIsNone),
        ("deleted_groups", txn.DeletedGroupsIsNone),
        ("deleted_arrays", txn.DeletedArraysIsNone),
        ("updated_arrays", txn.UpdatedArraysIsNone),
        ("updated_groups", txn.UpdatedGroupsIsNone),
        ("updated_chunks", txn.UpdatedChunksIsNone),
    ]
    for field_name, is_none_fn in required_vectors:
        if is_none_fn():
            issues.append(Issue(fname, field_name, "required field is absent"))

    return issues


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _get_snapshot_ids_from_repo(repo: object) -> list[bytes]:
    """Extract snapshot IDs from a parsed Repo FlatBuffer."""
    ids = []
    n = repo.SnapshotsLength()  # type: ignore[attr-defined]
    for i in range(n):
        snap = repo.Snapshots(i)  # type: ignore[attr-defined]
        sid = snap.Id()
        if sid is not None:
            ids.append(bytes(sid.Bytes()))
    return ids


def _get_manifest_ids_from_snapshot(snap: object) -> list[bytes]:
    """Extract manifest IDs referenced by nodes in a snapshot."""
    ids: set[bytes] = set()
    for i in range(snap.NodesLength()):  # type: ignore[attr-defined]
        node = snap.Nodes(i)  # type: ignore[attr-defined]
        from icepyck.generated.NodeData import NodeData

        if node.NodeDataType() == NodeData.Array:
            array_data = node.NodeData()
            if array_data is None:
                continue
            from icepyck.generated.ArrayNodeData import ArrayNodeData

            arr = ArrayNodeData()
            arr.Init(array_data.Bytes, array_data.Pos)
            for j in range(arr.ManifestsLength()):
                mref = arr.Manifests(j)
                oid = mref.ObjectId()
                if oid is not None:
                    ids.add(bytes(oid.Bytes()))

    # Also check manifest_files_v2
    n = snap.ManifestFilesV2Length()  # type: ignore[attr-defined]
    for i in range(n):
        mf = snap.ManifestFilesV2(i)  # type: ignore[attr-defined]
        mid = mf.Id()
        if mid is not None:
            ids.add(bytes(mid.Bytes()))

    return list(ids)


# ---------------------------------------------------------------------------
# Rich-formatted output
# ---------------------------------------------------------------------------


def print_report(
    path: str | Path,
    issues: list[Issue],
) -> None:
    """Print a rich-formatted verification report to the console."""
    from rich.console import Console
    from rich.table import Table
    from rich.text import Text

    console = Console()

    if not issues:
        console.print()
        console.print(
            Text(" PASS ", style="bold white on green"),
            Text(f" {path}", style="bold"),
        )
        console.print("  No spec violations found.", style="dim")
        console.print()
        return

    console.print()
    console.print(
        Text(" FAIL ", style="bold white on red"),
        Text(f" {path}", style="bold"),
    )
    console.print(
        f"  {len(issues)} spec violation{'s' if len(issues) != 1 else ''} found:",
        style="dim",
    )
    console.print()

    table = Table(show_header=True, header_style="bold", pad_edge=False)
    table.add_column("File", style="cyan", no_wrap=True)
    table.add_column("Field", style="yellow")
    table.add_column("Issue", style="red")

    for issue in issues:
        table.add_row(issue.file, issue.field or "-", issue.message)

    console.print(table)
    console.print()


def main(argv: list[str] | None = None) -> int:
    """CLI entry point for the spec verifier.

    Returns 0 if all repos pass, 1 if any violations are found.
    """
    import argparse

    parser = argparse.ArgumentParser(
        prog="icepyck-verify",
        description="Verify Icechunk V2 repository spec conformance.",
    )
    parser.add_argument(
        "paths",
        nargs="+",
        metavar="REPO_PATH",
        help="Path(s) to Icechunk repository root(s)",
    )
    args = parser.parse_args(argv)

    any_failed = False
    for repo_path in args.paths:
        path = Path(repo_path)
        if path.as_posix().startswith("s3://"):
            from icepyck.storage import S3Storage

            issues = verify_repo(path, storage=S3Storage(str(path)))
        else:
            issues = verify_repo(path)
        print_report(path, issues)
        if issues:
            any_failed = True

    return 1 if any_failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
