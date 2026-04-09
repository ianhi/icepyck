# /// script
# requires-python = ">=3.12"
# dependencies = ["icepyck", "icechunk>=2.0.0a0", "zarr>=3", "numpy"]
# [tool.uv]
# sources = {icepyck = {path = ".."}}
# prerelease = "allow"
# ///
"""Compare icepyck reads against the reference icechunk implementation."""

from pathlib import Path

import icechunk
import numpy as np
import zarr

import icepyck

ROOT = Path(__file__).resolve().parent.parent

# Compare across multiple repos with non-trivial data
repos = {
    "native-chunks": ROOT / "test-repos" / "native-chunks",
    "basic": ROOT / "test-repos" / "basic",
    "nested": ROOT / "test-repos" / "nested",
}

for name, repo_path in repos.items():
    print(f"\n=== {name} ===")

    # --- Read with icepyck ---
    pyck_repo = icepyck.open(repo_path)
    pyck_session = pyck_repo.readonly_session(branch="main")
    pyck_root = zarr.open_group(store=pyck_session.store, mode="r")

    # --- Read with icechunk ---
    storage = icechunk.local_filesystem_storage(str(repo_path))
    ref_repo = icechunk.Repository.open(storage)
    ref_session = ref_repo.readonly_session(branch="main")
    ref_root = zarr.open_group(store=ref_session.store, mode="r")

    # Find all arrays and compare
    for node in pyck_session.list_nodes():
        if node.node_type != "array":
            continue
        zarr_path = node.path.lstrip("/")
        pyck_arr = np.array(pyck_root[zarr_path][:])
        ref_arr = np.array(ref_root[zarr_path][:])
        match = np.array_equal(pyck_arr, ref_arr)
        print(
            f"  {node.path}: shape={pyck_arr.shape} dtype={pyck_arr.dtype} match={match}"
        )
        if match:
            print(f"    sample: {pyck_arr.flat[:5]}")
        else:
            print(
                f"    MISMATCH! icepyck={pyck_arr.flat[:5]} icechunk={ref_arr.flat[:5]}"
            )
