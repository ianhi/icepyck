# /// script
# requires-python = ">=3.12"
# dependencies = ["icepyck", "icechunk>=2.0.0a0", "zarr>=3", "numpy"]
# [tool.uv]
# sources = {icepyck = {path = ".."}}
# prerelease = "allow"
# ///
"""Compare icepyck reads against the reference icechunk implementation."""

import numpy as np
import zarr

import icechunk
import icepyck

REPO_PATH = "test-repos/basic"

# --- Read with icepyck ---
repo = icepyck.open(REPO_PATH)
session = repo.readonly_session(branch="main")
pyck_store = session.store
pyck_root = zarr.open_group(store=pyck_store, mode="r")
pyck_temps = np.array(pyck_root["group1/temperatures"][:])

# --- Read with icechunk ---
storage = icechunk.local_filesystem_storage(REPO_PATH)
ref_repo = icechunk.Repository.open(storage)
ref_session = ref_repo.readonly_session(branch="main")
ref_root = zarr.open_group(store=ref_session.store, mode="r")
ref_temps = np.array(ref_root["group1/temperatures"][:])

# --- Compare ---
print(f"icepyck temperatures: {pyck_temps[:5]}")
print(f"icechunk temperatures: {ref_temps[:5]}")
print(f"Arrays match: {np.array_equal(pyck_temps, ref_temps)}")

# Compare all arrays
for path in ["/group1/temperatures", "/group1/timestamps"]:
    zarr_path = path.lstrip("/")
    pyck_arr = np.array(pyck_root[zarr_path][:])
    ref_arr = np.array(ref_root[zarr_path][:])
    match = np.array_equal(pyck_arr, ref_arr)
    print(f"{path}: shape={pyck_arr.shape} dtype={pyck_arr.dtype} match={match}")
