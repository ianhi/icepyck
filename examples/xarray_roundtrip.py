# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "icepyck",
#     "icechunk>=2.0.0a0",
#     "zarr>=3",
#     "xarray",
#     "numpy",
#     "scipy",
#     "pooch",
# ]
# [tool.uv]
# sources = {icepyck = {path = ".."}}
# prerelease = "allow"
# ///
"""
Roundtrip test: write xarray tutorial data with icechunk, read back with icepyck.

This demonstrates that icepyck can correctly read any data written by the
reference icechunk implementation.
"""

import shutil
from pathlib import Path

import numpy as np
import xarray as xr

import icechunk
import icepyck

REPO_PATH = Path(__file__).resolve().parent.parent / "test-repos" / "xarray-air-temp"

# --- Step 1: Load xarray tutorial dataset ---
print("Loading air_temperature dataset...")
ds_original = xr.tutorial.load_dataset("air_temperature")
print(f"  {ds_original}")
print()

# --- Step 2: Write to icechunk repo using reference implementation ---
if REPO_PATH.exists():
    shutil.rmtree(REPO_PATH)

print(f"Writing to icechunk repo at {REPO_PATH}...")
storage = icechunk.local_filesystem_storage(str(REPO_PATH))
repo = icechunk.Repository.create(storage)
session = repo.writable_session("main")
ds_original.to_zarr(session.store, mode="w")
session.commit("write air_temperature dataset")
print("  Committed.")
print()

# --- Step 3: Read back with icechunk (reference) ---
print("Reading back with icechunk...")
ref_repo = icechunk.Repository.open(storage)
ref_session = ref_repo.readonly_session(branch="main")
ds_icechunk = xr.open_zarr(ref_session.store)
print(f"  {ds_icechunk}")
print()

# --- Step 4: Read back with icepyck (our implementation) ---
print("Reading back with icepyck...")
pyck_repo = icepyck.open(REPO_PATH)
pyck_session = pyck_repo.readonly_session(branch="main")
ds_icepyck = xr.open_zarr(pyck_session.store)
print(f"  {ds_icepyck}")
print()

# --- Step 5: Compare ---
print("Comparing datasets...")

# Check dimensions match
assert set(ds_original.dims) == set(ds_icepyck.dims), "Dimensions differ!"
print(f"  Dimensions match: {dict(ds_icepyck.dims)}")

# Check variables match
for var in ds_original.data_vars:
    orig = ds_original[var].values
    pyck = ds_icepyck[var].values
    match = np.allclose(orig, pyck, equal_nan=True)
    print(f"  {var}: shape={pyck.shape} dtype={pyck.dtype} match={match}")
    if not match:
        diff = np.abs(orig - pyck)
        print(f"    max diff: {np.nanmax(diff)}")

# Check coordinates match
for coord in ds_original.coords:
    orig = ds_original[coord].values
    pyck = ds_icepyck[coord].values
    if np.issubdtype(orig.dtype, np.floating):
        match = np.allclose(orig, pyck, equal_nan=True)
    else:
        match = np.array_equal(orig, pyck)
    print(f"  coord {coord}: dtype={pyck.dtype} match={match}")

print()
print("Roundtrip successful!")
