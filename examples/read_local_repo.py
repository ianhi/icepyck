# /// script
# requires-python = ">=3.12"
# dependencies = ["icepyck", "zarr>=3", "numpy"]
# [tool.uv]
# sources = {icepyck = {path = ".."}}
# ///
"""Read data from a local icechunk repository using icepyck + zarr."""

import icepyck
import zarr

# Open repository and get a read-only session
repo = icepyck.open("test-repos/basic")
print("Branches:", repo.list_branches())
print("Tags:", repo.list_tags())

session = repo.readonly_session(branch="main")

# Use zarr to read data
store = session.store
root = zarr.open_group(store=store, mode="r")

# Navigate the hierarchy
print("\nRepository contents:")
for node in session.list_nodes():
    print(f"  {node.path:40s} {node.node_type}")

# Read array data
temps = root["group1/temperatures"]
print(f"\ntemperatures: dtype={temps.dtype}, shape={temps.shape}")
print(f"  values: {temps[:]}")

timestamps = root["group1/timestamps"]
print(f"\ntimestamps: dtype={timestamps.dtype}, shape={timestamps.shape}")
print(f"  first 10: {timestamps[:10]}")
