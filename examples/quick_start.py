"""Quick start: create a repo, write data, read it back.

Usage::

    uv run examples/quick_start.py
"""

from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path

import numpy as np

import icepyck

tmp = Path(tempfile.mkdtemp())
repo_path = tmp / "my_repo"

# --- Create a new repository ---
repo = icepyck.Repository.init(repo_path)
print(f"Created repo at {repo_path}")
print(f"Branches: {repo.list_branches()}")

# --- Write data ---
ws = repo.writable_session(branch="main")

# Create an array with zarr v3 metadata
meta = json.dumps(
    {
        "zarr_format": 3,
        "node_type": "array",
        "shape": [10],
        "data_type": "float64",
        "chunk_grid": {"name": "regular", "configuration": {"chunk_shape": [10]}},
        "chunk_key_encoding": {
            "name": "default",
            "configuration": {"separator": "/"},
        },
        "fill_value": 0.0,
        "codecs": [{"name": "bytes", "configuration": {"endian": "little"}}],
    }
).encode()
ws.set_metadata("/temperatures", meta)
ws.set_chunk("/temperatures", (0,), np.arange(10, dtype="<f8").tobytes())

snap_id = ws.commit("Initial data")
print(f"\nCommitted: {snap_id}")

# --- Read it back ---
session = repo.readonly_session(branch="main")
print("\nNodes in snapshot:")
for node in session.list_nodes():
    print(f"  {node.path} ({node.node_type})")

# Read raw chunk data
chunk = repo.read_chunk("main", "/temperatures", (0,))
data = np.frombuffer(chunk, dtype="<f8")
print(f"\nTemperatures: {data}")

# --- Read via zarr ---
import zarr

store = session.store
root = zarr.open_group(store=store, mode="r")
print(f"\nVia zarr: {root['temperatures'][:]}")

# --- Commit log ---
print("\nCommit log:")
for entry in repo.log("main"):
    print(f"  {entry['id'][:12]}  {entry['message']}")

shutil.rmtree(tmp)
print("\nDone!")
