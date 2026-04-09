"""Conflict handling: detect and recover from concurrent writes.

Usage::

    uv run examples/conflict_handling.py
"""

from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path

import numpy as np

import icepyck
from icepyck import ConflictError

tmp = Path(tempfile.mkdtemp())
repo_path = tmp / "repo"
repo = icepyck.Repository.init(repo_path)

meta = json.dumps(
    {
        "zarr_format": 3,
        "node_type": "array",
        "shape": [4],
        "data_type": "float32",
        "chunk_grid": {"name": "regular", "configuration": {"chunk_shape": [4]}},
        "chunk_key_encoding": {
            "name": "default",
            "configuration": {"separator": "/"},
        },
        "fill_value": 0.0,
        "codecs": [{"name": "bytes", "configuration": {"endian": "little"}}],
    }
).encode()

# --- Two sessions from the same base ---
ws1 = repo.writable_session(branch="main")
ws2 = repo.writable_session(branch="main")

# Writer 1 commits first
ws1.set_metadata("/from_writer1", meta)
ws1.set_chunk("/from_writer1", (0,), np.array([1, 2, 3, 4], dtype="<f4").tobytes())
ws1.commit("Writer 1")
print("Writer 1 committed successfully")

# Writer 2 tries — conflict!
ws2.set_metadata("/from_writer2", meta)
ws2.set_chunk("/from_writer2", (0,), np.array([5, 6, 7, 8], dtype="<f4").tobytes())
try:
    ws2.commit("Writer 2")
    print("Writer 2 committed (unexpected)")
except ConflictError as e:
    print(f"Writer 2 got ConflictError: {e}")

# --- Retry pattern ---
print("\nRetrying with a new session...")
ws3 = repo.writable_session(branch="main")
ws3.set_metadata("/from_writer2", meta)
ws3.set_chunk("/from_writer2", (0,), np.array([5, 6, 7, 8], dtype="<f4").tobytes())
ws3.commit("Writer 2 (retry)")
print("Writer 2 retry succeeded!")

# Both arrays exist
nodes = {n.path for n in repo.readonly_session(branch="main").list_nodes()}
print(f"\nFinal nodes: {sorted(nodes)}")

shutil.rmtree(tmp)
