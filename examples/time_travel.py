"""Time travel: read old snapshots after new commits.

Usage::

    uv run examples/time_travel.py
"""

from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path

import numpy as np

import icepyck
from icepyck.ancestry import print_log

tmp = Path(tempfile.mkdtemp())
repo_path = tmp / "repo"
repo = icepyck.Repository.init(repo_path)

# --- Commit 1: initial values ---
ws = repo.writable_session(branch="main")
meta = json.dumps(
    {
        "zarr_format": 3,
        "node_type": "array",
        "shape": [5],
        "data_type": "float64",
        "chunk_grid": {"name": "regular", "configuration": {"chunk_shape": [5]}},
        "chunk_key_encoding": {
            "name": "default",
            "configuration": {"separator": "/"},
        },
        "fill_value": 0.0,
        "codecs": [{"name": "bytes", "configuration": {"endian": "little"}}],
    }
).encode()
ws.set_metadata("/sensor", meta)
ws.set_chunk(
    "/sensor", (0,), np.array([20.0, 21.0, 22.0, 23.0, 24.0], dtype="<f8").tobytes()
)
snap1 = ws.commit("Morning readings")

# --- Commit 2: updated values ---
ws.set_chunk(
    "/sensor", (0,), np.array([25.0, 26.0, 27.0, 28.0, 29.0], dtype="<f8").tobytes()
)
snap2 = ws.commit("Afternoon readings")

# --- Commit 3: more changes ---
ws.set_chunk(
    "/sensor", (0,), np.array([18.0, 17.0, 16.0, 15.0, 14.0], dtype="<f8").tobytes()
)
_snap3 = ws.commit("Evening readings")

# --- Show the log ---
print_log(repo_path, "main")

# --- Time travel to each snapshot ---
for snap_id, label in [(snap1, "Morning"), (snap2, "Afternoon")]:
    session = repo.readonly_session(snapshot=snap_id)
    chunk = repo.read_chunk(snap_id, "/sensor", (0,))
    data = np.frombuffer(chunk, dtype="<f8")
    print(f"{label} ({snap_id[:8]}...): {data}")

# Current (main)
chunk = repo.read_chunk("main", "/sensor", (0,))
data = np.frombuffer(chunk, dtype="<f8")
print(f"Evening (main):       {data}")

shutil.rmtree(tmp)
