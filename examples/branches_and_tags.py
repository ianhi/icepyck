"""Example: branch and tag management with icepyck.

Demonstrates creating branches, committing to them independently,
creating tags for releases, and viewing the commit ancestry.

Usage::

    uv run examples/branches_and_tags.py
"""

from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path

import numpy as np

import icepyck
from icepyck.ancestry import print_log
from icepyck.verify import print_report, verify_repo

# --- Create a repo with some initial data ---

tmp = Path(tempfile.mkdtemp())
repo_path = tmp / "demo"
repo = icepyck.Repository.init(repo_path)

ws = repo.writable_session(branch="main")
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
ws.commit("Initial temperature data")

# --- Tag the initial release ---

repo.create_tag("v1.0", "main")
print("Created tag 'v1.0' on main")
print(f"Tags: {repo.list_tags()}")

# --- Create a feature branch ---

repo.create_branch("experiment", "main")
print(f"\nBranches: {repo.list_branches()}")

# --- Commit different data to each branch ---
# Re-open after branch/tag ops to pick up new repo state
repo = icepyck.open(repo_path)

# On main: update the temperatures
ws_main = repo.writable_session(branch="main")
ws_main.set_chunk("/temperatures", (0,), (np.arange(10, dtype="<f8") * 2).tobytes())
ws_main.commit("Doubled temperatures on main")

# On experiment: add a new array
ws_exp = repo.writable_session(branch="experiment")
pressure_meta = json.dumps(
    {
        "zarr_format": 3,
        "node_type": "array",
        "shape": [5],
        "data_type": "float32",
        "chunk_grid": {"name": "regular", "configuration": {"chunk_shape": [5]}},
        "chunk_key_encoding": {
            "name": "default",
            "configuration": {"separator": "/"},
        },
        "fill_value": 0.0,
        "codecs": [{"name": "bytes", "configuration": {"endian": "little"}}],
    }
).encode()
ws_exp.set_metadata("/pressure", pressure_meta)
ws_exp.set_chunk(
    "/pressure",
    (0,),
    np.array([1013.0, 1012.5, 1011.0, 1010.5, 1009.0], dtype="<f4").tobytes(),
)
ws_exp.commit("Added pressure array on experiment")

# --- Show the diverged branches ---

print("\n--- Main branch log ---")
print_log(repo_path, "main")

print("--- Experiment branch log ---")
print_log(repo_path, "experiment")

# --- Show what each branch sees ---

repo2 = icepyck.open(repo_path)
main_paths = {n.path for n in repo2.readonly_session(branch="main").list_nodes()}
exp_paths = {n.path for n in repo2.readonly_session(branch="experiment").list_nodes()}
tag_paths = {n.path for n in repo2.readonly_session(tag="v1.0").list_nodes()}

print(f"main nodes: {sorted(main_paths)}")
print(f"experiment nodes: {sorted(exp_paths)}")
print(f"v1.0 tag nodes: {sorted(tag_paths)}")

# --- Verify spec compliance ---

print("\n--- Spec verification ---")
issues = verify_repo(repo_path)
print_report(repo_path, issues)

# --- Cleanup ---
print("\nDelete experiment branch:")
repo2.delete_branch("experiment")
print(f"  Branches after delete: {repo2.list_branches()}")

print("\nDelete tag v1.0:")
repo2.delete_tag("v1.0")
print(f"  Tags after delete: {repo2.list_tags()}")

# Try to recreate deleted tag
try:
    repo2.create_tag("v1.0", "main")
except KeyError as e:
    print(f"  Cannot recreate deleted tag: {e}")

shutil.rmtree(tmp)
print("\nDone!")
