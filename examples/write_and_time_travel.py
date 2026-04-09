"""Example: creating a repo, writing zarr data, committing, and time-traveling.

Run with:
    uv run python examples/write_and_time_travel.py
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import zarr

import icepyck

# --------------------------------------------------------------------------
# 1. Create a fresh repository
# --------------------------------------------------------------------------
repo_dir = Path(tempfile.mkdtemp()) / "my-repo"
repo_dir.mkdir()

repo = icepyck.Repository.init(repo_dir)
print(repo)
# Repository(branches=['main'])

# --------------------------------------------------------------------------
# 2. First commit: add temperature observations
# --------------------------------------------------------------------------
session = repo.writable_session(branch="main")
store = session.store

root = zarr.open_group(store=store, mode="a")
temps = root.create_array(
    "observations/temperature",
    shape=(5,),
    chunks=(5,),
    dtype="float64",
    fill_value=float("nan"),
)
temps[:] = np.array([18.2, 19.5, 21.0, 20.3, 17.8])

pressure = root.create_array(
    "observations/pressure",
    shape=(5,),
    chunks=(5,),
    dtype="float32",
    fill_value=0.0,
)
pressure[:] = np.array([1013.2, 1012.8, 1011.5, 1010.9, 1012.1], dtype="float32")

snap_v1 = session.commit("Initial weather observations")
print(f"\nCommit 1: {session.snapshot_id}")
print(session)

# --------------------------------------------------------------------------
# 3. Second commit: afternoon readings — all values change
# --------------------------------------------------------------------------
root["observations/temperature"][:] = np.array([22.1, 23.4, 25.0, 24.2, 21.9])
root["observations/pressure"][:] = np.array(
    [1010.5, 1009.8, 1008.2, 1007.6, 1009.0], dtype="float32"
)

snap_v2 = session.commit("Afternoon readings")
print(f"\nCommit 2: {session.snapshot_id}")

# --------------------------------------------------------------------------
# 4. Third commit: evening readings + add humidity
# --------------------------------------------------------------------------
root["observations/temperature"][:] = np.array([19.0, 18.5, 17.2, 16.8, 18.1])
root["observations/pressure"][:] = np.array(
    [1011.0, 1011.5, 1012.3, 1012.8, 1011.9], dtype="float32"
)

humidity = root.create_array(
    "observations/humidity",
    shape=(5,),
    chunks=(5,),
    dtype="float32",
    fill_value=0.0,
)
humidity[:] = np.array([65.0, 70.2, 55.8, 60.1, 72.5], dtype="float32")

snap_v3 = session.commit("Evening readings + humidity")
print(f"\nCommit 3: {session.snapshot_id}")

# --------------------------------------------------------------------------
# 5. Time travel: read data from each snapshot
# --------------------------------------------------------------------------
print("\n" + "=" * 60)
print("TIME TRAVEL")
print("=" * 60)

# Re-open to get a clean view
repo = icepyck.open(repo_dir)

# commit() returns Crockford Base32 strings like "RF238TWZTXGD49BDPXWG".
# You can use the full ID, or any unique prefix — just like git short SHAs.
print(f"\nSnapshot IDs: v1={snap_v1}, v2={snap_v2}, v3={snap_v3}")

# --- Latest (v3): has temperature, pressure, AND humidity ---
latest = repo.readonly_session(branch="main")
print(f"\nLatest (main): {latest}")
root_latest = zarr.open_group(store=latest.store, mode="r")
print(f"  temperature = {root_latest['observations/temperature'][:]}")
print(f"  pressure    = {root_latest['observations/pressure'][:]}")
print(f"  humidity    = {root_latest['observations/humidity'][:]}")

# --- v2: use full Crockford ID ---
v2 = repo.readonly_session(snapshot=snap_v2)
print(f"\nv2 (full ID): {v2}")
root_v2 = zarr.open_group(store=v2.store, mode="r")
print(f"  temperature = {root_v2['observations/temperature'][:]}")
print(f"  pressure    = {root_v2['observations/pressure'][:]}")
arrays_v2 = [n.path for n in v2.list_nodes() if n.node_type == "array"]
has_humidity = any("humidity" in a for a in arrays_v2)
print(f"  humidity?   = {has_humidity}")

# --- v1: use a SHORT prefix (as few chars as needed to be unique) ---
short = snap_v1[:6]
print(f"\nv1 (short prefix '{short}'): ", end="")
v1 = repo.readonly_session(snapshot=short)
print(v1)
root_v1 = zarr.open_group(store=v1.store, mode="r")
print(f"  temperature = {root_v1['observations/temperature'][:]}")
print(f"  pressure    = {root_v1['observations/pressure'][:]}")
arrays_v1 = [n.path for n in v1.list_nodes() if n.node_type == "array"]
has_humidity = any("humidity" in a for a in arrays_v1)
print(f"  humidity?   = {has_humidity}")

print("\n" + "=" * 60)
print("Each snapshot is immutable — old data is never overwritten.")
print("=" * 60)

# --------------------------------------------------------------------------
# 6. Show the full commit ancestry
# --------------------------------------------------------------------------
print("\n" + "=" * 60)
print("COMMIT LOG (main)")
print("=" * 60)

for entry in repo.log("main"):
    parent = entry["parent"]
    parent_short = parent[:8] + "..." if parent else "(root)"
    print(f"\n  {entry['id']}")
    print(f"  parent: {parent_short}")
    print(f"  {entry['time']}  {entry['message']}")
