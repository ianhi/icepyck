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
#     "rich",
# ]
# [tool.uv]
# sources = {icepyck = {path = ".."}}
# prerelease = "allow"
# ///
"""Benchmark icepyck vs icechunk: open times, metadata, and data reads."""

import time
import warnings
from pathlib import Path

import numpy as np
import xarray as xr
from rich.console import Console
from rich.table import Table

import icechunk
import icepyck

warnings.filterwarnings("ignore")
console = Console()

REPO_PATH = Path(__file__).resolve().parent.parent / "test-repos" / "xarray-air-temp"

# Ensure repo exists
if not REPO_PATH.exists():
    console.print("[red]Run create_test_data.py and xarray_roundtrip.py first[/red]")
    raise SystemExit(1)


def bench(label: str, fn, n: int = 3) -> float:
    """Run fn n times, return median time in seconds."""
    times = []
    for _ in range(n):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    return sorted(times)[n // 2]


# ── Open repo ──────────────────────────────────────────────

def pyck_open():
    return icepyck.open(REPO_PATH)

def ic_open():
    storage = icechunk.local_filesystem_storage(str(REPO_PATH))
    return icechunk.Repository.open(storage)


# ── Create session ─────────────────────────────────────────

pyck_repo = icepyck.open(REPO_PATH)
ic_storage = icechunk.local_filesystem_storage(str(REPO_PATH))
ic_repo = icechunk.Repository.open(ic_storage)

def pyck_session():
    return pyck_repo.readonly_session(branch="main")

def ic_session():
    return ic_repo.readonly_session(branch="main")


# ── Open dataset with xarray ──────────────────────────────

pyck_sess = pyck_repo.readonly_session(branch="main")
ic_sess = ic_repo.readonly_session(branch="main")

def pyck_open_ds():
    return xr.open_dataset(
        pyck_sess.store, engine="zarr", chunks=None, consolidated=False
    )

def ic_open_ds():
    return xr.open_dataset(
        ic_sess.store, engine="zarr", chunks=None, consolidated=False
    )


# ── Read full array (cold - new dataset each time) ────────

def pyck_read_air():
    ds = xr.open_dataset(
        pyck_sess.store, engine="zarr", chunks=None, consolidated=False
    )
    return ds["air"].values

def ic_read_air():
    ds = xr.open_dataset(
        ic_sess.store, engine="zarr", chunks=None, consolidated=False
    )
    return ds["air"].values


# ── Read single chunk (cold) ─────────────────────────────

import zarr

def pyck_read_chunk():
    root = zarr.open_group(store=pyck_sess.store, mode="r")
    return np.array(root["air"][0:730, 0:13, 0:27])

def ic_read_chunk():
    root = zarr.open_group(store=ic_sess.store, mode="r")
    return np.array(root["air"][0:730, 0:13, 0:27])


# ── Run benchmarks ────────────────────────────────────────

table = Table(title="⛏️🧊 icepyck vs icechunk benchmark (xarray-air-temp)")
table.add_column("Operation", style="bold")
table.add_column("icepyck", justify="right")
table.add_column("icechunk", justify="right")
table.add_column("ratio", justify="right")

benchmarks = [
    ("Open repo", pyck_open, ic_open),
    ("Create session", pyck_session, ic_session),
    ("xr.open_dataset", pyck_open_ds, ic_open_ds),
    ("Read air[:] (31MB)", pyck_read_air, ic_read_air),
    ("Read 1 chunk", pyck_read_chunk, ic_read_chunk),
]

for label, pyck_fn, ic_fn in benchmarks:
    t_pyck = bench(label, pyck_fn)
    t_ic = bench(label, ic_fn)
    ratio = t_pyck / t_ic if t_ic > 0 else float("inf")
    table.add_row(
        label,
        f"{t_pyck*1000:.1f} ms",
        f"{t_ic*1000:.1f} ms",
        f"{ratio:.1f}x",
    )

console.print()
console.print(table)

# Verify correctness
pyck_air = pyck_read_air()
ic_air = ic_read_air()
match = np.array_equal(pyck_air, ic_air)
console.print(f"\n  Data match: {'✅' if match else '❌'}")
console.print(f"  Array shape: {pyck_air.shape}, dtype: {pyck_air.dtype}")
