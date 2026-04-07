"""Benchmark icepyck vs icechunk on a local repo: write with icechunk, read with both."""

import shutil
import statistics
import tempfile
import time
import warnings
from pathlib import Path

import numpy as np
import xarray as xr
import zarr
from rich.console import Console
from rich.table import Table

import icechunk
import icepyck

warnings.filterwarnings("ignore")
console = Console()

# ── Step 1: Create the dataset and write to a fresh local icechunk repo ────────

console.print("[dim]Loading air_temperature dataset...[/dim]")
ds_original = xr.tutorial.load_dataset("air_temperature")

tmp_dir = tempfile.mkdtemp(prefix="icepyck_bench_")
REPO_PATH = Path(tmp_dir) / "air-temp-repo"

console.print(f"[dim]Writing icechunk repo to {REPO_PATH}...[/dim]")
storage = icechunk.local_filesystem_storage(str(REPO_PATH))
repo_write = icechunk.Repository.create(storage)
session_write = repo_write.writable_session("main")
ds_original.to_zarr(session_write.store, mode="w")
session_write.commit("write air_temperature dataset")
console.print("[dim]Write complete.[/dim]\n")

# ── Step 2: Open the same repo with both implementations ──────────────────────

# Pre-open repos so open-repo benchmark starts cold each iteration
def _pyck_open():
    return icepyck.open(REPO_PATH)

def _ic_open():
    _storage = icechunk.local_filesystem_storage(str(REPO_PATH))
    return icechunk.Repository.open(_storage)


# Pre-built repos for session/dataset/read benchmarks
pyck_repo = icepyck.open(REPO_PATH)
ic_storage = icechunk.local_filesystem_storage(str(REPO_PATH))
ic_repo = icechunk.Repository.open(ic_storage)


def _pyck_session():
    return pyck_repo.readonly_session(branch="main")

def _ic_session():
    return ic_repo.readonly_session(branch="main")


# Sessions used for dataset / read benchmarks
pyck_sess = pyck_repo.readonly_session(branch="main")
ic_sess = ic_repo.readonly_session(branch="main")


def _pyck_open_ds():
    return xr.open_dataset(
        pyck_sess.store, engine="zarr", chunks=None, consolidated=False
    )

def _ic_open_ds():
    return xr.open_dataset(
        ic_sess.store, engine="zarr", chunks=None, consolidated=False
    )


def _pyck_read_air():
    ds = xr.open_dataset(
        pyck_sess.store, engine="zarr", chunks=None, consolidated=False
    )
    return ds["air"].values

def _ic_read_air():
    ds = xr.open_dataset(
        ic_sess.store, engine="zarr", chunks=None, consolidated=False
    )
    return ds["air"].values


def _pyck_read_chunk():
    root = zarr.open_group(store=pyck_sess.store, mode="r")
    return np.array(root["air"][0:730, 0:13, 0:27])

def _ic_read_chunk():
    root = zarr.open_group(store=ic_sess.store, mode="r")
    return np.array(root["air"][0:730, 0:13, 0:27])


# ── Step 3: Benchmark helper ──────────────────────────────────────────────────

def bench(fn, n: int = 3) -> float:
    """Run fn n times, return median time in seconds."""
    times = []
    for _ in range(n):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    return statistics.median(times)


# ── Step 4: Run benchmarks and build table ────────────────────────────────────

benchmarks = [
    ("Open repo",         _pyck_open,       _ic_open),
    ("Create session",    _pyck_session,     _ic_session),
    ("xr.open_dataset",   _pyck_open_ds,     _ic_open_ds),
    ("Read air[:] (31MB)", _pyck_read_air,   _ic_read_air),
    ("Read 1 chunk",      _pyck_read_chunk,  _ic_read_chunk),
]

table = Table(title="🧊⛏️ Local benchmark (air_temperature, 31MB)")
table.add_column("Operation", style="bold")
table.add_column("icepyck", justify="right")
table.add_column("icechunk", justify="right")
table.add_column("ratio", justify="right")

results = []
for label, pyck_fn, ic_fn in benchmarks:
    console.print(f"[dim]  Benchmarking: {label}...[/dim]")
    t_pyck = bench(pyck_fn)
    t_ic = bench(ic_fn)
    ratio = t_pyck / t_ic if t_ic > 0 else float("inf")
    results.append((label, t_pyck, t_ic, ratio))
    color = "green" if ratio <= 1.0 else "yellow" if ratio <= 2.0 else "red"
    table.add_row(
        label,
        f"{t_pyck * 1000:.1f} ms",
        f"{t_ic * 1000:.1f} ms",
        f"[{color}]{ratio:.1f}x[/]",
    )

console.print()
console.print(table)

# ── Step 5: Verify correctness ────────────────────────────────────────────────

pyck_air = _pyck_read_air()
ic_air = _ic_read_air()
match = np.array_equal(pyck_air, ic_air)
console.print(f"\n  Data match: {'✅' if match else '❌'}")
console.print(f"  Array shape: {pyck_air.shape}, dtype: {pyck_air.dtype}")
console.print(
    "\n[dim]ratio = icepyck/icechunk. <1 = icepyck faster. >1 = icechunk faster.[/dim]"
)

# ── Step 6: Cleanup ───────────────────────────────────────────────────────────

shutil.rmtree(tmp_dir)
console.print(f"[dim]Cleaned up {tmp_dir}[/dim]")
