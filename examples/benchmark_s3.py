# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "icepyck",
#     "icechunk>=2.0.0a0",
#     "s3fs",
#     "zarr>=3",
#     "xarray",
#     "numpy",
#     "rich",
# ]
# [tool.uv]
# sources = {icepyck = {path = ".."}}
# prerelease = "allow"
# ///
"""Benchmark icepyck vs icechunk on S3 ERA5 data."""

import time
import warnings

import numpy as np
import xarray as xr
from rich.console import Console
from rich.table import Table

import icechunk
import icepyck

warnings.filterwarnings("ignore")
console = Console()

URL = "s3://icechunk-public-data/v1/era5_weatherbench2"
GROUP = "1x721x1440"


def bench(fn, n: int = 1) -> float:
    """Run fn n times, return median time."""
    times = []
    for _ in range(n):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    return sorted(times)[n // 2]


# ── Open repo ──────────────────────────────────────────────

def pyck_open():
    return icepyck.open(URL, anon=True)

def ic_open():
    storage = icechunk.s3_storage(
        bucket="icechunk-public-data",
        prefix="v1/era5_weatherbench2",
        region="us-east-1",
        anonymous=True,
    )
    return icechunk.Repository.open(storage=storage)


# ── Create session ─────────────────────────────────────────

pyck_repo = pyck_open()
ic_repo = ic_open()

def pyck_session():
    return pyck_repo.readonly_session(branch="main")

def ic_session():
    return ic_repo.readonly_session(branch="main")


# ── xr.open_dataset ───────────────────────────────────────

pyck_sess = pyck_session()
ic_sess = ic_session()

def pyck_open_ds():
    return xr.open_dataset(
        pyck_sess.store, group=GROUP, engine="zarr",
        chunks=None, consolidated=False,
    )

def ic_open_ds():
    return xr.open_dataset(
        ic_sess.store, group=GROUP, engine="zarr",
        chunks=None, consolidated=False,
    )


# ── Read a coordinate array (small, tests S3 chunk fetch) ─

def pyck_read_lat():
    ds = pyck_open_ds()
    return ds["latitude"].values

def ic_read_lat():
    ds = ic_open_ds()
    return ds["latitude"].values


# ── Read a single time slice (1 x 721 x 1440 = 4MB) ──────

import zarr

def pyck_read_slice():
    root = zarr.open_group(store=pyck_sess.store, mode="r", path=GROUP)
    return np.array(root["2m_temperature"][0, :, :])

def ic_read_slice():
    root = zarr.open_group(store=ic_sess.store, mode="r", path=GROUP)
    return np.array(root["2m_temperature"][0, :, :])


# ── Run benchmarks ────────────────────────────────────────

table = Table(title="⛏️🧊 icepyck vs icechunk — S3 ERA5 benchmark")
table.add_column("Operation", style="bold")
table.add_column("icepyck", justify="right")
table.add_column("icechunk", justify="right")
table.add_column("ratio", justify="right")

benchmarks = [
    ("Open repo (S3)", pyck_open, ic_open, 1),
    ("Create session (S3)", pyck_session, ic_session, 1),
    ("xr.open_dataset (S3)", pyck_open_ds, ic_open_ds, 1),
    ("Read latitude (721 floats)", pyck_read_lat, ic_read_lat, 1),
    ("Read 1 time slice (4MB)", pyck_read_slice, ic_read_slice, 1),
]

for label, pyck_fn, ic_fn, n in benchmarks:
    console.print(f"  benchmarking: {label}...", style="dim")
    t_pyck = bench(pyck_fn, n)
    t_ic = bench(ic_fn, n)
    ratio = t_pyck / t_ic if t_ic > 0 else float("inf")
    table.add_row(
        label,
        f"{t_pyck*1000:.0f} ms",
        f"{t_ic*1000:.0f} ms",
        f"[{'red' if ratio > 1.5 else 'yellow' if ratio > 1 else 'green'}]{ratio:.1f}x[/]",
    )

console.print()
console.print(table)
console.print()
console.print("[dim]ratio < 1 = icepyck faster, > 1 = icechunk faster[/dim]")
