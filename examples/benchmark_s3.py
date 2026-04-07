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
"""
Benchmark icepyck vs icechunk on S3: count sequential fetches.

Wall clock time on S3 is dominated by network latency, so the metric
that matters is how many sequential round-trips each operation requires.
"""

import time
import warnings
from unittest.mock import patch

import numpy as np
import xarray as xr
import zarr
from rich.console import Console
from rich.table import Table

import icepyck
from icepyck.storage import S3Storage

warnings.filterwarnings("ignore")
console = Console()

URL = "s3://icechunk-public-data/v1/era5_weatherbench2"
GROUP = "1x721x1440"


class CountingS3Storage(S3Storage):
    """S3Storage wrapper that counts read() calls."""

    def __init__(self, *args, **kwargs):  # type: ignore[no-untyped-def]
        super().__init__(*args, **kwargs)
        self.read_count = 0

    def read(self, path: str) -> bytes:
        self.read_count += 1
        return super().read(path)

    def reset(self) -> None:
        self.read_count = 0


# Set up counting storage
storage = CountingS3Storage(URL, anon=True)
repo = icepyck.Repository(storage=storage)

# ── Count fetches per operation ────────────────────────────

table = Table(title="⛏️🧊 icepyck S3 round-trips (ERA5)")
table.add_column("Operation", style="bold")
table.add_column("S3 GETs", justify="right")
table.add_column("Wall time", justify="right")
table.add_column("Notes", style="dim")

# 1. Open repo (already done in constructor, but let's measure fresh)
storage2 = CountingS3Storage(URL, anon=True)
t0 = time.perf_counter()
repo2 = icepyck.Repository(storage=storage2)
t1 = time.perf_counter()
table.add_row(
    "Open repo",
    str(storage2.read_count),
    f"{(t1-t0)*1000:.0f} ms",
    "reads $ROOT/repo",
)

# 2. Create session
storage2.reset()
t0 = time.perf_counter()
session = repo2.readonly_session(branch="main")
t1 = time.perf_counter()
table.add_row(
    "Create session",
    str(storage2.read_count),
    f"{(t1-t0)*1000:.0f} ms",
    "reads snapshot file",
)

# 3. xr.open_dataset (metadata only, no chunk data)
storage2.reset()
t0 = time.perf_counter()
ds = xr.open_dataset(
    session.store, group=GROUP, engine="zarr",
    chunks=None, consolidated=False,
)
t1 = time.perf_counter()
table.add_row(
    "xr.open_dataset (lazy)",
    str(storage2.read_count),
    f"{(t1-t0)*1000:.0f} ms",
    "metadata only, no chunk fetches",
)

# 4. Read coordinate (small array, triggers manifest + chunk loads)
storage2.reset()
t0 = time.perf_counter()
lat = ds["latitude"].values
t1 = time.perf_counter()
table.add_row(
    "Read latitude (721 floats)",
    str(storage2.read_count),
    f"{(t1-t0)*1000:.0f} ms",
    "manifest + chunk",
)

# 5. Read another coordinate (manifest may be cached)
storage2.reset()
t0 = time.perf_counter()
lon = ds["longitude"].values
t1 = time.perf_counter()
table.add_row(
    "Read longitude (1440 floats)",
    str(storage2.read_count),
    f"{(t1-t0)*1000:.0f} ms",
    "chunk only if manifest cached",
)

# 6. Read 1 time slice of 2m_temperature
root = zarr.open_group(store=session.store, mode="r", path=GROUP)
storage2.reset()
t0 = time.perf_counter()
temp_slice = np.array(root["2m_temperature"][0, :, :])
t1 = time.perf_counter()
table.add_row(
    "Read 1 time slice (4MB)",
    str(storage2.read_count),
    f"{(t1-t0)*1000:.0f} ms",
    "1×721×1440 float32",
)

console.print()
console.print(table)
console.print()
console.print(
    "[dim]Each S3 GET adds ~50-150ms of latency. "
    "Sequential GETs are the bottleneck.[/dim]"
)
console.print(
    f"[dim]latitude shape: {lat.shape}, "
    f"longitude shape: {lon.shape}, "
    f"temp slice shape: {temp_slice.shape}[/dim]"
)
