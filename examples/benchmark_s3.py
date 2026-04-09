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
Benchmark icepyck vs icechunk on S3 ERA5.

Measures wall time and (for icepyck) counts sequential S3 GETs.
On S3, wall time ≈ (sequential GETs) × latency per GET.
"""

import time
import warnings

import numpy as np
import xarray as xr
import zarr
from rich.console import Console
from rich.table import Table

warnings.filterwarnings("ignore")
console = Console()

URL = "s3://icechunk-public-data/v1/era5_weatherbench2"
GROUP = "1x721x1440"


def bench_icepyck() -> list[tuple[str, int, float]]:
    """Benchmark icepyck, returning (label, s3_gets, seconds)."""
    import icepyck
    from icepyck.storage import S3Storage

    class CountingStorage(S3Storage):
        def __init__(self, *a, **kw):  # type: ignore[no-untyped-def]
            super().__init__(*a, **kw)
            self.n = 0

        def read(self, path: str) -> bytes:
            self.n += 1
            return super().read(path)

        def reset(self) -> None:
            self.n = 0

    results = []
    s = CountingStorage(URL, anon=True)

    s.reset()
    t0 = time.perf_counter()
    repo = icepyck.Repository(storage=s)
    session = repo.readonly_session(branch="main")
    results.append(("Open + session", s.n, time.perf_counter() - t0))

    s.reset()
    t0 = time.perf_counter()
    _ds = xr.open_dataset(
        session.store,
        group=GROUP,
        engine="zarr",
        chunks=None,
        consolidated=False,
    )
    results.append(("xr.open_dataset", s.n, time.perf_counter() - t0))

    root = zarr.open_group(store=session.store, mode="r", path=GROUP)
    s.reset()
    t0 = time.perf_counter()
    _ = np.array(root["2m_temperature"][0, :, :])
    results.append(("Read 1 chunk (4MB)", s.n, time.perf_counter() - t0))

    return results


def bench_icechunk() -> list[tuple[str, int, float]]:
    """Benchmark icechunk, returning (label, s3_gets=-1, seconds)."""
    import icechunk

    results = []

    t0 = time.perf_counter()
    storage = icechunk.s3_storage(
        bucket="icechunk-public-data",
        prefix="v1/era5_weatherbench2",
        region="us-east-1",
        anonymous=True,
    )
    repo = icechunk.Repository.open(storage=storage)
    session = repo.readonly_session("main")
    results.append(("Open + session", -1, time.perf_counter() - t0))

    t0 = time.perf_counter()
    _ds = xr.open_dataset(
        session.store,
        group=GROUP,
        engine="zarr",
        chunks=None,
        consolidated=False,
    )
    results.append(("xr.open_dataset", -1, time.perf_counter() - t0))

    root = zarr.open_group(store=session.store, mode="r", path=GROUP)
    t0 = time.perf_counter()
    _ = np.array(root["2m_temperature"][0, :, :])
    results.append(("Read 1 chunk (4MB)", -1, time.perf_counter() - t0))

    return results


# Run icepyck first, then icechunk (separate to avoid loop conflicts)
console.print("[dim]Benchmarking icepyck...[/dim]")
pyck = bench_icepyck()
console.print("[dim]Benchmarking icechunk...[/dim]")
ic = bench_icechunk()

table = Table(title="🧊⛏️ S3 ERA5 benchmark (7TB dataset)")
table.add_column("Operation", style="bold")
table.add_column("icepyck", justify="right")
table.add_column("icechunk", justify="right")
table.add_column("S3 GETs", justify="right", style="dim")

for (label, gets, t_pyck), (_, _, t_ic) in zip(pyck, ic, strict=True):
    ratio = t_pyck / t_ic if t_ic > 0 else 0
    color = "green" if ratio <= 1 else "yellow" if ratio <= 2 else "red"
    table.add_row(
        label,
        f"{t_pyck * 1000:.0f} ms",
        f"{t_ic * 1000:.0f} ms [{color}]({ratio:.1f}x)[/]",
        str(gets) if gets >= 0 else "?",
    )

console.print()
console.print(table)
console.print(
    "\n[dim]ratio = icepyck/icechunk. <1 = icepyck faster. >1 = icechunk faster.[/dim]"
)
