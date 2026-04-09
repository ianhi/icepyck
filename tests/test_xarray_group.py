"""Regression test: xr.open_dataset with group= finds all arrays.

Verifies that icepyck's IcechunkReadStore exposes the same data_vars and
coords as the reference icechunk Store when xr.open_dataset is called with
a group= parameter pointing at a subgroup containing multiple arrays.

Background: on the ERA5 S3 dataset (group="1x721x1440") icepyck was
observed to return only 1 of 6 expected arrays.  This test reproduces the
structure locally so that any regression in list_dir / _node_dir_entries
handling of grouped stores is caught immediately.
"""

from __future__ import annotations

import numpy as np
import pytest

icechunk = pytest.importorskip("icechunk")
xr = pytest.importorskip("xarray")


@pytest.fixture
def grouped_repo(tmp_path):
    """Create a local icechunk repo with a nested group structure similar to ERA5."""
    path = tmp_path / "grouped"
    storage = icechunk.local_filesystem_storage(str(path))
    repo = icechunk.Repository.create(storage)
    session = repo.writable_session("main")
    store = session.store

    import zarr

    root = zarr.group(store)
    g = root.create_group("mygroup")

    # Coordinate arrays (1-D)
    g.create_array(
        "time", shape=(10,), dtype="i8", chunks=(10,), dimension_names=["time"]
    )
    g.create_array("lat", shape=(5,), dtype="f4", chunks=(5,), dimension_names=["lat"])
    g.create_array("lon", shape=(8,), dtype="f4", chunks=(8,), dimension_names=["lon"])

    # Data arrays (3-D, referencing coord dimensions)
    g.create_array(
        "temperature",
        shape=(10, 5, 8),
        dtype="f4",
        chunks=(5, 5, 8),
        dimension_names=["time", "lat", "lon"],
    )
    g.create_array(
        "pressure",
        shape=(10, 5, 8),
        dtype="f4",
        chunks=(5, 5, 8),
        dimension_names=["time", "lat", "lon"],
    )
    g.create_array(
        "humidity",
        shape=(10, 5, 8),
        dtype="f4",
        chunks=(5, 5, 8),
        dimension_names=["time", "lat", "lon"],
    )

    # Write data
    rng = np.random.default_rng(42)
    g["temperature"][:] = rng.standard_normal((10, 5, 8)).astype("f4")
    g["pressure"][:] = rng.standard_normal((10, 5, 8)).astype("f4")
    g["humidity"][:] = rng.standard_normal((10, 5, 8)).astype("f4")
    g["time"][:] = np.arange(10, dtype="i8")
    g["lat"][:] = np.linspace(-90, 90, 5).astype("f4")
    g["lon"][:] = np.linspace(0, 360, 8).astype("f4")

    session.commit("grouped data")

    return path


def test_xarray_group_finds_all_arrays(grouped_repo):
    """icepyck store should expose the same arrays as icechunk when using group=."""
    import icepyck

    # Open with icepyck
    pyck_repo = icepyck.open(grouped_repo)
    pyck_session = pyck_repo.readonly_session(branch="main")
    ds_pyck = xr.open_dataset(
        pyck_session.store,
        group="mygroup",
        engine="zarr",
        consolidated=False,
    )

    # Open with icechunk (reference implementation)
    ic_storage = icechunk.local_filesystem_storage(str(grouped_repo))
    ic_repo = icechunk.Repository.open(ic_storage)
    ic_session = ic_repo.readonly_session(branch="main")
    ds_ic = xr.open_dataset(
        ic_session.store,
        group="mygroup",
        engine="zarr",
        consolidated=False,
    )

    assert sorted(ds_pyck.data_vars) == sorted(ds_ic.data_vars), (
        f"data_vars mismatch: icepyck={sorted(ds_pyck.data_vars)!r}, "
        f"icechunk={sorted(ds_ic.data_vars)!r}"
    )
    assert sorted(ds_pyck.coords) == sorted(ds_ic.coords), (
        f"coords mismatch: icepyck={sorted(ds_pyck.coords)!r}, "
        f"icechunk={sorted(ds_ic.coords)!r}"
    )
