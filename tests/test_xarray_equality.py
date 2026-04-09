"""xr.testing.assert_equal comparisons between icepyck and icechunk stores.

Verifies that data read via icepyck's IcechunkReadStore is numerically and
structurally identical to data read via the reference icechunk Store, using
xr.testing.assert_equal as the comparison oracle.

Tests cover:
  - A simple flat dataset created in a tmp_path repo
  - The local xarray-air-temp test repo (skipped if absent)
  - A grouped dataset (arrays nested under a subgroup, opened with group=)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

icechunk = pytest.importorskip("icechunk")
xr = pytest.importorskip("xarray")

import icepyck

TEST_REPOS_DIR = Path(__file__).parent.parent / "test-repos"
XARRAY_AIR_TEMP_REPO = TEST_REPOS_DIR / "xarray-air-temp"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _open_pyck(path: Path, **kwargs):
    repo = icepyck.open(path)
    session = repo.readonly_session(branch="main")
    return xr.open_dataset(session.store, engine="zarr", consolidated=False, **kwargs)


def _open_ic(path: Path, **kwargs):
    storage = icechunk.local_filesystem_storage(str(path))
    repo = icechunk.Repository.open(storage)
    session = repo.readonly_session(branch="main")
    return xr.open_dataset(session.store, engine="zarr", consolidated=False, **kwargs)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def flat_repo(tmp_path):
    """Create a local icechunk repo containing a simple flat xarray Dataset."""
    path = tmp_path / "flat"
    storage = icechunk.local_filesystem_storage(str(path))
    repo = icechunk.Repository.create(storage)
    session = repo.writable_session("main")

    rng = np.random.default_rng(0)
    ds = xr.Dataset(
        {
            "temperature": xr.DataArray(
                rng.standard_normal((10, 5)).astype("f4"),
                dims=["time", "lat"],
            ),
            "pressure": xr.DataArray(
                rng.standard_normal((10, 5)).astype("f4"),
                dims=["time", "lat"],
            ),
        },
        coords={
            "time": np.arange(10, dtype="i8"),
            "lat": np.linspace(-90, 90, 5, dtype="f4"),
        },
    )
    ds.to_zarr(session.store, mode="w")
    session.commit("flat dataset")

    return path


@pytest.fixture
def grouped_repo(tmp_path):
    """Create a local icechunk repo with arrays nested under a subgroup."""
    path = tmp_path / "grouped"
    storage = icechunk.local_filesystem_storage(str(path))
    repo = icechunk.Repository.create(storage)
    session = repo.writable_session("main")

    import zarr

    rng = np.random.default_rng(42)
    root = zarr.group(session.store)
    g = root.create_group("mygroup")

    g.create_array(
        "time", shape=(10,), dtype="i8", chunks=(10,), dimension_names=["time"]
    )
    g.create_array("lat", shape=(5,), dtype="f4", chunks=(5,), dimension_names=["lat"])
    g.create_array("lon", shape=(8,), dtype="f4", chunks=(8,), dimension_names=["lon"])
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

    g["time"][:] = np.arange(10, dtype="i8")
    g["lat"][:] = np.linspace(-90, 90, 5).astype("f4")
    g["lon"][:] = np.linspace(0, 360, 8).astype("f4")
    g["temperature"][:] = rng.standard_normal((10, 5, 8)).astype("f4")
    g["pressure"][:] = rng.standard_normal((10, 5, 8)).astype("f4")

    session.commit("grouped dataset")

    return path


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_flat_dataset_equal(flat_repo):
    """icepyck and icechunk return identical flat datasets."""
    ds_pyck = _open_pyck(flat_repo)
    ds_ic = _open_ic(flat_repo)
    xr.testing.assert_equal(ds_pyck, ds_ic)


@pytest.mark.skipif(
    not XARRAY_AIR_TEMP_REPO.exists(),
    reason="xarray-air-temp test repo not available",
)
def test_xarray_air_temp_equal():
    """icepyck and icechunk return identical data for the xarray-air-temp repo."""
    ds_pyck = _open_pyck(XARRAY_AIR_TEMP_REPO)
    ds_ic = _open_ic(XARRAY_AIR_TEMP_REPO)
    xr.testing.assert_equal(ds_pyck, ds_ic)


def test_grouped_dataset_equal(grouped_repo):
    """icepyck and icechunk return identical datasets when using group=."""
    ds_pyck = _open_pyck(grouped_repo, group="mygroup")
    ds_ic = _open_ic(grouped_repo, group="mygroup")
    xr.testing.assert_equal(ds_pyck, ds_ic)
