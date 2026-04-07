# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "icepyck",
#     "icechunk>=2.0.0a0",
#     "s3fs",
#     "zarr>=3",
#     "xarray",
#     "numpy",
# ]
# [tool.uv]
# sources = {icepyck = {path = ".."}}
# prerelease = "allow"
# ///
"""Compare icepyck vs icechunk xarray output on ERA5."""

import warnings

import xarray as xr

warnings.filterwarnings("ignore")

URL = "s3://icechunk-public-data/v1/era5_weatherbench2"
GROUP = "1x721x1440"
KW = dict(group=GROUP, engine="zarr", chunks=None, consolidated=False)

# icepyck
print("Opening with icepyck...")
import icepyck

pyck_repo = icepyck.open(URL, anon=True)
pyck_session = pyck_repo.readonly_session(branch="main")
ds_pyck = xr.open_dataset(pyck_session.store, **KW)

# icechunk
print("Opening with icechunk...")
import icechunk

ic_storage = icechunk.s3_storage(
    bucket="icechunk-public-data",
    prefix="v1/era5_weatherbench2",
    region="us-east-1",
    anonymous=True,
)
ic_repo = icechunk.Repository.open(storage=ic_storage)
ic_session = ic_repo.readonly_session("main")
ds_ic = xr.open_dataset(ic_session.store, **KW)

# Compare
print("\n=== icepyck ===")
print(ds_pyck)
print("\n=== icechunk ===")
print(ds_ic)

print("\n=== Comparison ===")
print(f"dims match: {dict(ds_pyck.sizes) == dict(ds_ic.sizes)}")
print(f"coords match: {sorted(ds_pyck.coords) == sorted(ds_ic.coords)}")
print(f"data_vars match: {sorted(ds_pyck.data_vars) == sorted(ds_ic.data_vars)}")
print(f"attrs match: {ds_pyck.attrs == ds_ic.attrs}")

xr.testing.assert_equal(ds_pyck, ds_ic)
print("\nxr.testing.assert_equal: PASSED")
