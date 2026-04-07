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
"""Compare icepyck vs icechunk reading ERA5 from S3."""

import icechunk
import xarray as xr

import icepyck

URL = "s3://icechunk-public-data/v1/era5_weatherbench2"
GROUP = "1x721x1440"
OPEN_KW = dict(group=GROUP, engine="zarr", chunks=None, consolidated=False)

# --- icepyck ---
print("Opening with icepyck...")
pyck_repo = icepyck.open(URL, anon=True)
pyck_session = pyck_repo.readonly_session(branch="main")
ds_pyck = xr.open_dataset(pyck_session.store, **OPEN_KW)
print(f"icepyck: {dict(ds_pyck.sizes)}")
print(f"  coords: {sorted(ds_pyck.coords)}")
print(f"  data_vars: {sorted(ds_pyck.data_vars)}")

# --- icechunk ---
print("\nOpening with icechunk...")
storage = icechunk.s3_storage(
    bucket="icechunk-public-data",
    prefix="v1/era5_weatherbench2",
    region="us-east-1",
    anonymous=True,
)
ic_repo = icechunk.Repository.open(storage=storage)
ic_session = ic_repo.readonly_session("main")
ds_ic = xr.open_dataset(ic_session.store, **OPEN_KW)
print(f"icechunk: {dict(ds_ic.sizes)}")
print(f"  coords: {sorted(ds_ic.coords)}")
print(f"  data_vars: {sorted(ds_ic.data_vars)}")

# --- Compare ---
print("\n--- Comparison ---")
assert dict(ds_pyck.sizes) == dict(ds_ic.sizes), f"dims differ: {ds_pyck.sizes} vs {ds_ic.sizes}"
assert sorted(ds_pyck.coords) == sorted(ds_ic.coords), f"coords differ"
assert sorted(ds_pyck.data_vars) == sorted(ds_ic.data_vars), f"data_vars differ"
assert ds_pyck.attrs == ds_ic.attrs, f"attrs differ"

print("dims: MATCH")
print("coords: MATCH")
print("data_vars: MATCH")
print("attrs: MATCH")
print("\nAll metadata matches!")
