# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "icepyck",
#     "s3fs",
#     "zarr>=3",
#     "xarray",
#     "numpy",
# ]
# [tool.uv]
# sources = {icepyck = {path = ".."}}
# prerelease = "allow"
# ///
"""Read ERA5 WeatherBench2 from S3 using icepyck."""

import warnings

import xarray as xr

import icepyck

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

URL = "s3://icechunk-public-data/v1/era5_weatherbench2"

repo = icepyck.open(URL, anon=True)
session = repo.readonly_session(branch="main")
ds = xr.open_dataset(
    session.store,
    group="1x721x1440",
    engine="zarr",
    chunks=None,
    consolidated=False,
)
print(ds)
