# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "icepyck[s3]",
#     "zarr>=3",
#     "xarray",
#     "numpy",
# ]
# [tool.uv]
# sources = {icepyck = {path = ".."}}
# prerelease = "allow"
# ///
"""
Read ERA5 sample data from icechunk's public S3 repository.

This demonstrates reading a real-world icechunk dataset hosted on S3.
The ERA5 sample dataset is listed at https://icechunk.io/en/latest/sample-datasets/
"""

import icepyck
import xarray as xr

# Open the public ERA5 sample dataset (anonymous S3 access)
repo = icepyck.open("s3://earthmover-sample-data/icechunk/era5-demo", anon=True)
session = repo.readonly_session(branch="main")
ds = xr.open_zarr(session.store)
print(ds)
print(ds["2m_temperature"].sel(time="2020-01-01", method="nearest"))
