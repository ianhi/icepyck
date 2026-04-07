# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "icepyck",
#     "zarr>=3",
#     "xarray",
#     "s3fs",
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

NOTE: This requires S3 access (anonymous read). The icepyck S3 support
is not yet implemented — this script shows the intended API and will
work once S3 storage backends are added.
"""

# TODO: icepyck currently only supports local filesystem.
# Once S3 support is added, this script will work:
#
# import icepyck
# import xarray as xr
#
# repo = icepyck.open("s3://earthmover-sample-data/icechunk/era5-demo")
# session = repo.readonly_session(branch="main")
# ds = xr.open_zarr(session.store)
# print(ds)
# print(ds["2m_temperature"].sel(time="2020-01-01", method="nearest"))

print("S3 support not yet implemented. See read_local_repo.py for local usage.")
print("To read ERA5 from S3 with the reference icechunk package:")
print()
print("  import icechunk, zarr, xarray as xr")
print('  storage = icechunk.s3_storage("earthmover-sample-data/icechunk/era5-demo")')
print("  repo = icechunk.Repository.open(storage)")
print('  session = repo.readonly_session(branch="main")')
print("  ds = xr.open_zarr(session.store)")
