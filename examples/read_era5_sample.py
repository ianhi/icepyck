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
"""
Read from an S3-hosted icechunk repository.

NOTE: The public sample datasets at earthmover-sample-data are V1 repos.
icepyck only supports V2. This example uses a local repo but demonstrates
the S3 API. To read from a real V2 S3 repo, replace the URL below.
"""

from pathlib import Path

import icepyck
import zarr

# Demonstrate the S3 API with a local repo (since public samples are V1)
ROOT = Path(__file__).resolve().parent.parent
repo = icepyck.open(ROOT / "test-repos" / "native-chunks")
session = repo.readonly_session(branch="main")

root = zarr.open_group(store=session.store, mode="r")
print("Arrays:", list(root.keys()))
arr = root["data"]
print(f"data: dtype={arr.dtype}, shape={arr.shape}")
print(f"values: {arr[:]}")

# To read from S3, you would do:
# repo = icepyck.open("s3://your-bucket/path/to/v2-repo", anon=True)
# session = repo.readonly_session(branch="main")
# ds = xr.open_zarr(session.store)
