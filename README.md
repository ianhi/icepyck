# ⛏️🧊 icepyck

A read-only Python client for [Icechunk](https://icechunk.io) repositories, built entirely from the [Icechunk specification](https://icechunk.io/en/latest/spec/) without referencing the Rust or Python source code.

## Why

This project tests whether the Icechunk spec is complete enough for an independent implementor. A coding agent was given the spec, flatbuffers schemas, and test data — then asked to build a working reader. The result is a pure-Python Icechunk client that provides a zarr v3 Store, letting you read any Icechunk repository with standard zarr/xarray tooling.

Every moment of confusion and spec gap was logged as the implementation progressed — see [SPEC_GAPS.md](SPEC_GAPS.md) and [CONFUSION_LOG.md](CONFUSION_LOG.md).

## Install

```bash
uv add icepyck
# For S3 support:
uv add "icepyck[s3]"
```

For development:

```bash
git clone <repo-url> && cd icepyck
uv sync
```

## Quick start

```python
import icepyck
import zarr

repo = icepyck.open("test-repos/basic")
session = repo.readonly_session(branch="main")

root = zarr.open_group(store=session.store, mode="r")
temps = root["group1/temperatures"]
print(temps[:])  # numpy array
```

### Read from S3

```python
repo = icepyck.open("s3://earthmover-sample-data/icechunk/era5-demo", anon=True)
session = repo.readonly_session(branch="main")
ds = xr.open_zarr(session.store)
```

### Diff two snapshots

```bash
python -m icepyck.diff_display test-repos/basic main~1 main
```

```
╭─── Snapshot Diff: main~1 -> main ───╮
│ 0 added, 0 removed, 1 modified      │
╰──────────────────────────────────────╯
/
└── group1
    └── [~] temperatures  (1 chunk changed: (0,))
```

## What it can do

- Open local and S3-hosted Icechunk repositories
- List branches, tags, and nodes
- Provide a zarr v3 `Store` for reading data with zarr or xarray
- Diff two snapshots showing added/removed/modified nodes and chunks
- Validate reads against the reference `icechunk` package

## Project structure

| File | Purpose |
|------|---------|
| `icepyck/storage.py` | Storage abstraction (local filesystem, S3) |
| `icepyck/header.py` | Binary header parser (magic, version, compression) |
| `icepyck/crockford.py` | Crockford Base32 encode/decode for object IDs |
| `icepyck/repo.py` | Repo info reader (branches, tags, snapshots) |
| `icepyck/snapshot.py` | Snapshot reader (nodes, manifest refs) |
| `icepyck/manifest.py` | Manifest reader (chunk refs) |
| `icepyck/chunks.py` | Chunk data reader (inline, native, virtual) |
| `icepyck/store.py` | Zarr v3 ReadStore implementation |
| `icepyck/diff.py` | Snapshot diff engine |
| `icepyck/diff_display.py` | Rich terminal diff display |

## Spec conformance outputs

- **[SPEC_GAPS.md](SPEC_GAPS.md)** — formal gaps found in the spec, with severity ratings
- **[CONFUSION_LOG.md](CONFUSION_LOG.md)** — chronological log of every moment of confusion
- **[SPEC_FEEDBACK.md](SPEC_FEEDBACK.md)** — compiled feedback for spec authors
- **[SOURCES.md](SOURCES.md)** — every external source consulted (no icechunk source code)

## Development

```bash
uv sync
uv run create_test_data.py    # generate test repos
uv run pytest tests/ -v       # 109 tests
uv run ruff check icepyck/ --exclude icepyck/generated/
uv run mypy icepyck/ --exclude icepyck/generated/
```
