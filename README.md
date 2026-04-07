# 🧊⛏️ icepyck

**Can you build an Icechunk client from just the spec?**

This project answers that question. A coding agent (Claude) was given the [Icechunk specification](https://icechunk.io/en/latest/spec/), the flatbuffers schemas, and test data — no Rust or Python source code — and asked to build a working client. The result is a pure-Python read-only Icechunk client that provides a zarr v3 `Store`, so you can read any Icechunk repository with standard zarr/xarray tooling.

The real output isn't the library. It's the log of every spec gap and confusion moment found along the way.

## Why

The Icechunk project needs to know whether the spec is complete enough for independent implementation. This is a concrete test: if a capable agent with access to only the spec can ship a passing test suite, the spec is probably sufficient. If it can't, the gaps tell you exactly what's missing.

109 tests pass. Reads validate byte-for-byte against the reference `icechunk` package.

## Spec conformance outputs

These are the most useful artifacts in this repo:

- **[SPEC_GAPS.md](SPEC_GAPS.md)** — formal gaps found in the spec, with severity ratings (includes cases where the spec's enum values disagreed with the reference implementation)
- **[CONFUSION_LOG.md](CONFUSION_LOG.md)** — chronological log of every moment of confusion during implementation, with time-to-resolve
- **[SPEC_FEEDBACK.md](SPEC_FEEDBACK.md)** — compiled actionable feedback for spec authors
- **[SOURCES.md](SOURCES.md)** — every external source consulted (no icechunk source code)

## Install

```bash
uv add icepyck
```

For development:

```bash
git clone https://github.com/ianhi/icepyck && cd icepyck
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
import icepyck
import xarray as xr

repo = icepyck.open("s3://earthmover-sample-data/icechunk/era5-demo", anon=True)
session = repo.readonly_session(branch="main")
ds = xr.open_zarr(session.store, consolidated=False)
```

### CLI

```bash
uv tool install icepyck       # or: uvx icepyck ...

icepyck dig  test-repos/xarray-air-temp     # interactive TUI repo explorer
icepyck tree test-repos/basic               # node tree with shapes
icepyck diff test-repos/basic main~1 main   # diff two snapshots
icepyck info test-repos/basic               # branches, tags, snapshots
icepyck log  test-repos/basic               # snapshot history
icepyck show test-repos/basic main          # snapshot detail
```

```
🧊⛏️ test-repos/xarray-air-temp @ main
├── 📊 air int16 [2920, 25, 53] chunks=[730, 13, 27]
├── 📊 lat float32 [25] chunks=[25]
├── 📊 lon float32 [53] chunks=[53]
└── 📊 time float32 [2920] chunks=[2920]
```

`icepyck dig` opens a [Textual](https://textual.textualize.io/) TUI for interactive repo exploration, including a diff viewer between snapshots.

## What it can do

- Open local and S3-hosted Icechunk repositories
- Provide a zarr v3 `Store` for reading data with zarr or xarray
- List branches, tags, snapshots, and nodes
- Diff two snapshots showing added/removed/modified nodes and chunks
- Resolve short snapshot IDs (like git short hashes)
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

## Development

```bash
uv sync
uv run python create_test_data.py    # generate test repos
uv run pytest tests/ -v              # 109 tests
uv run ruff check icepyck/ --exclude icepyck/generated/
uv run mypy icepyck/ --exclude icepyck/generated/
```
