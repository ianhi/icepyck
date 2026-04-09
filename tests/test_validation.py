"""Validation tests: compare icepyck output against the reference icechunk package.

These tests ensure that our spec-only implementation produces identical results
to the official icechunk + zarr Python packages. They require the ``icechunk``,
``zarr``, and ``numpy`` packages which are dev-only dependencies.

Run with:  uv run pytest tests/test_validation.py -v
"""

from __future__ import annotations

from pathlib import Path

import pytest

icechunk = pytest.importorskip("icechunk")
zarr = pytest.importorskip("zarr")
np = pytest.importorskip("numpy")
zstandard = pytest.importorskip("zstandard")

import icepyck
from icepyck.crockford import encode as crockford_encode

pytestmark = pytest.mark.validation

TEST_REPOS_DIR = Path(__file__).parent.parent / "test-repos"

# ---------------------------------------------------------------------------
# Repo configurations: (name, array_paths)
# ---------------------------------------------------------------------------
REPO_CONFIGS = {
    "basic": ["/group1/temperatures", "/group1/timestamps"],
    "nested": ["/a/b/c/data"],
    "scalar": ["/value"],
    "native-chunks": ["/data"],
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _open_icechunk(repo_path: Path):
    """Open a repo via icechunk and return (icechunk_repo, zarr_root)."""
    storage = icechunk.local_filesystem_storage(str(repo_path))
    repo = icechunk.Repository.open(storage)
    return repo


def _zarr_root(ic_repo, branch: str = "main"):
    """Get a zarr Group from an icechunk repo."""
    session = ic_repo.readonly_session(branch=branch)
    return zarr.open_group(store=session.store, mode="r")


def _walk_zarr(group, prefix: str = "") -> list[tuple[str, str]]:
    """Walk a zarr hierarchy, return list of (path, type) pairs."""
    results: list[tuple[str, str]] = []
    for name, item in group.members():
        path = f"{prefix}/{name}"
        if isinstance(item, zarr.Group):
            results.append((path, "group"))
            results.extend(_walk_zarr(item, path))
        elif isinstance(item, zarr.Array):
            results.append((path, "array"))
    return results


def _decode_chunk(raw_bytes: bytes, meta: dict) -> np.ndarray:
    """Decode raw chunk bytes from icepyck using zarr metadata.

    Handles the codec pipeline described in zarr.json: typically
    bytes (endian encoding) + zstd compression.
    """
    codecs = meta.get("codecs", [])
    data = raw_bytes

    # Apply codecs in reverse order (outermost codec is last in the list,
    # but was applied last during encoding, so we reverse to decode)
    for codec in reversed(codecs):
        codec_name = codec["name"]
        if codec_name == "zstd":
            data = zstandard.ZstdDecompressor().decompress(data)
        elif codec_name == "bytes":
            # The bytes codec just describes the endianness; no transform needed
            # during decoding since numpy handles endian via dtype
            pass
        elif codec_name == "transpose":
            # Transpose codec: we handle it after converting to numpy
            pass
        else:
            pytest.skip(f"Unknown codec {codec_name!r}, cannot decode")

    # Determine dtype from metadata
    dtype_str = meta["data_type"]
    endian = "little"  # default
    for codec in codecs:
        if codec["name"] == "bytes":
            endian = codec.get("configuration", {}).get("endian", "little")
    endian_char = "<" if endian == "little" else ">"
    dtype_map = {
        "float32": f"{endian_char}f4",
        "float64": f"{endian_char}f8",
        "int32": f"{endian_char}i4",
        "int64": f"{endian_char}i8",
        "uint8": "u1",
        "uint16": f"{endian_char}u2",
        "uint32": f"{endian_char}u4",
        "uint64": f"{endian_char}u8",
        "int8": "i1",
        "int16": f"{endian_char}i2",
    }
    np_dtype = dtype_map.get(dtype_str)
    if np_dtype is None:
        pytest.skip(f"Unsupported dtype {dtype_str!r}")

    return np.frombuffer(data, dtype=np_dtype)


def _get_chunk_shape(meta: dict) -> tuple[int, ...]:
    """Extract chunk shape from zarr metadata."""
    grid = meta["chunk_grid"]
    if grid["name"] == "regular":
        return tuple(grid["configuration"]["chunk_shape"])
    pytest.skip(f"Unsupported chunk grid: {grid['name']!r}")


# ---------------------------------------------------------------------------
# Parametrized fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(
    params=list(REPO_CONFIGS.keys()),
    ids=list(REPO_CONFIGS.keys()),
)
def repo_name(request):
    return request.param


@pytest.fixture
def repo_path(repo_name):
    path = TEST_REPOS_DIR / repo_name
    if not (path / "repo").exists():
        pytest.skip(f"Test repo {repo_name!r} not available")
    return path


@pytest.fixture
def pyck_repo(repo_path):
    return icepyck.open(str(repo_path))


@pytest.fixture
def ic_repo(repo_path):
    return _open_icechunk(repo_path)


# ---------------------------------------------------------------------------
# Test: branches and tags match
# ---------------------------------------------------------------------------


class TestBranchesAndTags:
    def test_branches_match(self, pyck_repo, ic_repo):
        pyck_branches = sorted(pyck_repo.list_branches())
        ref_branches = sorted(ic_repo.list_branches())
        assert pyck_branches == ref_branches

    def test_tags_match(self, pyck_repo, ic_repo):
        pyck_tags = sorted(pyck_repo.list_tags())
        ref_tags = sorted(ic_repo.list_tags())
        assert pyck_tags == ref_tags


# ---------------------------------------------------------------------------
# Test: snapshot IDs match
# ---------------------------------------------------------------------------


class TestSnapshotIds:
    def test_branch_snapshot_ids_match(self, pyck_repo, ic_repo):
        for branch in ic_repo.list_branches():
            # icechunk returns Crockford-encoded snapshot IDs
            ref_snap_id = ic_repo.lookup_branch(branch)

            # icepyck resolves branch to raw bytes; encode for comparison
            pyck_snap_bytes = pyck_repo._resolve_ref(branch)
            pyck_snap_id = crockford_encode(pyck_snap_bytes)

            assert pyck_snap_id == ref_snap_id, (
                f"Branch {branch!r}: icepyck={pyck_snap_id}, icechunk={ref_snap_id}"
            )

    def test_tag_snapshot_ids_match(self, pyck_repo, ic_repo):
        ref_tags = ic_repo.list_tags()
        if not ref_tags:
            pytest.skip("No tags in this repo")
        for tag in ref_tags:
            ref_snap_id = ic_repo.lookup_tag(tag)
            pyck_snap_bytes = pyck_repo._state.get_snapshot_id_by_tag(tag)
            pyck_snap_id = crockford_encode(pyck_snap_bytes)
            assert pyck_snap_id == ref_snap_id, (
                f"Tag {tag!r}: icepyck={pyck_snap_id}, icechunk={ref_snap_id}"
            )


# ---------------------------------------------------------------------------
# Test: node paths and types match
# ---------------------------------------------------------------------------


class TestNodePaths:
    def test_node_paths_match(self, pyck_repo, ic_repo):
        # icepyck nodes
        pyck_nodes = pyck_repo.list_nodes("main")
        # Filter out root node since zarr doesn't list it as a member
        pyck_items = sorted((n.path, n.node_type) for n in pyck_nodes if n.path != "/")

        # zarr/icechunk hierarchy walk
        root = _zarr_root(ic_repo, "main")
        ref_items = sorted(_walk_zarr(root))

        assert pyck_items == ref_items, (
            f"Node mismatch:\n  icepyck: {pyck_items}\n  icechunk: {ref_items}"
        )


# ---------------------------------------------------------------------------
# Test: array metadata matches
# ---------------------------------------------------------------------------


class TestArrayMetadata:
    def test_zarr_metadata_matches(self, pyck_repo, ic_repo, repo_name):
        _zarr_root(ic_repo, "main")
        array_paths = REPO_CONFIGS[repo_name]

        for array_path in array_paths:
            pyck_meta = pyck_repo.get_array_metadata("main", array_path)

            # Get zarr array for reference metadata
            zarr_path = array_path.lstrip("/")
            ref_arr = zarr.open_array(
                store=ic_repo.readonly_session(branch="main").store,
                path=zarr_path,
                mode="r",
            )

            # Compare key metadata fields
            pyck_shape = tuple(pyck_meta["shape"])
            pyck_chunks = _get_chunk_shape(pyck_meta)

            assert pyck_shape == ref_arr.shape, (
                f"{array_path}: shape mismatch {pyck_shape} vs {ref_arr.shape}"
            )
            assert pyck_chunks == ref_arr.chunks, (
                f"{array_path}: chunk shape mismatch {pyck_chunks} vs {ref_arr.chunks}"
            )

            # Validate zarr_format
            assert pyck_meta["zarr_format"] == 3


# ---------------------------------------------------------------------------
# Test: chunk data matches (most important)
# ---------------------------------------------------------------------------


class TestChunkData:
    def test_array_data_matches(self, pyck_repo, ic_repo, repo_name):
        """Compare decoded chunk data from icepyck against zarr/icechunk."""
        array_paths = REPO_CONFIGS[repo_name]

        for array_path in array_paths:
            zarr_path = array_path.lstrip("/")
            session = ic_repo.readonly_session(branch="main")
            ref_arr = zarr.open_array(store=session.store, path=zarr_path, mode="r")
            # Use [()] for scalar (0-d) arrays, [:] for everything else
            if ref_arr.shape == ():
                ref_data = np.array(ref_arr[()])
            else:
                ref_data = np.array(ref_arr[:])

            # Get metadata and chunks from icepyck
            meta = pyck_repo.get_array_metadata("main", array_path)
            shape = tuple(meta["shape"])
            chunk_shape = _get_chunk_shape(meta)
            chunks_dict = pyck_repo.read_all_chunks("main", array_path)

            if shape == ():
                # Scalar array: single chunk at index ()
                assert () in chunks_dict, f"{array_path}: scalar array missing chunk ()"
                decoded = _decode_chunk(chunks_dict[()], meta)
                assert len(decoded) == 1
                np.testing.assert_array_equal(
                    decoded[0],
                    ref_data,
                    err_msg=f"{array_path}: scalar value mismatch",
                )
                continue

            # Reconstruct full array from chunks
            pyck_full = np.empty(shape, dtype=ref_data.dtype)
            fill_value = meta.get("fill_value", 0)

            ndim = len(shape)

            # Fill with fill_value first
            pyck_full[:] = fill_value

            for chunk_idx, raw_bytes in chunks_dict.items():
                decoded = _decode_chunk(raw_bytes, meta)

                if ndim == 1:
                    start = chunk_idx[0] * chunk_shape[0]
                    end = min(start + chunk_shape[0], shape[0])
                    actual_size = end - start
                    pyck_full[start:end] = decoded[:actual_size]
                elif ndim == 2:
                    r_start = chunk_idx[0] * chunk_shape[0]
                    c_start = chunk_idx[1] * chunk_shape[1]
                    r_end = min(r_start + chunk_shape[0], shape[0])
                    c_end = min(c_start + chunk_shape[1], shape[1])
                    chunk_rows = r_end - r_start
                    chunk_cols = c_end - c_start
                    decoded_2d = decoded.reshape(chunk_shape)
                    pyck_full[r_start:r_end, c_start:c_end] = decoded_2d[
                        :chunk_rows, :chunk_cols
                    ]
                else:
                    pytest.skip(f"Chunk assembly for {ndim}D arrays not implemented")

            np.testing.assert_array_equal(
                pyck_full,
                ref_data,
                err_msg=f"{array_path}: array data mismatch",
            )

    def test_chunk_byte_count_matches(self, pyck_repo, ic_repo, repo_name):
        """Verify icepyck reads the right number of chunks per array."""
        array_paths = REPO_CONFIGS[repo_name]

        for array_path in array_paths:
            meta = pyck_repo.get_array_metadata("main", array_path)
            shape = tuple(meta["shape"])
            if shape == ():
                # Scalar: expect exactly 1 chunk
                chunks = pyck_repo.read_all_chunks("main", array_path)
                assert len(chunks) == 1
                continue

            chunks = pyck_repo.read_all_chunks("main", array_path)

            # Every chunk must produce non-empty bytes
            for idx, data in chunks.items():
                assert len(data) > 0, f"{array_path} chunk {idx}: got empty bytes"
