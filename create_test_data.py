# /// script
# requires-python = ">=3.12"
# dependencies = ["icechunk>=2.0.0a0", "zarr>=3", "numpy"]
# [tool.uv]
# prerelease = "allow"
# ///

"""Create test icechunk repositories for spec conformance testing."""

import icechunk
import zarr
import numpy as np
import shutil
from pathlib import Path

TEST_DIR = Path(__file__).parent / "test-repos"

def create_basic_repo():
    """A simple repo with one group and two arrays."""
    path = TEST_DIR / "basic"
    if path.exists():
        shutil.rmtree(path)

    storage = icechunk.local_filesystem_storage(str(path))
    repo = icechunk.Repository.create(storage)
    session = repo.writable_session("main")
    store = session.store

    root = zarr.group(store)
    root.create_group("group1")
    root["group1"].create_array("temperatures", shape=(100,), dtype="f4", chunks=(10,))
    root["group1"].create_array("timestamps", shape=(100,), dtype="i8", chunks=(50,))

    root["group1/temperatures"][:] = np.random.randn(100).astype("f4")
    root["group1/timestamps"][:] = np.arange(100, dtype="i8")

    session.commit("initial data")
    print(f"Created basic repo at {path}")

    # Second commit to test parent snapshots
    session = repo.writable_session("main")
    store = session.store
    root = zarr.open_group(store=store)
    root["group1/temperatures"][0:10] = np.zeros(10, dtype="f4")
    session.commit("update first 10 temperatures")
    print("  Added second commit")

    # Create a tag
    snap_id = repo.lookup_branch("main")
    repo.create_tag("v1", snap_id)
    print("  Created tag 'v1'")


def create_nested_repo():
    """A repo with nested groups to test hierarchy traversal."""
    path = TEST_DIR / "nested"
    if path.exists():
        shutil.rmtree(path)

    storage = icechunk.local_filesystem_storage(str(path))
    repo = icechunk.Repository.create(storage)
    session = repo.writable_session("main")
    store = session.store

    root = zarr.group(store)
    root.create_group("a")
    root["a"].create_group("b")
    root["a/b"].create_group("c")
    root["a/b/c"].create_array("data", shape=(5, 5), dtype="i4", chunks=(5, 5))
    root["a/b/c/data"][:] = np.arange(25, dtype="i4").reshape(5, 5)

    session.commit("nested structure")
    print(f"Created nested repo at {path}")


def create_scalar_repo():
    """A repo with a scalar (0-dimensional) array."""
    path = TEST_DIR / "scalar"
    if path.exists():
        shutil.rmtree(path)

    storage = icechunk.local_filesystem_storage(str(path))
    repo = icechunk.Repository.create(storage)
    session = repo.writable_session("main")
    store = session.store

    root = zarr.group(store)
    root.create_array("value", shape=(), dtype="f8", chunks=())
    root["value"][()] = 42.0

    session.commit("scalar data")
    print(f"Created scalar repo at {path}")


def create_native_chunks_repo():
    """A repo with native (non-inline) chunks stored in chunks/ directory."""
    path = TEST_DIR / "native-chunks"
    if path.exists():
        shutil.rmtree(path)

    storage = icechunk.local_filesystem_storage(str(path))
    # Setting inline_chunk_threshold_bytes=0 forces all chunks to native storage
    config = icechunk.RepositoryConfig(inline_chunk_threshold_bytes=0)
    repo = icechunk.Repository.create(storage, config=config)
    session = repo.writable_session("main")
    store = session.store

    root = zarr.group(store)
    root.create_array("data", shape=(100,), dtype="f4", chunks=(50,))
    root["data"][:] = np.arange(100, dtype="f4")

    session.commit("native chunks")
    print(f"Created native-chunks repo at {path}")


def create_many_commits_repo():
    """A repo with many commits to test ops log overflow and overwritten/ directory.

    Uses num_updates_per_repo_info_file=5 so the ops log overflows after 5 updates,
    creating backup files in overwritten/ and exercising the linked list chain.
    """
    path = TEST_DIR / "many-commits"
    if path.exists():
        shutil.rmtree(path)

    storage = icechunk.local_filesystem_storage(str(path))
    config = icechunk.RepositoryConfig(num_updates_per_repo_info_file=5)
    repo = icechunk.Repository.create(storage, config=config)

    for i in range(15):
        session = repo.writable_session("main")
        store = session.store
        if i == 0:
            root = zarr.group(store)
            root.create_array("counter", shape=(1,), dtype="i4", chunks=(1,))
        root = zarr.open_group(store=store)
        root["counter"][0] = i
        session.commit(f"commit {i}")

    print(f"Created many-commits repo at {path} (15 commits, overflow at 5)")


if __name__ == "__main__":
    TEST_DIR.mkdir(exist_ok=True)
    create_basic_repo()
    create_nested_repo()
    create_scalar_repo()
    create_native_chunks_repo()
    create_many_commits_repo()
    print("\nAll test repos created. Use these to test your spec-only reader.")
