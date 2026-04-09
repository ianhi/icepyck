"""Tests for IcechunkReadStore.list_dir and _node_dir_entries.

These tests focus specifically on directory listing of sub-groups, which is
the scenario exercised when xarray opens a zarr dataset with group= parameter.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

zarr = pytest.importorskip("zarr")

import icepyck

TEST_REPOS_DIR = Path(__file__).parent.parent / "test-repos"
BASIC_REPO = TEST_REPOS_DIR / "basic"

pytestmark = pytest.mark.skipif(
    not (BASIC_REPO / "repo").exists(),
    reason="basic test repo not available",
)


def _list_dir_sync(store, prefix: str) -> list[str]:
    """Run list_dir synchronously and return sorted results."""

    async def _run():
        return [e async for e in store.list_dir(prefix)]

    return asyncio.run(_run())


@pytest.fixture
def basic_store():
    repo = icepyck.open(str(BASIC_REPO))
    session = repo.readonly_session(branch="main")
    return session.store


class TestListDirRoot:
    """list_dir("") or list_dir("/") on the root group."""

    def test_root_includes_group(self, basic_store):
        entries = _list_dir_sync(basic_store, "")
        assert "group1/" in entries

    def test_root_includes_zarr_json(self, basic_store):
        entries = _list_dir_sync(basic_store, "")
        assert "zarr.json" in entries

    def test_root_no_duplicates(self, basic_store):
        entries = _list_dir_sync(basic_store, "")
        assert len(entries) == len(set(entries))


class TestListDirSubGroup:
    """list_dir("group1/") — listing a sub-group that contains arrays."""

    def test_subgroup_includes_zarr_json(self, basic_store):
        entries = _list_dir_sync(basic_store, "group1/")
        assert "zarr.json" in entries

    def test_subgroup_includes_all_arrays(self, basic_store):
        """Both arrays inside group1/ must appear in list_dir output."""
        entries = _list_dir_sync(basic_store, "group1/")
        # basic repo has group1/temperatures and group1/timestamps
        assert "temperatures/" in entries, f"entries={entries}"
        assert "timestamps/" in entries, f"entries={entries}"

    def test_subgroup_no_duplicates(self, basic_store):
        entries = _list_dir_sync(basic_store, "group1/")
        assert len(entries) == len(set(entries)), f"duplicate entries: {entries}"

    def test_subgroup_no_trailing_slash_prefix(self, basic_store):
        """list_dir should normalise a prefix without trailing slash."""
        entries_slash = _list_dir_sync(basic_store, "group1/")
        entries_no_slash = _list_dir_sync(basic_store, "group1")
        assert set(entries_slash) == set(entries_no_slash)

    def test_subgroup_does_not_include_nested_zarr_json(self, basic_store):
        """list_dir on a group should not directly expose nested zarr.json paths."""
        entries = _list_dir_sync(basic_store, "group1/")
        # zarr.json for arrays should not be flat-listed; only immediate children
        assert "temperatures/zarr.json" not in entries
        assert "timestamps/zarr.json" not in entries

    def test_subgroup_array_entry_has_trailing_slash(self, basic_store):
        """Directory entries for arrays should include trailing slash."""
        entries = _list_dir_sync(basic_store, "group1/")
        # entries for sub-arrays must end in '/'
        array_entries = [e for e in entries if e != "zarr.json"]
        assert all(e.endswith("/") for e in array_entries), f"entries={entries}"


class TestListDirArrayLevel:
    """list_dir on an array path — should include zarr.json and c/."""

    def test_array_level_includes_zarr_json(self, basic_store):
        entries = _list_dir_sync(basic_store, "group1/temperatures/")
        assert "zarr.json" in entries

    def test_array_level_includes_chunk_dir(self, basic_store):
        entries = _list_dir_sync(basic_store, "group1/temperatures/")
        assert "c/" in entries


class TestZarrGroupOpenAtSubPath:
    """Open a zarr group at a sub-path (simulates xarray group= parameter)."""

    def test_zarr_group_members_at_subpath(self, basic_store):
        """zarr.open_group with path='group1' must find all arrays inside."""
        grp = zarr.open_group(store=basic_store, path="group1", mode="r")
        member_names = {name for name, _ in grp.members()}
        assert "temperatures" in member_names, f"members={member_names}"
        assert "timestamps" in member_names, f"members={member_names}"

    def test_zarr_group_member_count_at_subpath(self, basic_store):
        """All 2 arrays inside group1 should be discovered."""
        grp = zarr.open_group(store=basic_store, path="group1", mode="r")
        member_names = {name for name, _ in grp.members()}
        assert len(member_names) == 2, (
            f"Expected 2, got {len(member_names)}: {member_names}"
        )
