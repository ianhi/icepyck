"""Tests for Phase 1: storage write, header round-trip, ID generation."""

from __future__ import annotations

import tempfile
from pathlib import Path

from icepyck.crockford import decode, encode
from icepyck.header import (
    Compression,
    FileType,
    build_bytes,
    parse_bytes,
)
from icepyck.ids import content_hash_id12, generate_id8, generate_id12
from icepyck.storage import LocalStorage


class TestStorageWrite:
    def test_local_write_read_roundtrip(self, tmp_path: Path) -> None:
        storage = LocalStorage(tmp_path)
        storage.write("test.bin", b"hello world")
        assert storage.read("test.bin") == b"hello world"

    def test_local_write_creates_subdirs(self, tmp_path: Path) -> None:
        storage = LocalStorage(tmp_path)
        storage.write("a/b/c/deep.bin", b"deep data")
        assert storage.read("a/b/c/deep.bin") == b"deep data"
        assert storage.exists("a/b/c/deep.bin")

    def test_local_write_overwrites(self, tmp_path: Path) -> None:
        storage = LocalStorage(tmp_path)
        storage.write("file.bin", b"v1")
        storage.write("file.bin", b"v2")
        assert storage.read("file.bin") == b"v2"

    def test_local_write_list_prefix(self, tmp_path: Path) -> None:
        storage = LocalStorage(tmp_path)
        storage.write("chunks/aaa", b"data1")
        storage.write("chunks/bbb", b"data2")
        files = storage.list_prefix("chunks")
        assert sorted(files) == ["chunks/aaa", "chunks/bbb"]


class TestHeaderBuild:
    def test_roundtrip_no_compression(self) -> None:
        payload = b"test payload data"
        raw = build_bytes(
            payload, FileType.SNAPSHOT, compression=Compression.NONE
        )
        header, decoded = parse_bytes(raw)
        assert header.file_type == FileType.SNAPSHOT
        assert header.compression == Compression.NONE
        assert header.spec_version == 2
        assert header.implementation == "icepyck"
        assert decoded == payload

    def test_roundtrip_zstd(self) -> None:
        payload = b"test payload data" * 100
        raw = build_bytes(
            payload, FileType.MANIFEST, compression=Compression.ZSTD
        )
        header, decoded = parse_bytes(raw)
        assert header.file_type == FileType.MANIFEST
        assert header.compression == Compression.ZSTD
        assert decoded == payload

    def test_roundtrip_all_file_types(self) -> None:
        payload = b"some fb data"
        for ft in [
            FileType.SNAPSHOT,
            FileType.MANIFEST,
            FileType.TRANSACTION_LOG,
            FileType.REPO_INFO,
        ]:
            raw = build_bytes(payload, ft, compression=Compression.NONE)
            header, decoded = parse_bytes(raw)
            assert header.file_type == ft
            assert decoded == payload

    def test_impl_name_truncated_to_24(self) -> None:
        raw = build_bytes(
            b"x",
            FileType.SNAPSHOT,
            compression=Compression.NONE,
            implementation="a" * 50,
        )
        header, _ = parse_bytes(raw)
        assert len(header.implementation) == 24

    def test_impl_name_padded(self) -> None:
        raw = build_bytes(
            b"x",
            FileType.SNAPSHOT,
            compression=Compression.NONE,
            implementation="hi",
        )
        # The raw impl field is bytes 12:36
        impl_field = raw[12:36]
        assert impl_field == b"hi" + b" " * 22


class TestIdGeneration:
    def test_id12_length(self) -> None:
        assert len(generate_id12()) == 12

    def test_id8_length(self) -> None:
        assert len(generate_id8()) == 8

    def test_id12_unique(self) -> None:
        ids = {generate_id12() for _ in range(100)}
        assert len(ids) == 100

    def test_id8_unique(self) -> None:
        ids = {generate_id8() for _ in range(100)}
        assert len(ids) == 100

    def test_content_hash_deterministic(self) -> None:
        data = b"hello world"
        assert content_hash_id12(data) == content_hash_id12(data)

    def test_content_hash_different_inputs(self) -> None:
        assert content_hash_id12(b"a") != content_hash_id12(b"b")

    def test_content_hash_length(self) -> None:
        assert len(content_hash_id12(b"anything")) == 12

    def test_id12_crockford_roundtrip(self) -> None:
        raw = generate_id12()
        encoded = encode(raw)
        assert len(encoded) == 20
        assert decode(encoded) == raw

    def test_id8_crockford_roundtrip(self) -> None:
        raw = generate_id8()
        encoded = encode(raw)
        assert len(encoded) == 13
        assert decode(encoded) == raw
