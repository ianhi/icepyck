"""Binary header parser for Icechunk metadata files.

All Icechunk metadata files share a 39-byte header:
  - 12 bytes magic: ICE🧊CHUNK (hex: 49 43 45 F0 9F A7 8A 43 48 55 4E 4B)
  - 24 bytes: implementation name (UTF-8, left-aligned, right-space-padded)
  - 1 byte: spec version (1 or 2)
  - 1 byte: file type
  - 1 byte: compression algorithm (0=none, 1=zstd)
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from pathlib import Path

import zstandard

MAGIC = b"ICE\xf0\x9f\xa7\x8aCHUNK"
HEADER_SIZE = 39  # 12 + 24 + 1 + 1 + 1


class FileType(IntEnum):
    """Icechunk file type byte values.

    NOTE: The spec document says the ordering is:
        1=Snapshot, 2=Manifest, 3=Attributes, 4=RepoInfo,
        5=TransactionLog, 6=Chunk
    However, actual binary data from icechunk (ic-2.0.0-alpha.7) uses
    different values for TransactionLog and RepoInfo:
        TransactionLog = 4 (spec says 5)
        RepoInfo = 6 (spec says 4)
    We use the ACTUAL values observed in test data.
    """

    SNAPSHOT = 1
    MANIFEST = 2
    ATTRIBUTES = 3
    TRANSACTION_LOG = 4  # spec says 5, actual data says 4
    # 5 is unused / unknown
    REPO_INFO = 6  # spec says 4, actual data says 6


class Compression(IntEnum):
    NONE = 0
    ZSTD = 1


@dataclass(frozen=True)
class Header:
    """Parsed Icechunk file header."""

    implementation: str
    spec_version: int
    file_type: FileType
    compression: Compression


def parse_bytes(raw: bytes) -> tuple[Header, bytes]:
    """Parse raw Icechunk file bytes, returning (header, decompressed payload)."""
    if len(raw) < HEADER_SIZE:
        raise ValueError(
            f"File too short: {len(raw)} bytes, need at least {HEADER_SIZE}"
        )

    magic = raw[:12]
    if magic != MAGIC:
        raise ValueError(f"Bad magic: {magic!r}, expected {MAGIC!r}")

    impl_name = raw[12:36].rstrip().decode("utf-8")
    spec_version = raw[36]
    file_type = FileType(raw[37])
    compression = Compression(raw[38])

    payload = raw[HEADER_SIZE:]

    if compression == Compression.ZSTD:
        dctx = zstandard.ZstdDecompressor()
        payload = dctx.decompress(payload, max_output_size=64 * 1024 * 1024)

    header = Header(
        implementation=impl_name,
        spec_version=spec_version,
        file_type=file_type,
        compression=compression,
    )

    return header, payload


def parse_file(path: str | Path) -> tuple[Header, bytes]:
    """Parse an Icechunk file, returning (header, decompressed payload)."""
    raw = Path(path).read_bytes()
    return parse_bytes(raw)
