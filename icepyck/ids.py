"""ID generation utilities for Icechunk objects.

ObjectId12 (12 bytes, 20 Crockford chars) — snapshots, manifests, chunks.
ObjectId8 (8 bytes, 13 Crockford chars) — nodes.
"""

from __future__ import annotations

import hashlib
import os


def generate_id12() -> bytes:
    """Generate a random 12-byte ObjectId12."""
    return os.urandom(12)


def generate_id8() -> bytes:
    """Generate a random 8-byte ObjectId8."""
    return os.urandom(8)


def content_hash_id12(data: bytes) -> bytes:
    """Content-addressed ObjectId12 from SHA-256 truncated to 12 bytes."""
    return hashlib.sha256(data).digest()[:12]
