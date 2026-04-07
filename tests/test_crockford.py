"""Tests for Crockford Base32 encoding/decoding."""

import pytest

from icepyck.crockford import encode, decode


class TestEncode:
    def test_known_value(self):
        """Encode a known 12-byte value to its expected Crockford string."""
        data = bytes.fromhex("0b1cc8d6787580f0e33a6534")
        assert encode(data) == "1CECHNKREP0F1RSTCMT0"

    def test_empty_bytes(self):
        assert encode(b"") == ""

    def test_single_byte(self):
        # 0xFF = 11111111, right-padded to 10 bits: 1111111100
        # 5-bit groups: 11111 11100 -> V W -> but let's just check roundtrip
        result = encode(b"\xff")
        assert len(result) == 2
        assert decode(result) == b"\xff"

    def test_result_is_uppercase(self):
        result = encode(b"\x00\x01\x02\x03\x04\x05\x06\x07\x08\x09\x0a\x0b")
        assert result == result.upper()


class TestDecode:
    def test_known_value(self):
        """Decode a known Crockford string to its expected bytes."""
        assert decode("1CECHNKREP0F1RSTCMT0") == bytes.fromhex("0b1cc8d6787580f0e33a6534")

    def test_lowercase_accepted(self):
        """Lowercase input should decode identically to uppercase."""
        upper = decode("1CECHNKREP0F1RSTCMT0")
        lower = decode("1cechnkrep0f1rstcmt0")
        assert upper == lower

    def test_invalid_character_raises(self):
        with pytest.raises(ValueError, match="Invalid Crockford Base32 character"):
            decode("INVALID-CHAR!")

    def test_empty_string(self):
        assert decode("") == b""


class TestRoundtrip:
    def test_12_byte_roundtrip(self):
        data = bytes.fromhex("0b1cc8d6787580f0e33a6534")
        assert decode(encode(data)) == data

    def test_various_lengths(self):
        """Roundtrip works for various byte lengths."""
        for length in [1, 2, 4, 8, 12, 16, 32]:
            data = bytes(range(length))
            assert decode(encode(data)) == data, f"Roundtrip failed for {length} bytes"

    def test_all_zeros(self):
        data = b"\x00" * 12
        assert decode(encode(data)) == data

    def test_all_ones(self):
        data = b"\xff" * 12
        assert decode(encode(data)) == data
