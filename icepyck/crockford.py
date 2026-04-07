"""Crockford Base32 encoding/decoding for Icechunk ObjectId12 values.

Icechunk encodes 12-byte ObjectId12 values as Crockford Base32 strings
for use as file names in the storage layout.

Alphabet: 0123456789ABCDEFGHJKMNPQRSTVWXYZ (no I, L, O, U)

Note: Icechunk uses right-padding (LSB) for the extra bits when the
bit count is not a multiple of 5, unlike some Crockford implementations
that left-pad (MSB). For 12 bytes (96 bits), this produces 20 characters
(100 bits) with 4 zero-padding bits appended on the right.
"""

ALPHABET = "0123456789ABCDEFGHJKMNPQRSTVWXYZ"
_DECODE_MAP = {c: i for i, c in enumerate(ALPHABET)}
# Also accept lowercase
_DECODE_MAP.update({c.lower(): i for i, c in enumerate(ALPHABET)})


def encode(data: bytes) -> str:
    """Encode bytes to a Crockford Base32 uppercase string.

    Processes 5 bits at a time from the input bytes to produce
    base-32 characters. Padding bits are added on the right (LSB side).
    """
    total_bits = len(data) * 8
    num_chars = (total_bits + 4) // 5  # ceil(total_bits / 5)
    padded_bits = num_chars * 5
    pad = padded_bits - total_bits  # extra zero bits on the right

    n = int.from_bytes(data, "big") << pad

    chars = []
    for i in range(num_chars - 1, -1, -1):
        chars.append(ALPHABET[(n >> (i * 5)) & 0x1F])

    return "".join(chars)


def decode(s: str) -> bytes:
    """Decode a Crockford Base32 string to bytes.

    Reverses the right-padded encoding: strips the padding bits
    from the LSB side to recover the original bytes.
    """
    n = 0
    for c in s:
        val = _DECODE_MAP.get(c)
        if val is None:
            raise ValueError(f"Invalid Crockford Base32 character: {c!r}")
        n = (n << 5) | val

    total_bits = len(s) * 5
    num_bytes = total_bits // 8
    pad = total_bits - num_bytes * 8  # padding bits on the right

    n >>= pad  # strip the right-side padding

    return n.to_bytes(num_bytes, "big")
