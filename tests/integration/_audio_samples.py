"""Synthetic audio bytes for Task 07 tests (never real diary recordings).

Each builder returns bytes whose container signature (magic bytes) matches the
declared MIME type so the server-side content check passes; the payloads are
deliberately tiny and are not valid decodable recordings. WAV gets a correct
RIFF/WAVE container header (44-byte canonical form) with silence payload.
"""

from __future__ import annotations

import struct

WAV_MIME = "audio/wav"
WEBM_MIME = "audio/webm"
OGG_MIME = "audio/ogg"
MP3_MIME = "audio/mpeg"
MP4_MIME = "audio/mp4"

#: Signed 16-bit little-endian, mono, 8 kHz — a canonical minimal PCM WAV.
_WAV_FMT_CHUNK = struct.pack("<HHIIHH", 1, 1, 8000, 16000, 2, 16)


def wav_bytes(payload_size: int = 400) -> bytes:
    """A minimal WAV: RIFF/WAVE header, ``fmt `` and one ``data`` chunk."""
    payload = b"\x00\x00" * (payload_size // 2)
    riff_size = 4 + (8 + len(_WAV_FMT_CHUNK)) + (8 + len(payload))
    return (
        b"RIFF"
        + struct.pack("<I", riff_size)
        + b"WAVE"
        + b"fmt "
        + struct.pack("<I", len(_WAV_FMT_CHUNK))
        + _WAV_FMT_CHUNK
        + b"data"
        + struct.pack("<I", len(payload))
        + payload
    )


def webm_bytes() -> bytes:
    """EBML magic of a WebM container plus padding."""
    return b"\x1a\x45\xdf\xa3" + bytes(96)


def ogg_bytes() -> bytes:
    """Ogg capture pattern plus padding."""
    return b"OggS" + bytes(96)


def mp3_bytes() -> bytes:
    """An ID3-tagged MP3 stream (valid MPEG sync would also be accepted)."""
    return b"ID3" + bytes(96)


def mp4_bytes() -> bytes:
    """ISO-BMFF ``ftyp`` header of an M4A file plus padding."""
    return b"\x00\x00\x00\x18ftypM4A " + bytes(96)


def not_audio_bytes() -> bytes:
    """Bytes that match no supported container signature."""
    return b"plain text that is definitely not audio content" * 4
