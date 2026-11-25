"""Window encoding for the segment API: stereo 48 kHz PCM -> Ogg Opus whose decoded length is exactly the window's.

Consecutive windows are played back to back by independent decoders, so every window must decode to exactly its own samples and
must not start or end with a codec transient. Both come from the Ogg Opus container (RFC 7845), which ffmpeg/libmpv honour by default:

* the encoder gets `pre` samples of the preceding audio and `post` samples of the following audio around the window (the mix renders
  a few seconds of context anyway); the OpusHead pre-skip is 312 (libopus lookahead) + len(pre), so the decoder drops the warm-up and
  the window starts at its first sample;
* the last page's granule position is cut right after the window's last sample, so the decoder drops the end padding and the
  post-roll. The packets are 20 ms, 30 s is exactly 1500 of them: a normal window has no padding at all.

ffmpeg's libopus only produces the packets; the container is written here because ffmpeg cannot set a pre-skip beyond the lookahead.
"""
import struct
import subprocess
import zlib

import numpy as np

from .core import SR

BITRATE = "160k"
LOOKAHEAD = 312      # libopus encoder delay at 48 kHz (ffmpeg's initial_padding)
FRAME = 960          # 20 ms
PAGE_PACKETS = 25    # 0.5 s per page
PRE_MS, POST_MS = 40, 20   # context the caller should pass (more is cut off)

_REV8 = bytes(int(f"{i:08b}"[::-1], 2) for i in range(256))


def _crc(b):
    """Ogg CRC-32 (poly 0x04C11DB7, MSB first, init 0, no final xor) through zlib's reflected CRC-32: reverse the bits of every byte,
    run zlib with init/xor cancelled, reverse the bits of the result."""
    raw = zlib.crc32(b.translate(_REV8), 0xFFFFFFFF) ^ 0xFFFFFFFF
    return int(f"{raw:032b}"[::-1], 2)


def _page(flags, granule, serial, seq, packets):
    lacing = b"".join(bytes([255] * (len(p) // 255) + [len(p) % 255]) for p in packets)
    body = b"".join(packets)
    head = b"OggS" + struct.pack("<BBqIIIB", 0, flags, granule, serial, seq, 0, len(lacing)) + lacing
    crc = _crc(head + body)
    return head[:22] + struct.pack("<I", crc) + head[26:] + body


def _packets(ogg):
    """Packets of an Ogg stream written by ffmpeg (one logical stream, pages whole)."""
    out, cur, pos = [], b"", 0
    while pos < len(ogg):
        assert ogg[pos:pos + 4] == b"OggS"
        nseg = ogg[pos + 26]
        lacing = ogg[pos + 27:pos + 27 + nseg]
        pos += 27 + nseg
        for lace in lacing:
            cur += ogg[pos:pos + lace]
            pos += lace
            if lace < 255:
                out.append(cur)
                cur = b""
    return out


def _opus_packets(pcm):
    """(2, n) float32 -> list of 20 ms Opus packets (the first two Ogg packets, OpusHead and OpusTags, dropped)."""
    r = subprocess.run(["ffmpeg", "-v", "error", "-f", "f32le", "-ar", str(SR), "-ac", "2", "-i", "-", "-c:a", "libopus", "-b:a", BITRATE,
                        "-vbr", "on", "-application", "audio", "-frame_duration", "20", "-f", "ogg", "-"],
                       input=np.ascontiguousarray(pcm.T, dtype=np.float32).tobytes(), capture_output=True, check=True)
    return _packets(r.stdout)[2:]


def encode_opus(pcm, pre=None, post=None):
    """pcm (2, n) float32 at SR: the window. pre / post: (2, k) float32 audio just before / after it (or None at the clip's ends).
    Returns Ogg Opus bytes that decode to exactly `pcm`."""
    assert SR == 48000
    p = 0 if pre is None else min(pre.shape[1], PRE_MS * SR // 1000)
    q = 0 if post is None else min(post.shape[1], POST_MS * SR // 1000)
    parts = ([pre[:, pre.shape[1] - p:]] if p else []) + [pcm] + ([post[:, :q]] if q else [])
    pk = _opus_packets(np.concatenate(parts, axis=1))
    skip = LOOKAHEAD + p
    end = skip + pcm.shape[1]           # decoded samples up to the window's last one, counted from the stream start
    pk = pk[:-(-end // FRAME)]
    assert len(pk) * FRAME >= end, "encoder produced fewer samples than the window"
    serial = 0x5EED0001
    opus_head = b"OpusHead" + struct.pack("<BBHIhB", 1, 2, skip, SR, 0, 0)
    opus_tags = b"OpusTags" + struct.pack("<I", 7) + b"studio " + struct.pack("<I", 0)
    out = [_page(2, 0, serial, 0, [opus_head]), _page(0, 0, serial, 1, [opus_tags])]
    seq = 2
    for i in range(0, len(pk), PAGE_PACKETS):
        chunk = pk[i:i + PAGE_PACKETS]
        last = i + PAGE_PACKETS >= len(pk)
        granule = end if last else (i + len(chunk)) * FRAME
        out.append(_page(4 if last else 0, granule, serial, seq, chunk))
        seq += 1
    return b"".join(out)
