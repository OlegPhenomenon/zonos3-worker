"""AAC-LC .m4a encoding of a finished window (ffmpeg)."""
import os
import subprocess
import tempfile

import numpy as np

from .core import SR

BITRATE = "160k"  # the lab's review mixes use the same


def encode_m4a(pcm, workdir=None):
    """(2, n) float32 at SR -> AAC .m4a bytes. The file is written next to `workdir` (not /tmp) because the mp4 muxer needs a seekable
    output; ffmpeg's edit list removes the encoder delay so the decoded length is n samples."""
    with tempfile.TemporaryDirectory(dir=workdir) as d:
        path = os.path.join(d, "w.m4a")
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "f32le", "-ar", str(SR), "-ac", "2", "-i", "-", "-c:a", "aac", "-b:a", BITRATE,
                        "-movflags", "+faststart", path], input=np.ascontiguousarray(pcm.T, dtype=np.float32).tobytes(), check=True)
        with open(path, "rb") as f:
            return f.read()
