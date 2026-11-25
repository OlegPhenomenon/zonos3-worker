"""Shared basics of the studio mix: sample rate, audio io, frame levels, convolution, the clip (stems) container and the
dry reference voice.

Everything runs at SR = 48000 Hz float32, the rate of the stems the worker writes (`mix.flac`, `speech.flac`, `bg.flac`).
The lab (tool/studio_dub_lab) ran at 44100; constants given in seconds / Hz carry over, constants given in samples are
derived from SR here.
"""
import json
import os
import pickle
import subprocess
from functools import cached_property
from pathlib import Path

import numpy as np

SR = 48000
FR = SR // 100  # 10 ms analysis frames (level curves, activity)
HERE = Path(__file__).resolve().parent
ASSETS = HERE / "assets"


# ----------------------------------------------------------------------------- io

def load_audio(path, channels=None, sr=SR):
    """(channels, n) float32 at sr. flac/wav at the right rate by libsndfile, anything else through ffmpeg.
    channels=None keeps the file's layout, 1 mixes down, 2 up/down-mixes."""
    path = str(path)
    try:
        import soundfile as sf
        x, rate = sf.read(path, dtype="float32", always_2d=True)
        if rate != sr:
            raise RuntimeError("rate")
        x = np.ascontiguousarray(x.T)
        if channels is not None and channels != x.shape[0]:
            x = x.mean(0, keepdims=True) if channels == 1 else np.repeat(x[:1], channels, 0) if x.shape[0] == 1 else x[:channels]
        return x
    except Exception:
        ch = channels or 2
        raw = subprocess.run(["ffmpeg", "-v", "error", "-i", path, "-ac", str(ch), "-ar", str(sr), "-f", "f32le", "-"],
                             check=True, capture_output=True).stdout
        return np.ascontiguousarray(np.frombuffer(raw, np.float32).reshape(-1, ch).T)


def read_range(path, i0, i1, channels=2):
    """Samples [i0, i1) of a 48 kHz flac/wav as (channels, i1 - i0) float32 (zero-padded outside the file; seeks, no full decode)."""
    import soundfile as sf
    with sf.SoundFile(str(path)) as f:
        a, b = max(0, int(i0)), min(f.frames, int(i1))
        f.seek(a)
        x = f.read(max(0, b - a), dtype="float32", always_2d=True).T if b > a else np.zeros((f.channels, 0), np.float32)
    if x.shape[0] != channels:
        x = x.mean(0, keepdims=True) if channels == 1 else np.repeat(x[:1], channels, 0)
    return np.pad(x, ((0, 0), (a - int(i0), int(i1) - b)))


def fit(x, total):
    """Trim or zero-pad the last axis to `total` samples."""
    n = x.shape[-1]
    return x[..., :total] if n >= total else np.pad(x, [(0, 0)] * (x.ndim - 1) + [(0, total - n)])


# ----------------------------------------------------------------------------- levels

def frame_power(x, fr=FR):
    """Mean power per `fr`-sample frame of every channel: (channels, frames), float64. 1-D input -> (1, frames)."""
    x = x[None] if x.ndim == 1 else x
    n = x.shape[1] // fr
    xr = x[:, : n * fr].reshape(x.shape[0], n, fr)
    return np.einsum("cnf,cnf->cn", xr, xr, dtype=np.float64) / fr


def level_db(p, a, b):
    """Loudness of the active part of [a, b] s of a 10 ms frame-power vector p: mean power of the frames within 20 dB of the window
    peak (nan if empty). Same rule as the lab's dub_mix.level_db."""
    w = p[max(0, int(a * 100)):max(0, int(b * 100) + 1)]
    if len(w) < 2 or w.max() < 1e-10:
        return float("nan")
    db = 10 * np.log10(w + 1e-12)
    return float(10 * np.log10(w[db >= db.max() - 20].mean()))


def pow2(n):
    return 1 << max(int(n - 1), 1).bit_length()


# ----------------------------------------------------------------------------- convolution

def conv_full(x, h):
    """Linear convolution, output len(x) + len(h) - 1 (a long tail rings out). float32 in, float32 out."""
    if len(h) == 1:
        return x * h[0]
    n = len(x) + len(h) - 1
    size = pow2(n)
    return np.fft.irfft(np.fft.rfft(x, size) * np.fft.rfft(h, size), size)[:n].astype(np.float32)


def conv_same(x, h, delay=0):
    """Linear convolution truncated to len(x), the result advanced by `delay` samples (linear-phase FIR: delay = len(h)//2).
    Blocked overlap-add so that long inputs with short filters stay cheap; blocks of digital silence are skipped."""
    n, m = len(x), len(h)
    size = pow2(max(4 * m, 1 << 15))
    step = size - m + 1
    hf = np.fft.rfft(h, size)
    out = np.zeros(n + m, np.float32)
    for s in range(0, n, step):
        blk = x[s:s + step]
        if not blk.any():
            continue
        y = np.fft.irfft(np.fft.rfft(blk, size) * hf, size)[: len(blk) + m - 1]
        out[s:s + len(y)] += y
    return out[delay:delay + n]


# ----------------------------------------------------------------------------- speech activity

def frame_db(x, sr=SR, hop_s=0.01):
    h = int(round(hop_s * sr))
    n = len(x) // h
    xr = x[: n * h].reshape(n, h)
    return 10 * np.log10(np.einsum("nf,nf->n", xr, xr, dtype=np.float64) / h + 1e-12)


def runs(mask):
    """[(start, stop)) index pairs of the True runs of a boolean vector."""
    m = np.concatenate([[False], np.asarray(mask, bool), [False]])
    d = np.diff(m.astype(np.int8))
    return list(zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)))


def speech_regions(x, sr=SR, span_db=30.0, join_s=0.3, min_s=0.15):
    """Blind voice activity: 10 ms frames within span_db of the 98th percentile level, gaps shorter than join_s closed, regions
    shorter than min_s dropped. Returns [(start_s, end_s)]. (lab: channel_match.speech_regions)"""
    db = frame_db(x, sr)
    if not len(db):
        return []
    on = db >= np.percentile(db, 98) - span_db
    merged = []
    for a, b in runs(on):
        a, b = a * 0.01, b * 0.01
        if merged and a - merged[-1][1] < join_s:
            merged[-1][1] = b
        else:
            merged.append([a, b])
    return [(a, b) for a, b in merged if b - a >= min_s]


# ----------------------------------------------------------------------------- clip

class Clip:
    """The stems of one clip (`stems_dir`: mix.flac, speech.flac, bg.flac; stereo, 48 kHz, equal length) and the analysis folder.
    Arrays load lazily and are cached on the object."""

    def __init__(self, stems_dir):
        self.dir = Path(stems_dir)
        self.analysis_dir = self.dir / "analysis"

    def _load(self, name):
        return load_audio(self.dir / name, 2, SR)

    @cached_property
    def mix2(self):
        return self._load("mix.flac")

    @cached_property
    def speech2(self):
        return fit(self._load("speech.flac"), self.n)

    @cached_property
    def bg2(self):
        return fit(self._load("bg.flac"), self.n)

    @cached_property
    def speech(self):
        """Mono (L+R)/2 of the speech stem."""
        return (0.5 * (self.speech2[0] + self.speech2[1])).astype(np.float32)

    @cached_property
    def n(self):
        import soundfile as sf
        return sf.info(str(self.dir / "mix.flac")).frames

    @property
    def duration_s(self):
        return self.n / SR

    # analysis documents: pickled dicts / arrays, one file per stage
    def save_doc(self, name, obj):
        self.analysis_dir.mkdir(parents=True, exist_ok=True)
        tmp = self.analysis_dir / (name + ".pkl.tmp")
        with open(tmp, "wb") as f:
            pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(tmp, self.analysis_dir / (name + ".pkl"))

    def load_doc(self, name):
        with open(self.analysis_dir / (name + ".pkl"), "rb") as f:
            return pickle.load(f)

    def has_doc(self, name):
        return (self.analysis_dir / (name + ".pkl")).exists()


# ----------------------------------------------------------------------------- dry reference voice

class Reference:
    """A fixed dry voice (Zonos output, several speakers, ~2 min, mono, SR) with its speech regions. The lab compares the original
    voice of a scene with the dub track of the same clip ("what does the channel add to a dry voice"); the realtime product has no
    dub yet when the clip is analysed, so every comparison uses this reference instead. The channel estimates depend on the dub only
    through its dry decay statistics, its long-term spectrum and the nominal regions of its speech."""

    def __init__(self):
        name = os.environ.get("STUDIO_MIX_REFERENCE", "dry_reference")  # alternative reference voices (tests / gender-specific tone)
        self.sig = load_audio(ASSETS / (name + ".flac"), 1, SR)[0]
        self.regions = [tuple(r) for r in json.load(open(ASSETS / (name + ".json")))["regions"]]
        self.duration_s = len(self.sig) / SR

    def excerpt(self, seconds):
        """(signal, regions) of the first whole regions that add up to about `seconds` of material, regions re-based to 0."""
        end = 0.0
        for a, b in self.regions:
            if b > seconds and end > 0:
                break
            end = b
        end = min(end + 0.5, self.duration_s)
        i1 = int(end * SR)
        return self.sig[:i1], [(a, min(b, end)) for a, b in self.regions if a < end]


_REFERENCE = []


def reference():
    if not _REFERENCE:
        _REFERENCE.append(Reference())
    return _REFERENCE[0]
