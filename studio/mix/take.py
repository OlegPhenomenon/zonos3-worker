"""A dub take as the backend sends it: WAV bytes (any rate / channel count) -> trimmed mono 48 kHz float32 time-scaled by PSOLA.

PSOLA (pitch-synchronous overlap-add on the voice's own pitch marks, Praat through parselmouth) is the lab's time-scaler
(`iso_fit.psola`): unlike ffmpeg's rubberband it keeps onsets, shimmer and harmonics-to-noise of the voice at tempo 0.8-1.3.
"""
import io

import numpy as np
import parselmouth
import soundfile as sf
from parselmouth.praat import call, run
from scipy.signal import resample_poly

from .core import SR

NO_STRETCH = 0.03  # tempo within this of 1 is left alone (lab iso_fit.NO_STRETCH)
TRIM_DB = 40.0  # silence trim: frames within this of the take's loudest frame are the voice (lab build_iso_dub.trimmed)
TRIM_PAD_FRAMES = (2, 3)  # frames kept before the first / after the last voiced frame


def decode_wav(data):
    """WAV bytes -> mono float32 at SR."""
    x, rate = sf.read(io.BytesIO(data), dtype="float32", always_2d=True)
    x = x.mean(1)
    if rate != SR:
        g = np.gcd(int(rate), SR)
        x = resample_poly(x, SR // g, int(rate) // g).astype(np.float32)
    return np.ascontiguousarray(x, dtype=np.float32)


def trim_silence(x):
    """Zonos pads ~0.2 s of silence at the start and up to ~0.4 s at the end; the slot must start with the voice. Idempotent."""
    hop = SR // 100
    n = len(x) // hop
    if not n:
        return x
    db = 20 * np.log10(np.sqrt(np.einsum("nf,nf->n", x[: n * hop].reshape(n, hop), x[: n * hop].reshape(n, hop), dtype=np.float64) / hop) + 1e-9)
    on = np.flatnonzero(db > db.max() - TRIM_DB)
    if not len(on):
        return x
    return x[max(0, on[0] - TRIM_PAD_FRAMES[0]) * hop:min(len(x), (on[-1] + TRIM_PAD_FRAMES[1]) * hop)]


def psola(x, tempo, sr=SR):
    """Time-scale mono float audio by `tempo` (2.0 = twice as fast) keeping pitch and formants. The pitch range is taken from the
    take itself, so female and male voices both get right pitch marks."""
    run("random_initializeWithSeedUnsafelyButPredictably (1)")  # Praat's overlap-add draws random numbers: make the take reproducible
    snd = parselmouth.Sound(x.astype(np.float64), sampling_frequency=sr)
    f0 = snd.to_pitch(0.01, 60, 600).selected_array["frequency"]
    f0 = f0[f0 > 0]
    floor, ceiling = ((max(60.0, 0.75 * np.percentile(f0, 10)), min(700.0, 1.5 * np.percentile(f0, 90))) if len(f0) >= 20 else (75.0, 500.0))
    manip = call(snd, "To Manipulation", 0.01, floor, ceiling)
    tier = call("Create DurationTier", "d", 0, snd.duration)
    call(tier, "Add point", 0.0, 1 / tempo)
    call([manip, tier], "Replace duration tier")
    return call(manip, "Get resynthesis (overlap-add)").values[0].astype(np.float32)


def prepare(wav_bytes, tempo):
    """Take WAV bytes -> the dry mono take at SR, silence trimmed and stretched by `tempo` (< 1 slower, > 1 faster)."""
    x = trim_silence(decode_wav(wav_bytes))
    if abs(tempo - 1.0) >= NO_STRETCH and len(x) > SR // 20:
        x = psola(x, tempo)
    return x
