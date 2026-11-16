"""Median F0 (Hz) of voiced frames: normalized autocorrelation (FFT, all frames at once) on 40 ms frames of a 2x
decimated signal, 70-300 Hz, voicing threshold 0.5. ~60 ms for a 5 s chunk.
Import median_f0(wav_bytes) or run: python3 f0.py a.wav b.wav ..."""
import io, sys, wave

import numpy as np


def median_f0(wav: bytes) -> float:
    with wave.open(io.BytesIO(wav)) as w:
        sr = w.getframerate()
        x = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32)
    x = x[: len(x) // 2 * 2].reshape(-1, 2).mean(1)  # 2x decimation (box filter): F0 band stays far below Nyquist
    sr //= 2
    n, hop = int(0.04 * sr), int(0.01 * sr)
    if len(x) <= n:
        return float('nan')
    frames = np.lib.stride_tricks.sliding_window_view(x, n)[::hop]
    frames = frames - frames.mean(1, keepdims=True)
    frames = frames[frames.std(1) >= 200]  # drop silence
    if not len(frames):
        return float('nan')
    spec = np.fft.rfft(frames, 2 * n)
    ac = np.fft.irfft(spec * spec.conj(), 2 * n)[:, :n]
    ac /= ac[:, :1] + 1e-9
    lo, hi = int(sr / 300), int(sr / 70)
    lag = lo + ac[:, lo:hi].argmax(1)
    voiced = ac[np.arange(len(ac)), lag] > 0.5
    return round(float(np.median(sr / lag[voiced])), 1) if voiced.any() else float('nan')


if __name__ == '__main__':
    for p in sys.argv[1:]:
        print(p, median_f0(open(p, 'rb').read()))
