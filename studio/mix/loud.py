"""In-process EBU R128 loudness (ITU-R BS.1770-4 K-weighting, 400 ms blocks with 75 % overlap, -70 LUFS absolute and -10 LU relative
gate) and 4x-oversampled true peak, for stereo float32 audio at SR = 48000 Hz. Agrees with `ffmpeg -af ebur128=peak=true`.

Units: LUFS / LU / dBTP as floats; signals are (2, n) float32 (a 1-D mono input counts as one channel).
"""
import numpy as np
from concurrent.futures import ThreadPoolExecutor

from scipy import signal

from .core import SR

# BS.1770-4 K-weighting at 48 kHz: high-shelf pre-filter, then the RLB high-pass
_K1 = (np.array([1.53512485958697, -2.69169618940638, 1.19839281085285]), np.array([1.0, -1.69065929318241, 0.73248077421585]))
_K2 = (np.array([1.0, -2.0, 1.0]), np.array([1.0, -1.99004745483398, 0.99007225036621]))
_SOS = signal.tf2sos(np.convolve(_K1[0], _K2[0]), np.convolve(_K1[1], _K2[1]))
_SUB = SR // 10  # 100 ms sub-block; a gating block is 4 of them (400 ms, hop 100 ms)
_OFFSET = -0.691


def _sub_energy(x2, workers=1):
    """Sum over channels of the mean K-weighted power of every 100 ms sub-block: (sub-blocks,) float64. With workers > 1 the clip is
    cut into chunks filtered in threads, each started 1 s early (the K filters forget in far less) so the result equals the serial one."""
    x2 = x2[None] if x2.ndim == 1 else x2
    n = x2.shape[1] // _SUB
    if n == 0:
        return np.zeros(0)

    def run(c0, c1):
        a = max(0, c0 * _SUB - SR)
        # +1e-20: a decaying filter state would otherwise run through denormal floats (30x slower) in digital silence
        k = signal.sosfilt(_SOS, x2[:, a:c1 * _SUB].astype(np.float64) + 1e-20, axis=1)[:, c0 * _SUB - a:]
        k = k.reshape(k.shape[0], c1 - c0, _SUB)
        return np.einsum("cnf,cnf->n", k, k) / _SUB

    step = -(-n // workers) if workers > 1 else n
    bounds = [(c, min(n, c + step)) for c in range(0, n, step)]
    if len(bounds) == 1:
        return run(*bounds[0])
    with ThreadPoolExecutor(len(bounds)) as ex:
        return np.concatenate(list(ex.map(lambda b: run(*b), bounds)))


def _gated(z):
    """Gated integrated loudness (LUFS) of block mean powers z (summed over channels); -inf when nothing passes the gates."""
    if not len(z):
        return float("-inf")
    z = z[_OFFSET + 10 * np.log10(np.maximum(z, 1e-30)) > -70.0]
    if not len(z):
        return float("-inf")
    thr = _OFFSET + 10 * np.log10(z.mean()) - 10.0
    z = z[_OFFSET + 10 * np.log10(z) > thr]
    return _OFFSET + 10 * np.log10(z.mean()) if len(z) else float("-inf")


def block_power(x2, workers=1):
    """Mean K-weighted power (summed over channels) of every 400 ms block, hop 100 ms: (blocks,) float64. Blocks start at 0, the first
    one ends at 400 ms (the ffmpeg ebur128 grid)."""
    e = _sub_energy(x2, workers)
    return np.convolve(e, np.ones(4) / 4, "valid") if len(e) >= 4 else np.zeros(0)


def integrated_lufs(x2, workers=1):
    """EBU R128 integrated loudness (LUFS) of (2, n) float32 audio; -inf for silence or less than 400 ms."""
    return float(_gated(block_power(x2, workers)))


def momentary_lufs(x2):
    """400 ms momentary loudness (LUFS) per 100 ms hop, ungated: (blocks,)."""
    return _OFFSET + 10 * np.log10(np.maximum(block_power(x2), 1e-30))


_TAPS = 16  # taps per fractional phase of the 4x interpolator (Kaiser-windowed sinc, beta 7): inter-sample peaks within ~0.02 dB
_PHASES = np.stack([np.sinc(np.arange(-_TAPS // 2 + 1, _TAPS // 2 + 1) - d / 4) *
                    np.i0(7.0 * np.sqrt(np.maximum(1 - ((np.arange(-_TAPS // 2 + 1, _TAPS // 2 + 1) - d / 4) / (_TAPS / 2)) ** 2, 0))) / np.i0(7.0)
                    for d in (1, 2, 3)], axis=1).astype(np.float32)  # (taps, 3): the three fractional positions 1/4, 1/2, 3/4


def block_true_peaks(x2, starts, blk, workers=1):
    """True peak (linear amplitude, max over channels) of the blocks [s, s + blk) for every s in starts: the larger of the sample peak and the
    peaks of the 4x-oversampled signal between the block's samples (linear-phase interpolators of _TAPS taps; samples outside x2 are
    zeros). (len(starts),) float32."""
    x2 = x2[None] if x2.ndim == 1 else x2
    pad = _TAPS // 2
    xp = np.pad(x2, ((0, 0), (pad, pad + 1)))
    view = np.lib.stride_tricks.sliding_window_view(xp, _TAPS, axis=1)  # window j = samples j - pad .. j + pad - 1 of x2
    out = np.empty(len(starts), np.float32)
    ar = np.arange(blk)
    step = max(1, min(1024, 2 ** 18 // blk))

    def chunk(a):
        idx = starts[a:a + step, None] + ar[None, :] + 1
        w = view[:, idx]  # (ch, k, blk, taps): samples i - pad + 1 .. i + pad of every block sample i
        y = np.einsum("ckbt,tp->ckbp", w, _PHASES, optimize=False)
        out[a:a + step] = np.maximum(np.abs(w[..., pad - 1]).max(axis=(0, 2)), np.abs(y).max(axis=(0, 2, 3)))

    jobs = range(0, len(starts), step)
    if workers > 1 and len(jobs) > 1:
        with ThreadPoolExecutor(workers) as ex:
            list(ex.map(chunk, jobs))
    else:
        for a in jobs:
            chunk(a)
    return out


def true_peak_db(x2, workers=1):
    """True peak (dBTP) of (2, n) float32 audio: 4x oversampling, maximum absolute value over channels. Only the 1 ms blocks whose sample
    peak is within 3 dB of the loudest one are oversampled (an inter-sample overshoot of more than 3 dB over the block's own sample
    peak does not occur in band-limited audio)."""
    x2 = x2[None] if x2.ndim == 1 else x2
    n = x2.shape[1]
    if n == 0:
        return float("-inf")
    blk = SR // 1000
    nc = -(-n // blk)
    pk = np.abs(np.pad(x2, ((0, 0), (0, nc * blk - n)))).reshape(x2.shape[0], nc, blk).max(axis=(0, 2))
    top = float(pk.max())
    if top <= 0:
        return float("-inf")
    cand = np.flatnonzero(pk >= top * 0.7)
    return float(20 * np.log10(max(top, float(block_true_peaks(x2, cand * blk, blk, workers).max()))))
