"""Room-tone bed: the scene's own noise floor, measured on the ORIGINAL mix and synthesised under the dub.

The separated M&E is digital silence under speech, the ambience (mic noise, room tone, crowd murmur) leaves with the original
voice. `analyze` measures it once (per 1/3-octave band the minimum statistics of the smoothed STFT power of the original mix over a
window that holds one real pause, bias-corrected on synthetic noise, plus the real L/R coherence of the floor frames);
`bed_window` synthesises the bed for any absolute sample range as a PURE function of the sample position: frames of an overlap-add
synthesis (N = 4096, hop 1024) whose noise comes from a counter-based generator seeded per block of frames, so two adjacent
windows concatenated are bit-identical to one call.

Units: sample rate core.SR (48 kHz), x2 is (2, n) float32; powers are per-bin mean power (white noise of variance s gives s).
"""
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import scipy.fft as sfft

from .core import SR

N = 4096  # STFT / synthesis frame (85 ms at 48 kHz)
H = 1024  # hop (21 ms)
SMOOTH = 4  # frames of the power smoothing
WIN = np.hanning(N).astype(np.float32)
WIN_E = float((WIN.astype(np.float64) ** 2).sum())  # power normalisation of the analysis window
SYN = np.sqrt(WIN).astype(np.float32)  # synthesis window: overlapped hann powers sum to a constant at 75 % overlap
BAND_MARGIN = 2.0  # floor frames: smoothed band power within 3 dB of the floor
SEED = 5
BLOCK = 16  # frames per noise block (one seeded generator each)
_PAD_S = 2.0  # the floor window is the longest speech run plus this


# ----------------------------------------------------------------------------- bands

def band_edges(sr=SR):
    """Band edges in Hz: [0, 100, 100 * 2^(1/3), ...] 1/3 octave up to Nyquist (last edge above Nyquist)."""
    e = [0.0]
    k = 0
    while 100 * 2 ** (k / 3) < sr / 2 / 2 ** (1 / 6):
        e.append(100 * 2 ** (k / 3))
        k += 1
    e.append(sr / 2 + 1)
    return np.array(e)


def _band_layout(edges, sr=SR):
    """First rfft bin of every band and the number of bins in it (the bands partition the bins contiguously)."""
    f = np.fft.rfftfreq(N, 1 / sr)
    first = np.searchsorted(f, edges[:-1], side="left")
    return first, np.diff(np.append(first, len(f)))


def longest_run_s(regions, gap=0.3):
    """Longest stretch of speech (s) without a pause of `gap` seconds; regions = [(start_s, end_s)]."""
    iv = sorted(regions)
    if not iv:
        return 0.0
    a, b = iv[0]
    best = 0.0
    for s, e in iv[1:]:
        if s - b < gap:
            b = max(b, e)
        else:
            best = max(best, b - a)
            a, b = s, e
    return max(best, b - a)


# ----------------------------------------------------------------------------- floor tracking

def _frames(x, a, b):
    """Frames a..b-1 of mono x as an (b-a, N) strided view."""
    return np.lib.stride_tricks.sliding_window_view(x[a * H: (b - 1) * H + N], N)[::H]


def _pmap(fn, items, workers):
    """fn over items in `workers` threads (numpy / FFT release the GIL), results in order."""
    if workers <= 1 or len(items) <= 1:
        return [fn(i) for i in items]
    with ThreadPoolExecutor(workers) as ex:
        return list(ex.map(fn, items))


def _band_power(x, lay, workers):
    """(frames, bands) mean power per bin of the Hann-windowed STFT of mono x (white noise of variance s gives s). float64."""
    nb = (len(x) - N) // H + 1
    view = np.lib.stride_tricks.sliding_window_view(x, N)[::H]
    first, counts = lay
    norm = WIN_E * counts

    def chunk(a):
        spec = sfft.rfft(view[a:min(nb, a + 256)] * WIN, axis=1)
        return np.add.reduceat(spec.real ** 2 + spec.imag ** 2, first, axis=1, dtype=np.float64) / norm

    return np.concatenate(_pmap(chunk, list(range(0, nb, 256)), workers))


def _moving_mean(p, w):
    """Centered mean over w frames, per column."""
    c = np.cumsum(np.pad(p, ((w // 2, w - w // 2 - 1), (0, 0)), mode="edge"), axis=0)
    c = np.vstack([np.zeros((1, p.shape[1])), c])
    return (c[w:] - c[:-w]) / w


def _sliding_min(p, w):
    """Minimum over a w-frame window per column (van Herk / Gil-Werman, O(n)). The window is centered; within w/2 of either end it
    keeps its full length and sits at the end instead of shrinking (a half window may hold no pause)."""
    n = len(p)
    if n <= w:
        return np.broadcast_to(p.min(0), p.shape).copy()
    q = np.pad(p, ((0, -(-n // w) * w - n), (0, 0)), mode="edge")
    blocks = q.reshape(-1, w, p.shape[1])
    fwd = np.minimum.accumulate(blocks, axis=1).reshape(-1, p.shape[1])
    bwd = np.minimum.accumulate(blocks[:, ::-1], axis=1)[:, ::-1].reshape(-1, p.shape[1])
    m = np.minimum(bwd[: n - w + 1], fwd[w - 1: n])
    return m[np.clip(np.arange(n) - w // 2, 0, n - w)]


def _min_bias(lay, w_frames, workers, seed=3):
    """E[smoothed band power] / E[sliding minimum of it] for stationary white noise, per band (the bias of the tracker)."""
    rng = np.random.default_rng(seed)
    nf = 4 * w_frames + SMOOTH
    x = rng.standard_normal(nf * H + N).astype(np.float32)
    p = _moving_mean(_band_power(x, lay, workers), SMOOTH)
    m = _sliding_min(p, w_frames)
    lo, hi = w_frames, len(p) - w_frames
    return p[lo:hi].mean(0) / m[lo:hi].mean(0)


def _coherence(x2, p, fl, lay, workers):
    """Real L/R coherence per band over the frames at the floor (both channels within BAND_MARGIN of their floor); nan where fewer than
    30 frame-bands qualify. Returns (rho, frame counts)."""
    first = lay[0]
    nbands = len(first)
    ok = (p[0] <= BAND_MARGIN * fl[0]) & (p[1] <= BAND_MARGIN * fl[1])
    views = [np.lib.stride_tricks.sliding_window_view(x2[c], N)[::H] for c in (0, 1)]
    frames_ok = np.flatnonzero(ok.any(1))

    def chunk(a):
        idx = frames_ok[a:a + 256]
        spl, spr = (sfft.rfft(v[idx] * WIN, axis=1) for v in views)
        sel = ok[idx]
        out = np.empty((3, nbands))
        for i, v in enumerate((spl.real ** 2 + spl.imag ** 2, spr.real ** 2 + spr.imag ** 2, spl.real * spr.real + spl.imag * spr.imag)):
            out[i] = (np.add.reduceat(v, first, axis=1, dtype=np.float64) * sel).sum(0)
        return out

    acc = sum(_pmap(chunk, list(range(0, len(frames_ok), 256)), workers), np.zeros((3, nbands)))
    cnt = ok.sum(0).astype(np.float64)
    rho = acc[2] / np.sqrt(acc[0] * acc[1] + 1e-30)
    return np.where(cnt >= 30, np.clip(rho, 0, 1), np.nan), cnt


def analyze(clip, regions, workers=8):
    """Floor + width of the original mix (clip.mix2). The window of the minimum tracker is the longest speech run (regions, s) + 2 s.
    Returns a picklable dict: edges (bands+1,), window_frames, bias (bands,), floor (2, frames, bands) float32 power per bin,
    rho (bands,) L/R coherence of the floor, rho_frames, n_samples, sr."""
    x2 = np.ascontiguousarray(clip.mix2)
    n = x2.shape[1]
    edges = band_edges(SR)
    lay = _band_layout(edges)
    w = int(round((longest_run_s(regions) + _PAD_S) * SR / H)) | 1
    nb = (n - N) // H + 1
    x2 = x2[:, : (nb - 1) * H + N]
    bias = _min_bias(lay, w, workers)
    pw = [_band_power(x2[c], lay, workers) for c in (0, 1)]
    fl = [_moving_mean(_sliding_min(_moving_mean(p, SMOOTH), w) * bias, max(w // 2, 1)) for p in pw]
    pw = [_moving_mean(p, SMOOTH) for p in pw]
    rho, cnt = _coherence(x2, pw, fl, lay, workers)
    # unreliable bands take the nearest measured neighbour (mono-ish floors are smooth across frequency)
    if np.isnan(rho).all():
        rho = np.zeros(len(edges) - 1)
    else:
        idx = np.arange(len(rho))
        good = ~np.isnan(rho)
        rho = np.interp(idx, idx[good], rho[good])
    return {"edges": edges, "window_frames": w, "bias": bias, "floor": np.stack(fl).astype(np.float32), "rho": rho,
            "rho_frames": cnt, "n_samples": n, "sr": SR}


# ----------------------------------------------------------------------------- synthesis

_TABLES = {}
_TRI_SD = float(np.sqrt(2 * 65536.0 ** 2 / 12))  # std of the sum of two uniform int16


def _tables(edges, rho_bands):
    """Interpolation of the band values to rfft bins over log frequency: (i0, i1, wt, a, b) with a = sqrt(rho), b = sqrt(1 - rho)."""
    key = (len(edges), float(edges[1]), float(rho_bands.sum()))
    t = _TABLES.get(key)
    if t is None:
        f = np.fft.rfftfreq(N, 1 / SR)
        centres = np.sqrt(edges[:-1] * edges[1:])
        centres[0] = edges[1] / 2
        lf = np.log(np.maximum(f, 1.0))
        lc = np.log(centres)
        rho = np.interp(lf, lc, rho_bands)
        i1 = np.clip(np.searchsorted(lc, lf), 1, len(lc) - 1)
        i0 = i1 - 1
        wt = np.clip((lf - lc[i0]) / (lc[i1] - lc[i0]), 0, 1).astype(np.float32)
        t = _TABLES[key] = (i0, i1, wt, np.sqrt(rho).astype(np.float32), np.sqrt(1 - rho).astype(np.float32))
    return t


def _noise_frames(est, b, tab):
    """Synthesis frames t in [b*BLOCK, (b+1)*BLOCK) clipped to the clip: (2, m, N) float32 windowed noise frames of the floor."""
    i0, i1, wt, a, bb = tab
    nfr = est["floor"].shape[1]
    t0, t1 = b * BLOCK, min(nfr, (b + 1) * BLOCK)
    m = t1 - t0
    nbin = N // 2 + 1
    # raw 64-bit words of the block's own generator -> 4 x int16 each; two summed give a triangular variate per real/imag part (the
    # irfft of 2049 such bins is Gaussian by the central limit theorem, the cost is a tenth of standard_normal)
    raw = np.random.PCG64([SEED, b]).random_raw(-(-2 * 3 * BLOCK * nbin * 2 // 4)).view(np.int16)
    k = 2 * 3 * BLOCK * nbin
    r = raw[0:2 * k:2].astype(np.int32)
    r += raw[1:2 * k:2]
    r = r.astype(np.float32).reshape(2, 3, BLOCK, nbin)[:, :, :m]
    z = r[0] + 1j * r[1]  # complex, variance (2 * 65536^2 / 12) per part; normalised in the scale below
    scale = np.float32(np.sqrt(N / 2) / np.sqrt(2) / _TRI_SD) * SYN
    out = np.empty((2, m, N), np.float32)
    for c in (0, 1):
        lfb = np.log(est["floor"][c, t0:t1].astype(np.float32) + np.float32(1e-30))
        gain = np.exp(np.float32(0.5) * (lfb[:, i0] * (1 - wt) + lfb[:, i1] * wt))
        spec = gain * (a * z[2] + bb * z[c])
        out[c] = sfft.irfft(spec, N, axis=1) * scale
    return out, t0


_POOL = []


def bed_window(est, i0, n):
    """The room-tone bed for absolute samples [i0, i0 + n) of the clip: (2, n) float32. Pure in the absolute position: the result of
    any split of a range into windows concatenates to the result of the whole range (bit-identical). Samples outside the clip's
    frame coverage (before 0, past the analysed length) are zero."""
    out = np.zeros((2, n), np.float32)
    nfr = est["floor"].shape[1]
    f0 = max(0, (i0 - N) // H + 1)  # first frame that reaches i0
    f1 = min(nfr, (i0 + n - 1) // H + 1)  # one past the last frame that starts before the window end
    if n <= 0 or f1 <= f0 or i0 + n <= 0:
        return out
    tab = _tables(est["edges"], est["rho"])
    nslots = (f1 - f0) + N // H - 1
    acc = np.zeros((2, nslots, H), np.float32)  # absolute slot s covers samples [s*H, (s+1)*H), buffer slot = s - f0
    blocks = list(range(f0 // BLOCK, (f1 - 1) // BLOCK + 1))
    if not _POOL:
        _POOL.append(ThreadPoolExecutor(4))
    # a slot sums the frames s, s-1, s-2, s-3 in this order for every window: bit-identical overlap-add
    for fr, t0 in _POOL[0].map(lambda b: _noise_frames(est, b, tab), blocks):
        lo, hi = max(t0, f0), min(t0 + fr.shape[1], f1)
        seg = fr[:, lo - t0: hi - t0].reshape(2, hi - lo, N // H, H)
        for j in range(N // H):
            acc[:, lo - f0 + j: hi - f0 + j] += seg[:, :, j]
    flat = acc.reshape(2, -1)
    s0 = i0 - f0 * H
    a, b = max(0, -s0), min(n, flat.shape[1] - s0)
    out[:, a:b] = flat[:, s0 + a: s0 + b]
    return out
