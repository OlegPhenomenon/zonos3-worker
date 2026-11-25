"""Mastering of the dub mix: dialogue focus (M&E duck), tonal balance, loudness and a look-ahead true-peak limiter (Netflix dubbing
guide: the dub is mastered to the original's loudness and balance, dialogue stays in front, no peak over -2 dBTP).

The lab masters the FINISHED whole mix, measuring it against the original mix (tonal_match, loudness_and_peak). The product renders
30 s windows long before any other window exists, so the finishing is a pure function of the window with every number taken from
the ORIGINAL at analysis time (`analyze`):
  * loudness: the original mix's LUFS is the target; the gain is the one at which the limiter, run on the original mix (the dub mix
    is close to it: the voice follows the original's levels over the original's own M&E), delivers exactly that loudness, lowered
    where the limiter would take more than 3 dB for more than MAX_HEAVY_LIMIT_S seconds per minute (hot masters);
  * tone: a fixed gentle high shelf (calibrated on the lab's tonal_match of standup_en / panel_en, which both took ~1 dB off 4-13 kHz);
  * peaks: a limiter whose memory is ~100 ms, detecting TRUE peaks (4x oversampled) so a single pass holds the ceiling.
`finish` therefore concatenates seamlessly when each window carries MIN_PRE_S / MIN_POST_S of context that the caller crops.

Units: seconds, dB, samples at core.SR (48 kHz); mixes are (2, n) float32; level vectors are 10 ms frames (core.FR).
"""
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import scipy.fft as sfft
from scipy import signal

from . import loud
from .core import FR, SR, frame_power

TRUE_PEAK_DB = -2.0  # Netflix sound specification: maximum true peak -2 dBTP
CEILING_DB = -2.6  # limiter ceiling: the AAC encoder overshoots the true peak by 0.6-1.1 dB (measured on the two lab clips), the delivery limit after decode is -1 dBTP
RELEASE_MS = 80.0
MAX_HEAVY_LIMIT_S = 1.0  # seconds per minute of programme in which the limiter may take more than 3 dB
CREST_DB = 2.0  # the dub mix is peakier than the original at equal loudness (lab panel_en: 2.3 s/min over 3 dB where the original mix gives 0);
# the limiter simulation on the original mix raises its peaks by this. Fitted on the lab's standup_en / panel_en results (see report).
VBR_CAP_DB = 10.0  # a voice 10 dB over the M&E is clearly intelligible: the original's VBR is not matched above this
VBR_TOL_DB = 1.0  # the dub may sit this much below the original's voice-to-background ratio before the M&E is ducked
DUCK_MAX_DB = 3.0
PAN_LIMIT_DB = 3.0
TONE_SHELF_HZ = 5500.0  # high shelf corner of the fixed tonal correction
TONE_SHELF_DB = -1.3  # calibrated: lab tonal_match EQ of standup_en / panel_en averaged -0.5 .. -1.5 dB over 3.2-12.8 kHz
MIN_PRE_S = 1.0  # context the caller must give before the part to keep: limiter release 80 ms (1 s = 12 e-foldings) + shelf
MIN_POST_S = 0.1  # and after it: look-ahead 3 ms + smoothing 3 ms
BLK = SR // 1000  # limiter control block: 1 ms

_EQ_CENTRES = 50 * 2 ** (np.arange(0, 107) / 12)  # 50 Hz .. 21 kHz, 1/12 octave (the lab's EQ grid; reference spectrum of the doc)
_LTAS_N = 4096


# ----------------------------------------------------------------------------- limiter

def _block_peaks(x2):
    """Peak |sample| over channels per 1 ms block: (n // BLK,) float32."""
    n = x2.shape[1] // BLK
    return np.abs(x2[:, : n * BLK]).reshape(x2.shape[0], n, BLK).max(axis=(0, 2))


def _release(dm, decay, chunk=8192):
    """Exponential release of the reduction dm = 1 - gain (>= 0): d[i] = max(dm[i], d[i-1] * decay), vectorised as a running maximum
    of dm[j] * decay^-j (in chunks, carrying the state, to keep decay^-j finite)."""
    out = np.empty_like(dm)
    j = np.arange(chunk)
    scale = decay ** (-j.astype(np.float64))
    carry = 0.0
    for s in range(0, len(dm), chunk):
        seg = dm[s:s + chunk].astype(np.float64) * scale[: len(dm[s:s + chunk])]
        seg[0] = max(seg[0], carry * decay)
        d = np.maximum.accumulate(seg) / scale[: len(seg)]
        out[s:s + len(d)] = d
        carry = d[-1]
    return out


_HANN7 = np.hanning(9)[1:-1] / np.hanning(9)[1:-1].sum()


def _limiter_gain(pk, ceiling_db, release_ms):
    """Gain per 1 ms block for block peaks pk: gain reduced 3 ms ahead of a peak (minimum over +-3 blocks), exponential release,
    7-tap Hann smoothing. float64, <= 1."""
    thr = 10 ** (ceiling_db / 20)
    g = np.minimum(1.0, thr / np.maximum(pk, 1e-9))
    gm = g.copy()
    for s in range(1, 4):
        gm[s:] = np.minimum(gm[s:], g[:-s])
        gm[:-s] = np.minimum(gm[:-s], g[s:])
    gr = 1.0 - _release(1.0 - gm, float(np.exp(-1.0 / release_ms)))
    return np.convolve(np.pad(gr, 3, mode="edge"), _HANN7, mode="valid")


def _gain_to_samples(gs, n):
    """Block gain (control points at block centres) interpolated linearly to n samples, float32."""
    t = np.arange(len(gs)) * BLK + BLK / 2
    return np.interp(np.arange(n), t, gs).astype(np.float32)


def _stats(gs):
    red = -20 * np.log10(np.maximum(gs, 1e-6))
    return {"max_reduction_db": round(float(red.max()), 2) if len(red) else 0.0, "seconds_over_1db": round(float((red > 1).sum() / 1000), 1),
            "seconds_over_3db": round(float((red > 3).sum() / 1000), 1)}


def limit_peaks(x2, ceiling_db, release_ms):
    """Look-ahead peak limiter on a stereo (2, n) signal (lab limit_peaks): 1 ms blocks, gain reduced 3 ms ahead of a peak,
    exponential release (e-folding release_ms), 7 ms Hann smoothing; never lets a block exceed the ceiling. The gain curve at a
    sample depends on the ~5 release time constants before it, nothing after it (but 6 ms). Returns (limited float32, stats)."""
    gs = _limiter_gain(_block_peaks(x2), ceiling_db, release_ms)
    return x2 * _gain_to_samples(gs, x2.shape[1]), _stats(gs)


def _true_peak_blocks(x2, pk, floor, workers=4):
    """Block peaks pk raised to the 4x-oversampled (true) peak of the blocks whose sample peak is >= floor (others cannot overshoot
    the ceiling that is at most 1.43 x floor)."""
    cand = np.flatnonzero(pk >= floor)
    if not len(cand):
        return pk
    out = pk.copy()
    out[cand] = loud.block_true_peaks(x2, cand * BLK, BLK, workers)
    return out


# ----------------------------------------------------------------------------- tone

def _high_shelf_sos(f0, gain_db, sr=SR):
    """RBJ high shelf (slope 1) as one second-order section."""
    a_ = 10 ** (gain_db / 40)
    w0 = 2 * np.pi * f0 / sr
    cs, al = np.cos(w0), np.sin(w0) / 2 * np.sqrt(2)
    sq = 2 * np.sqrt(a_) * al
    b = [a_ * ((a_ + 1) + (a_ - 1) * cs + sq), -2 * a_ * ((a_ - 1) + (a_ + 1) * cs), a_ * ((a_ + 1) + (a_ - 1) * cs - sq)]
    a = [(a_ + 1) - (a_ - 1) * cs + sq, 2 * ((a_ - 1) - (a_ + 1) * cs), (a_ + 1) - (a_ - 1) * cs - sq]
    return signal.tf2sos(b, a)


def _ltas_db(x, workers):
    """Long-term spectrum (dB) of mono x at the 1/12-octave EQ centres: mean power spectrum of the 4096-sample Hann frames (hop 2048)
    above -55 dBFS, Gaussian-smoothed over 1/4 octave in log frequency (lab: dub_mix.ltas / smooth_log)."""
    win = np.hanning(_LTAS_N).astype(np.float32)
    nfr = (len(x) - _LTAS_N) // (_LTAS_N // 2) + 1
    if nfr < 1:
        return np.full(len(_EQ_CENTRES), -200.0)
    view = np.lib.stride_tricks.sliding_window_view(x, _LTAS_N)[:: _LTAS_N // 2]
    gate = 10 ** (-55.0 / 10)

    def chunk(a):
        v = view[a:a + 1024]
        v = v[np.einsum("ij,ij->i", v, v) / _LTAS_N > gate]
        if not len(v):
            return np.zeros(_LTAS_N // 2 + 1), 0
        s = sfft.rfft(v * win, axis=1)
        return (s.real ** 2 + s.imag ** 2).sum(0, dtype=np.float64), len(v)

    with ThreadPoolExecutor(workers) as ex:
        res = list(ex.map(chunk, range(0, nfr, 1024)))
    p = sum(r[0] for r in res) / max(sum(r[1] for r in res), 1)
    f = np.fft.rfftfreq(_LTAS_N, 1 / SR)[1:]
    w = np.exp(-0.5 * (np.log2(f[None, :] / _EQ_CENTRES[:, None]) / 0.25) ** 2)
    return 10 * np.log10((w * p[None, 1:]).sum(1) / w.sum(1) + 1e-20)


# ----------------------------------------------------------------------------- analysis

def _peaks_threaded(x2, workers):
    """_block_peaks of a long mix in threads (chunks on block boundaries)."""
    n = x2.shape[1] // BLK
    step = -(-n // workers)
    with ThreadPoolExecutor(workers) as ex:
        return np.concatenate(list(ex.map(lambda c: _block_peaks(x2[:, c * BLK:min(n, c + step) * BLK]), range(0, n, step))))


def _gated_from_sub(e):
    """Gated integrated loudness of 100 ms sub-block energies e."""
    return loud._gated(np.convolve(e, np.ones(4) / 4, "valid")) if len(e) >= 4 else float("-inf")


def _solve(pk, e, target_lufs, dur_min, crest_db=0.0):
    """Gain (dB) for the original mix (block peaks pk, K-weighted 100 ms energies e): the largest g such that the limited mix reaches
    target_lufs AND the limiter takes more than 3 dB for at most MAX_HEAVY_LIMIT_S per minute. Returns (gain_db, info)."""
    nsub = len(e)
    nb = min(len(pk), nsub * 100) // 100 * 100

    def run(g_db):
        g = 10 ** (g_db / 20)
        gs = _limiter_gain(pk * (g * 10 ** (crest_db / 20)), CEILING_DB, RELEASE_MS)
        gsq = (gs[:nb] ** 2).reshape(-1, 100).mean(1)
        lu = _gated_from_sub(e[: len(gsq)] * g * g * gsq)
        heavy = float((gs < 10 ** (-3 / 20)).sum() / 1000) / dur_min
        return lu, heavy, gs

    def bisect(pred, lo=-8.0, hi=8.0, it=9):
        for _ in range(it):
            mid = 0.5 * (lo + hi)
            lo, hi = (mid, hi) if pred(mid) else (lo, mid)
        return lo

    with ThreadPoolExecutor(2) as ex:
        ft = ex.submit(bisect, lambda g: run(g)[0] < target_lufs)
        fh = ex.submit(bisect, lambda g: run(g)[1] <= MAX_HEAVY_LIMIT_S)
        g_t, g_h = ft.result(), fh.result()
    g = min(g_t, g_h)
    lu, heavy, gs = run(g)
    return g, {"gain_loudness_db": round(g_t, 2), "gain_heavy_db": round(g_h, 2), "predicted_lufs": round(float(lu), 2),
               "predicted_heavy_s_per_min": round(heavy, 2), "predicted_limiter": _stats(gs),
               "reason": "original loudness reached" if g_t <= g_h else "target lowered: the limiter would take > 3 dB for more than %g s/min" % MAX_HEAVY_LIMIT_S}


def analyze(clip, regions, workers=8):
    """Everything the finishing takes from the ORIGINAL. Returns a picklable dict:
      target_lufs (original mix integrated loudness), orig_true_peak_db, gain_db (static gain of the finishing, see module doc),
      gain_info (both constraints and the prediction), crest_db, ceiling_db, release_ms, shelf {f0_hz, gain_db} (fixed tonal correction),
      voice_peak_db (peak of the original speech stem inside the regions: ceiling of the dub voice limiter),
      ltas_hz / ltas_db (long-term spectrum of the original mix at 1/12-octave centres: the tonal reference), duration_s."""
    mix = clip.mix2
    e = loud._sub_energy(mix, workers)
    target = loud._gated(np.convolve(e, np.ones(4) / 4, "valid")) if len(e) >= 4 else -23.0
    with ThreadPoolExecutor(3) as ex:
        f_pk = ex.submit(_peaks_threaded, mix, max(1, workers // 2))
        f_tp = ex.submit(loud.true_peak_db, mix, max(1, workers // 2))
        f_lt = ex.submit(_ltas_db, 0.5 * (mix[0] + mix[1]), max(1, workers // 2))
        sp = clip.speech2
        voice_peak = max([float(np.abs(sp[:, int(a * SR):int(b * SR)]).max()) for a, b in regions if b > a] or [1e-6])
        pk, tp, ltas = f_pk.result(), f_tp.result(), f_lt.result()
    gain, info = _solve(pk, e, target, clip.duration_s / 60, CREST_DB)
    return {"target_lufs": float(target), "orig_true_peak_db": float(tp), "gain_db": float(gain), "gain_info": info,
            "crest_db": CREST_DB, "ceiling_db": CEILING_DB, "release_ms": RELEASE_MS, "shelf": {"f0_hz": TONE_SHELF_HZ, "gain_db": TONE_SHELF_DB},
            "voice_peak_db": float(20 * np.log10(voice_peak + 1e-12)), "ltas_hz": _EQ_CENTRES.copy(), "ltas_db": ltas,
            "duration_s": float(clip.duration_s)}


# ----------------------------------------------------------------------------- finishing

def finish(mix, doc, pre, post):
    """Master a window mix (2, n) float32 that carries `pre` leading and `post` trailing context samples (at least MIN_PRE_S /
    MIN_POST_S seconds; `pre` a multiple of 48 samples, or the same `pre` for every window of a clip, so the 1 ms limiter grids of
    adjacent windows coincide): fixed tonal shelf, loudness gain, true-peak limiter at doc['ceiling_db']. Returns only the middle
    part, (2, n - pre - post) float32."""
    x = mix.astype(np.float32, copy=False)
    sh = doc["shelf"]
    if abs(sh["gain_db"]) > 1e-3:
        x = signal.sosfilt(_high_shelf_sos(sh["f0_hz"], sh["gain_db"]), x, axis=1).astype(np.float32)
    x = x * np.float32(10 ** (doc["gain_db"] / 20))
    thr = 10 ** (doc["ceiling_db"] / 20)
    pk = _true_peak_blocks(x, _block_peaks(x), 0.7 * thr)
    gs = _limiter_gain(pk, doc["ceiling_db"], doc["release_ms"])
    out = x * _gain_to_samples(gs, x.shape[1])
    return out[:, pre: x.shape[1] - post]


# ----------------------------------------------------------------------------- dialogue focus

def apply_gain_curve(x, g_db):
    """x * 10^(g/20) with the 10 ms grid g_db (dB per frame, frame k centred at (k + 0.5) FR) interpolated linearly to samples.
    x is (n,) or (channels, n); frames past the curve hold its last value."""
    t = np.arange(len(g_db)) * FR + FR / 2
    gain = (10 ** (np.interp(np.arange(x.shape[-1]), t, g_db) / 20)).astype(np.float32)
    return x * gain


def apply_duck(bg2, g_db):
    """The M&E (2, n) lowered by the duck curve g_db (dB >= 0, 10 ms frames)."""
    return apply_gain_curve(bg2, -g_db)


def _smooth_gain(g, att, rel):
    """One-pole smoothing of a duck gain (dB >= 0, 10 ms frames) with attack / release in frames."""
    out = np.empty_like(g)
    prev = 0.0
    ka, kr = 1 - np.exp(-1 / att), 1 - np.exp(-1 / rel)
    for i, v in enumerate(g.tolist()):
        prev += (ka if v > prev else kr) * (v - prev)
        out[i] = prev
    return out


def _smooth_db(p):
    """100 ms integration (syllable-level masking) of 10 ms frame powers, dB. The SAME function measures original and dub."""
    return 10 * np.log10(np.convolve(p, np.ones(10) / 10, "same") + 1e-12)


def _part_vbr(lv, lb, f0, f1):
    """Median voice-to-background ratio (dB) over the active frames (within 20 dB of the part's peak) of [f0, f1); None when the part
    has no voice or the M&E is silent there (nothing to preserve)."""
    v, b = lv[f0:f1], lb[f0:f1]
    if len(v) < 3 or v.max() < -90:
        return None
    act = v >= v.max() - 20
    if act.sum() < 3 or b[act].max() < -90:
        return None
    return float(np.median(v[act] - b[act]))


def duck_curve(p_stem_mean, p_bg_orig, voice2, bg2, parts):
    """M&E duck curve (dB >= 0, one value per 10 ms frame, frame 0 = sample 0 of the arrays) that keeps the dub's voice-to-background
    ratio (VBR) at the original's, per part: target = median VBR (100 ms integration, active frames) of the original speech vs the
    original M&E in the part's slot, at most VBR_CAP_DB; got = the same statistic of the rendered voice vs the M&E in the mix over the
    dub span. A shortfall over VBR_TOL_DB ducks the M&E over the span by the shortfall (<= DUCK_MAX_DB), attack 50 ms / release 300 ms.
    p_stem_mean / p_bg_orig: 10 ms frame powers (channel mean) of the ORIGINAL speech stem / M&E over the same time span as the window;
    voice2 / bg2: (2, n) rendered voice / M&E of the window; parts: [{"o0", "o1", "d0", "d1"}] seconds from the window start
    (original slot, dub span). Needs MIN_PRE_S of context before the first part to be seam-free (release 300 ms)."""
    nf = voice2.shape[1] // FR
    lv = _smooth_db(frame_power(voice2).mean(0)[:nf])
    lb = _smooth_db(frame_power(bg2).mean(0)[:nf])
    so, bo = _smooth_db(p_stem_mean), _smooth_db(p_bg_orig)
    deficit = np.zeros(nf)
    for p in parts:
        t = _part_vbr(so, bo, int(p["o0"] * 100), int(p["o1"] * 100) + 1)
        f0, f1 = int(p["d0"] * 100), min(int(p["d1"] * 100) + 1, nf)
        got = _part_vbr(lv, lb, f0, f1)
        if t is None or got is None:
            continue
        short = min(t, VBR_CAP_DB) - VBR_TOL_DB - got
        if short > 0:
            deficit[f0:f1] = np.maximum(deficit[f0:f1], min(short, DUCK_MAX_DB))
    return _smooth_gain(deficit, 5, 30)


# ----------------------------------------------------------------------------- stereo

def speaker_balance(clip, start_s, end_s):
    """L/R balance (dB, L - R, clipped +-PAN_LIMIT_DB) of the original voice over the active frames (within 20 dB of the slot's loudest
    frame) of the slot [start_s, end_s] of the speech stem; 0 for a silent slot."""
    a, b = int(start_s * 100), int(end_s * 100) + 1
    w = frame_power(clip.speech2[:, a * FR: b * FR])
    tot = w.sum(0)
    if not len(tot) or tot.max() <= 1e-10:
        return 0.0
    sel = tot >= tot.max() * 10 ** -2
    return float(np.clip(10 * np.log10((w[0][sel].sum() + 1e-12) / (w[1][sel].sum() + 1e-12)), -PAN_LIMIT_DB, PAN_LIMIT_DB))
