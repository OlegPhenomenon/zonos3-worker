"""Recording defects of the original per acoustic scene - and what the dub must NOT copy from them.

Port of the lab's `recording_defects.py` (Netflix Mixing Style Guide for Dubbed Content: do not copy technical defects of the
original - clipping, wind / handling noise, a bandwidth that is low only because of the recording or its codec, heavy noise - keep
deliberate futzes: phone, TV, radio, speakerphone, intercom, helmet). The channel matcher copies EVERYTHING the stem shows; this
module decides, per scene of the channel document, what is a defect, what is a futz and what treatment the mixer applies. All
numbers are measured from the audio of the scene (speech stem, original mix, speech slots = the nominal speech regions, pauses =
the gaps between them); no per-clip tuning. Differences to the lab, all forced by the product: one speaker "all" (the speaker
reference spectrum is the LTAS of the whole clip), slots = the blind speech regions, no summary rows; sample-count constants are
re-derived from 48 kHz (frames of 0.094 s / 0.375 s, 1 ms = 48 samples); the true-peak oversampling only visits 8192-sample blocks
whose sample peak is >= TP_BLOCK_MIN.

API
    analyze(clip, chan, regions, workers=8) -> dict
        {"rule", "params", "reference_hp_hz", "codec_lp_hz", "scenes": [{"id","start","end","speech_s","measures","flags",
         "verdict","treatment"}]}  (lab defects.json). verdict "clean" | "futz" | "defect"; treatment = {"relax_hp_hz",
         "relax_lp_hz", "copy_dynamics", "bed": {"highpass_hz", "gain_db"}, "futz": None | {"kind","f_lo_hz","f_hi_hz"}, "reason"}.
        `chan` = channel document (scenes with id, start, end, speech_s, borrowed_from, dynamics), `regions` = core.speech_regions
        in seconds.

Method (the lab's module docstring, short)
1. CLIPPING: >= 3 consecutive samples within 0.1 % of the scene's ceiling per channel of the mix (events/min), full-scale clamps,
   4x-oversampled true-peak overs; stem overshoot above full scale. Flag when the mix has >= 1 event/min and >= 3 events.
   copy_dynamics = False when the scene is hot AND the fitted compressor is limiter-like (ratio >= 10 or release < 10 ms).
2. WIND: 20-200 Hz bursts >= 6 dB over the quiet reference of the pauses, voice band not up, noise-like (spectral flatness), in
   >= 5 % of the pause frames -> bed high-pass at 200 Hz.
3. BANDWIDTH: stem LTAS over the speech regions (1/6 octave smoothing, 1/12 octave grid). FUTZ = steep stable high-pass edge at
   200-500 Hz AND steep stable low-pass edge at 2.5-8 kHz; anything else that limits the band is a DEFECT (relax_hp_hz /
   relax_lp_hz); scenes with < 5 s of speech inherit from the nearest long scene by LTAS signature.
4. NOISE: speech-band SNR of the stem floor and of the mix floor in the pauses < 20 dB -> bed attenuated (gain_db <= 0).
"""
import math
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp

import numpy as np
from scipy import fft as sfft
from scipy.signal import resample_poly
from threadpoolctl import threadpool_limits

from . import core

SR = core.SR


def _fast(n):
    return int(sfft.next_fast_len(int(round(n)), real=True))


# ----------------------------------------------------------------------------- clipping
CLIP_RUN = 3  # consecutive samples on the ceiling = a clipped event (QC convention)
CLIP_TOL = 1e-3  # "on the ceiling" = within 0.1 % of it
FULL_SCALE = 1.0 - 2.0 ** -15  # s16 clamp of the mix file
EVENT_MERGE_S = 0.001  # samples / overs closer than 1 ms are one event
OVERSAMPLE = 4  # true peak (ITU-R BS.1770)
CLIP_MIN_EVENTS = 3
CLIP_EVENTS_PER_MIN = 1.0
PILE_BAND_DB = 0.5  # 1 ms peak histogram: top band / next band
LIMITER_RATIO = 10.0
RELEASE_WAVE_MS = 10.0
STEM_OVERS = 1.0
TP_BLOCK = 8192  # true peak: only blocks whose sample peak is >= TP_BLOCK_MIN are oversampled (an over needs large neighbours)
TP_BLOCK_MIN = 0.5
TP_MARGIN = 64

# ----------------------------------------------------------------------------- wind
LF_BAND = (20.0, 200.0)
MID_BAND = (300.0, 3400.0)
LF_EDGE_HZ = 200.0
LF_FRAME = _fast(16384 * SR / 44100)  # 0.375 s (the lab: 0.37 s)
LF_HOP = LF_FRAME // 2
PAUSE_MARGIN_S = 0.3
LF_REF_PCT = 25.0
BURST_DB = 6.0
MID_DB = 3.0
FLAT_MIN_DB = -6.0
WIND_SHARE = 0.05
WIND_MIN_FRAMES = 3
MIN_PAUSE_FRAMES = 12

# ----------------------------------------------------------------------------- bandwidth
THIN_SPEECH_S = 5.0
NFFT = _fast(4096 * SR / 44100)  # 0.094 s (the lab: 0.093 s)
SMOOTH_OCT = 1 / 6
GRID = 60.0 * 2.0 ** (np.arange(0, 100) / 12)
GRID = GRID[GRID < 20000.0]
STEP_PER_OCT = 12
HALF_OCT = 6
STEEP_DB_OCT = 24.0
EDGE_DEPTH_DB = 20.0
EDGE_FREQ_DB = 10.0
LP_SEARCH = (1500.0, 7000.0)
LP_MAX_HZ = 9000.0
HP_SEARCH = (70.0, 700.0)
HP_ABS_MIN_HZ = 200.0
HP_DEFICIT_DB = 10.0
FUTZ_HP_BAND = (200.0, 500.0)
FUTZ_LP_BAND = (2500.0, 8000.0)
STABLE_OCT = 1 / 3
ROLLOFF_DB = 10.0
ROLLOFF_BAND = (3000.0, 8000.0)
CODEC_SEARCH = (12000.0, 19500.0)
CODEC_LP_MAX_HZ = 16000.0
SIG_BAND = (200.0, 4000.0)
MIN_HALF_ROWS = 4

# ----------------------------------------------------------------------------- noise
SNR_MIN_DB = 20.0
SNR_TARGET_DB = 20.0
SNR_BAND = (300.0, 3400.0)
MIN_FLOOR_FRAMES = 98  # 4.6 s of quiet frames (the lab: 100 frames of 46 ms)
FLOOR_GATE_DB = 30.0
FLOOR_PCT = 10.0
THIRD = 100.0 * 2.0 ** (np.arange(0, 22) / 3)  # 1/3-octave centres 100 Hz .. 12.7 kHz
OCT_REPORT = (125.0, 250.0, 500.0, 1000.0, 2000.0, 4000.0, 8000.0)
NONZERO_MS = 1e-12  # a stem frame below -120 dBFS mean square is the separator's digital silence

RULE = ("Per scene: clipping = >=1 clipped event/min in the mix (>=3 samples within 0.1 % of the ceiling, or true-peak overs); wind = "
        "20-200 Hz bursts >=6 dB over the pause floor, noise-like, voice band not up, in >=5 % of pause frames; noise = speech-band "
        "SNR of the stem's own floor < 20 dB. Bandwidth (speech >= 5 s): futz = steep (>=24 dB/oct, sustained 20 dB) high-pass edge at 200-500 Hz AND steep "
        "low-pass edge at 2.5-8 kHz, both stable between the two halves of the scene within 1/3 octave; anything else that limits "
        "the band (low-pass only, high-pass only against the same speaker elsewhere, gradual roll-off >=10 dB, drifting edge) is a "
        "defect: the dub keeps the low end / highs the clip's recording had (relax_hp_hz, relax_lp_hz). Scenes under 5 s of speech "
        "inherit a futz of the nearest long scene by LTAS signature and never become a futz on their own data.")


def params():
    return {k: (list(v) if isinstance(v, tuple) else v) for k, v in globals().items() if k.isupper() and isinstance(v, (int, float, str, tuple))}


FREQ = np.fft.rfftfreq(NFFT, 1 / SR)
_STATE = {}  # forked workers read the clip arrays from here


# ----------------------------------------------------------------------------- 1. clipping

def runs(mask):
    """(start, length) of the runs of True in a boolean array."""
    d = np.diff(np.concatenate([[0], mask.astype(np.int8), [0]]))
    s = np.flatnonzero(d == 1)
    e = np.flatnonzero(d == -1)
    return s, e - s


def merge_events(idx, gap):
    if len(idx) == 0:
        return 0
    return int(1 + (np.diff(idx) > gap).sum())


def true_peak_overs(x, sr=SR):
    """(true peak dBTP, number of 4x oversampled excursions above 0 dBFS merged within 1 ms) of mono x; only the blocks whose
    sample peak is >= TP_BLOCK_MIN are oversampled."""
    nb = len(x) // TP_BLOCK
    if nb == 0:
        return 20 * math.log10(max(float(np.abs(x).max()) if len(x) else 1e-9, 1e-9)), 0
    bm = np.abs(x[:nb * TP_BLOCK]).reshape(nb, TP_BLOCK).max(1)
    loud = np.flatnonzero(bm >= TP_BLOCK_MIN)
    peak = float(bm.max()) if len(bm) else 0.0
    idx = []
    xp = np.pad(np.asarray(x, np.float32), (TP_MARGIN, TP_MARGIN))
    for g0 in range(0, len(loud), 64):
        sel = loud[g0:g0 + 64]
        blocks = np.stack([xp[b * TP_BLOCK: b * TP_BLOCK + TP_BLOCK + 2 * TP_MARGIN] for b in sel])
        y = resample_poly(blocks, OVERSAMPLE, 1, axis=1).astype(np.float32)[:, TP_MARGIN * OVERSAMPLE: (TP_MARGIN + TP_BLOCK) * OVERSAMPLE]
        ay = np.abs(y)
        peak = max(peak, float(ay.max()))
        for r, b in enumerate(sel):
            s, _ = runs(ay[r] > 1.0)
            idx.extend((s + b * TP_BLOCK * OVERSAMPLE) // OVERSAMPLE)
    return 20 * math.log10(max(peak, 1e-9)), merge_events(np.array(idx), int(EVENT_MERGE_S * sr))


def clip_channel(x, sr=SR, full=True):
    """Clipping measures of one mono channel of a scene (module docstring, 1)."""
    n = len(x)
    minutes = max(n / sr / 60.0, 1e-6)
    a = np.abs(x)
    c = float(a.max()) if n else 0.0
    out = {"peak": round(c, 4), "peak_dbfs": round(20 * math.log10(max(c, 1e-9)), 2)}
    if c <= 0:
        return {**out, "run_share": 0.0, "events_per_min": 0.0, "fs_events_per_min": 0.0, "overs_per_min": 0.0, "tp_dbtp": None, "pile_up": None}
    s, ln = runs(a >= c * (1 - CLIP_TOL))
    ok = ln >= CLIP_RUN
    out["run_share"] = float(ln[ok].sum() / n)
    out["events_per_min"] = round(merge_events(s[ok], int(EVENT_MERGE_S * sr)) / minutes, 3)
    fs = np.flatnonzero(a >= FULL_SCALE)
    out["fs_events_per_min"] = round(merge_events(fs, int(EVENT_MERGE_S * sr)) / minutes, 3)
    ms = int(EVENT_MERGE_S * sr)
    m = n // ms * ms
    pk = a[:m].reshape(-1, ms).max(1)
    db = 20 * np.log10(c / np.maximum(pk, 1e-9))
    top = int((db < PILE_BAND_DB).sum())
    nxt = int(((db >= PILE_BAND_DB) & (db < 2 * PILE_BAND_DB)).sum())
    out["pile_up"] = round(top / nxt, 2) if top + nxt >= 20 and nxt > 0 else None
    if full and c >= 0.5:
        tp, overs = true_peak_overs(x, sr)
        out["tp_dbtp"] = round(tp, 2)
        out["overs_per_min"] = round(overs / minutes, 3)
    else:
        out["tp_dbtp"] = None
        out["overs_per_min"] = 0.0
    out["run_share"] = round(out["run_share"], 7)
    return out


def stem_clip(stem, sr=SR):
    """Clipping measures of the voice stem of a scene (a separation model output: the overshoot above full scale is reported)."""
    st = {"peak": round(float(np.abs(stem).max()), 4) if len(stem) else 0.0}
    st["overs_share"] = round(float((np.abs(stem) > STEM_OVERS).mean()), 6) if len(stem) else 0.0
    st["run_share"] = clip_channel(stem, sr, full=False)["run_share"]
    return st


def clip_summary(mx, st, n, sr=SR):
    """Clipping measures of a scene from its mix channels' clip_channel() dicts `mx`, the stem's stem_clip() `st` and its length n."""
    worst = lambda k: max((m.get(k) or 0.0) for m in mx)
    events = max(worst("events_per_min"), worst("fs_events_per_min"), worst("overs_per_min"))
    minutes = max(n / sr / 60.0, 1e-6)
    return {"mix": mx, "stem": st, "mix_events_per_min": round(events, 3), "mix_events": int(round(events * minutes)), "mix_run_share": worst("run_share"),
            "mix_tp_dbtp": max([m["tp_dbtp"] for m in mx if m.get("tp_dbtp") is not None], default=None)}


# ----------------------------------------------------------------------------- 2. wind / handling noise

def gaps(rows_s, a, b, margin=PAUSE_MARGIN_S):
    """Pause intervals of [a, b]: complement of the slots padded by margin."""
    iv = sorted((max(s - margin, a), min(e + margin, b)) for s, e in rows_s if e + margin > a and s - margin < b)
    out, t = [], a
    for s, e in iv:
        if s > t:
            out.append((t, s))
        t = max(t, e)
    if t < b:
        out.append((t, b))
    return out


def _frame_starts(i0, i1, size, hop):
    """Start samples of the frames of `size` that fit in [i0, i1) at `hop`."""
    return np.arange(i0, i1 - size + 1, hop) if i1 - size >= i0 else np.zeros(0, int)


_LF = {}


def _lf_tables():
    if "win" not in _LF:
        f = np.fft.rfftfreq(LF_FRAME, 1 / SR)
        _LF["lf"] = (f >= LF_BAND[0]) & (f <= LF_BAND[1])
        _LF["mid"] = (f >= MID_BAND[0]) & (f <= MID_BAND[1])
        _LF["win"] = np.hanning(LF_FRAME).astype(np.float32)
    return _LF


def lf_measures(mix2, sr, slots, a, b):
    """Wind / handling noise measures in the pauses of scene [a, b] (module docstring, 2). mix2 = the scene's (2, n) mix."""
    t = _lf_tables()
    starts = np.concatenate([_frame_starts(int(g0 * sr), int(g1 * sr), LF_FRAME + 1, LF_HOP) for g0, g1 in gaps(slots, a, b)] or [np.zeros(0, int)])
    n = len(starts)
    out = {"pause_frames": n}
    if n < MIN_PAUSE_FRAMES:
        return {**out, "lf_level_db": None, "lf_excess_db": None, "wind_share": None, "spans": [], "coherence": None}
    L, M, flat, XL, XR = np.empty(n), np.empty(n), np.empty(n), [], []
    for g0 in range(0, n, 128):
        ss = starts[g0:g0 + 128]
        seg = np.stack([mix2[:, s:s + LF_FRAME] for s in ss]) * t["win"]  # (frames, 2, LF_FRAME)
        X = sfft.rfft(seg, axis=2)
        P = (np.abs(X) ** 2).mean(1)
        p = P[:, t["lf"]]
        L[g0:g0 + len(ss)] = 10 * np.log10(p.sum(1) + 1e-20)
        M[g0:g0 + len(ss)] = 10 * np.log10(P[:, t["mid"]].sum(1) + 1e-20)
        flat[g0:g0 + len(ss)] = 10 * np.log10(np.exp(np.log(p + 1e-20).mean(1)) / (p.mean(1) + 1e-20))
        XL.append(X[:, 0, t["lf"]])
        XR.append(X[:, 1, t["lf"]])
    tt = starts / sr
    l_ref, m_ref = np.percentile(L, LF_REF_PCT), np.percentile(M, LF_REF_PCT)
    wind = (L - l_ref >= BURST_DB) & (M - m_ref < MID_DB) & (flat >= FLAT_MIN_DB)
    out["lf_level_db"] = round(float(l_ref), 1)
    out["lf_excess_db"] = round(float((L[wind] - l_ref).mean()), 1) if wind.any() else 0.0
    out["lf_p90_minus_ref_db"] = round(float(np.percentile(L, 90) - l_ref), 2)
    out["wind_share"] = round(float(wind.mean()), 3)
    out["wind_frames"] = int(wind.sum())
    spans = []
    for tj in tt[wind]:
        if spans and tj - spans[-1][1] <= LF_FRAME / sr:
            spans[-1][1] = tj + LF_FRAME / sr
        else:
            spans.append([tj, tj + LF_FRAME / sr])
    out["spans"] = [[round(float(s), 2), round(float(e), 2)] for s, e in spans]
    if wind.any():
        XL, XR = np.concatenate(XL)[wind], np.concatenate(XR)[wind]
        out["coherence"] = round(float(np.mean(np.abs((XL * XR.conj()).sum(0)) ** 2 / ((np.abs(XL) ** 2).sum(0) * (np.abs(XR) ** 2).sum(0) + 1e-20))), 2)
    else:
        out["coherence"] = None
    out["flat_db_median"] = round(float(np.median(flat)), 1)
    return out


# ----------------------------------------------------------------------------- 3. bandwidth

def regions_of(rows_s, a, b):
    return [(max(s, a), min(e, b)) for s, e in rows_s if e > a and s < b and min(e, b) - max(s, a) >= 0.1]


_WIN = {}


def region_power(x, regions, sr=SR, workers=1):
    """Per region: (power spectrum summed over the frames of NFFT (hann, half overlap) inside it, frame count); a region shorter than
    NFFT carries no frame (zeros, 0)."""
    if "w" not in _WIN:
        _WIN["w"] = np.hanning(NFFT).astype(np.float32)
    out = []
    hop = NFFT // 2
    for a, b in regions:
        i0, i1 = int(a * sr), min(int(b * sr), len(x))
        acc = np.zeros(NFFT // 2 + 1)
        if i1 - i0 < NFFT:
            out.append((acc, 0))
            continue
        fr = np.lib.stride_tricks.sliding_window_view(np.asarray(x[i0:i1], np.float32), NFFT)[::hop]
        for s0 in range(0, len(fr), 512):
            sp = sfft.rfft(fr[s0:s0 + 512] * _WIN["w"], axis=1, workers=workers)
            acc += (sp.real ** 2 + sp.imag ** 2).sum(0)
        out.append((acc, len(fr)))
    return out


def sum_power(parts):
    """Mean power spectrum + frame count of region_power() entries."""
    n = sum(c for _, c in parts)
    return (sum(p for p, _ in parts) / max(n, 1) if parts else np.zeros(NFFT // 2 + 1)), n


def to_grid_db(p, oct_=SMOOTH_OCT):
    """Power spectrum -> dB on GRID, power averaged in a +-oct_/2 box (cumulative sums); level not normalised."""
    cum = np.concatenate([[0.0], np.cumsum(p)])
    lo = np.searchsorted(FREQ, GRID * 2 ** (-oct_ / 2), side="left")
    hi = np.searchsorted(FREQ, GRID * 2 ** (oct_ / 2), side="right")
    hi = np.maximum(hi, lo + 1)
    return 10 * np.log10((cum[hi] - cum[lo]) / (hi - lo) + 1e-20)


def level_1k(l):
    m = (GRID >= 800) & (GRID <= 1250)
    return float(10 * np.log10(np.mean(10 ** (l[m] / 10))))


def norm_db(p):
    l = to_grid_db(p)
    return l - level_1k(l)


def gidx(f):
    return int(np.argmin(np.abs(np.log2(GRID / f))))


def lp_edge(l):
    """Lowest sustained steep fall of the normalised LTAS l inside LP_SEARCH (the first start whose half-octave slope is
    >= STEEP_DB_OCT and whose drop is sustained). -> dict(valid, f_hz, slope_db_oct, depth_db); when nothing is steep enough
    the steepest start is reported with valid False."""
    i0, i1 = gidx(LP_SEARCH[0]), gidx(LP_SEARCH[1])
    best = (-1e9, None)
    for i in range(i0, min(i1, len(GRID) - HALF_OCT - 1) + 1):
        s = (l[i] - l[i + HALF_OCT]) / 0.5
        j1 = min(i + 2 * STEP_PER_OCT, len(GRID) - 1, gidx(14000.0))
        depth = float(l[i] - l[i + STEP_PER_OCT: j1 + 1].max())
        if s >= STEEP_DB_OCT and depth >= EDGE_DEPTH_DB:
            best = (s, i)
            break
        if s > best[0]:
            best = (s, i)
    s, i = best
    j1 = min(i + 2 * STEP_PER_OCT, len(GRID) - 1, gidx(14000.0))
    depth = float(l[i] - l[i + STEP_PER_OCT: j1 + 1].max())
    k = i
    while k < len(GRID) - 1 and l[k] > l[i] - EDGE_FREQ_DB:
        k += 1
    f = float(GRID[k])
    return {"valid": bool(s >= STEEP_DB_OCT and depth >= EDGE_DEPTH_DB and f <= LP_MAX_HZ), "f_hz": round(f, 0),
            "slope_db_oct": round(float(s), 1), "depth_db": round(depth, 1)}


def hp_edge(l, ref=None):
    """Highest sustained steep rise of the LF part inside HP_SEARCH. With ref (normalised LTAS of the same speakers elsewhere)
    the deficit of the scene against it is used (an edge of the CHANNEL); without, the absolute LTAS, and then the edge counts
    only above HP_ABS_MIN_HZ."""
    if ref is not None:
        d = l - ref
        sig = d - np.median(d[(GRID >= 500) & (GRID <= 3000)])  # a level difference of the two spectra is not a filter
    else:
        sig = l
    i0, i1 = gidx(HP_SEARCH[0]), gidx(HP_SEARCH[1])
    need = EDGE_DEPTH_DB if ref is None else HP_DEFICIT_DB + 5.0
    best = (-1e9, None)
    for i in range(i1, i0 - 1, -1):
        s = (sig[i + HALF_OCT] - sig[i]) / 0.5
        top = float(sig[i + HALF_OCT: i + HALF_OCT + STEP_PER_OCT + 1].max())
        depth = float(top - sig[max(i - HALF_OCT, 0): i + 1].min())
        if s >= STEEP_DB_OCT and depth >= need and sig[max(i - HALF_OCT, 0): i + 1].min() <= -HP_DEFICIT_DB * (ref is not None):
            best = (s, i)
            break
        if s > best[0]:
            best = (s, i)
    s, i = best
    top = float(sig[i + HALF_OCT: i + HALF_OCT + STEP_PER_OCT + 1].max())
    depth = float(top - sig[max(i - HALF_OCT, 0): i + 1].min())
    k = i
    while k < len(GRID) - 1 and sig[k] < top - EDGE_FREQ_DB:
        k += 1
    f = float(GRID[k])
    valid = s >= STEEP_DB_OCT and depth >= need
    if ref is None:
        valid = valid and f >= HP_ABS_MIN_HZ
    else:
        valid = valid and float(sig[max(i - HALF_OCT, 0): i + 1].min()) <= -HP_DEFICIT_DB
    return {"valid": bool(valid), "f_hz": round(f, 0), "slope_db_oct": round(float(s), 1), "depth_db": round(depth, 1),
            "against": "reference" if ref is not None else "absolute"}


def abs_hp_hz(l):
    """Frequency where the absolute LTAS rises to 10 dB below its passband (90th percentile of 200-3400 Hz)."""
    band = (GRID >= 200) & (GRID <= 3400)
    p = np.percentile(l[band], 90)
    for i in range(gidx(HP_SEARCH[0]), len(GRID)):
        if l[i] >= p - EDGE_FREQ_DB:
            return float(GRID[i])
    return None


def f20(l):
    """Lowest / highest frequency (60 Hz - 12 kHz) where the LTAS is within 20 dB of its 1 kHz level."""
    sel = np.flatnonzero((l >= -20.0) & (GRID <= 12000.0))
    if len(sel) == 0:
        return None, None
    return float(GRID[sel[0]]), float(GRID[sel[-1]])


def codec_edge(l):
    """Steepest fall of l between CODEC_SEARCH (the source's own low-pass), Hz or None."""
    i0, i1 = gidx(CODEC_SEARCH[0]), gidx(CODEC_SEARCH[1])
    best = (-1e9, None)
    for i in range(i0, min(i1, len(GRID) - HALF_OCT)):
        s = (l[i] - l[i + HALF_OCT]) / 0.5
        if s > best[0]:
            best = (s, i)
    return float(GRID[best[1]]) if best[1] is not None and best[0] >= STEEP_DB_OCT else None


def futz_kind(f_lo, f_hi):
    if f_hi <= 4000.0:
        return "phone"
    if f_lo >= 350.0:
        return "speaker"
    return "radio" if f_lo >= 250.0 else "tv"


def band_report(parts, ref_db, halves=True):
    """All bandwidth measures of a scene from region_power() `parts` (the stem's per-region power sums)."""
    p, n = sum_power(parts)
    out = {"ltas_frames": n}
    if n < 4:
        return {**out, "ok": False}
    l = norm_db(p)
    lo20, hi20 = f20(l)
    lp = lp_edge(l)
    hp = hp_edge(l, ref_db)
    out.update({"ok": True, "f_lo20_hz": lo20, "f_hi20_hz": hi20, "lp": lp, "hp": hp, "ltas_db": l, "power": p})
    if ref_db is not None:
        d = l - ref_db
        d = d - np.median(d[(GRID >= 500) & (GRID <= 3000)])
        band = (GRID >= ROLLOFF_BAND[0]) & (GRID <= ROLLOFF_BAND[1])
        out["rolloff_db"] = round(float(-d[band].mean()), 1)
    if halves and len(parts) >= 2 * MIN_HALF_ROWS:
        edges = []
        k = len(parts) // 2
        for part in (parts[:k], parts[k:]):  # first / second half of the scene in time: a drift shows, noise does not
            ph, _ = sum_power(part)
            lh = norm_db(ph)
            edges.append((lp_edge(lh), hp_edge(lh, ref_db)))
        out["halves"] = [{"lp": e[0], "hp": e[1]} for e in edges]
        out["stable_lp"] = None if not lp["valid"] else bool(all(e[0]["valid"] for e in edges) and abs(math.log2(edges[0][0]["f_hz"] / edges[1][0]["f_hz"])) <= STABLE_OCT)
        out["stable_hp"] = None if not hp["valid"] else bool(all(e[1]["valid"] for e in edges) and abs(math.log2(edges[0][1]["f_hz"] / edges[1][1]["f_hz"])) <= STABLE_OCT)
    return out


# ----------------------------------------------------------------------------- 4. noise

def _third_matrix():
    """(bins, bands) averaging matrix: mean power per 1/3-octave band (THIRD)."""
    m = np.zeros((len(FREQ), len(THIRD)))
    for j, c in enumerate(THIRD):
        sel = (FREQ >= c / 2 ** (1 / 6)) & (FREQ < c * 2 ** (1 / 6))
        if sel.any():
            m[sel, j] = 1.0 / sel.sum()
    return m


THIRD_M = _third_matrix()


def band_third_power(p):
    """Power spectrum (rfft bins) (or (frames, bins)) -> mean power per 1/3-octave band (THIRD)."""
    return p @ THIRD_M


_BIAS = {}


def floor_bias():
    """Ratio mean / FLOOR_PCT-th percentile of the 1/3-octave band power of stationary Gaussian noise, per band and for the
    300-3400 Hz sum, through the same frames (so the percentile floor is unbiased for a stationary floor)."""
    if "b" not in _BIAS:
        rng = np.random.default_rng(3)
        x = rng.standard_normal(NFFT * 2 * 400).astype(np.float32)
        fr = np.lib.stride_tricks.sliding_window_view(x, NFFT)[:: NFFT // 2][:400]
        P = np.abs(sfft.rfft(fr * np.hanning(NFFT).astype(np.float32), axis=1)) ** 2
        B = band_third_power(P)
        _BIAS["b"] = B.mean(0) / np.percentile(B, FLOOR_PCT, axis=0)
    return _BIAS["b"]


def frame_ms(frames):
    return np.einsum("ij,ij->i", frames, frames, dtype=np.float64) / NFFT


def speech_ms(stem, regions):
    """Median mean-square of the NFFT frames of the stem inside the regions (the speech level the gate is relative to)."""
    v = []
    hop = NFFT // 2
    for a, b in regions:
        i0, i1 = int(a * SR), min(int(b * SR), len(stem))
        if i1 - i0 >= NFFT:
            v.append(frame_ms(np.lib.stride_tricks.sliding_window_view(np.asarray(stem[i0:i1], np.float32), NFFT)[::hop]))
    return float(np.median(np.concatenate(v))) if v else None


def pause_frames(x, slots, a, b, stem=None, gate_ms=None):
    """(frames, NFFT) of mono x inside the pauses of [a, b]. With stem and gate_ms only frames where the stem (the original voice,
    which holds all speech including the undubbed one) has a mean square below gate_ms: unlabelled speech and crosstalk do not
    count as floor."""
    out = []
    hop = NFFT // 2
    for g0, g1 in gaps(slots, a, b):
        i0, i1 = int(g0 * SR), min(int(g1 * SR), len(x))
        if i1 - i0 >= NFFT:
            fr = np.lib.stride_tricks.sliding_window_view(np.asarray(x[i0:i1], np.float32), NFFT)[::hop]
            if stem is not None and gate_ms is not None:
                fr = fr[frame_ms(np.lib.stride_tricks.sliding_window_view(np.asarray(stem[i0:i1], np.float32), NFFT)[::hop]) < gate_ms]
            out.append(fr)
    return np.concatenate(out) if out else np.zeros((0, NFFT), np.float32)


def floor_third(frames):
    """Floor power per 1/3-octave band (bias corrected 10th percentile over the frames) or None."""
    if len(frames) == 0:
        return None
    B = []
    w = np.hanning(NFFT).astype(np.float32)
    for s0 in range(0, len(frames), 1024):
        sp = sfft.rfft(frames[s0:s0 + 1024] * w, axis=1)
        B.append(band_third_power(sp.real ** 2 + sp.imag ** 2))
    return np.percentile(np.concatenate(B), FLOOR_PCT, axis=0) * floor_bias()


def speech_snr(p_speech, floor):
    """Speech-band SNR (dB), the 1 kHz speech reference level (dB) of the stem LTAS and the floor shape."""
    s = band_third_power(p_speech)
    sel = (THIRD >= SNR_BAND[0] / 2 ** (1 / 6)) & (THIRD <= SNR_BAND[1] * 2 ** (1 / 6))
    snr = 10 * np.log10(s[sel].sum() / max(floor[sel].sum(), 1e-20))
    ref = 10 * np.log10(s[np.argmin(np.abs(THIRD - 1000))] + 1e-20)
    shape = {}
    for c in OCT_REPORT:
        sl = (THIRD >= c / 2 ** 0.5 * 0.99) & (THIRD < c * 2 ** 0.5 * 0.99)
        shape[int(c)] = round(float(10 * np.log10(floor[sl].mean() + 1e-20) - ref), 1)
    return round(float(snr), 1), shape


def noise_measures(p_speech, stem_floor, n_stem, mix_floor, n_mix, clip_floor=None):
    """Noise of the recording (stem floor, the voice's own: decides the flag) and the program floor of the mix (everything
    in the pauses: mic noise + room + audience + music bed; this is what room_tone puts under the dub; informational).
    stem_floor / mix_floor: 1/3-octave floor power (None if no frames); n_*: frames behind them."""
    out = {"snr_db": None, "snr_stem_frames": int(n_stem)}
    if stem_floor is not None and n_stem >= MIN_FLOOR_FRAMES:
        out["snr_db"], out["stem_floor_db_re_speech_1k"] = speech_snr(p_speech, stem_floor)
    used = "scene"
    if mix_floor is None or n_mix < MIN_FLOOR_FRAMES:
        mix_floor, used = clip_floor, "clip"
    if mix_floor is not None:
        out["snr_mix_db"], out["floor_db_re_speech_1k"] = speech_snr(p_speech, mix_floor)
        out["floor_source"], out["floor_frames"] = used, int(n_mix)
    return out


# ----------------------------------------------------------------------------- decision

def decide(m, thin, ref_hp_hz, codec_hz, limiter_like, nearest=None):
    """measures -> (verdict, flags, treatment). m: measure dict of one scene (clipping, wind, bandwidth, noise)."""
    bw = m["bandwidth"]
    flags, reasons = [], []
    relax_hp = relax_lp = None
    futz = None
    lp, hp = (bw.get("lp"), bw.get("hp")) if bw.get("ok") else (None, None)
    if thin:
        flags.append("thin")
    if bw.get("ok"):
        lp_v = bool(lp["valid"])
        hp_v = bool(hp["valid"])
        stable_ok = (bw.get("stable_lp") in (True, None)) and (bw.get("stable_hp") in (True, None))
        in_futz = (hp_v and FUTZ_HP_BAND[0] <= hp["f_hz"] <= FUTZ_HP_BAND[1] and lp_v and FUTZ_LP_BAND[0] <= lp["f_hz"] <= FUTZ_LP_BAND[1])
        if in_futz and not thin and stable_ok:
            futz = {"kind": futz_kind(hp["f_hz"], lp["f_hz"]), "f_lo_hz": hp["f_hz"], "f_hi_hz": lp["f_hz"]}
            flags.append("futz")
            reasons.append(f"band-pass {hp['f_hz']:.0f}-{lp['f_hz']:.0f} Hz, edges {hp['slope_db_oct']:.0f}/{lp['slope_db_oct']:.0f} dB/oct, stable between the two halves in time")
        else:
            if lp_v:
                flags.append("lowpass")
                relax_lp = codec_hz
                reasons.append(f"low-pass edge {lp['f_hz']:.0f} Hz ({lp['slope_db_oct']:.0f} dB/oct) without a matching high-pass: recording/codec limit")
            if hp_v:
                flags.append("highpass")
                relax_hp = ref_hp_hz
                reasons.append(f"high-pass edge {hp['f_hz']:.0f} Hz ({hp['slope_db_oct']:.0f} dB/oct) vs the same speaker elsewhere (reference {ref_hp_hz:.0f} Hz): recording/mic limit")
            if in_futz and not thin and not stable_ok:
                flags.append("drift")
                reasons.append("both edges present but they drift between the halves of the scene: not a constant channel")
            if (bw.get("stable_lp") is False or bw.get("stable_hp") is False) and "drift" not in flags:
                flags.append("drift")
            if not lp_v and (bw.get("rolloff_db") or 0) >= ROLLOFF_DB:
                flags.append("rolloff")
                relax_lp = codec_hz
                reasons.append(f"gradual roll-off {bw['rolloff_db']:.0f} dB below the same voice elsewhere over 3-8 kHz")
    if thin and nearest and nearest.get("futz") and not futz:
        futz = dict(nearest["futz"])
        flags.append("futz")
        reasons.append(f"inherits the futz of scene {nearest['id']} (signature distance {nearest['distance_db']:.1f} dB)")
    if thin and nearest:
        relax_hp = relax_hp if relax_hp is not None else nearest.get("relax_hp_hz")
        relax_lp = relax_lp if relax_lp is not None else nearest.get("relax_lp_hz")
    c = m["clipping"]
    hot = c["mix_events_per_min"] >= CLIP_EVENTS_PER_MIN and c["mix_events"] >= CLIP_MIN_EVENTS
    if hot:
        flags.append("clipping")
        reasons.append(f"mix clipped: {c['mix_events_per_min']:.1f} events/min, true peak {c['mix_tp_dbtp']} dBTP")
    copy_dyn = not ((hot or c["stem"]["overs_share"] > 0) and limiter_like)
    if not copy_dyn:
        reasons.append("fitted 'compressor' is limiter-like on a hot signal: dynamics not copied")
    w = m["wind"]
    bed_hp = None
    if w.get("wind_share") is not None and w["wind_share"] >= WIND_SHARE and w.get("wind_frames", 0) >= WIND_MIN_FRAMES:
        flags.append("wind")
        bed_hp = LF_EDGE_HZ
        reasons.append(f"LF bursts {w['lf_excess_db']:.0f} dB over the pause floor in {100 * w['wind_share']:.0f} % of the pauses")
    nz = m["noise"]
    gain = 0.0
    snr = min([v for v in (nz.get("snr_db"), nz.get("snr_mix_db")) if v is not None], default=None)
    if thin and nearest and nearest.get("noise_gain_db") is not None:  # the floor is a property of the recording, not of 1 s of speech
        gain = nearest["noise_gain_db"]
        reasons.append(f"bed gain {gain:.0f} dB from the nearest long scene {nearest['id']}" if gain < 0 else "bed untouched (nearest long scene)")
        if gain < 0:
            flags.append("noise")
    elif snr is not None and snr < SNR_MIN_DB:
        flags.append("noise")
        gain = -float(SNR_TARGET_DB - snr)
        src = "stem floor" if nz.get("snr_db") == snr else "mix floor (room + audience + music bed in the pauses)"
        reasons.append(f"speech-band SNR {snr:.0f} dB vs the {src}: bed {gain:.0f} dB")
    verdict = "futz" if futz else ("defect" if [f for f in flags if f not in ("thin", "futz")] else "clean")
    if not reasons:
        reasons.append("no defect measured")
    treat = {"relax_hp_hz": relax_hp, "relax_lp_hz": relax_lp, "copy_dynamics": bool(copy_dyn),
             "bed": {"highpass_hz": bed_hp, "gain_db": round(gain, 1)}, "futz": futz, "reason": "; ".join(reasons)}
    return verdict, flags, treat


# ----------------------------------------------------------------------------- scenes

def scene_floors(stem, mix2, slots, a, b):
    """Pause floors of scene [a, b]: (mix floor, mix frames, stem floor, stem frames), see noise_measures."""
    gate = speech_ms(stem, regions_of(slots, a, b))
    gate = gate * 10 ** (-FLOOR_GATE_DB / 10) if gate else None
    fr = pause_frames(mix2.mean(0), slots, a, b, stem, gate)
    sfr = pause_frames(stem, slots, a, b)
    sfr = sfr[frame_ms(sfr) > NONZERO_MS]
    return floor_third(fr), len(fr), floor_third(sfr), len(sfr)


def _measure_task(args):
    """Worker: one independent piece of the measures of scene i ("mix0" / "mix1": clipping of a mix channel, "stem": clipping of the
    stem, "rest": wind, per-region LTAS power, pause floors). The pieces of a scene run in parallel (a long scene is one task of each)."""
    i, kind = args
    g = _STATE
    sc = g["scenes"][i]
    a, b = sc["start"], min(sc["end"], g["n"] / SR)
    i0, i1 = int(a * SR), int(b * SR)
    with threadpool_limits(1):
        if kind in ("mix0", "mix1"):
            return clip_channel(g["mix2"][int(kind[-1]), i0:i1], SR)
        if kind == "stem":
            return stem_clip(g["stem"][i0:i1])
        rel = [(x - a, y - a) for x, y in g["slots"] if y > a and x < b]
        return {"wind": lf_measures(g["mix2"][:, i0:i1], SR, rel, 0.0, b - a), "parts": region_power(g["stem"], regions_of(g["slots"], a, b)),
                "floors": scene_floors(g["stem"], g["mix2"], g["slots"], a, b)}


def signature_distance(l_a, l_b):
    """rms (dB) of the difference of two normalised LTAS (GRID) over SIG_BAND, on 1/3-octave spacing."""
    sel = (GRID >= SIG_BAND[0]) & (GRID <= SIG_BAND[1])
    idx = np.flatnonzero(sel)[::4]
    d = l_a[idx] - l_b[idx]
    return float(np.sqrt(np.mean((d - np.median(d)) ** 2)))


def level_lin_1k(p):
    m = (FREQ >= 800) & (FREQ <= 1250)
    return float(p[m].mean())


def clip_reference(per_scene):
    """Normalised LTAS (dB, GRID) of the one speaker "all" over the WHOLE clip: all scenes, frame weighted, each scene's spectrum
    normalised at 1 kHz (the speaker's typical channel). None when there is no spectrum."""
    ps = [(p, n) for p, n in per_scene if n]
    if not ps:
        return None
    p = sum(q / max(level_lin_1k(q), 1e-20) * n for q, n in ps) / sum(n for _, n in ps)
    return norm_db(p)


def analyze(clip, chan, regions, workers=8):
    """See the module docstring."""
    scenes = chan["scenes"]
    stem, mix2 = clip.speech, clip.mix2
    slots = [tuple(r) for r in regions]
    _STATE.update(scenes=scenes, n=clip.n, stem=stem, mix2=mix2, slots=slots)
    tasks = [(i, k) for i in range(len(scenes)) for k in ("rest", "mix0", "mix1", "stem")]
    order = sorted(range(len(tasks)), key=lambda t: scenes[tasks[t][0]]["start"] - scenes[tasks[t][0]]["end"])  # longest scenes first
    with ProcessPoolExecutor(max(1, min(workers, len(tasks))), mp_context=mp.get_context("fork")) as ex:
        done = list(ex.map(_measure_task, [tasks[t] for t in order], chunksize=1))
    res = [None] * len(tasks)
    for t, r in zip(order, done):
        res[t] = r
    jobs = []
    for i, sc in enumerate(scenes):
        rest, m0, m1, st = res[4 * i: 4 * i + 4]
        n = min(int(sc["end"] * SR), clip.n) - int(sc["start"] * SR)
        jobs.append(({"clipping": clip_summary([m0, m1], st, n), "wind": rest["wind"]}, rest["floors"], rest["parts"]))
    # clip-wide pause floor: all pauses of the clip
    g_all = speech_ms(stem, regions_of(slots, 0.0, clip.n / SR))
    fr_all = pause_frames(mix2.mean(0), slots, 0.0, clip.n / SR, stem, g_all * 10 ** (-FLOOR_GATE_DB / 10))
    if len(fr_all) > 3000:
        fr_all = fr_all[:: len(fr_all) // 3000]
    floor_clip = floor_third(fr_all)
    _STATE.clear()
    ref = clip_reference([sum_power(parts) for _, _, parts in jobs])
    meas = {}
    for sc, (m, (flo, n_mix, sflo, n_stem), parts) in zip(scenes, jobs):
        bw = band_report(parts, ref)
        m["bandwidth"] = bw
        m["noise"] = noise_measures(bw["power"], sflo, n_stem, flo, n_mix, floor_clip) if bw.get("ok") else {"snr_db": None}
        meas[sc["id"]] = m
    long_ids = [s["id"] for s in scenes if s["speech_s"] >= THIN_SPEECH_S]
    longest = max(scenes, key=lambda s: s["speech_s"])
    l_long = meas[longest["id"]]["bandwidth"]["ltas_db"]
    ref_hp = float(round(abs_hp_hz(l_long)))
    codec = codec_edge(l_long)
    codec_hz = float(round(min(codec if codec else CODEC_LP_MAX_HZ, CODEC_LP_MAX_HZ), -2))
    res = {}
    for sc in sorted(scenes, key=lambda s: s["speech_s"] < THIN_SPEECH_S):  # long scenes first (a thin scene needs their verdicts)
        m = meas[sc["id"]]
        thin = sc["speech_s"] < THIN_SPEECH_S
        dyn = (sc.get("dynamics") or {}).get("compressor")  # a scene without a fitted channel has no dynamics
        limiter_like = bool(dyn and dyn["active"] and (dyn["ratio"] >= LIMITER_RATIO or dyn["release_ms"] < RELEASE_WAVE_MS))
        nearest = None
        if thin and m["bandwidth"].get("ok"):
            cand = [(signature_distance(m["bandwidth"]["ltas_db"], meas[i]["bandwidth"]["ltas_db"]), i) for i in long_ids if meas[i]["bandwidth"].get("ok")]
            if cand:
                dist, j = min(cand)
                tj = res[j]["treatment"]
                nearest = {"id": j, "distance_db": dist, "futz": tj["futz"], "relax_hp_hz": tj["relax_hp_hz"], "relax_lp_hz": tj["relax_lp_hz"],
                           "noise_gain_db": tj["bed"]["gain_db"]}
                m["nearest_long_scene"] = {"id": j, "distance_db": round(dist, 1), "channel_match_borrowed_from": sc.get("borrowed_from")}
        verdict, flags, treat = decide(m, thin, ref_hp, codec_hz, limiter_like, nearest)
        res[sc["id"]] = {"id": sc["id"], "start": sc["start"], "end": sc["end"], "speech_s": sc["speech_s"], "measures": m,
                         "flags": flags, "verdict": verdict, "treatment": treat}
    out_scenes = []
    for sc in scenes:
        r = res[sc["id"]]
        bw = r["measures"]["bandwidth"]
        bw.pop("ltas_db", None)
        bw.pop("power", None)
        out_scenes.append(r)
    return {"rule": RULE, "params": params(), "reference_hp_hz": ref_hp, "codec_lp_hz": codec_hz, "scenes": out_scenes}
