"""Level following (lab `dub_mix.level_curve` / `apply_gain_curve`) on window-local arrays: every ~1 s segment of a dub part gets the
loudness of the original voice in the same slot.

All times are seconds relative to the first sample of the arrays (the window's origin); frame powers are 10 ms frames.
A part is {"o0","o1": original slot, "d0","d1": placed dub span, "err": dub end - original end}.
"""
import numpy as np

from .core import FR, level_db

SEG_S = 1.0  # level is followed per ~1 s of a part
LEVEL_CLAMP_DB = 6.0  # per-segment gain may differ from the window's median gain by at most this
LOOSE_END_S = 0.5  # parts whose dub misses the slot end by more than this are not cut up: their time maps are too loose


def segments(parts):
    """~1 s segments of every part: (part index, orig window, dub window)."""
    segs = []
    for pi, p in enumerate(parts):
        n = max(1, int(round((p["d1"] - p["d0"]) / SEG_S))) if abs(p["err"]) <= LOOSE_END_S else 1
        for k in range(n):
            f0, f1 = k / n, (k + 1) / n
            segs.append((pi, (p["o0"] + (p["o1"] - p["o0"]) * f0, p["o0"] + (p["o1"] - p["o0"]) * f1),
                         (p["d0"] + (p["d1"] - p["d0"]) * f0, p["d0"] + (p["d1"] - p["d0"]) * f1)))
    return segs


def curve(parts, p_orig, p_dub, total_s):
    """Gain curve (dB per 10 ms frame) that gives every dub segment the loudness of the original voice in its slot."""
    nf = int(total_s * 100) + 2
    if not parts:
        return np.zeros(nf)
    segs = segments(parts)
    dev = np.array([level_db(p_orig, *o) - level_db(p_dub, *d) for _, o, d in segs])
    part_dev = np.array([level_db(p_orig, p["o0"], p["o1"]) - level_db(p_dub, p["d0"], p["d1"]) for p in parts])
    pidx = np.array([s[0] for s in segs])
    dev = np.where(np.isfinite(dev), dev, part_dev[pidx])
    ok = np.isfinite(part_dev)
    g0 = float(np.median(part_dev[ok])) if ok.any() else 0.0
    dev = np.where(np.isfinite(dev), dev, g0)
    sm = dev.copy()  # light smoothing along the part so a 1 s segment does not jump
    for i in range(len(dev)):
        lo, hi = max(0, i - 1), min(len(dev) - 1, i + 1)
        if pidx[lo] == pidx[i] and pidx[hi] == pidx[i]:
            sm[i] = 0.25 * dev[lo] + 0.5 * dev[i] + 0.25 * dev[hi]
    gain = g0 + np.clip(sm - g0, -LEVEL_CLAMP_DB, LEVEL_CLAMP_DB)
    g = np.full(nf, np.nan)
    for (_, _, d), gv in sorted(zip(segs, gain), key=lambda s: s[0][2][0]):
        g[max(0, int(d[0] * 100)):max(0, int(d[1] * 100))] = gv
    idx = np.where(np.isfinite(g), np.arange(nf), 0)  # hold the previous value across gaps, then ramp (100 ms triangle)
    np.maximum.accumulate(idx, out=idx)
    g = np.where(np.isfinite(g[idx]), g[idx], g0)
    k = np.bartlett(11)
    k /= k.sum()
    return np.convolve(np.pad(g, 5, mode="edge"), k, mode="valid")


def apply(x, g_db):
    """x * 10^(g/20) along the last axis, the 10 ms grid interpolated linearly to samples."""
    t = np.arange(len(g_db)) * FR + FR / 2
    gain = (10 ** (np.interp(np.arange(x.shape[-1]), t, g_db) / 20)).astype(np.float32)
    return x * gain
