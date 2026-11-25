"""Perspective and movement of the original voice per part: reverberation (wet), high-frequency tilt and L/R position over time,
measured on the separated original speech stem inside the part's slot (lab: perspective.py, Netflix dubbing guide: "follow the
original voice's distance cues and stereo position").

Every wet / tilt cue is the part's DEPARTURE from what the scene's voice does on average (the scene's median over its parts is the
neutral position the channel already models), shrunk by empirical-Bayes weights w = tau2 / (tau2 + sigma2 / m). In the product the
clip is analysed before parts exist: `analyze` uses the nominal speech regions as stand-in parts to get the scene floors, the
scene neutrals and the pooled sigma2 / tau2; `part_cue` then measures ONE part from its slot (features only for the slot) and
needs nothing but the document. The pan is the absolute L/R balance over time (movement).

Units: seconds, dB, Hz; frames are 10 ms (core.FR) hops of a 1024-sample (21 ms) Hann window of the stem's mid (L+R)/2.
"""
import numpy as np
import scipy.fft as sfft
from concurrent.futures import ThreadPoolExecutor

from .core import FR, SR, frame_power, read_range

# ----------------------------------------------------------------------------- constants (lab values; seconds / Hz carry over)
FRAME_WIN = 1024  # 21 ms window (the lab's 1024 samples at 44.1 kHz were 23 ms): resolves the 60-300 ms decay window, bin 47 Hz
BAND_W = (100.0, 8000.0)  # broadband activity / decay level
BAND_HF = (4000.0, 10000.0)  # high-frequency roll-off numerator
BAND_MID = (300.0, 2000.0)  # denominator: the voiced body that distance does not remove
FLOOR_PCT = 10.0  # noise floor = this percentile of the scene's 10 ms frames
ACTIVE_DB = 20.0  # active frame: within this of the slot's 95th percentile
ACTIVE_FLOOR_DB = 10.0  # and at least this far over the noise floor
NET_MIN = 0.01  # net band power never below this fraction of the gross power
TILT_BLOCK_FRAMES = 25  # active frames per block for the tilt standard error
DECAY_FROM_S, DECAY_END_S = 0.06, 0.30  # decay window after a speech end
ANCHOR_DROP_DB = 6.0
DECAY_PRE_S = 0.2
DECAY_MIN_ABOVE_FLOOR_DB = 20.0
CENSOR_DB = 3.0
JOIN_FRAMES = 15  # active runs closer than 150 ms are one run
PAN_HOP_FRAMES = 10  # 100 ms pan bins
PAN_MIN_ACTIVE = 5
PAN_MEDIAN = 5
PAN_RAW_CLIP_DB = 20.0
PAN_RAMP_S = 0.1
WET_CLAMP_DB = 6.0
TILT_CLAMP_DB = 4.0
MIN_GROUP = 3  # parts with evidence needed for a median to be a neutral reference
MIN_DOF = 5  # replicate degrees of freedom needed to estimate sigma2 of the clip
TILT_CORNER_HZ = 3000.0
TILT_REF_HZ = 8000.0  # the filter gain here equals tilt_db exactly
TILT_TAPS = 2047
TAIL_AFTER_S = 3.0  # decay events are searched this far past the slot end (or up to the next part)
PRE_S = 1.0  # context before the slot (kept for the floor / active statistics of the lab's slot extension)

_BINS = np.fft.rfftfreq(FRAME_WIN, 1.0 / SR)
_BIN_RANGES = [(int(np.searchsorted(_BINS, lo, "left")), int(np.searchsorted(_BINS, hi, "left"))) for lo, hi in (BAND_W, BAND_HF, BAND_MID)]
_WIN = np.hanning(FRAME_WIN).astype(np.float32)
_NORM = float((_WIN.astype(np.float64) ** 2).sum()) * FRAME_WIN


# ----------------------------------------------------------------------------- frame features

def features(stem2, f0, f1, workers=1):
    """Per 10 ms frame f0..f1-1 of the stereo stem (2, n): mid band powers W (100-8k), A (4-10k), B (300-2k) and channel powers L, R.
    Frame k is centred at (k + 0.5) FR; samples outside the stem count as zeros. Returns dict of float64 vectors of f1 - f0."""
    n = stem2.shape[1]
    f1 = min(f1, n // FR)
    nf = max(f1 - f0, 0)
    out = {k: np.zeros(nf) for k in "WABLR"}
    if not nf:
        return out
    s0 = f0 * FR + FR // 2 - FRAME_WIN // 2
    s1 = (f1 - 1) * FR + FR // 2 - FRAME_WIN // 2 + FRAME_WIN
    a, b = max(s0, 0), min(s1, n)
    mid = np.pad(0.5 * (stem2[0, a:b] + stem2[1, a:b]), (a - s0, s1 - b))
    view = np.lib.stride_tricks.sliding_window_view(mid, FRAME_WIN)[::FR]

    def chunk(c0):
        c1 = min(nf, c0 + 2048)
        spec = sfft.rfft(view[c0:c1] * _WIN, axis=1)
        p = (spec.real ** 2 + spec.imag ** 2) / np.float32(_NORM)
        return [p[:, lo:hi].sum(1, dtype=np.float64) for lo, hi in _BIN_RANGES]

    starts = list(range(0, nf, 2048))
    if workers > 1 and len(starts) > 1:
        with ThreadPoolExecutor(workers) as ex:
            res = list(ex.map(chunk, starts))
    else:
        res = [chunk(c0) for c0 in starts]
    for i, k in enumerate("WAB"):
        out[k] = np.concatenate([r[i] for r in res])
    pw = frame_power(stem2[:, f0 * FR: f1 * FR])
    out["L"], out["R"] = pw[0], pw[1]
    return out


def _floors(F, i0, i1):
    """Noise floor (power) per feature over frames [i0, i1) of a scene."""
    return {k: float(np.percentile(F[k][i0:i1], FLOOR_PCT)) for k in F}


def _net(p, fl):
    return np.maximum(p - fl, NET_MIN * p)


def _db(p):
    return 10.0 * np.log10(np.asarray(p, dtype=np.float64) + 1e-20)


# ----------------------------------------------------------------------------- one slot

def _active_mask(F, i0, i1, fl):
    """Active frames of [i0, i1): broadband level within ACTIVE_DB of the slot's p95 and ACTIVE_FLOOR_DB over the scene floor."""
    w = _db(F["W"][i0:i1])
    if not len(w):
        return np.zeros(0, bool)
    thr = max(np.percentile(w, 95) - ACTIVE_DB, _db(fl["W"]) + ACTIVE_FLOOR_DB)
    return w >= thr


def _measure_tilt(F, i0, i1, fl, act):
    """(tilt_raw_db, n_active_frames, n_blocks, sigma2_blocks_db2) of the active frames of the slot, or None."""
    n = int(act.sum())
    if n < 3:
        return None
    a = _net(F["A"][i0:i1], fl["A"])[act]
    b = _net(F["B"][i0:i1], fl["B"])[act]
    raw = float(_db(a.sum()) - _db(b.sum()))
    nb = n // TILT_BLOCK_FRAMES
    s2 = float("nan")
    if nb >= 2:
        ab = a[: nb * TILT_BLOCK_FRAMES].reshape(nb, -1).sum(1)
        bb = b[: nb * TILT_BLOCK_FRAMES].reshape(nb, -1).sum(1)
        s2 = float(np.var(_db(ab) - _db(bb), ddof=1))
    return raw, n, nb, s2


def _measure_decay(F, i0, iR, fl):
    """Decay events inside the frames [i0, iR) (the slot extended past its end). Returns the event values (dB re the active level,
    floor-net, censored ones at the floor level) and the number of censored events."""
    w = F["W"]
    wdb = _db(w[i0:iR])
    if len(wdb) < 3:
        return [], 0
    act = _active_mask(F, i0, iR, fl)
    idx = np.flatnonzero(act)
    if not len(idx):
        return [], 0
    cut = np.flatnonzero(np.diff(idx) > JOIN_FRAMES)
    runs = list(zip(idx[np.concatenate([[0], cut + 1])], idx[np.concatenate([cut, [len(idx) - 1]])]))
    f0, f1 = int(round(DECAY_FROM_S * 100)), int(round(DECAY_END_S * 100))
    pre_n = int(round(DECAY_PRE_S * 100))
    floor_db = _db(fl["W"])
    vals, cens = [], 0
    for k, (s, e) in enumerate(runs):
        nxt = runs[k + 1][0] if k + 1 < len(runs) else len(wdb) + f1  # no later run: the rest of the extended slot is free
        if e + f1 >= len(wdb) or nxt <= e + f1 + 1:
            continue  # decay window not fully inside, or the next speech starts within it
        pre = float(np.percentile(wdb[max(e - pre_n, 0): e + 1], 90))
        keep = np.flatnonzero(wdb[max(e - pre_n, 0): e + 1] >= pre - ANCHOR_DROP_DB)  # speech end: last frame near the active level
        e = max(e - pre_n, 0) + int(keep[-1])
        if e + f1 >= len(wdb) or nxt <= e + f1 + 1:
            continue
        if pre < floor_db + DECAY_MIN_ABOVE_FLOOR_DB:
            continue
        post = float(w[i0 + e + f0: i0 + e + f1 + 1].mean())
        if post < fl["W"] * 10 ** (CENSOR_DB / 10):
            cens += 1
            post_net = fl["W"]
        else:
            post_net = post - fl["W"]
        vals.append(float(_db(post_net) - pre))
    return vals, cens


def _measure_pan(F, i0, i1, fl, act):
    """Pan track of the slot: [[t_rel_s, bal_db], ...] on a 100 ms grid and the share of measured bins."""
    nb = max(1, (i1 - i0) // PAN_HOP_FRAMES)
    L = _net(F["L"][i0:i1], fl["L"])
    R = _net(F["R"][i0:i1], fl["R"])
    m = min(nb * PAN_HOP_FRAMES, len(act))
    sl = np.zeros(nb * PAN_HOP_FRAMES, bool)
    sl[:m] = act[:m]
    a = sl.reshape(nb, PAN_HOP_FRAMES)
    pad = np.zeros(nb * PAN_HOP_FRAMES)
    pl, pr = pad.copy(), pad.copy()
    pl[:m], pr[:m] = L[:m], R[:m]
    sum_l = (pl.reshape(nb, -1) * a).sum(1)
    sum_r = (pr.reshape(nb, -1) * a).sum(1)
    measured = a.sum(1) >= PAN_MIN_ACTIVE
    vals = np.where(measured, np.clip(_db(sum_l) - _db(sum_r), -PAN_RAW_CLIP_DB, PAN_RAW_CLIP_DB), np.nan)
    ok = np.flatnonzero(measured)
    if not len(ok):
        track = np.zeros(nb)
        share = 0.0
    else:
        v = vals[ok]
        h = PAN_MEDIAN // 2
        sm = np.array([np.median(v[max(0, i - h): i + h + 1]) for i in range(len(v))])
        track = np.interp(np.arange(nb), ok, sm)  # hold (edge values) over unmeasured bins, leading ones back-filled
        share = len(ok) / nb
    t = (np.arange(nb) + 0.5) * PAN_HOP_FRAMES / 100.0
    return [[round(float(x), 2), round(float(y), 2)] for x, y in zip(t, track)], share


def _measure_slot(F, i0, i1, iR, fl):
    """Raw measurements of one slot (frames [i0, i1) of F, decay events searched up to iR >= i1): tilt, decay events, pan track."""
    act = _active_mask(F, i0, i1, fl)
    tilt = _measure_tilt(F, i0, i1, fl, act)
    ev, cens = _measure_decay(F, i0, iR, fl)
    track, share = _measure_pan(F, i0, i1, fl, act)
    return {"tilt": tilt, "events": ev, "censored": cens, "track": track, "pan_share": share, "n_active": int(act.sum())}


def _raw_values(meas):
    """(raw_wet, m_wet, s2_wet, raw_tilt, m_tilt, s2_tilt) of a slot measurement; nan = no evidence."""
    ev = meas["events"]
    t = meas["tilt"]
    return (float(np.median(ev)) if ev else np.nan, len(ev), float(np.var(ev, ddof=1)) if len(ev) >= 2 else np.nan,
            t[0] if t else np.nan, t[2] if t else 0, t[3] if t else np.nan)


# ----------------------------------------------------------------------------- shrinkage

def _pooled_sigma2(m, s2):
    """Pooled within-part variance of the replicates: sum (m-1) s2 / sum (m-1) over parts with a variance, and its dof."""
    ok = np.isfinite(s2) & (m >= 2)
    dof = float((m[ok] - 1).sum())
    return (float((s2[ok] * (m[ok] - 1)).sum() / dof) if dof > 0 else float("nan")), dof


def _shrink_constants(dev, m, s2):
    """Empirical-Bayes constants of the departures `dev` of the stand-in parts: {sigma2, dof, tau2, fallback}."""
    good = np.isfinite(dev) & (m >= 1)
    sig2, dof = _pooled_sigma2(m, s2)
    info = {"sigma2": sig2, "dof": dof, "tau2": float("nan"), "fallback": False}
    if dof < MIN_DOF or not np.isfinite(sig2) or good.sum() < 3:
        info["fallback"] = True
        return info
    d = dev[good]
    var_d = (1.4826 * np.median(np.abs(d - np.median(d)))) ** 2
    info["tau2"] = float(max(var_d - float(np.mean(sig2 / m[good])), 0.0))
    return info


def _weight(info, m):
    """Shrink weight of a part with m replicates (0 when there are none)."""
    if m < 1:
        return 0.0
    if info["fallback"]:
        return m / (m + 1.0)
    tau2 = info["tau2"]
    return tau2 / (tau2 + info["sigma2"] / m) if tau2 > 0 else 0.0


# ----------------------------------------------------------------------------- analysis

def _slot_frames(start_s, end_s, next_start_s, nfr):
    """(i0, i1, iR) frames of a slot: the slot, and the slot extended to 50 ms before the next part (at most TAIL_AFTER_S past the end)."""
    i0 = max(0, int(start_s * 100))
    i1 = min(nfr, int(max(end_s, start_s + 0.2) * 100))
    iR = i1 + int(TAIL_AFTER_S * 100)
    if next_start_s is not None:
        iR = min(iR, int(next_start_s * 100) - 5)
    return i0, i1, min(nfr, max(iR, i1))


def _scene_of(scenes, t):
    for s in scenes:
        if s["start"] <= t < s["end"]:
            return s["id"]
    return scenes[-1]["id"]


def analyze(clip, chan, regions, workers=8):
    """Scene floors of the stem features, per-scene neutral (median) wet / tilt / pan and the shrinkage constants.
    chan = channel document (needs chan["scenes"] = [{"id", "start", "end"} ...], seconds); regions = nominal speech regions
    [(start_s, end_s)] used as stand-in parts. Returns a picklable dict:
      {"sr", "scenes": [{"id", "start", "end"}], "floors": {scene id: {"W","A","B","L","R": power}},
       "scene_neutral": {scene id: {"wet_db", "tilt_db" (None without MIN_GROUP parts with evidence), "pan_db", "n_parts"}},
       "shrink": {"wet": {...}, "tilt": {...}}  (sigma2, dof, tau2, fallback), "stats": {...}}"""
    stem = clip.speech2
    F = features(stem, 0, stem.shape[1] // FR, workers)
    nfr = len(F["W"])
    scenes = [{"id": s["id"], "start": float(s["start"]), "end": float(s["end"])} for s in chan["scenes"]]
    floors = {s["id"]: _floors(F, int(s["start"] * 100), min(nfr, max(int(s["end"] * 100), int(s["start"] * 100) + 1))) for s in scenes}
    regs = sorted(regions)
    starts = [a for a, _ in regs]
    scene_of, meas = [], []
    for k, (a, b) in enumerate(regs):
        sc = _scene_of(scenes, 0.5 * (a + b))
        scene_of.append(sc)
        i0, i1, iR = _slot_frames(a, b, starts[k + 1] if k + 1 < len(regs) else None, nfr)
        meas.append(_measure_slot(F, i0, i1, iR, floors[sc]))
    raw = np.array([_raw_values(m) for m in meas], float).reshape(-1, 6)
    raw_wet, m_wet, s2_wet, raw_tilt, m_tilt, s2_tilt = raw.T

    neutral = {}
    dev_w = np.full(len(regs), np.nan)
    dev_t = np.full(len(regs), np.nan)
    for s in scenes:
        mem = [i for i, sc in enumerate(scene_of) if sc == s["id"]]
        pans = [y for i in mem for _, y in meas[i]["track"]]
        nt = {"wet_db": None, "tilt_db": None, "pan_db": round(float(np.median(pans)), 2) if pans else 0.0, "n_parts": len(mem)}
        for key, rv, dev in (("wet_db", raw_wet, dev_w), ("tilt_db", raw_tilt, dev_t)):
            v = [rv[i] for i in mem if np.isfinite(rv[i])]
            if len(v) >= MIN_GROUP:
                nt[key] = float(np.median(v))
                for i in mem:
                    dev[i] = rv[i] - nt[key]
        neutral[s["id"]] = nt
    shrink = {"wet": _shrink_constants(dev_w, m_wet, s2_wet),
              "tilt": _shrink_constants(dev_t, np.where(np.isfinite(dev_t), np.maximum(m_tilt, 1.0), 0.0), s2_tilt)}
    return {"sr": SR, "scenes": scenes, "floors": floors, "scene_neutral": neutral, "shrink": shrink,
            "stats": {"n_regions": len(regs), "wet_parts_with_evidence": int(np.isfinite(dev_w).sum()),
                      "tilt_parts_with_evidence": int(np.isfinite(dev_t).sum())}}


# ----------------------------------------------------------------------------- one part at render time

def part_cue(clip, doc, start_s, end_s, next_start_s=None):
    """Cue of one part measured on the speech stem inside its original slot [start_s, end_s] (decay events are searched up to
    next_start_s, at most TAIL_AFTER_S past the slot). Returns {'wet_db', 'tilt_db', 'pan_db_track': [[t_rel_s, bal_db], ...]};
    wet_db > 0 = more reverberant than the scene's neutral, tilt_db > 0 = brighter, pan bal_db = 10log10(L / R)."""
    nfr = clip.n // FR
    i0, i1, iR = _slot_frames(start_s, end_s, next_start_s, nfr)
    sc = _scene_of(doc["scenes"], 0.5 * (start_s + end_s))
    fl = doc["floors"][sc]
    f0 = max(0, i0 - int(PRE_S * 100))
    base = max(0, f0 - 3)  # read only the samples the frames f0..iR touch (the stem of a 10-min clip is 230 MB)
    stem = read_range(clip.dir / "speech.flac", base * FR, (iR + 3) * FR, 2)
    F = features(stem, f0 - base, iR - base)
    meas = _measure_slot(F, i0 - f0, i1 - f0, iR - f0, fl)
    raw_wet, m_wet, _, raw_tilt, m_tilt, _ = _raw_values(meas)
    neu = doc["scene_neutral"][sc]
    wet = tilt = 0.0
    if neu["wet_db"] is not None and np.isfinite(raw_wet):
        wet = float(np.clip(_weight(doc["shrink"]["wet"], m_wet) * (raw_wet - neu["wet_db"]), -WET_CLAMP_DB, WET_CLAMP_DB))
    if neu["tilt_db"] is not None and np.isfinite(raw_tilt):
        tilt = float(np.clip(_weight(doc["shrink"]["tilt"], max(m_tilt, 1.0)) * (raw_tilt - neu["tilt_db"]), -TILT_CLAMP_DB, TILT_CLAMP_DB))
    return {"wet_db": round(wet, 2), "tilt_db": round(tilt, 2), "pan_db_track": meas["track"]}


# ----------------------------------------------------------------------------- application

def tilt_response(freqs, tilt_db, f0=TILT_CORNER_HZ, f_ref=TILT_REF_HZ):
    """Magnitude of the first-order high shelf |H|^2 = (1 + (G f/f0)^2) / (1 + (f/f0)^2) with the asymptote G solved so that
    |H(f_ref)| = tilt_db."""
    t2 = 10.0 ** (tilt_db / 10.0)
    x2 = (f_ref / f0) ** 2
    g2 = max((t2 * (1.0 + x2) - 1.0) / x2, 0.0)
    f2 = (np.asarray(freqs, np.float64) / f0) ** 2
    return np.sqrt((1.0 + g2 * f2) / (1.0 + f2))


_TILT_FIR = {}


def tilt_filter(x, tilt_db, sr=SR):
    """High shelf of tilt_db at TILT_REF_HZ (corner TILT_CORNER_HZ) on mono x: linear-phase FIR (TILT_TAPS), delay compensated
    (zero latency), same length, float32. tilt_db = 0 returns a copy."""
    x = np.asarray(x, np.float32)
    if abs(tilt_db) < 1e-6 or len(x) == 0:
        return x.copy()
    key = (round(float(tilt_db), 3), sr)
    h = _TILT_FIR.get(key)
    if h is None:
        nfft = 1 << ((TILT_TAPS - 1).bit_length() + 2)  # frequency grid of the prototype
        mag = tilt_response(np.fft.rfftfreq(nfft, 1.0 / sr), tilt_db)
        p = np.fft.irfft(mag, nfft)  # zero-phase, symmetric around 0
        half = TILT_TAPS // 2
        h = _TILT_FIR[key] = (np.concatenate([p[-half:], p[: half + 1]]) * np.hanning(TILT_TAPS)).astype(np.float32)
    half = TILT_TAPS // 2
    nf = sfft.next_fast_len(len(x) + TILT_TAPS - 1, real=True)
    y = sfft.irfft(sfft.rfft(x, nf) * sfft.rfft(h, nf), nf)
    return y[half: half + len(x)].astype(np.float32)


def _balance_knots(parts, limit_db):
    """Knots (time s, balance dB) of the pan movement along the dub: per part its original track time-stretched to the dub span,
    held and then ramped over PAN_RAMP_S into the next part."""
    kt, kb = [], []

    def push(t, b):
        if kt and t <= kt[-1]:
            t = kt[-1] + 1e-4
        kt.append(t)
        kb.append(b)

    for p in parts:
        tr = p.get("pan_db_track")
        if not tr:
            continue
        tr = np.asarray(tr, float)
        d0, d1 = p["d0"], p["d1"]
        dur_o = max(p["o1"] - p["o0"], 1e-6)
        td = d0 + np.clip(tr[:, 0] / dur_o, 0.0, 1.0) * (d1 - d0)  # time stretch by the span ratio
        bal = np.clip(tr[:, 1], -limit_db, limit_db)
        if not kt:
            push(max(d0 - PAN_RAMP_S, 0.0), 0.0)
        else:  # hold the previous value, then ramp into this part
            push(max(d0 - PAN_RAMP_S, kt[-1] + 1e-4), kb[-1])
        push(d0, float(bal[0]))
        for t, b in zip(td, bal):
            push(float(t), float(b))
        push(d1, float(bal[-1]))
    if kt:
        push(kt[-1] + PAN_RAMP_S, 0.0)
    return kt, kb


def pan_gains(total_samples, parts, sr=SR, limit_db=6.0):
    """Per-sample (gl, gr) float32 of the original voice's stereo movement along the dub of a window of total_samples samples.
    parts = [{"d0", "d1" (dub span, s from the window start), "o0", "o1" (original slot, s; only o1 - o0 is used),
    "pan_db_track": [[t_rel_s, bal_db], ...]}] in time order. Balance is clipped to +-limit_db and split gl = 10^(bal/40),
    gr = 10^(-bal/40); gain 1.0 before the first part and after the last (ramped over PAN_RAMP_S)."""
    kt, kb = _balance_knots(parts, limit_db)
    if not kt:
        return np.ones(total_samples, np.float32), np.ones(total_samples, np.float32)
    grid_n = int(total_samples / sr * 200) + 2  # 5 ms control grid
    b = np.interp(np.arange(grid_n) / 200.0, kt, kb, left=0.0, right=0.0)
    s = np.arange(total_samples) / sr * 200.0
    g = np.arange(grid_n)
    return (np.interp(s, g, 10.0 ** (b / 40.0)).astype(np.float32), np.interp(s, g, 10.0 ** (-b / 40.0)).astype(np.float32))


def pan_gains_part(n, pan_db_track, dur_orig_s, limit_db=6.0, sr=SR):
    """Per-sample (gl, gr) float32 for ONE part of n samples (its dub span): the original track [[t_rel_s, bal_db], ...] of a slot of
    dur_orig_s seconds time-stretched over the n samples, clipped to +-limit_db. The part is its own window: no ramp inside it,
    the ramps between parts belong to the caller's crossfades (use pan_gains for a window of several parts)."""
    if n <= 0 or not pan_db_track:
        return np.ones(max(n, 0), np.float32), np.ones(max(n, 0), np.float32)
    tr = np.asarray(pan_db_track, float)
    t = np.clip(tr[:, 0] / max(dur_orig_s, 1e-6), 0.0, 1.0) * n
    bal = np.clip(tr[:, 1], -limit_db, limit_db)
    b = np.interp(np.arange(n), t, bal)
    return (10.0 ** (b / 40.0)).astype(np.float32), (10.0 ** (-b / 40.0)).astype(np.float32)
