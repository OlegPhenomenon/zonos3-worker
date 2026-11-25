"""Non-linear part of the channel of a voice stem (compressor / limiter + saturation): measured and re-applied. Port of the lab's
`channel_dynamics.py` made windowed-safe for the realtime renderer.

s (the stem) = NL(y) + residual, y = linear-channel prediction of the stem (IR * dry reference). NL = compressor, then a
memoryless saturation. Everything is measured from the audio of the scene; there are no presets.

API
    fit_dynamics(y, s, spans, sr=SR, x_ref=None) -> dict   (picklable, plain python / numpy scalars)
        y, s float32 mono, same length, time aligned to ~ms (a residual envelope lag of +-25 ms is measured and removed, reported
        as align_ms). spans = [(a, b)] speech regions, seconds. x_ref = (signal, spans): a dry voice through the scene's linear
        channel; its level statistics fix the saturation's peak reference (see "Reference levels").
        Result keys: compressor {active, threshold_db (re lref), ratio, knee_db, attack_ms, release_ms, makeup_db},
        saturation {active, drive, bias, curve}, ref {lref_db, sat_peak}, measures, confidence, improvement, notes, align_ms, levels.
    apply_dynamics(x, params, sr=SR, level_db=None) -> float32 array, same length, no delay
        x: mono (n,) float32 dub channel at the level the level-following leaves it (absolute: the stem's level, as the dub is
        level-followed to the original part by part). A pure local function: no statistic of x is measured. Compressor threshold
        = lref + threshold_db with lref = params["ref"]["lref_db"] (the stem's active speech level at fit time; `level_db`
        replaces it, e.g. the level a part was followed to), saturation peak reference P = params["ref"]["sat_peak"]
        (linear amplitude). State memory: memory_s(params) seconds (10 time constants of the gain smoother; the
        window caller supplies that much pre-roll, the first samples of a window without it carry a gain error that decays
        within that time).
    memory_s(params) -> seconds of pre-roll after which a window's output equals whole-signal processing (to e^-10 of any gain step).
    preroll_samples(params, sr=SR) -> the same in samples, rounded up to the 1 ms detector grid (windows must start on that grid).

Definitions (shared by fit and apply)
  * Frames: 1 ms blocks (h = round(sr/1000) samples) give peak |x| and mean square; 20 / 50 / 400 ms statistics are aggregates.
  * Active speech level (lref) of a signal: 20 ms RMS levels (dB) of blocks fully inside the speech regions (all signal if no
    regions) that are not digital silence (> -100 dBFS); active = within 20 dB of their 95th percentile; lref = median of the
    active levels. Compressor threshold is re lref of the compressor INPUT (y when fitting).
  * Compressor: peak detector (1 ms max |x|, dB) -> static curve (threshold, ratio, soft knee) -> gain reduction smoothed in
    the dB domain by a branching one-pole (attack when reduction grows, release when it falls; numba loop) -> makeup. The 1 ms gain is linearly interpolated to samples and
    delayed by one frame so no sample sees a gain computed from the future (output is not delayed).
  * Saturation: v = P (tanh(d (u + b)) - tanh(d b)) / (d (1 - tanh(d b)^2)), u = x / P; unit small-signal gain, d = drive
    (-> 0: identity, large: hard clip), b = bias (asymmetry, even harmonics). P = peak reference.
  * Reference levels: lab: P = 99.9th percentile of the 1 ms peaks of the saturator input over active frames of the signal being
    processed (so drive is relative to the scene's own peaks at any level). Here: the dry reference voice x_ref is scaled to the
    level lref, passed through the fitted compressor, and P = that percentile of ITS output - the same rule on a stand-in dub.
  * Fit: closed loop on 4 ms power envelopes of s vs the model output (compressor gain computed from y; residual power floor
    rho fitted as nuisance) plus the distribution of the 50 ms waveform crest factor. Robust soft-L1 loss. Pieces (0.5 s of speech
    each, 1 s of signal before them warm up the detector) are taken evenly from the scene up to FIT_BUDGET_S. `active` decision
    (held out): pieces alternate into halves A/B; models {identity, c, s, cs} are fitted on one half and scored on the other;
    significant in BOTH directions (one-sided cluster t-test p < 0.01), stage curves of the two halves agree; every stage must
    pass an effect test (it must change the measures of NL(y) by more than EFFECT_Z sampling errors of the stem's measures).

Differences from the lab: thresholds / peak reference come from stored reference levels instead of the processed signal;
compressor release is limited to RELEASE_MAX_MS (state memory); FIT_BUDGET_S, NFEV smaller (compute budget).
"""
import math

import numpy as np
from numba import njit
from scipy import stats
from scipy.optimize import least_squares

from . import core

SR = core.SR
FIT_BUDGET_S = 40.0  # compute budget of the fit (speech seconds used), not a signal property
PIECE = 500  # frames (ms) per fit piece; multiple of 50
PRE = 1000  # frames of signal before each piece (detector warm-up)
ALPHA = 0.01  # significance level of the held-out test
NFEV = 25  # function evaluations allowed per least-squares fit
CS_P_MAX = 0.2  # compressor + saturation is fitted jointly only if c or s alone has a held-out p-value (both directions) below this
EFFECT_Z = 2.0  # a stage must change the measures of NL(y) by this many sampling errors of the stem's measures
SNR_MARGIN_DB = 10.0  # frames closer than this to the noise floor of y are not fitted (bias < 0.5 dB)
RELEASE_MAX_MS = 150.0  # slowest release the fit may choose: bounds the state memory of a window (memory_s = 8 x this)
NAMES = ("thr", "minv", "knee", "lat", "lrel", "G", "rho", "ld", "b")
TH0 = np.array([-20.0, 0.5, 6.0, math.log(5.0), math.log(100.0), 0.0, -60.0, 0.0, 0.0])
LO = np.array([-60.0, 0.01, 0.0, math.log(0.25), math.log(5.0), -40.0, -100.0, math.log(0.05), -1.0])
HI = np.array([20.0, 1.0, 24.0, math.log(100.0), math.log(RELEASE_MAX_MS), 40.0, -3.0, math.log(60.0), 1.0])
COMP_RATIO_MAX = 10.0  # ratios above this are limiter territory: the fit then only clamps the loudest peaks and flattens the
# per-part level; peak control is the mastering limiter's job, so such a stage is rejected


def plausible_compressor(c):
    """A fitted compressor stage that a real dialogue chain could contain: ratio <= COMP_RATIO_MAX."""
    return bool(c["active"]) and c["ratio"] <= COMP_RATIO_MAX


FREE = {"0": (5, 6), "c": (0, 1, 2, 3, 4, 5, 6), "s": (5, 6, 7, 8), "cs": tuple(range(9))}


# ----------------------------------------------------------------------------- frames, levels
def _frames(x, sr):
    h = max(1, int(round(sr * 0.001)))
    n = -(-len(x) // h)
    xp = np.zeros(n * h, np.float32)
    xp[:len(x)] = x
    X = xp.reshape(n, h)
    return X, np.einsum("ij,ij->i", X, X) / h, np.abs(X).max(1), h


def _db(p):
    return 10 * np.log10(np.maximum(p, 1e-14))


def _dba(a):
    return 20 * np.log10(np.maximum(a, 1e-7))


def _span_mask(spans, n, h, sr):
    m = np.zeros(n, bool)
    if spans is None:
        m[:] = True
    else:
        for a, b in spans:
            m[int(math.ceil(a * sr / h)):int(b * sr / h)] = True
    return m


def _blocks(a, k, red):
    nb = len(a) // k
    return red(a[:nb * k].reshape(nb, k), axis=1)


def _ref_level(pw, m, k=20):
    """(lref dB, active 20 ms blocks, P95 dB) or None."""
    L = _db(_blocks(pw, k, np.mean))
    ok = _blocks(m, k, np.all) & (L > -100)
    if ok.sum() < 5:
        return None
    p95 = float(np.percentile(L[ok], 95))
    act = ok & (L >= p95 - 20)
    return float(np.median(L[act])), act, p95


def _active_frames(act, n, k=20):
    a = np.zeros(n, bool)
    a[:len(act) * k] = np.repeat(act, k)
    return a


def _lag(pwy, pws, m, maxlag=25):
    """Frames k such that the envelope es[n + k] ~ ey[n] (s lags y by k frames), over speech frames."""
    ey, es = np.sqrt(pwy) * m, np.sqrt(pws) * m
    ey, es = ey - ey[m].mean() * m, es - es[m].mean() * m
    N = 1 << int(len(ey) + maxlag).bit_length()
    c = np.fft.irfft(np.fft.rfft(es, N) * np.conj(np.fft.rfft(ey, N)), N)
    c = np.concatenate([c[-maxlag:], c[:maxlag + 1]])
    return int(np.argmax(c)) - maxlag


# ----------------------------------------------------------------------------- measures
def _hd(xp, starts, h, sr):
    """Harmonic indicator: median over voiced 40 ms frames of 10 log10(E(2f0..5f0) / E(f0)), +-0.15 f0 bands."""
    L = 40 * h
    nfft = 8192
    P = np.abs(np.fft.rfft(xp[starts[:, None] * h + np.arange(L)] * np.hanning(L), nfft, axis=1)) ** 2
    ac = np.fft.irfft(P, nfft, axis=1)
    lo, hi = int(sr / 400), int(sr / 70)
    a = ac[:, lo:hi] / (ac[:, :1] + 1e-20)
    lag = lo + a.argmax(1)
    voiced = a.max(1) > 0.5
    if voiced.sum() < 5:
        return float("nan")
    cs = np.concatenate([np.zeros((len(P), 1)), np.cumsum(P, 1)], 1)
    f0 = (nfft / lag)[:, None]
    ks = np.arange(1, 6)[None, :]
    lo_i, hi_i = np.round((ks - 0.15) * f0).astype(int), np.round((ks + 0.15) * f0).astype(int)
    E = np.take_along_axis(cs, hi_i, 1) - np.take_along_axis(cs, lo_i, 1)
    return float(np.median(10 * np.log10((E[voiced, 1:].sum(1) + 1e-20) / (E[voiced, 0] + 1e-20))))


def _arrays(x, sr, spans):
    """Per-frame distributions behind the measures: crest (50 ms), 400 ms levels (gated), 20 ms active levels."""
    X, pw, pk, h = _frames(x, sr)
    m = _span_mask(spans, len(pw), h, sr)
    r = _ref_level(pw, m)
    if r is None:
        return None
    lref, act, p95 = r
    p50, pk50 = _blocks(pw, 50, np.mean), _blocks(pk, 50, np.max)
    ok = _blocks(m, 50, np.all) & (_db(p50) >= p95 - 20)
    cs = np.concatenate([[0.0], np.cumsum(pw)])
    st = np.arange(0, len(pw) - 400, 100)
    L4 = _db((cs[st + 400] - cs[st]) / 400)
    L4 = L4[np.array([m[i:i + 400].all() for i in st]) & (L4 > -70)]  # absolute gate
    return {"crest": 20 * np.log10(np.maximum(pk50, 1e-7) / np.sqrt(np.maximum(p50, 1e-14)))[ok], "L4": L4,
            "L20": _db(_blocks(pw, 20, np.mean))[act], "PK": _dba(pk[_active_frames(act, len(pk)) & m]), "lref": lref, "act": act, "X": X, "h": h}


QS = (5, 10, 25, 75, 90, 95)


def _core(crest, L4, L20, PK):
    """(crest median, crest P95, LRA, PLR = P99.9 of the 1 ms peaks - loudness, level quantiles re the median active level)."""
    loud = _db(np.mean(10 ** (L4 / 10)))
    L4 = L4[L4 >= loud - 20]  # relative gate (EBU style)
    return [float(np.median(crest)), float(np.percentile(crest, 95)), float(np.percentile(L4, 95) - np.percentile(L4, 10)),
            float(np.percentile(PK, 99.9) - loud)] + \
        [float(np.percentile(L20, p) - np.median(L20)) for p in QS]


def _measures(x, sr, spans):
    A = _arrays(x, sr, spans)
    if A is None:
        return None
    c = _core(A["crest"], A["L4"], A["L20"], A["PK"])
    act, h = A["act"], A["h"]
    c20 = np.nonzero(act[0::2][:len(act) // 2] & act[1::2][:len(act) // 2])[0] * 2  # 40 ms blocks, both halves active
    sel = c20[np.linspace(0, len(c20) - 1, min(len(c20), 400)).round().astype(int)] * 20 if len(c20) else c20
    return {"crest_med_db": c[0], "crest_p95_db": c[1], "lra_db": c[2], "plr_db": c[3],
            "hd_db": _hd(A["X"].ravel(), sel, h, sr) if len(sel) else float("nan"),
            "level_q": {str(p): v for p, v in zip(QS, c[4:])}, "lref_db": A["lref"]}


def _boot_se(A, B=40, seed=0):
    """Standard error of _core of the distributions in A: block bootstrap over ~2 s blocks."""
    rng = np.random.default_rng(seed)
    arrs = [(A[k], g) for k, g in (("crest", 40), ("L4", 20), ("L20", 100), ("PK", 2000))]
    groups = [np.array_split(np.arange(len(a)), max(1, len(a) // g)) for a, g in arrs]
    out = []
    for _ in range(B):
        rs = [a[np.concatenate([gr[i] for i in rng.integers(0, len(gr), len(gr))])] for (a, _g), gr in zip(arrs, groups)]
        out.append(_core(*rs))
    return np.std(out, axis=0)


# ----------------------------------------------------------------------------- compressor / saturator kernels
def _reduction(Ld, thr, minv, knee):
    """Static curve: gain reduction (dB >= 0) at detector level Ld (dB); minv = 1 / ratio; quadratic soft knee."""
    over = Ld - thr
    if knee > 1e-6:
        g = np.where(over <= -knee / 2, 0.0, np.where(over >= knee / 2, over, (over + knee / 2) ** 2 / (2 * knee)))
    else:
        g = np.maximum(over, 0.0)
    return (1 - minv) * g


@njit(cache=True)
def _smooth_rows(R, aa, ar):
    """Branching one-pole along axis 0 of R (T, K), K independent columns: coefficient aa when the input is above the state
    (reduction grows), ar when below. Starts from the first row."""
    T, K = R.shape
    out = np.empty_like(R)
    for k in range(K):
        st = R[0, k]
        out[0, k] = st
        for t in range(1, T):
            d = R[t, k] - st
            st += (aa if d > 0.0 else ar) * d
            out[t, k] = st
    return out


def _coef(ms, h, sr):
    return 1 - math.exp(-h / (sr * 1e-3 * ms))


def _smooth_1d(r, att_ms, rel_ms, h, sr):
    """The branching one-pole over one signal r (frames), started from r[0]."""
    return _smooth_rows(np.ascontiguousarray(r, np.float64)[:, None], _coef(att_ms, h, sr), _coef(rel_ms, h, sr))[:, 0]


def _sat(z, P, d, b):
    if d < 1e-6:
        return z
    t0 = math.tanh(d * b)
    return (np.tanh(d * (z / P + b)) - t0) * (P / (d * (1 - t0 * t0)))


def _sat_peak(zmax, zmin, P, d, b):
    """Largest |output| of the saturation over a frame from the frame's extreme input samples (the curve is monotone)."""
    return np.maximum(np.abs(_sat(zmax, P, d, b)), np.abs(_sat(zmin, P, d, b)))


def _compress(x, c, lref, sr):
    """x through the compressor stage c with the threshold re lref (dB): float32 (n,)."""
    X, pw, pk, h = _frames(x, sr)
    n = len(pk)
    r = _reduction(_dba(pk), lref + c["threshold_db"], 1 / c["ratio"], c["knee_db"])
    gdb = c["makeup_db"] - _smooth_1d(r, c["attack_ms"], c["release_ms"], h, sr)
    g = np.power(10.0, gdb / 20).astype(np.float32)
    pos = (np.arange(n) + 2) * h  # gain of frame i is valid from the end of frame i + 1: no look-ahead
    return x * np.interp(np.arange(len(x)), pos, g).astype(np.float32)


def memory_s(params):
    """Pre-roll (s) after which windowed processing equals whole-signal processing (gain smoother, 10 time constants)."""
    c = params["compressor"]
    return 10e-3 * max(c["attack_ms"], c["release_ms"]) if c["active"] and plausible_compressor(c) else 0.0


def preroll_samples(params, sr=SR):
    """memory_s as a whole number of 1 ms detector frames: the pre-roll to cut off a window processed on its own. The window must
    start on the frame grid of the whole signal (a multiple of round(sr / 1000) samples, true for a start at a whole millisecond
    of a 48 kHz signal) or the 1 ms peak detector sees different blocks and the gain differs by up to ~0.2 % of the signal."""
    h = max(1, int(round(sr * 0.001)))
    return -(-int(math.ceil(memory_s(params) * sr)) // h) * h


def apply_dynamics(x, params, sr=SR, level_db=None):
    """Pass the mono dub channel x through the fitted compressor and saturation, see the module docstring."""
    x = np.asarray(x, np.float32)
    c, sa, ref = params["compressor"], params["saturation"], params.get("ref")
    if c["active"] and not plausible_compressor(c):
        c = dict(c, active=False)
    if ref is None or not (c["active"] or sa["active"]):
        return x.copy()
    out = _compress(x, c, ref["lref_db"] if level_db is None else level_db, sr) if c["active"] else x.copy()
    if sa["active"]:
        out = _sat(out, ref["sat_peak"], sa["drive"], sa["bias"]).astype(np.float32)
    return out


def warmup():
    """Compile / load the numba gain smoother (a few hundred ms on the first call of a process, cached on disk afterwards);
    call once at service start so the first window does not pay for it."""
    c = {"active": True, "threshold_db": -10.0, "ratio": 2.0, "knee_db": 0.0, "attack_ms": 5.0, "release_ms": 50.0, "makeup_db": 0.0}
    apply_dynamics(np.zeros(SR // 10, np.float32), {"compressor": c, "saturation": {"active": False}, "ref": {"lref_db": -20.0, "sat_peak": None}})


def _nl(y, params, lref, spans, sr):
    """Fit-time NL(y): the lab's rule, the saturation peak reference measured on the processed signal itself (active frames of y)."""
    c, sa = params["compressor"], params["saturation"]
    out = _compress(y, c, lref, sr) if c["active"] else y.copy()
    if sa["active"]:
        _, pw, _, h = _frames(y, sr)
        r = _ref_level(pw, _span_mask(spans, len(pw), h, sr))
        P = float(np.percentile(_frames(out, sr)[2][_active_frames(r[1], len(pw))], 99.9))
        out = _sat(out, P, sa["drive"], sa["bias"]).astype(np.float32)
    return out


# ----------------------------------------------------------------------------- fit data
def _ma4(a):
    c = np.cumsum(a, 1)
    o = c.copy()
    o[:, 4:] = c[:, 4:] - c[:, :-4]
    return o / 4


def _psi(z):
    return np.sqrt(2 * (np.sqrt(1 + z * z) - 1))  # soft-L1 as a residual transform: sum psi^2 = soft-L1 loss


def _crest(pk, pw):
    K, S = pk.shape
    g = S // 50
    return 20 * np.log10(np.maximum(pk.reshape(K, g, 50).max(2), 1e-7) / np.sqrt(np.maximum(pw.reshape(K, g, 50).mean(2), 1e-14)))


class _Set:
    """Fit pieces: K rows, each PRE warm-up frames + PIECE frames. Row-subset with take()."""
    ROWS = ("Ld", "Yp", "Ypw", "Ypk", "Ymax", "Ymin", "obs", "mask", "cs", "cmask")

    def __init__(self, Y, pwy, pky, pws, pks, m, starts, lref, floor, h, sr):
        K = len(starts)
        t = np.asarray(starts)[:, None] + np.arange(-PRE, PIECE)[None, :]
        tp = t[:, PRE:]
        self.h, self.sr, self.lref, self.K = h, sr, lref, K
        self.Ld = _dba(pky[np.clip(t, 0, len(pky) - 1)])
        self.Yp, self.Ypw, self.Ypk = Y[tp], pwy[tp], pky[tp]
        self.Ymax, self.Ymin = self.Yp.max(2), self.Yp.min(2)
        self.ones = np.ones(h, np.float32)
        self.obs = _db(_ma4(pws[tp]))
        ok = m[tp] & (_db(_ma4(self.Ypw)) >= floor + SNR_MARGIN_DB)
        ok[:, :3] = False
        self.mask = ok
        self.cs = _crest(pks[tp], pws[tp])
        self.cmask = (m[tp].reshape(K, -1, 50).mean(2) >= 0.5) & (_db(pws[tp].reshape(K, -1, 50).mean(2)) >= floor + SNR_MARGIN_DB)
        self.se, self.sc = 1.0, 1.0

    def take(self, rows):
        o = object.__new__(_Set)
        o.__dict__.update(self.__dict__)
        for k in self.ROWS:
            setattr(o, k, getattr(self, k)[rows])
        o.K = len(o.cs)
        return o

    def fwd(self, th, comp, sat):
        """Predicted 4 ms power envelope (dB) and 50 ms crest (dB) per piece frame."""
        h, sr = self.h, self.sr
        gdb = np.full(self.Ypw.shape, th[5])
        if comp:
            r = _reduction(self.Ld, self.lref + th[0], th[1], th[2])
            rs = _smooth_rows(np.ascontiguousarray(r.T, np.float64), _coef(math.exp(th[3]), h, sr), _coef(math.exp(th[4]), h, sr)).T
            gdb = gdb - 0.5 * (rs[:, PRE - 2:-2] + rs[:, PRE - 1:-1])  # gain delayed 1.5 frames at the frame centre (matches apply)
        g = (10 ** (gdb / 20)).astype(np.float32)
        if sat:
            P = float(np.percentile((self.Ypk * g)[self.mask], 99.9))
            d, b = math.exp(th[7]), float(th[8])
            t0 = math.tanh(d * b)
            c = P / (d * (1 - t0 * t0))  # output = c (tanh(d (z / P + b)) - t0)
            u = self.Yp * (g * np.float32(d / P))[:, :, None]  # u = d z / P
            u += np.float32(d * b)
            np.tanh(u, out=u)
            sq = np.einsum("ksh,ksh->ks", u, u).astype(np.float64)
            sm = (u @ self.ones).astype(np.float64)
            pw = c * c * (sq - 2 * t0 * sm + self.h * t0 * t0) / self.h
            ep = _sat_peak(self.Ymax * g, self.Ymin * g, P, d, b)
            pk = ep
        else:
            pw, pk = self.Ypw * g * g, self.Ypk * g
        rho = 10 ** ((self.lref + th[6]) / 10)
        return _db(_ma4(pw) + rho), _crest(pk, pw)

    def parts(self, th, comp, sat):
        pe, pc = self.fwd(th, comp, sat)
        re = (self.obs - pe)[self.mask]
        rc = (self.cs - pc)[self.cmask]
        return _psi(re / self.se) / math.sqrt(max(len(re), 1)), _psi(rc / self.sc) / math.sqrt(max(len(rc), 1))

    def resid(self, th, comp, sat):
        return np.concatenate(self.parts(th, comp, sat))

    def row_loss(self, th, comp, sat):
        """Per-piece contribution to  mean(psi_env^2) + mean(psi_crest^2)."""
        e, c = self.parts(th, comp, sat)
        return (np.bincount(np.nonzero(self.mask)[0], e * e, self.K) + np.bincount(np.nonzero(self.cmask)[0], c * c, self.K))


def _fit(D, kind, th0=None):
    comp, sat = "c" in kind, "s" in kind
    free = list(FREE[kind])
    th = TH0.copy() if th0 is None else th0.copy()

    def full(v):
        t = th.copy()
        t[free] = v
        return D.resid(t, comp, sat)
    if th0 is None:  # coarse grid for the non-convex starts, G from the median offset
        best = None
        grid = [(a, b, 0.0) for a in (-30, -20, -10, 0) for b in (0.6, 0.3, 0.1)] if comp else [(0, 0.5, 0.0)]
        for g in grid:
            for d in ((0.5, 1.5, 4.0) if sat else (0.0,)):
                t = th.copy()
                t[0], t[1], t[7] = g[0], g[1], math.log(d) if sat else 0.0
                t[5] = 0.0
                pe, _ = D.fwd(t, comp, sat)
                t[5] = float(np.median((D.obs - pe)[D.mask]))
                l = float(np.sum(D.resid(t, comp, sat) ** 2))
                if best is None or l < best[0]:
                    best = (l, t)
        th = best[1]
    r = least_squares(full, th[free], bounds=(LO[free], HI[free]), x_scale=np.array([10, .3, 5, 1, 1, 3, 10, 1, .5])[free],
                      diff_step=1e-3, max_nfev=NFEV, xtol=1e-8, ftol=1e-9)
    th[free] = r.x
    return th, float(np.sum(r.fun ** 2))


def _curve_gap(ta, tb, kind):
    """Stage-curve agreement of two fits: (rms difference, rms of the mean curve) for compressor / saturator."""
    lev = np.linspace(-40, 20, 121)
    out = {}
    if "c" in kind:
        ca, cb = (_reduction(lev, t[0], t[1], t[2]) for t in (ta, tb))
        out["c"] = (float(np.sqrt(np.mean((ca - cb) ** 2))), float(np.sqrt(np.mean(((ca + cb) / 2) ** 2))))
    if "s" in kind:
        u = np.linspace(0, 1.5, 61)
        sa_, sb_ = (_sat(u, 1.0, math.exp(t[7]), t[8]) - u for t in (ta, tb))
        out["s"] = (float(np.sqrt(np.mean((sa_ - sb_) ** 2))), float(np.sqrt(np.mean(((sa_ + sb_) / 2) ** 2))))
    return out


def _ttest(d):
    K = len(d)
    if K < 3 or np.std(d) == 0:
        return (0.0, 1.0)
    t = float(np.mean(d) / (np.std(d, ddof=1) / math.sqrt(K)))
    return t, float(stats.t.sf(t, K - 1))


def _mk(th, kind):
    """Parameter dict (compressor / saturation stages) of the fit vector th for the stages in `kind`."""
    p = _identity_params([])
    if "c" in kind:
        p["compressor"] = {"active": True, "threshold_db": float(th[0]), "ratio": float(1 / th[1]), "knee_db": float(th[2]),
                           "attack_ms": float(math.exp(th[3])), "release_ms": float(math.exp(th[4])), "makeup_db": float(th[5])}
    if "s" in kind:
        p["saturation"] = {"active": True, "drive": float(math.exp(th[7])), "bias": float(th[8]), "curve": "tanh_asym"}
    return {"compressor": p["compressor"], "saturation": p["saturation"]}


def _identity_params(notes, conf=0.0, extra=None):
    p = {"compressor": {"active": False, "threshold_db": 0.0, "ratio": 1.0, "knee_db": 0.0, "attack_ms": 1.0, "release_ms": 100.0,
                        "makeup_db": 0.0},
         "saturation": {"active": False, "drive": 0.0, "bias": 0.0, "curve": "tanh_asym"},
         "measures": {}, "confidence": conf, "improvement": {}, "notes": notes}
    if extra:
        p.update(extra)
    return p


def _reference_levels(params, lref, x_ref):
    """params["ref"]: lref_db and, when the saturation is active, the peak reference P from the dry voice x_ref = (signal, spans)
    scaled to level lref and passed through the fitted compressor (None when there is no reference signal)."""
    ref = {"lref_db": float(lref), "sat_peak": None}
    if params["saturation"]["active"] and x_ref is not None:
        sig, spans = np.asarray(x_ref[0], np.float32), x_ref[1]
        _, pw, _, h = _frames(sig, SR)
        r = _ref_level(pw, _span_mask(spans, len(pw), h, SR))
        if r is not None:
            sig = sig * np.float32(10 ** ((lref - r[0]) / 20))
            out = _compress(sig, params["compressor"], lref, SR) if params["compressor"]["active"] else sig
            act = _active_frames(r[1], len(pw))
            ref["sat_peak"] = float(np.percentile(_frames(out, SR)[2][act], 99.9))
    if params["saturation"]["active"] and ref["sat_peak"] is None:
        params["saturation"] = dict(params["saturation"], active=False)
        params["notes"].append("saturation dropped: no reference voice level for its peak reference")
    params["ref"] = ref


def fit_dynamics(y, s, spans, sr=SR, x_ref=None):
    """Fit the compressor / saturation that turns y into s, see the module docstring."""
    y, s = np.asarray(y, np.float32), np.asarray(s, np.float32)
    notes = []
    Y, pwy, pky, h = _frames(y, sr)
    n = len(pwy)
    m = _span_mask(spans, n, h, sr)
    ry = _ref_level(pwy, m)
    if ry is None:
        return _identity_params(["no speech in y: identity"])
    lref, _, p95 = ry
    lag = _lag(pwy, _frames(s, sr)[1], m)
    if lag:
        s = np.concatenate([s[lag * h:], np.zeros(lag * h, np.float32)]) if lag > 0 else np.concatenate([np.zeros(-lag * h, np.float32), s[:lag * h]])
    _, pws, pks, _ = _frames(s, sr)
    L20 = _db(_blocks(pwy, 20, np.mean))
    out20 = ~_blocks(m, 20, np.any) & (L20 > -100)
    floor = float(np.median(L20[out20])) if out20.sum() >= 20 else float(np.percentile(L20[_blocks(m, 20, np.all)], 5))
    if out20.sum() < 20:
        notes.append("no gaps outside the speech regions: noise floor of y = P5 of speech levels")
    starts = []
    for a, b in (spans if spans is not None else [(0, n * h / sr)]):
        f0, f1 = int(a * sr / h), int(b * sr / h)
        starts += list(range(f0, f1 - PIECE + 1, PIECE))
    kmax = int(FIT_BUDGET_S * 1000 / PIECE)
    if len(starts) > kmax:
        starts = [starts[i] for i in np.unique(np.linspace(0, len(starts) - 1, kmax).round().astype(int))]
    meas = {"y": _measures(y, sr, spans), "s": _measures(s, sr, spans)}
    base = _identity_params(notes, 0.0, {"measures": meas, "align_ms": lag * h / sr * 1000, "levels": {"lref_y_db": lref, "floor_y_db": floor},
                                         "ref": {"lref_db": float(lref), "sat_peak": None}})
    if len(starts) < 6:
        notes.append("fewer than 6 half-second pieces of speech: no held-out test possible -> identity")
        return base
    D = _Set(Y, pwy, pky, pws, pks, m, starts, lref, floor, h, sr)
    t0, _ = _fit(D, "0")  # scales from the identity model
    r = (D.obs - D.fwd(t0, False, False)[0])[D.mask]
    D.se = max(1.4826 * float(np.median(np.abs(r - np.median(r)))), 0.05)
    rc = (D.cs - D.fwd(t0, False, False)[1])[D.cmask]
    D.sc = max(1.4826 * float(np.median(np.abs(rc - np.median(rc)))), 0.05) if len(rc) else 1.0
    half = {"A": D.take(np.arange(0, D.K, 2)), "B": D.take(np.arange(1, D.K, 2))}
    fits = {(k, kd): _fit(half[k], kd)[0] for k in half for kd in ("0", "c", "s")}
    tr_te = (("A", "B"), ("B", "A"))
    stat = {}
    for kd in ("c", "s", "cs"):
        comp, sat = "c" in kd, "s" in kd
        if kd == "cs":  # the joint model is only fitted when one of its stages shows some held-out evidence on its own
            if min(max(stat["c"]["p"]), max(stat["s"]["p"])) > CS_P_MAX:
                continue
            for k in half:
                t = fits[(k, "c")].copy()
                t[7:] = fits[(k, "s")][7:]
                fits[(k, "cs")] = _fit(half[k], "cs", t)[0]
        d, I, L = [], [], []
        for tr, te in tr_te:
            l0 = half[te].row_loss(fits[(tr, "0")], False, False)
            lm = half[te].row_loss(fits[(tr, kd)], comp, sat)
            d.append(l0 - lm)
            I.append(float((l0.sum() - lm.sum()) / l0.sum()))
            L.append(lm)
        pv = [_ttest(x)[1] for x in d]
        gap = _curve_gap(fits[("A", kd)], fits[("B", kd)], kd)
        agree = all(g[0] < g[1] for g in gap.values())
        stat[kd] = {"I": I, "p": pv, "agree": agree, "gap": gap, "loss": np.concatenate(L), "sig": all(p < ALPHA for p in pv) and min(I) > 0 and agree}
    sig = [kd for kd in stat if stat[kd]["sig"]]
    best_any = min(stat, key=lambda kd: max(stat[kd]["p"]))
    simple = [kd for kd in sig if len(kd) == 1]
    kind = min(simple, key=lambda kd: stat[kd]["loss"].sum()) if simple else ""  # simplest first; cs only if paired-significantly better
    if "cs" in sig and (not kind or _ttest(stat[kind]["loss"] - stat["cs"]["loss"])[1] < ALPHA):
        kind = "cs"
    z, th = {}, (fits[("A", kind)] if kind else t0)
    if kind:  # effect size: a stage must move the measures of NL(y) by more than the sampling error of the same measures of s
        se_s = float(np.sqrt(np.mean(_boot_se(_arrays(s, sr, spans)) ** 2)))

        def vec(t, k):
            A = _arrays(_nl(y, _mk(t, k), lref, spans, sr), sr, spans)
            return np.array(_core(A["crest"], A["L4"], A["L20"], A["PK"]))
        while kind:
            th, _ = _fit(D, kind, th)
            full = vec(th, kind)
            z = {st: float(np.sqrt(np.mean((full - vec(th, kind.replace(st, ""))) ** 2)) / se_s) for st in kind}
            weak = [st for st in kind if z[st] < EFFECT_Z]
            if not weak:
                break
            notes.append(f"stage '{min(weak, key=z.get)}' dropped: effect {min(z.values()):.2f} x the sampling error of the stem's measures (< {EFFECT_Z})")
            kind = kind.replace(min(weak, key=z.get), "")
            if kind and not stat[kind]["sig"]:
                kind = ""
    if "c" in kind and not plausible_compressor(_mk(th, kind)["compressor"]):
        notes.append(f"compressor rejected: ratio {1 / th[1]:.1f}:1 > {COMP_RATIO_MAX:g}:1 (limiter-like fit)")
        kind = kind.replace("c", "")
        if kind and not stat[kind]["sig"]:
            kind = ""
        if kind:
            th, _ = _fit(D, kind, th)
        else:
            th = t0
    if kind:  # final polish: robust scales from the fitted model's own residual (the identity scales were set by the effect being fitted)
        mad = lambda a: max(1.4826 * float(np.median(np.abs(a - np.median(a)))), 0.05) if len(a) else 1.0
        D.se, D.sc = min(D.se, mad((D.obs - D.fwd(th, "c" in kind, "s" in kind)[0])[D.mask])), min(D.sc, mad((D.cs - D.fwd(th, "c" in kind, "s" in kind)[1])[D.cmask]))
        th, _ = _fit(D, kind, th)
    st = stat[kind or best_any]
    base["improvement"] = {"model": kind or "identity", "candidate": best_any, "heldout_rel_improvement": st["I"], "p_values": st["p"], "alpha": ALPHA,
                           "curves_agree": st["agree"], "curve_gap_rms_vs_mean_rms": st["gap"], "effect_over_sampling_error": z,
                           "level_gain_db_identity": float(t0[5]), "n_pieces": int(D.K), "speech_s_fitted": D.K * PIECE / 1000}
    if not kind:
        base["confidence"] = float(max(st["p"]))
        notes.append("no stage is both significant on held-out pieces (both directions) and larger than the sampling error: identity")
        return base
    base.update(_mk(th, kind))
    base["confidence"] = float((1 - max(st["p"])) * min(st["I"]) / max(st["I"]))
    ra = lambda t, c_, s_: ((D.obs - D.fwd(t, c_, s_)[0])[D.mask], (D.cs - D.fwd(t, c_, s_)[1])[D.cmask])
    (e0, c0), (e1, c1) = ra(t0, False, False), ra(th, "c" in kind, "s" in kind)
    rms = lambda a: float(np.sqrt(np.mean(a ** 2)))
    base["improvement"].update({"env_err_db_rms": {"identity": rms(e0), "fitted": rms(e1)}, "crest_err_db_rms": {"identity": rms(c0), "fitted": rms(c1)},
                                "residual_floor_db_re_lref": float(th[6])})
    _reference_levels(base, lref, x_ref)
    base["measures"]["nl_y"] = _measures(_nl(y, base, lref, spans, sr), sr, spans)
    return base
