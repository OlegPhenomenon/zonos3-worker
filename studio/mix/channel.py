"""Channel of the original voice per scene: room tail, tone filter, dynamics. Port of the lab's `channel_match.py`.

The separated original voice (`clip.speech`) is "dry actor + a channel": room reverberation, microphone / phone / radio band,
compressor, saturation. The channel is measured from the audio itself, per scene (`scenes.detect`), and the dub (dry studio speech)
is passed through the same channel at render time. The clip is analysed BEFORE any dub exists, so every comparison the lab made
with the dub track (dry decay statistics, long-term spectrum, nominal dub regions) uses the fixed dry reference voice
`core.reference()` instead.

API
    estimate(clip, scene_doc, regions, workers=8) -> dict
        scene_doc = scenes.detect(...), regions = core.speech_regions(clip.speech). Result = the lab's channels.json schema
        {"sr", "duration_s", "scene_detection", "scenes": [{id, start, end, speech_s, n_atoms, signature_noise, borrowed_from,
        note, tail (per octave band: band_hz, tail, rt60_s, level_db, fit_rms_db, dry_rms_db, ... ; last item: ir_length_s),
        report (rt60_s, edt_s, t20_s, t30_s, drr_db, f_lo/f_hi_hz (+_3db/_20db), band_rt60_s, eq_edges, ...), model_drr_db,
        n_events, eq_db_1_6_oct, dynamics (dynamics.fit_dynamics document or None), ir_len_s, levels, slopes (per octave band,
        None = no tail), "_h" (float32 ndarray (m,): the linear channel IR at SR, direct sound + tone + tail), "_fir" (float32
        ndarray (EQ_TAPS,): the tone filter alone, i.e. the IR without the tail)}]}. Does not save anything.
    The primitives rooms / defects need (band_env_db, decay_curves, decay_stat, decay_stat_se, minus_ambient, ltas, noise_psd,
    smooth_db, eq_filter, octave_weights, window_regions, enough_events, synth_ir, conv and the constants) are module level.

Method (unchanged from the lab, see channel_match.py)
1. Reverberation, blind: free decays after speech ends are collected per octave band from the stem (level after the phrase end
   relative to the level before it; low quantile over events) and the same statistic is measured on the dry reference voice. A
   parametric tail (per band: level at 70 ms + exponential decay) is applied to the reference and fitted by analysis by synthesis
   until the reference's decay statistics equal the stem's.
2. Tone: long-term average speech spectrum of the stem over that of the reference through the tail (1/6 octave smoothing),
   minimum-phase FIR, no limit on its depth.
3. Dynamics (compressor / saturation): dynamics.py, fitted on a dereverberated copy of the stem.
Scenes with too little evidence of their own borrow the tail of the most similar scene that has enough (the tone stays their own);
scenes without regions take the whole channel of the most similar scene.

Parallelism: a persistent forkserver pool of `workers` processes (`prepare(workers)` starts it; the stem is shared memory), the scenes'
work is a task graph (envelope slices -> decay statistics -> tail fit -> tone / IR -> dynamics fit; stem spectra and the dereverberated
chunks run beside the tail fit), so a clip with one long scene still uses all workers.

Differences from the lab (all measured, see the hand-off report): 48 kHz; the reference voice replaces the dub; the tail fit uses
the linearity of the simulation in the tail level (one convolution per slope set instead of one per level candidate); band
envelopes use FFT 1024 / hop 240 samples (21 ms / 5 ms); dynamics.py explains its own changes.
"""
import math
import multiprocessing as mp
import warnings
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from multiprocessing import shared_memory

import numpy as np
import scipy.fft as sfft
from nara_wpe.wpe import wpe_v8
from scipy import signal as sps
from scipy.signal import fftconvolve
from threadpoolctl import threadpool_limits

from . import core, dynamics, scenes as scenes_mod

SR = core.SR

# ----------------------------------------------------------------------------- analysis constants
# Analysis geometry and measurement definitions; none of them is tuned to a clip.
OCT_CENTRES = 125.0 * 2.0 ** np.arange(0, 8)  # 125 Hz .. 16 kHz octave bands (cross-faded, partition of unity)
ENV_HOP_S = 0.005  # envelope frames for the decay analysis
ENV_HOP = int(round(ENV_HOP_S * SR))
ENV_WIN = 1024  # their analysis window (21 ms)
TAU_S = (0.01, 0.02, 0.04, 0.07, 0.1, 0.15, 0.2, 0.3, 0.4, 0.6, 0.8, 1.0, 1.4, 1.8, 2.4)  # times after a speech end that are compared
PRE_WIN_S = 0.2  # level before the end = 90th percentile of this window
DECAY_QUANTILE = 0.25  # events: low quantile of the decay curves (contamination only raises a tail); same for stem and reference
Z_TAIL = 2.5  # a band gets a tail when the stem's curve is above the dry reference's by this many standard errors at >= 2 time points
MIN_EVENTS = 8  # a time point needs at least this many events
FLOOR_MARGIN_DB = 6.0  # curve points closer than this to the stem's noise floor are not fitted (noise, not tail)
RT_MIN_S, RT_MAX_S = 0.03, 12.0  # numerical safety of the fixed-point iteration only
T_MIX_S = 0.005  # the modelled diffuse tail starts here
T_REF_S = 0.07  # tail level is quoted at this time
DIRECT_S = 0.0025  # direct sound window (+-) around the main IR peak for the direct-to-reverberant ratio
EDGE_DB = 6.0  # band edge = where the smoothed channel magnitude is this far below its passband level
EDGE_SMOOTH_OCT = 1 / 3
EQ_SMOOTH_OCT = 1 / 6
EQ_TAPS = 8192
TAIL_MAX_S = 14.0  # longest synthetic tail
AMBIENT_AFTER_S = 1.5
ANCHOR_SEARCH_S = 0.25  # the nominal end of a region is refined within +-this
ANCHOR_DROP_DB = 15.0
LTAS_N = 4096
REF_FIT_S = 45.0  # seconds of the reference voice used per scene (whole regions)
DYN_BODY_S = 8.0  # dynamics fit: stretches of the scene that are dereverberated (WPE) and fitted ...
DYN_LEAD_S = 1.0  # ... each with this much signal before it for the WPE prediction
WPE_RATE = 16000  # WPE runs on a 16 kHz copy (32 ms window, 16 ms hop); its gain mask is applied to the full-band signal
WPE_NFFT, WPE_HOP = 512, 256
WPE_TAIL_S, WPE_ITERS, WPE_DELAY = 0.5, 2, 2

pow2 = core.pow2


# ----------------------------------------------------------------------------- band envelopes and decay events

def octave_weights(nfft, sr=SR):
    """Partition of unity over rfft bins: one cos^2 cross-faded weight per octave band centre (lowest / highest are shelves)."""
    f = np.maximum(np.fft.rfftfreq(nfft, 1 / sr), 1.0)
    pos = np.log2(f / OCT_CENTRES[0])
    nb = len(OCT_CENTRES)
    ws = np.zeros((nb, len(f)))
    for b in range(nb):
        d = pos - b
        w = np.where(np.abs(d) < 1, np.cos(0.5 * np.pi * d) ** 2, 0.0)
        if b == 0:
            w = np.where(d < 0, 1.0, w)
        if b == nb - 1:
            w = np.where(d > 0, 1.0, w)
        ws[b] = w
    return ws


_BW = {}


def _env_consts():
    if "w" not in _BW:
        _BW["w"] = np.ascontiguousarray(octave_weights(ENV_WIN, SR).T, dtype=np.float32)  # (bins, bands)
        _BW["win"] = np.hanning(ENV_WIN).astype(np.float32)
    return _BW["w"], _BW["win"]


def _frames(x):
    """Strided view (frames, ENV_WIN) of x at hop ENV_HOP (zero-padded to one window when shorter)."""
    if len(x) < ENV_WIN:
        x = np.pad(x, (0, ENV_WIN - len(x)))
    return np.lib.stride_tricks.sliding_window_view(x, ENV_WIN)[::ENV_HOP]


def band_env_db(x, sr=SR):
    """Octave-band power envelope in dB, 5 ms frames (hann 1024 window): (frames, bands) float32."""
    w, win = _env_consts()
    view = _frames(np.asarray(x, np.float32))
    out = np.empty((len(view), len(OCT_CENTRES)), np.float32)
    for s0 in range(0, len(view), 2048):
        X = sfft.rfft(view[s0:s0 + 2048] * win, axis=1)
        out[s0:s0 + 2048] = 10 * np.log10((X.real * X.real + X.imag * X.imag) @ w + 1e-20)
    return out


def decay_curves(db, regions, sr=SR):
    """Free-decay curves after speech ends. Every region end is refined (same rule on every signal compared) to the last 5 ms
    frame within +-ANCHOR_SEARCH_S of it whose broadband level is within ANCHOR_DROP_DB of the level before it. Per band:
    level at anchor + TAU_S minus the level before the anchor (90th percentile of the PRE_WIN_S before it); NaN where the next
    speech starts within the window or the band is weaker than 15 dB over the signal's 5th percentile. Returns
    (events x taus x bands curves, events x bands pre-levels, per-band floor = 5th percentile of the band level)."""
    hop = ENV_HOP_S
    off = ENV_WIN / 2 / sr
    nfr = len(db)
    live = db.max(1) > -120.0  # digital silence (zero padding) is not a noise floor
    floor = np.percentile(db[live], 5, axis=0) if live.any() else np.full(db.shape[1], -200.0)
    # ambient level: what the band does in long pauses (>= AMBIENT_AFTER_S after a speech end, until 0.2 s before the next
    # speech): stationary noise / crowd leakage plateau. 20th percentile (intermittent bursts raise, never lower it).
    t = np.arange(nfr) * hop + off
    ends = np.array([e for _, e in regions])
    starts = np.array([a for a, _ in regions])
    if len(ends):
        k = np.searchsorted(ends, t, side="right") - 1  # last region that ended before each frame
        sx = np.concatenate([starts, [np.inf]])
        k2 = np.searchsorted(sx, t, side="right")  # first region starting after each frame
        pause = (k >= 0) & (k2 == k + 1) & (t - ends[np.maximum(k, 0)] >= AMBIENT_AFTER_S) & (sx[k2] - t >= 0.2) & live
        if pause.sum() >= 100:
            floor = np.percentile(db[pause], 20, axis=0)
    bb = 10 * np.log10((10 ** (db / 10)).sum(1) + 1e-20)
    ev = np.full((len(regions), len(TAU_S), db.shape[1]), np.nan)
    pre = np.full((len(regions), db.shape[1]), np.nan)
    ns = int(round(ANCHOR_SEARCH_S / hop))
    npre = int(PRE_WIN_S / hop)
    taus = np.array(TAU_S)
    dtau = np.round(taus / hop).astype(int)
    for k, (a, e) in enumerate(regions):
        nxt = regions[k + 1][0] if k + 1 < len(regions) else float("inf")
        ic = int(round((e - off) / hop))
        lo, hi = ic - ns, ic + ns
        if lo - npre < 0 or hi >= nfr:
            continue
        ref = np.percentile(bb[lo - int(0.1 / hop): lo + int(0.1 / hop) + 1], 90)
        hot = np.where(bb[lo:hi + 1] >= ref - ANCHOR_DROP_DB)[0]
        if not len(hot):
            continue
        ia = lo + int(hot[-1])
        t_anchor = ia * hop + off
        p = np.percentile(db[max(ia - npre, 0): ia + 1], 90, axis=0)
        okb = p > floor + 15.0
        pre[k] = np.where(okb, p, np.nan)
        reach = t_anchor + taus + 0.05 <= nxt  # taus up to the first one that would run into the next speech
        nt = int(np.argmin(reach)) if not reach.all() else len(taus)
        idx = ia + dtau[:nt]
        nt = min(nt, int(np.searchsorted(idx, nfr)))  # frames beyond the end of the signal are not measured
        ev[k, :nt] = np.where(okb[None, :], db[idx[:nt]] - p[None, :], np.nan)
    return ev, pre, floor


def _nanquantile0(ev, q):
    """np.nanquantile(ev, q, axis=0) (linear interpolation) with NaN for all-NaN columns, and the count of finite values."""
    cnt = np.isfinite(ev).sum(0)
    srt = np.sort(ev, axis=0)  # NaN sorts last
    pos = np.maximum(cnt - 1, 0) * q
    lo = np.floor(pos).astype(int)
    hi = np.minimum(lo + 1, np.maximum(cnt - 1, 0))
    a = np.take_along_axis(srt, lo[None], 0)[0]
    b = np.take_along_axis(srt, hi[None], 0)[0]
    return np.where(cnt > 0, a + (b - a) * (pos - lo), np.nan), cnt


def decay_stat(ev, q=DECAY_QUANTILE):
    """Quantile over events per (tau, band) with the number of events: (stat, count)."""
    st, cnt = _nanquantile0(ev, q)
    return np.where(cnt >= MIN_EVENTS, st, np.nan), cnt


def minus_ambient(stat, floor_rel):
    """Decay curve (dB re the level before the end) with the stationary ambient (floor_rel dB re that level, per band) removed in
    the power domain; points that are not at least 3 dB above the ambient become NaN."""
    p = 10 ** (stat / 10) - 10 ** (floor_rel[None, :] / 10)
    with np.errstate(all="ignore"):
        return np.where(p > 10 ** (floor_rel[None, :] / 10), 10 * np.log10(np.maximum(p, 1e-30)), np.nan)


def decay_stat_se(ev, q=DECAY_QUANTILE, boots=60, seed=3):
    """Bootstrap (over events) standard error of the decay_stat quantiles, (taus, bands); NaN where not measured."""
    rng = np.random.default_rng(seed)
    n = ev.shape[0]
    acc = []
    for _ in range(boots):
        st, _ = decay_stat(ev[rng.integers(0, n, n)], q)
        acc.append(st)
    with np.errstate(all="ignore"):
        acc = np.array(acc)
        ok = np.isfinite(acc)
        mean = np.where(ok, acc, 0).sum(0) / np.maximum(ok.sum(0), 1)
        var = (np.where(ok, acc - mean, 0) ** 2).sum(0) / np.maximum(ok.sum(0), 1)
        return np.where(ok.any(0), np.sqrt(var), np.nan)


def window_regions(regions, t0, t1):
    """Regions [(a, b)] lying inside [t0, t1] s."""
    return [(a, b) for a, b in regions if a >= t0 and b <= t1]


def enough_events(cnt):
    """A scene can be judged when at least one mid-time point (0.1 .. 0.4 s) has MIN_EVENTS events in some band."""
    j = [TAU_S.index(t) for t in (0.1, 0.2, 0.4)]
    return int(cnt[j].max()) >= MIN_EVENTS


# ----------------------------------------------------------------------------- tail model

_NOISE = {}


def _noise_bank():
    """Unit-power band-limited noise per octave band times the raised-cosine onset, for TAIL_MAX_S (fixed seed): float32
    (bands, n), and the time axis. Built once per process (before the worker pool forks)."""
    if not _NOISE:
        n = int(TAIL_MAX_S * SR)
        nfft = sfft.next_fast_len(n, real=True)
        ws = octave_weights(nfft, SR)
        spec = sfft.rfft(np.random.default_rng(11).standard_normal(nfft))
        t = np.arange(n, dtype=np.float32) / SR
        onset = np.clip((t - T_MIX_S) / 0.005, 0, 1)
        onset = 0.5 - 0.5 * np.cos(np.pi * onset)
        bank = np.empty((len(OCT_CENTRES), n), np.float32)
        for b in range(len(OCT_CENTRES)):
            nz = sfft.irfft(spec * ws[b], nfft)[:n]
            bank[b] = nz / math.sqrt(float(np.mean(nz ** 2))) * onset
        _NOISE.update(bank=bank, t=t)
    return _NOISE["bank"], _NOISE["t"]


def synth_tail(levels_db, slopes_db_s, length_s, sr=SR):
    """Diffuse tail without the direct pulse: per octave band, band-limited noise with power per sample
    10^((level + slope * (t - T_REF_S)) / 10) from T_MIX_S on (raised-cosine onset over 5 ms). float32 (n,)."""
    bank, t = _noise_bank()
    n = min(int(length_s * sr), bank.shape[1])
    tt = t[:n]
    tail = np.zeros(n, np.float32)
    k = math.log(10) / 20
    for b, (lv, sl) in enumerate(zip(levels_db, slopes_db_s)):
        if lv is None or sl is None:
            continue
        tail += bank[b, :n] * np.exp(np.float32(k * (lv - sl * T_REF_S)) + np.float32(k * sl) * tt)
    return tail


def synth_ir(levels_db, slopes_db_s, length_s, sr=SR):
    """Unit direct pulse + diffuse tail (see synth_tail). `levels_db` / `slopes_db_s` are per OCT_CENTRES band (None = no tail)."""
    h = synth_tail(levels_db, slopes_db_s, length_s, sr)
    h[0] += 1.0
    return h


def conv(x, h):
    """Linear convolution of x with h, same length as x (tails past the end are cut). float32."""
    return fftconvolve(np.asarray(x, np.float32), np.asarray(h, np.float32))[:len(x)].astype(np.float32)


def tail_length(slopes):
    """Length (s) of the synthetic tail for the per-band slopes (None = no tail): 1.3 x the longest RT60, 0.25 .. TAIL_MAX_S."""
    return float(np.clip(1.3 * max([60.0 / -s for s in slopes if s is not None] + [0.0]), 0.25, TAIL_MAX_S))


class _Ref:
    """The reference voice excerpt as the dry side of the tail fit: its STFT at the envelope geometry, band powers, decay statistics.
    Built once per process (before the pool forks); the simulation of the tail is linear in the tail signal, which lets the
    level search reuse one convolution for all level candidates."""

    def __init__(self, seconds):
        self.sig, self.regions = core.reference().excerpt(seconds)
        self.sig = np.ascontiguousarray(self.sig, np.float32)
        self.n = len(self.sig)
        w, win = _env_consts()
        self.w, self.win = w, win
        view = _frames(self.sig)
        self.nfr = len(view)
        self.X = np.concatenate([sfft.rfft(view[s:s + 2048] * win, axis=1) for s in range(0, self.nfr, 2048)])
        self.A = (self.X.real * self.X.real + self.X.imag * self.X.imag) @ w  # (frames, bands) power of the dry reference
        db = 10 * np.log10(self.A + 1e-20)
        ev, _, _ = decay_curves(db, self.regions)
        self.stat, _ = decay_stat(ev)
        self.se = decay_stat_se(ev)
        self.ffts = {}

    def fft(self, size):
        if size not in self.ffts:
            self.ffts[size] = sfft.rfft(self.sig, size)
        return self.ffts[size]

    def tail_signal(self, tail):
        """conv(reference, tail) truncated to the excerpt: float32 (n,)."""
        size = sfft.next_fast_len(self.n + len(tail) - 1, real=True)
        return sfft.irfft(self.fft(size) * sfft.rfft(tail, size), size)[:self.n].astype(np.float32)


_REF = {}


def _ref():
    if "r" not in _REF:
        _REF["r"] = _Ref(REF_FIT_S)
    return _REF["r"]


def fit_tail(ev_stem_stat, ev_stem_se, cnt_stem, stem_floor_rel, sr=SR):
    """Analysis by synthesis: find per band (level at T_REF_S, slope) of a diffuse tail such that the decay statistic of the
    reference voice through the tail equals the stem's (region ends are refined on every simulated signal exactly as on the
    stem). ev_stem_stat (taus, bands) = stem curve quantile (NaN = not measured) with the stem's stationary ambient already
    removed (minus_ambient); stem_floor_rel (bands) = (ambient level - median pre-level) dB: curve points below ambient +
    FLOOR_MARGIN_DB are noise. Returns (levels, slopes, info) with None where the stem is not distinguishable from the dry voice."""
    ref = _ref()
    nb = len(OCT_CENTRES)
    taus = np.array(TAU_S)
    dry_stat = ref.stat
    dry_se = np.where(dry_stat < stem_floor_rel[None, :], 0.0, ref.se)  # below the stem's noise floor the dry curve is unobservable
    dry_stat = np.maximum(dry_stat, stem_floor_rel[None, :])
    valid = np.isfinite(ev_stem_stat) & np.isfinite(dry_stat) & (ev_stem_stat > (stem_floor_rel + FLOOR_MARGIN_DB)[None, :])
    levels = [None] * nb
    slopes = [None] * nb
    info = [{"band_hz": float(c), "tail": False, "points": int(valid[:, b].sum()), "events": int(cnt_stem[:, b].max())}
            for b, c in enumerate(OCT_CENTRES)]
    active = []
    for b in range(nb):
        v = valid[:, b]
        if v.sum() < 2:
            info[b]["note"] = "not enough measurable points"
            continue
        z = (ev_stem_stat[v, b] - dry_stat[v, b]) / np.sqrt(ev_stem_se[v, b] ** 2 + dry_se[v, b] ** 2 + 1e-6)
        if (z > Z_TAIL).sum() < 2:  # the stem does not decay slower than the dry voice by more than the statistics can resolve
            info[b]["note"] = "indistinguishable from the dry dub"
            continue
        active.append(b)
        v2 = v & (taus >= 0.05)
        sl = float(np.polyfit(taus[v2], ev_stem_stat[v2, b], 1)[0]) if v2.sum() >= 2 else -60.0 / 0.5
        slopes[b] = float(np.clip(sl, -60.0 / RT_MIN_S, -60.0 / RT_MAX_S))
        levels[b] = -60.0
    if not active:
        return levels, slopes, info
    noise = 2 * ev_stem_se ** 2  # expected squared deviation of a curve point from the truth

    def simulate(lv, sl, grid):
        """Decay statistics of the reference through the tail (lv + g, sl) for every level shift g of the grid. The simulated
        signal is y = dry + k T with T = conv(dry, tail(lv, sl)) and k = 10^(g/20): its band power is A + 2k C + k^2 B (A, B, C
        = band sums of |X_dry|^2, |X_T|^2, Re(X_dry conj X_T)), so one convolution serves all shifts."""
        tail = synth_tail(lv, sl, tail_length(sl), sr)
        T = ref.tail_signal(tail)
        XT = np.concatenate([sfft.rfft(_frames(T)[s:s + 2048] * ref.win, axis=1) for s in range(0, ref.nfr, 2048)])
        B = (XT.real * XT.real + XT.imag * XT.imag) @ ref.w
        C = (ref.X.real * XT.real + ref.X.imag * XT.imag) @ ref.w
        out = []
        for g in grid:
            k = 10 ** (g / 20)
            ev, _, _ = decay_curves(10 * np.log10(ref.A + 2 * k * C + k * k * B + 1e-20), ref.regions, sr)
            out.append(decay_stat(ev)[0])
        return out

    def level_search(grid):
        """Per band the LOWEST level whose curve error is within the statistical noise of the best one: where the tail is
        louder than the speech the offset curves saturate and cannot tell how much louder - the smallest consistent tail is taken."""
        costs = np.full((len(grid), nb), np.inf)
        for gi, sim in enumerate(simulate(levels, slopes, grid)):
            for b in active:
                v = valid[:, b] & np.isfinite(sim[:, b])
                if v.any():
                    costs[gi, b] = float(((ev_stem_stat[v, b] - sim[v, b]) ** 2).sum())
        for b in active:
            v = valid[:, b]
            tol = float(noise[v, b].sum()) if v.any() else 0.0
            ok = np.where(costs[:, b] <= costs[:, b].min() + tol)[0]
            levels[b] = levels[b] + float(grid[ok[0]])  # grid ascending: lowest level

    level_search(np.arange(-20, 41, 4.0))  # absolute -80 .. -20 dB per sample
    for step in (2.0, 1.0, 1.0):
        sim = simulate(levels, slopes, [0.0])[0]
        for b in active:
            v = valid[:, b] & np.isfinite(sim[:, b]) & (taus >= 0.05)
            if v.sum() >= 3:
                d = ev_stem_stat[v, b] - sim[v, b]
                A = np.vstack([np.ones(v.sum()), taus[v] - T_REF_S]).T
                slopes[b] = float(np.clip(slopes[b] + 0.7 * np.clip(np.linalg.lstsq(A, d, rcond=None)[0][1], -30, 30), -60.0 / RT_MIN_S, -60.0 / RT_MAX_S))
        level_search(np.arange(-3 * step, 3 * step + 0.1, step))
    length = tail_length(slopes)
    sim = simulate(levels, slopes, [0.0])[0]
    for b in active:
        v = valid[:, b] & np.isfinite(sim[:, b])
        info[b].update({"tail": True, "rt60_s": round(60.0 / -slopes[b], 3), "level_db": round(levels[b], 1),
                        "fit_rms_db": round(float(np.sqrt(np.mean((ev_stem_stat[v, b] - sim[v, b]) ** 2))), 2) if v.any() else None,
                        "dry_rms_db": round(float(np.sqrt(np.mean((ev_stem_stat[v, b] - dry_stat[v, b]) ** 2))), 2) if v.any() else None})
    info.append({"ir_length_s": round(length, 2)})
    return levels, slopes, info


# ----------------------------------------------------------------------------- tone (long-term spectrum ratio)

def ltas(x, regions, sr=SR, max_frames=4000):
    """Mean power spectrum of the frames of x inside the speech regions (hann 4096, half overlap): (spectrum, frame count). When
    the regions hold more than max_frames frames, every k-th frame is used (the average has converged long before)."""
    win = np.hanning(LTAS_N).astype(np.float32)
    acc = np.zeros(LTAS_N // 2 + 1)
    total = sum(max((min(int(b * sr), len(x)) - int(a * sr) - LTAS_N) // (LTAS_N // 2) + 1, 0) for a, b in regions)
    k = max(1, -(-total // max_frames))
    cnt = 0
    for a, b in regions:
        i0, i1 = int(a * sr), min(int(b * sr), len(x))
        if i1 - i0 < LTAS_N:
            continue
        fr = np.lib.stride_tricks.sliding_window_view(np.asarray(x[i0:i1], np.float32), LTAS_N)[:: (LTAS_N // 2) * k]
        for s0 in range(0, len(fr), 1024):
            X = sfft.rfft(fr[s0:s0 + 1024] * win, axis=1)
            acc += (X.real * X.real + X.imag * X.imag).sum(0, dtype=np.float64)
        cnt += len(fr)
    return acc / max(cnt, 1), cnt


def noise_psd(x, sr=SR):
    """Stationary-noise power spectrum of x (same bins as ltas): per bin the 10th percentile over all frames (speech pauses give
    the minimum), divided by 0.105 = the 10th percentile of an exponentially distributed periodogram -> the mean noise power."""
    win = np.hanning(LTAS_N).astype(np.float32)
    fr = np.lib.stride_tricks.sliding_window_view(np.asarray(x, np.float32), LTAS_N)[:: LTAS_N // 2]
    if len(fr) > 4000:
        fr = fr[:: len(fr) // 4000]
    X = sfft.rfft(fr * win, axis=1)
    return np.percentile(X.real * X.real + X.imag * X.imag, 10, axis=0) / 0.105


def smooth_db(p, sr=SR, oct_=EQ_SMOOTH_OCT):
    """Power spectrum (rfft bins of LTAS_N) -> dB, averaged over a +-oct_/2 octave box around each bin."""
    f = np.fft.rfftfreq(LTAS_N, 1 / sr)
    cum = np.concatenate([[0.0], np.cumsum(p)])
    lo = np.searchsorted(f, f * 2 ** (-oct_ / 2), side="left")
    hi = np.searchsorted(f, f * 2 ** (oct_ / 2), side="right")
    hi = np.maximum(hi, np.arange(len(f)) + 1)
    lo = np.minimum(lo, np.arange(len(f)))
    return 10 * np.log10((cum[hi] - cum[lo]) / (hi - lo) + 1e-20)


def eq_filter(p_stem, p_dub, sr=SR, taps=EQ_TAPS, p_noise=None):
    """Minimum-phase FIR whose magnitude = smoothed long-term spectrum of the stem over that of the dub (dB, no limit), overall
    gain normalised to 0 dB over 300-3000 Hz. Returns (fir, gain_db at LTAS bins, freq)."""
    f = np.fft.rfftfreq(LTAS_N, 1 / sr)
    if p_noise is not None:  # the stem's additive noise is not part of the voice's tone: remove it. Both spectra are floored at 10 dB
        lim = 0.1 * p_noise  # under that noise (detection limit): where neither has signal the filter is flat, where only the stem
        p_stem = np.maximum(p_stem - p_noise, lim)  # lacks it (band limit) it cuts, where only the dub lacks it it boosts to the limit
        p_dub = np.maximum(p_dub, lim)
    g = smooth_db(p_stem, sr) - smooth_db(p_dub, sr)
    ref = (f >= 300) & (f <= 3000)
    g = g - g[ref].mean()
    g[0] = g[1]
    nfft = pow2(taps * 4)
    fo = np.fft.rfftfreq(nfft, 1 / sr)
    mag = np.interp(fo, f, g)  # dB
    logm = mag / 20 * math.log(10)
    c = sfft.irfft(logm, nfft)
    w = np.zeros(nfft)
    w[0] = 1
    w[1: nfft // 2] = 2
    w[nfft // 2] = 1
    hmin = sfft.irfft(np.exp(sfft.rfft(c * w)), nfft)[:taps]
    fade = np.ones(taps)
    k = taps // 4
    fade[-k:] = 0.5 + 0.5 * np.cos(np.pi * np.arange(k) / k)
    return hmin * fade, g, f


def eq_edges(g_db, p_dub, sr=SR, span_db=45.0):
    """Band edges of the tone filter, speech-weighted: from the 1 kHz centre outwards, the first frequency (inside the band where
    the dub has speech energy: within span_db of its peak) at which the gain is edge_db / 3 dB / 20 dB below the passband (0 dB =
    median gain 300-3000 Hz). None = the filter does not cut within the dub's own band on that side."""
    f = np.fft.rfftfreq(LTAS_N, 1 / sr)
    sm = smooth_db(p_dub, sr)
    live = (sm >= sm[(f > 100) & (f < 8000)].max() - span_db) & (f >= 40)
    out = {}
    c = int(np.searchsorted(f, 1000))
    for tag, db in (("", EDGE_DB), ("_3db", 3.0), ("_20db", 20.0)):
        lo = hi = None
        for i in range(c, 0, -1):
            if live[i] and g_db[i] < -db:
                lo = float(round(f[i]))
                break
        for i in range(c, len(f)):
            if live[i] and g_db[i] < -db:
                hi = float(round(f[i]))
                break
        out["f_lo" + tag + "_hz"] = lo
        out["f_hi" + tag + "_hz"] = hi
    return out


# ----------------------------------------------------------------------------- IR metrics

def edc_db(h):
    """Schroeder backward-integrated energy decay curve of an (already noise-free) IR, dB re total, from the main peak."""
    i0 = int(np.argmax(np.abs(h)))
    e = (h[i0:].astype(np.float64)) ** 2
    c = np.cumsum(e[::-1])[::-1]
    return 10 * np.log10(c / c[0] + 1e-30), i0


def decay_times(h, sr=SR):
    """EDT (0..-10 dB x6), T20 (-5..-25 x3), T30 (-5..-35 x2) from the EDC, and RT60 = the longest-range one the curve reaches.
    Missing (None) when the curve does not reach that level."""
    d, _ = edc_db(h)
    t = np.arange(len(d)) / sr

    def slope(lo, hi):
        if d.min() > hi:
            return None
        i_lo = int(np.argmax(d <= lo))
        i_hi = int(np.argmax(d <= hi))
        if i_hi - i_lo < 8:
            return None
        a, b = np.polyfit(t[i_lo:i_hi], d[i_lo:i_hi], 1)
        return float(-60.0 / a) if a < 0 else None

    edt = slope(0, -10)
    t20 = slope(-5, -25)
    t30 = slope(-5, -35)
    rt = t30 if t30 is not None else (t20 if t20 is not None else edt)
    src = "T30" if t30 is not None else ("T20" if t20 is not None else "EDT")
    return {"edt_s": edt, "t20_s": t20, "t30_s": t30, "rt60_s": rt, "rt60_from": src}


def drr_db(h, sr=SR):
    """Direct-to-reverberant ratio (dB): energy within +-DIRECT_S of the main peak over the rest."""
    i0 = int(np.argmax(np.abs(h)))
    w = int(DIRECT_S * sr)
    e = h.astype(np.float64) ** 2
    d = e[max(i0 - w, 0): i0 + w + 1].sum()
    r = e.sum() - d
    return float(10 * math.log10(d / r)) if r > 0 else float("inf")


def band_edges(h, sr=SR, edge_db=EDGE_DB):
    """Passband of the channel from its smoothed magnitude response (1/3-octave power smoothing): level of the passband =
    95th percentile of the smoothed response; f_lo / f_hi = outermost frequencies within edge_db of it (-3 dB, -20 dB too).
    The smoothing is done on log-frequency bins of 1/48 octave (power summed per bin), identical to the lab's per-bin Gaussian
    to within the bin width. Returns (dict, centres Hz, smoothed response dB)."""
    nfft = pow2(max(len(h), 1 << 15))
    H = np.abs(sfft.rfft(np.asarray(h, np.float32), nfft)) ** 2
    f = np.fft.rfftfreq(nfft, 1 / sr)[1:]
    H = H[1:].astype(np.float64)
    centres = 40 * 2 ** (np.arange(0, 9.2, 1 / 12))
    centres = centres[centres < sr / 2 * 0.98]
    lb = np.log2(np.maximum(f, 1.0))
    q0 = lb.min()
    nbin = int((lb.max() - q0) * 48) + 1
    ib = ((lb - q0) * 48).astype(int)
    S = np.bincount(ib, H, nbin)
    N = np.bincount(ib, None, nbin).astype(np.float64)
    lc = q0 + (np.arange(nbin) + 0.5) / 48
    w = np.exp(-0.5 * ((lc[None, :] - np.log2(centres)[:, None]) / EDGE_SMOOTH_OCT) ** 2)
    sm = 10 * np.log10((w * S[None]).sum(1) / (w * N[None]).sum(1) + 1e-30)
    ref = float(np.percentile(sm, 95))
    out = {"passband_db": round(ref, 1)}
    for tag, db in (("", edge_db), ("_3db", 3.0), ("_20db", 20.0)):
        ok = np.where(sm >= ref - db)[0]
        out["f_lo" + tag + "_hz"] = float(round(centres[ok[0]])) if len(ok) else None
        out["f_hi" + tag + "_hz"] = float(round(centres[ok[-1]])) if len(ok) else None
    return out, centres, sm


def ir_report(h, sr=SR):
    """RT60 family, DRR, passband edges and peak lag of an IR."""
    r = decay_times(h, sr)
    r["drr_db"] = round(drr_db(h, sr), 1)
    r.update(band_edges(h, sr)[0])
    i0 = int(np.argmax(np.abs(h)))
    r["peak_lag_ms"] = round(i0 / sr * 1000, 2)
    for k in ("edt_s", "t20_s", "t30_s", "rt60_s"):
        if r[k] is not None:
            r[k] = round(r[k], 3)
    return r


def band_rt60(h, sr=SR):
    """RT60 (T20 extrapolated, else EDT) of the IR per octave band, from the Schroeder curve of the band-filtered IR."""
    nfft = sfft.next_fast_len(len(h), real=True)
    H = sfft.rfft(np.asarray(h, np.float32), nfft)
    ws = octave_weights(nfft, sr).astype(np.float32)
    out = []
    for b in range(len(OCT_CENTRES)):
        hb = sfft.irfft(H * ws[b], nfft)[: len(h)]
        d = decay_times(hb, sr)
        out.append(d["t20_s"] if d["t20_s"] is not None else d["edt_s"])
    return [None if v is None else round(v, 3) for v in out]


# ----------------------------------------------------------------------------- dry reference of the stem (dynamics fit)

def dereverb(x):
    """Blind dereverberation of mono x (48 kHz) by WPE (nara_wpe, single channel) on a 16 kHz copy: the late reverberation that is
    linearly predictable from the signal's own past WPE_TAIL_S seconds is removed; the resulting |Z| / |Y| gain mask (0..1,
    bins 0-8 kHz, the 8 kHz value above) is applied to the full-band STFT (1536 / 768 at 48 kHz = the same 31.25 Hz bins and 16 ms
    hop). Returns float32 of the same length."""
    n = len(x)
    x16 = sps.resample_poly(x, 1, SR // WPE_RATE)
    _, _, Y = sps.stft(x16.astype(np.float64), WPE_RATE, window="hann", nperseg=WPE_NFFT, noverlap=WPE_NFFT - WPE_HOP)
    taps = int(math.ceil(WPE_TAIL_S * WPE_RATE / WPE_HOP))
    Z = wpe_v8(Y[:, None, :], taps=taps, delay=WPE_DELAY, iterations=WPE_ITERS, statistics_mode="full")[:, 0, :]
    M = np.minimum(np.abs(Z) / (np.abs(Y) + 1e-12), 1.0)
    k = SR // WPE_RATE
    _, _, X = sps.stft(x.astype(np.float64), SR, window="hann", nperseg=WPE_NFFT * k, noverlap=(WPE_NFFT - WPE_HOP) * k)
    nf = min(X.shape[1], M.shape[1])
    M = np.concatenate([M, np.repeat(M[-1:], X.shape[0] - M.shape[0], axis=0)], axis=0)
    _, d = sps.istft(X[:, :nf] * M[:, :nf], SR, window="hann", nperseg=WPE_NFFT * k, noverlap=(WPE_NFFT - WPE_HOP) * k)
    return np.pad(d[:n], (0, max(n - len(d), 0))).astype(np.float32)


def dynamics_chunks(sreg, dur, budget_s):
    """Stretches [(a, b)] s of a scene (regions sreg, duration dur) to dereverberate and fit. Bodies of about DYN_BODY_S start at
    regions spread evenly over the scene's speech (enough of them to hold about budget_s of speech); regions starting within
    DYN_LEAD_S before a body are included as context. Every edge lies in the middle of a pause between regions, so a stretch
    holds whole regions only. Scenes shorter than one body are taken whole."""
    n = len(sreg)
    if dur <= DYN_BODY_S + DYN_LEAD_S:
        return [(0.0, dur)]
    cut = [0.0] + [(sreg[i][1] + sreg[i + 1][0]) / 2 for i in range(n - 1)] + [dur]  # cut[i] / cut[i + 1]: edges around region i
    starts = np.array([a for a, _ in sreg])
    speech = sum(b - a for a, b in sreg)
    m = int(np.clip(math.ceil(budget_s / (0.75 * DYN_BODY_S)), 1, n))
    cum = np.cumsum([b - a for a, b in sreg])
    spans = []
    for j in range(m):
        i = min(int(np.searchsorted(cum, speech * (j + 0.5) / m)), n - 1)
        first = int(np.searchsorted(starts, starts[i] - DYN_LEAD_S))  # context regions
        last = int(np.searchsorted(starts, starts[i] + DYN_BODY_S, side="right")) - 1  # last region starting inside the body
        spans.append((first, max(last, i)))
    merged = []
    for f, l in sorted(spans):
        if merged and f <= merged[-1][1] + 1:
            merged[-1] = (merged[-1][0], max(l, merged[-1][1]))
        else:
            merged.append((f, l))
    return [(cut[f], cut[l + 1]) for f, l in merged]


def dynamics_inputs(sreg, dur, pieces):
    """Spans (s, on the concatenated time line) of the regions inside the stretches `pieces` [(a, b)] of a scene."""
    spans, off = [], 0.0
    for a, b in pieces:
        spans += [(ra - a + off, rb - a + off) for ra, rb in sreg if ra >= a and rb <= b]
        off += int(b * SR) / SR - int(a * SR) / SR
    return spans


def fit_chunk_dynamics(stem_chunks, derev_chunks, spans, fir):
    """Dynamics from the stem chunks (s) and their dereverberated copies (y = fir * dereverberated stem), concatenated, with the
    regions `spans` of the concatenation."""
    ys = [fftconvolve(d, fir.astype(np.float32))[:len(d)].astype(np.float32) for d in derev_chunks]
    ref_sig, ref_regs = core.reference().excerpt(20.0)
    x_ref = (fftconvolve(ref_sig, fir.astype(np.float32))[:len(ref_sig)].astype(np.float32), ref_regs)
    return dynamics.fit_dynamics(np.concatenate(ys), np.concatenate([np.asarray(c, np.float32) for c in stem_chunks]), spans, SR, x_ref=x_ref)


# ----------------------------------------------------------------------------- worker tasks (run in the pool processes)
# The pool is persistent and started with forkserver (the analysis runs other stages in threads of the parent: no fork of a
# multithreaded process). The stem travels as shared memory; every task reads its slice of it by sample range.

_ATTACHED = {}


def _attach(name, n):
    """The shared mono stem (float32, n samples) of the current estimate, mapped once per worker."""
    if _ATTACHED.get("name") != name:
        if "shm" in _ATTACHED:
            _ATTACHED["arr"] = None
            _ATTACHED["shm"].close()
        shm = shared_memory.SharedMemory(name=name)
        _ATTACHED.update(name=name, shm=shm, arr=np.ndarray((n,), np.float32, buffer=shm.buf))
    return _ATTACHED["arr"]


def _t_env(name, n, i0, i1, fw):
    with sfft.set_workers(fw):
        return band_env_db(_attach(name, n)[i0:i1])


def _t_spectra(name, n, i0, i1, sreg):
    """Stem long-term spectrum inside the regions and its stationary noise spectrum (independent of the tail)."""
    seg = _attach(name, n)[i0:i1]
    p_stem, n_s = ltas(seg, sreg)
    return p_stem, n_s, noise_psd(seg)


def _t_dereverb(name, n, i0, i1):
    return dereverb(_attach(name, n)[i0:i1])


def _t_evidence(db, sreg):
    """Decay statistics of a scene from its band envelope."""
    ev, pre, floor = decay_curves(db, sreg)
    stat, cnt = decay_stat(ev)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # a band that is never above its floor has no pre-level at all
        floor_rel = floor - np.nanmedian(pre, axis=0)
    return {"stat": stat, "cnt": cnt, "ev": ev, "floor_rel": floor_rel}


def _t_fit(evid, fw):
    """Tail of a scene (levels, slopes, info) from its decay statistics."""
    with sfft.set_workers(fw), threadpool_limits(fw):
        se = decay_stat_se(evid["ev"])
        return fit_tail(minus_ambient(evid["stat"], evid["floor_rel"]), se, evid["cnt"], evid["floor_rel"])


def _t_tone(tail, p_stem, p_noise, borrowed, fw):
    """Tone filter, IR and reports of a scene for a tail: dict(h, fir, eq_db, p_dub, model_drr_db, report)."""
    levels, slopes, _ = tail
    ref = _ref()
    with sfft.set_workers(fw):
        tl = synth_tail(levels, slopes, tail_length(slopes))
        p_dub, _ = ltas(ref.tail_signal(tl) + ref.sig, ref.regions)  # the tone filter only has to close what the tail leaves open
        fir, g_db, _ = eq_filter(p_stem, p_dub, SR, p_noise=p_noise)
        ir = tl.copy()
        ir[0] += 1.0
        h = fftconvolve(fir.astype(np.float32), ir).astype(np.float32)
    energy = float(np.sum(tl.astype(np.float64) ** 2))
    report = {**ir_report(h), "band_rt60_s": band_rt60(h)}
    if not borrowed:
        report["eq_edges"] = eq_edges(g_db, p_dub)
    return {"h": h, "fir": fir.astype(np.float32), "eq_db": g_db, "model_drr_db": None if energy <= 0 else round(-10 * math.log10(energy), 1),
            "report": report}


def _t_dynamics(name, n, ranges, derev, spans, fir):
    """Dynamics fit on the stem chunks at the absolute sample `ranges`."""
    stem = _attach(name, n)
    return fit_chunk_dynamics([stem[i0:i1] for i0, i1 in ranges], derev, spans, fir)


def _worker_init():
    """Pool workers: one BLAS thread each, constants built, numba kernel loaded."""
    threadpool_limits(1)
    _noise_bank()
    _ref()
    dynamics.warmup()


def _ping():
    return True


# ----------------------------------------------------------------------------- the pool and the estimate

_POOL = {}


def _pool(workers):
    """The persistent worker pool (created on first use, recreated when `workers` changes)."""
    if _POOL.get("workers") != workers or _POOL["pool"]._broken:
        if "pool" in _POOL:
            _POOL["pool"].shutdown()
        ctx = mp.get_context("forkserver")
        ctx.set_forkserver_preload([__name__])
        _POOL.update(workers=workers, pool=ProcessPoolExecutor(workers, mp_context=ctx, initializer=_worker_init))
        wait([_POOL["pool"].submit(_ping) for _ in range(workers)])  # spawn all workers now
    return _POOL["pool"]


def prepare(workers=8):
    """Start the worker pool and build the per-process constants (tail noise bank, reference voice analysis, numba kernel). Call
    once at service start (about 1 s, parallel); otherwise the first estimate pays for it."""
    _pool(workers)


def _eq_samples(g_db):
    f = np.fft.rfftfreq(LTAS_N, 1 / SR)
    return [round(float(v), 1) for v in g_db[np.searchsorted(f, [100, 200, 400, 800, 1600, 3200, 6400, 12800, 16000])]]


def _env_slices(i0, i1, workers):
    """Sample ranges [(a, b)] whose band envelopes concatenate to the envelope of [i0, i1) (frames split, window overlap kept)."""
    n = i1 - i0
    if n < 2 * ENV_WIN:
        return [(i0, i1)]
    nfr = (n - ENV_WIN) // ENV_HOP + 1
    k = int(np.clip(round(n / SR / 60.0), 1, workers))
    f = np.linspace(0, nfr, k + 1).round().astype(int)
    return [(i0 + a * ENV_HOP, i0 + (b - 1) * ENV_HOP + ENV_WIN) for a, b in zip(f[:-1], f[1:])]


def estimate(clip, scene_doc, regions, workers=8):
    """Channel of every scene, see the module docstring. The work of every scene is split over the pool as a task graph:
    envelope slices -> decay statistics -> tail fit -> tone / IR -> dynamics fit, with the stem spectra and the dereverberated
    chunks computed beside the tail fit."""
    sc_list = scene_doc["scenes"]
    n_sc = len(sc_list)
    pool = _pool(workers)
    stem = clip.speech
    shm = shared_memory.SharedMemory(create=True, size=stem.nbytes)
    try:
        np.ndarray(stem.shape, np.float32, buffer=shm.buf)[:] = stem
        S = _run_graph(pool, shm.name, len(stem), sc_list, [(float(a), float(b)) for a, b in regions], workers)
    finally:
        shm.close()
        shm.unlink()
    out_scenes = []
    for i, sc in enumerate(sc_list):
        t0, t1 = sc["start"], sc["end"]
        rec = {"id": sc["id"], "start": round(t0, 2), "end": round(t1, 2), "speech_s": round(sc["speech_s"], 1), "n_atoms": sc.get("n_atoms"),
               "signature_noise": sc.get("signature_noise"), "borrowed_from": None, "dynamics": None}
        s = S.get(i)
        if s is None:
            rec["note"] = "pending"
            rec["_h"], rec["_fir"] = np.ones(1, np.float32), None
        else:
            if s["donor"] is not None:
                rec["borrowed_from"] = sc_list[s["donor"]]["id"]
            elif not s["good"]:
                rec["borrowed_from"] = "none available: tail kept as measured"
            tone = s["tone"]
            rec.update({"tail": s["tail"][2], "report": tone["report"], "model_drr_db": tone["model_drr_db"], "n_events": len(s["sreg"]),
                        "eq_db_1_6_oct": _eq_samples(tone["eq_db"]), "levels": s["tail"][0], "slopes": s["tail"][1], "dynamics": s["dyn"]})
            rec["_h"], rec["_fir"] = tone["h"], tone["fir"]
        rec["ir_len_s"] = round(len(rec["_h"]) / SR, 2)
        out_scenes.append(rec)
    own_ids = [i for i in range(n_sc) if i in S]
    for i, rec in enumerate(out_scenes):
        if rec.get("note") != "pending":
            continue
        sig = sc_list[i]["signature"]
        cand = [(scenes_mod.signature_distance(sig, sc_list[j]["signature"]), j) for j in own_ids if sig is not None and sc_list[j]["signature"] is not None]
        if not cand:  # no signature: the temporally nearest scene with an estimate
            cand = [(abs(0.5 * (sc_list[i]["start"] + sc_list[i]["end"]) - 0.5 * (sc_list[j]["start"] + sc_list[j]["end"])), j) for j in own_ids]
        if not cand:
            rec["note"] = "no estimate anywhere: neutral (identity) channel"
            continue
        donor = out_scenes[min(cand)[1]]
        rec.update({k: donor.get(k) for k in ("tail", "report", "model_drr_db", "eq_db_1_6_oct", "dynamics", "levels", "slopes")})
        rec["borrowed_from"] = donor["id"]
        rec["note"] = "too little speech of its own: whole channel (tail, tone, dynamics) of the most similar scene"
        rec["_h"], rec["_fir"] = donor["_h"], donor["_fir"]
        rec["ir_len_s"] = donor["ir_len_s"]
    return {"key": clip.dir.name, "sr": SR, "duration_s": round(clip.duration_s, 2), "scenes": out_scenes,
            "scene_detection": {k: scene_doc.get(k) for k in ("candidate_boundaries", "noise_db", "params")}}


def _run_graph(pool, name, n, sc_list, regions, workers):
    """Run the per-scene task graph on the pool; returns {scene index: state} for the scenes with >= 2 regions (keys: sreg, good,
    donor, tail, tone, dyn)."""
    S = {}
    for i, sc in enumerate(sc_list):
        t0, t1 = sc["start"], sc["end"]
        sreg = [(a - t0, b - t0) for a, b in window_regions(regions, t0, t1)]
        if sc["speech_s"] <= 0 or len(sreg) < 2:
            continue
        i0, i1 = int(t0 * SR), int(t1 * SR)
        slices = _env_slices(i0, i1, workers)
        pieces = dynamics_chunks(sreg, (i1 - i0) / SR, dynamics.FIT_BUDGET_S)
        ranges = [(i0 + int(a * SR), i0 + int(b * SR)) for a, b in pieces]
        S[i] = {"sreg": sreg, "i0": i0, "i1": i1, "slices": slices, "db": [None] * len(slices), "ranges": ranges,
                "spans": dynamics_inputs(sreg, (i1 - i0) / SR, pieces), "derev": [None] * len(ranges), "done": set()}
    heavy = sum(1 for s in S.values() if s["i1"] - s["i0"] >= 60 * SR)  # scenes long enough to need the cores
    fw = max(1, workers // max(heavy, 1))  # FFT / BLAS threads of the tail fit, the longest serial step
    pending = {}

    def submit(kind, i, k, fn, *args):
        pending[pool.submit(fn, *args)] = (kind, i, k)

    for i, s in S.items():
        for k, (a, b) in enumerate(s["slices"]):
            submit("env", i, k, _t_env, name, n, a, b, fw)
        submit("spec", i, 0, _t_spectra, name, n, s["i0"], s["i1"], s["sreg"])
        for k, (a, b) in enumerate(s["ranges"]):
            submit("derev", i, k, _t_dereverb, name, n, a, b)
    decided = False

    def once(i, tag):
        """True the first time (scene i, tag) is asked."""
        if tag in S[i]["done"]:
            return False
        S[i]["done"].add(tag)
        return True

    def advance():
        nonlocal decided
        for i, s in S.items():
            if all(d is not None for d in s["db"]) and once(i, "evidence"):
                submit("evid", i, 0, _t_evidence, np.concatenate(s["db"]), s["sreg"])
        if not decided and all("evid" in s for s in S.values()):
            decided = True
            good = [i for i, s in S.items() if enough_events(s["evid"]["cnt"])]
            for i, s in S.items():
                s["good"] = i in good
                sig = sc_list[i]["signature"]
                cand = [(scenes_mod.signature_distance(sig, sc_list[j]["signature"]), j) for j in good
                        if not s["good"] and sig is not None and sc_list[j]["signature"] is not None]
                s["donor"] = min(cand)[1] if cand else None
                if s["good"] or s["donor"] is None:
                    once(i, "fit")
                    submit("fit", i, 0, _t_fit, s["evid"], fw)
        for i, s in S.items():
            if decided and "tail" not in s and s["donor"] is not None and "tail" in S[s["donor"]]:
                s["tail"] = S[s["donor"]]["tail"]
            if "tail" in s and "spec" in s and once(i, "tone"):
                submit("tone", i, 0, _t_tone, s["tail"], s["spec"][0], s["spec"][2], s["donor"] is not None, fw)
            if "tone" in s and all(d is not None for d in s["derev"]) and once(i, "dyn"):
                submit("dyn", i, 0, _t_dynamics, name, n, s["ranges"], s["derev"], s["spans"], s["tone"]["fir"])

    while pending:
        done, _ = wait(pending, return_when=FIRST_COMPLETED)
        for f in done:
            kind, i, k = pending.pop(f)
            r = f.result()
            s = S[i]
            if kind == "env":
                s["db"][k] = r
            elif kind == "derev":
                s["derev"][k] = r
            elif kind == "spec":
                s["spec"] = r
            elif kind == "evid":
                s["evid"] = r
            elif kind == "fit":
                s["tail"] = r
            elif kind == "tone":
                s["tone"] = r
            elif kind == "dyn":
                s["dyn"] = r
        advance()
    return S
