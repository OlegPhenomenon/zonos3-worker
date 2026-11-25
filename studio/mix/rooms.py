"""The room of the original voice as TWO simultaneous reverbs (early reflections + late tail), cross-checked on the M&E.

Port of the lab's `room_layers.py` (Netflix Mixing Style Guide for Dubbed Content: "more than one reverb simultaneously: early
reflections + late tail (decorrelated), fitted per scene; use clues from the M&E to cross-check the room estimate"). Method, per
scene of the channel document:

1. Voice statistics: free-decay curves after the phrase ends (5 ms octave-band envelopes, region ends refined to an "anchor", low
   quantile over events, stationary ambient removed), plus the anchor-free `nominal` version (time origin = the slot end).
2. Two-component model per octave band fitted by analysis by synthesis against the dry reference voice `core.reference()`
   (the product has no dub when the clip is analysed): simulate = reference voice * (delta + early + late), measure the same
   statistics, move the parameters until they equal the stem's. The reference is used as a pack of REF_EVENTS isolated events
   (the last REF_LEAD_S of a reference region + REF_PAUSE_S of digital silence, each convolved on its own): the reference's own pauses
   are too short for the long-tau points, and the pack keeps one simulation at ~25 ms. late: diffuse noise (level at T_REF_L_S, slope); early: sparse
   taps 5..100 ms = the late exponential continued back (prior) whose level is moved only when the data object strongly.
3. M&E cross-check: decay events of the background stem (broadband level falls >= 12 dB within 50 ms after a 150 ms plateau); the
   low quartile of the per-band RT60 over events is the room estimate of the M&E.
4. Decision per scene (voice / M&E agree within 2x -> voice, disagree -> blend, voice implausible -> me, nothing -> none = dry),
   stereo decorrelation of the late tail (L/R coherence of the mix's room tone, capped at 0.4), tone FIR closing what the room
   leaves open (stem LTAS over reference-voice LTAS through the room).

Units: seconds, Hz, dB; sample rate core.SR (48 kHz). Arrays float32 unless noted; IR layers are (2, m) channel-first.

API
    analyze(clip, chan, regions, workers=8, tone=None) -> dict
        {"sr", "floor_coherence" (per octave band or None), "scenes": [{"id","start","end", "early_params","late_params",
          "voice_rt60_s","voice_plausible","voice_reasons","voice_events","me_rt60_s","me_events","me_detail",
          "me_rt60_by_band_s","agreement","agreement_ratio","source","borrowed_from","rho_late","note","n_sims",
          "early_late_energy_db","corr", "_fir" (taps,), "_early" (2, m), "_late" (2, m)}]}
        The lab's rooms.json schema; the wav files of the lab are the arrays `_fir`, `_early`, `_late` (WITHOUT the direct pulse;
        application per channel c: y_c = conv(x, fir * (delta + early[c] + late[c]))). A "none" scene has early / late = (2, 1)
        zeros. `chan` = the channel document (scenes with id, start, end, borrowed_from), `regions` = core.speech_regions(speech)
        in seconds, `tone` = the room_tone.analyze() document (its L/R coherence `rho` per band sets the decorrelation of the
        late tails; None = RHO_DEFAULT).
    floor_rho(tone) -> list   room_tone document -> coherence per octave band
"""
import math
import multiprocessing as mp
import warnings
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

import numpy as np
from scipy import fft as sfft
from threadpoolctl import threadpool_limits

from . import channel as cm
from . import core

SR = core.SR
OCT = cm.OCT_CENTRES
NB = len(OCT)
TAU = np.array(cm.TAU_S)

# ----------------------------------------------------------------------------- model constants (seconds / dB: carried over from the lab)
T_E0_S, T_E1_S = 0.005, 0.100  # early window: the direct sound's window ends at 5 ms; the late layer starts after the early window
T_XFADE_S = (0.080, 0.100)  # early fades out / late fades in, power-complementary (cos / sin), over the last 20 ms of the window
ECHO_DENS = 1000.0  # echoes per second at which reflections fuse into diffuse reverb (Schroeder 1962) ...
T_ECHO_MIX_S = 0.030  # ... reached here: density = ECHO_DENS * (t / T_ECHO_MIX_S)^2
ECHO_MIN_DENS = 150.0  # ... but never sparser than this
T_REF_E_S = 0.020  # early level is quoted at this time
T_REF_L_S = 0.100  # late level is quoted at this time (the end of the early window)
RT_MIN_S = cm.RT_MIN_S
RT_MAX_E_S = 1.0  # an early layer is only 95 ms long: slower than this it is flat
RT_MAX_L_S = 6.0  # late RT60 limit = what the 2.4 s statistic horizon can carry and the bank length
LATE_BANK_S = 8.0  # length of the stored band noises (>= 1.3 * RT_MAX_L_S)
EQ_TAPS = cm.EQ_TAPS
SEED = 17
RHO_DEFAULT = 0.3  # inter-channel correlation of the late tail when room_tone gives none
RHO_LATE_MAX = 0.4  # diffuse field: decorrelated
LATE_TAUS = TAU >= T_REF_L_S - 1e-9  # taus carried by the late layer
EARLY_TAUS = TAU <= T_E1_S + 1e-9  # taus carried by the early layer
ANCHOR_SEARCH_S = 0.25  # the nominal end of a region is refined within +-this (channel_match.decay_curves)
SIM_FLOOR_DB = -100.0  # "no noise": the floor of the simulations (the stem's ambient is handled on the stem's side)

# ----------------------------------------------------------------------------- plausibility / decision constants
IMPL_RT_S, IMPL_LEVEL_DB = 3.0, -60.0  # RT60 > 3 s with level below -60 dB is the level-search floor, not a room
IMPL_FIT_RMS_DB = 10.0  # a fit that misses the stem's curve by more than this describes nothing
AGREE_RATIO = 2.0  # voice vs M&E RT60 disagree beyond this factor
MIN_EVENTS = cm.MIN_EVENTS
RT_BANDS = (1, 6)  # octave bands 250 Hz .. 4 kHz carry the voice

# ----------------------------------------------------------------------------- M&E event constants
ME_FALL_DB = 12.0
ME_FALL_S = 0.050
ME_PLATEAU_S = 0.150
ME_ABOVE_FLOOR_DB = 15.0
ME_PLATEAU_FLAT_DB = 6.0
ME_MEAS_S = (0.050, 0.600)
ME_MIN_SPAN_S = 0.100
ME_MIN_FALL_DB = 3.0
ME_END_ABOVE_FLOOR_DB = 6.0
ME_RISE_DB = 6.0
ME_GATE_S = 0.100

# ----------------------------------------------------------------------------- speed settings
TONE_REF_S = 45.0  # seconds of the dry reference voice the tone FIR is computed on (core.Reference.excerpt)
SIM_NFFT_STEP = 12000  # simulation IR lengths are rounded up to this many samples so that FFT sizes (and cached spectra) repeat

_ENV = {}  # band-envelope tables
_BANKS = {}  # Bank by rho
_REF = {}  # the reference excerpt and its statistics
_G = {}  # state shared with the forked workers (set before the pool is created)


# ----------------------------------------------------------------------------- band envelope

def _env_tables():
    if "w" not in _ENV:
        _ENV["w"] = np.ascontiguousarray(cm.octave_weights(cm.ENV_WIN, SR).T.astype(np.float32))
        _ENV["win"] = np.hanning(cm.ENV_WIN).astype(np.float32)
        _ENV["hop"] = int(round(cm.ENV_HOP_S * SR))
    return _ENV["w"], _ENV["win"], _ENV["hop"]


def env_db(x, frames=None, workers=1, chunk=4096):
    """Octave-band power envelope in dB (channel_match.band_env_db): frames of cm.ENV_WIN samples every cm.ENV_HOP_S, (frames, NB)
    float32. x (n,). frames = optional sorted frame indices: only those are computed (rows in that order)."""
    w, win, hop = _env_tables()
    x = np.asarray(x, np.float32)
    if len(x) < cm.ENV_WIN:
        x = np.pad(x, (0, cm.ENV_WIN - len(x)))
    nfr = (len(x) - cm.ENV_WIN) // hop + 1
    view = np.lib.stride_tricks.sliding_window_view(x, cm.ENV_WIN)[::hop]
    n_out = nfr if frames is None else len(frames)
    out = np.empty((n_out, NB), np.float32)
    for s0 in range(0, n_out, chunk):
        blk = (view[s0:s0 + chunk] if frames is None else view[np.asarray(frames[s0:s0 + chunk])]) * win
        p = np.abs(sfft.rfft(blk, axis=1, workers=workers, overwrite_x=True))
        p *= p
        out[s0:s0 + len(blk)] = 10 * np.log10(p @ w + 1e-20)
    return out


def env_db_parallel(x, workers):
    """env_db of a long signal on `workers` threads (the FFT releases the GIL)."""
    _, _, hop = _env_tables()
    nfr = (len(x) - cm.ENV_WIN) // hop + 1
    if workers <= 1 or nfr < 20000:
        return env_db(x)
    cuts = np.linspace(0, nfr, workers * 2 + 1).astype(int)
    parts = [(cuts[i], cuts[i + 1]) for i in range(len(cuts) - 1)]

    def one(ab):
        a, b = ab
        return env_db(x[a * hop: (b - 1) * hop + cm.ENV_WIN])

    with ThreadPoolExecutor(workers) as ex, threadpool_limits(1):
        res = list(ex.map(one, parts))
    return np.concatenate(res)


# ----------------------------------------------------------------------------- synthesis

class Bank:
    """Unit-power band-limited building blocks of the two layers, generated once per rho (fixed seeds) so a simulation only scales
    them by the parameter envelopes. late[c]: (NB, bank samples), channel 1 exactly rho_b-correlated with channel 0 over the bank;
    early[c]: (NB, T_E1_S samples) octave-band-filtered tap trains (each channel its own tap set, density
    ECHO_DENS * (t / T_ECHO_MIX_S)^2)."""

    def __init__(self, rho=None, sr=SR):
        self.sr = sr
        self.rho = np.full(NB, RHO_DEFAULT) if rho is None else np.clip(np.asarray(rho, float), 0.0, RHO_LATE_MAX)
        base = _bank_base()
        r = self.rho.astype(np.float32)[:, None]
        self.late = np.stack([base["x0"], r * base["x0"] + np.sqrt(1 - r ** 2) * base["x1o"]])  # channel 1: exactly rho_b-correlated
        self.early = base["early"]


    def fade(self, n):
        t = np.arange(n) / self.sr
        u = np.clip((t - T_XFADE_S[0]) / (T_XFADE_S[1] - T_XFADE_S[0]), 0.0, 1.0)
        return t, np.sin(0.5 * np.pi * u), np.cos(0.5 * np.pi * u)

    def _layer(self, blocks, t, fade, t_ref, levels, slopes):
        act = [b for b in range(NB) if levels[b] is not None and slopes[b] is not None]
        out = np.zeros(len(t))
        if not act:
            return out
        lv = np.array([levels[b] for b in act])[:, None]
        sl = np.array([slopes[b] for b in act])[:, None]
        amp = np.exp((0.5 * math.log(10) / 10) * (lv + sl * (t[None, :] - t_ref))) * fade[None, :]
        return np.einsum("bn,bn->n", blocks[act, :len(t)], amp.astype(np.float32), dtype=np.float64)

    def render_early(self, c, levels, slopes):
        """Early layer of channel c (no direct pulse), length T_E1_S + 1 sample. levels / slopes: per band, None = absent."""
        n = self.early.shape[2]
        t, _, fo = self.fade(n)
        return self._layer(self.early[c], t, fo, T_REF_E_S, levels, slopes)

    def render_late(self, c, levels, slopes, length_s):
        n = int(min(length_s, LATE_BANK_S) * self.sr)
        t, fi, _ = self.fade(n)
        return self._layer(self.late[c], t, fi, T_REF_L_S, levels, slopes)


def _bank_base():
    """The rho-independent noise of the Bank (generated once per process, fixed seed): late x0 and x1o (unit power, exactly
    orthogonal bands over the bank length) and the early tap trains (2, NB, T_E1_S samples)."""
    if "x0" not in _BANKS:
        sr = SR
        rng = np.random.default_rng(SEED)
        nl = int(LATE_BANK_S * sr)
        nfft = cm.pow2(nl)
        ws = cm.octave_weights(nfft, sr)
        spec = {c: sfft.rfft(rng.standard_normal(nfft)) for c in (0, 1)}
        x0s = np.empty((NB, nl), np.float32)
        x1s = np.empty((NB, nl), np.float32)
        for b in range(NB):
            x0 = sfft.irfft(spec[0] * ws[b], nfft)[:nl]
            x0 /= math.sqrt(float(np.mean(x0 ** 2)))
            x1 = sfft.irfft(spec[1] * ws[b], nfft)[:nl]
            x1 -= x0 * float(np.dot(x1, x0) / np.dot(x0, x0))  # exactly orthogonal to channel 0
            x1 *= math.sqrt(float(np.mean(x0 ** 2)) / float(np.mean(x1 ** 2)))
            x0s[b], x1s[b] = x0, x1
        ne = int(T_E1_S * sr) + 1
        nfft_e = cm.pow2(4 * ne)
        wse = cm.octave_weights(nfft_e, sr)
        t = np.arange(ne) / sr
        rate = np.maximum(ECHO_DENS * (t / T_ECHO_MIX_S) ** 2, ECHO_MIN_DENS)
        win = t >= T_E0_S
        early = np.empty((2, NB, ne), np.float32)
        for c in (0, 1):
            r2 = np.random.default_rng(SEED + 100 + c)
            hit = (r2.random(ne) < np.minimum(rate / sr, 1.0)) & win
            train = np.where(hit, r2.choice([-1.0, 1.0], ne) * np.sqrt(sr / rate), 0.0)
            sp = sfft.rfft(train, nfft_e)
            for b in range(NB):
                z = np.where(win, sfft.irfft(sp * wse[b], nfft_e)[:ne], 0.0)
                early[c, b] = z / math.sqrt(float(np.mean(z[win] ** 2)) + 1e-30)
        _BANKS.update(x0=x0s, x1o=x1s, early=early)
    return _BANKS


def get_bank(rho):
    return Bank(rho)


def late_length(slopes):
    rts = [60.0 / -s for s in slopes if s is not None]
    return float(np.clip(1.3 * max(rts), 0.25, LATE_BANK_S)) if rts else 0.0


def mono_ir(bank, early, late, c=0, cap_s=None):
    """delta + early + late of channel c (the fit's simulation IR), float32. cap_s: longest IR the caller can observe (s)."""
    ln = late_length(late[1]) if any(v is not None for v in late[1]) else 0.0
    if cap_s is not None:
        ln = min(ln, cap_s)
    n = max(int(ln * bank.sr), int(T_E1_S * bank.sr) + 1)
    h = np.zeros(n)
    h[0] = 1.0
    if any(v is not None for v in early[0]):
        e = bank.render_early(c, *early)
        h[:len(e)] += e
    if ln > 0:
        l = bank.render_late(c, *late, ln)
        h[:len(l)] += l
    return h.astype(np.float32)


# ----------------------------------------------------------------------------- decay events
# Vectorised re-statements of channel_match.decay_curves for the SIMULATIONS (floor fixed at SIM_FLOOR_DB, so no percentile of the
# signal) and of room_layers.nominal_curves.

def _silent_nanmedian(a, axis):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return np.nanmedian(a, axis=axis)


def nominal_curves(db, regions, floor):
    """Anchor-free decay curves: level at (nominal region end + tau) minus the 90th percentile of the PRE_WIN_S before the end, per
    band (events, taus, bands); NaN where the band is not floor + 15 dB live before the end or the next region starts within the
    window. Same event rules as cm.decay_curves, but the time origin is the slot end, NOT the refined anchor."""
    hop = cm.ENV_HOP_S
    off = cm.ENV_WIN / 2 / SR
    nfr = len(db)
    npre = int(cm.PRE_WIN_S / hop)
    tau_f = np.round(TAU / hop).astype(int)
    ev = np.full((len(regions), len(TAU), NB), np.nan)
    pre = np.full((len(regions), NB), np.nan)
    for k, (a, e) in enumerate(regions):
        nxt = regions[k + 1][0] if k + 1 < len(regions) else float("inf")
        ic = int(round((e - off) / hop))
        if ic - npre < 0 or ic >= nfr:
            continue
        p = np.percentile(db[ic - npre: ic + 1], 90, axis=0)
        okb = p > floor + 15.0
        pre[k] = np.where(okb, p, np.nan)
        ok = (e + TAU + 0.05 <= nxt) & (ic + tau_f < nfr)
        ok = np.cumprod(ok).astype(bool)  # the lab loop breaks at the first tau that does not fit
        if ok.any():
            ev[k, ok] = np.where(okb[None, :], db[ic + tau_f[ok]] - p[None, :], np.nan)
    return ev, pre


# ----------------------------------------------------------------------------- the dry reference voice (a simulation target)
# The statistics compare the decay AFTER speech ends. The dry reference's natural pauses are short (0.3-0.5 s), so the long-tau points
# (>= 0.6 s) would have < MIN_EVENTS events. The reference is therefore used as a pack of REF_EVENTS isolated events: the last
# REF_LEAD_S of a reference region (speech) followed by REF_PAUSE_S of digital silence (exactly what a pause of the dry reference is),
# every event simulated on its own (a ring-out beyond the pause never reaches the next event's statistic window).

REF_EVENTS = 12
REF_LEAD_S = 1.5
REF_PAUSE_S = 2.9  # > ANCHOR_SEARCH_S + TAU_S[-1] + 0.05


def _event_frames(Y, rows, cols, workers=1):
    """Band power (dB) of the envelope frames `cols` (same shape as rows, frame indices) of the events Y (N, L): (*cols.shape, NB)."""
    w, win, hop = _env_tables()
    view = np.lib.stride_tricks.sliding_window_view(Y, cm.ENV_WIN, axis=1)  # (N, L - win + 1, win)
    blk = view[rows.ravel(), cols.ravel() * hop] * win
    p = np.abs(sfft.rfft(blk, axis=1, workers=workers, overwrite_x=True))
    p *= p
    out = 10 * np.log10(p @ w + 1e-20)
    return out.reshape(*cols.shape, NB).astype(np.float64)


def sim_events(Y, end_s):
    """(anchored curves, nominal curves), both (events, taus, bands), of the simulated events Y (N, L): the same statistic as
    channel_match.decay_curves / nominal_curves at the fixed noise floor SIM_FLOOR_DB, for events that end at end_s (all alike) and
    are followed by silence longer than the longest tau."""
    hop = cm.ENV_HOP_S
    off = cm.ENV_WIN / 2 / SR
    npre = int(cm.PRE_WIN_S / hop)
    ns = int(round(ANCHOR_SEARCH_S / hop))
    k100 = int(0.1 / hop)
    tau_f = np.round(TAU / hop).astype(int)
    n = len(Y)
    ic = int(round((end_s - off) / hop))
    lo, hi = ic - ns, ic + ns
    base = lo - npre
    cols = np.arange(base, hi + 1)
    rows = np.repeat(np.arange(n)[:, None], len(cols), 1)
    db1 = _event_frames(Y, rows, np.broadcast_to(cols, rows.shape))  # (N, frames, NB)
    bb = 10 * np.log10((10 ** (db1 / 10)).sum(2) + 1e-20)
    ref = np.percentile(bb[:, lo - k100 - base: lo + k100 + 1 - base], 90, axis=1)
    hot = bb[:, lo - base: hi + 1 - base] >= (ref - cm.ANCHOR_DROP_DB)[:, None]
    has = hot.any(1)
    ia = lo + (hot.shape[1] - 1 - np.argmax(hot[:, ::-1], axis=1))  # last hot frame
    win_idx = (ia - base)[:, None] + np.arange(-npre, 1)[None, :]
    p = np.percentile(db1[np.arange(n)[:, None], win_idx], 90, axis=1)  # (N, NB)
    p_nom = np.percentile(db1[:, ic - npre - base: ic + 1 - base], 90, axis=1)
    c2 = np.concatenate([ia[:, None] + tau_f[None, :], np.full((n, 1), 0) + ic + tau_f[None, :]], axis=1)
    db2 = _event_frames(Y, np.repeat(np.arange(n)[:, None], c2.shape[1], 1), c2)
    nt = len(TAU)
    okb, okn = p > SIM_FLOOR_DB + 15.0, p_nom > SIM_FLOOR_DB + 15.0
    ev = np.where(okb[:, None, :], db2[:, :nt] - p[:, None, :], np.nan)
    nev = np.where(okn[:, None, :], db2[:, nt:] - p_nom[:, None, :], np.nan)
    ev[~has] = np.nan
    return ev, nev


def reference_sim():
    """The reference event pack every fit simulates: dict(x (N, S) float32 speech ends, end_s, dry = (stat, se) of the unprocessed
    pack, spec = cache of the events' spectra by FFT size)."""
    if "x" not in _REF:
        ref = core.reference()
        s_n = int(REF_LEAD_S * SR)
        cand = [(a, b) for a, b in ref.regions if b - a >= REF_LEAD_S + 0.05]
        pick = np.unique(np.linspace(0, len(cand) - 1, REF_EVENTS).round().astype(int))
        fade = np.minimum(np.arange(s_n) / (0.01 * SR), 1.0).astype(np.float32)
        x = np.stack([ref.sig[int(round(cand[k][1] * SR)) - s_n: int(round(cand[k][1] * SR))] * fade for k in pick]).astype(np.float32)
        _REF.update(x=x, end_s=REF_LEAD_S, n_total=s_n + int(REF_PAUSE_S * SR), spec={})
        ev0, _ = sim_events(_pad_events(x), REF_LEAD_S)
        _REF["dry"] = (cm.decay_stat(ev0)[0], cm.decay_stat_se(ev0))
    return _REF


def _pad_events(x, n_total=None):
    n_total = n_total or _REF["n_total"]
    out = np.zeros((len(x), n_total), np.float32)
    k = min(x.shape[1], n_total)
    out[:, :k] = x[:, :k]
    return out


def ref_through(ref, h):
    """The reference events (N, n_total) convolved with the mono IR h (each event on its own, the ring-out beyond the pause is cut)."""
    x = ref["x"]
    s_n = x.shape[1]
    m = -(-len(h) // SIM_NFFT_STEP) * SIM_NFFT_STEP
    nfft = sfft.next_fast_len(s_n + m - 1, real=True)
    if nfft not in ref["spec"]:
        ref["spec"][nfft] = sfft.rfft(x, nfft, axis=1)
    y = sfft.irfft(ref["spec"][nfft] * sfft.rfft(h, nfft)[None, :], nfft, axis=1)
    return _pad_events(y.astype(np.float32))


# ----------------------------------------------------------------------------- voice statistics and fit

def voice_stats(db, sreg):
    """Decay statistics of the stem scene's band envelope db. `stat`: exactly channel_match.estimate_scene's (anchored, ambient
    removed). `nom`: the same events with the time origin at the nominal slot end (nominal_curves), ambient removed the same way.
    Why both: the anchored curve is blind to the tail LEVEL once it is above ~-15 dB under the speech peak (the anchor moves with
    the tail); the nominal-end curve has 1 dB / dB sensitivity but inherits the slot-end timing error of the stem's ASR timestamps (a
    LOW quantile makes that error lower the estimate, never raise it)."""
    ev, pre, floor = cm.decay_curves(db, sreg, SR)
    stat, cnt = cm.decay_stat(ev)
    floor_rel = floor - _silent_nanmedian(pre, 0)
    se = cm.decay_stat_se(ev)
    nev, npre = nominal_curves(db, sreg, floor)
    nstat, ncnt = cm.decay_stat(nev)
    nfloor_rel = floor - _silent_nanmedian(npre, 0)
    nse = cm.decay_stat_se(nev)
    return {"stat": cm.minus_ambient(stat, floor_rel), "se": np.where(np.isfinite(se), se, 1.0), "cnt": cnt, "floor_rel": floor_rel,
            "nom": {"stat": cm.minus_ambient(nstat, nfloor_rel), "se": np.where(np.isfinite(nse), nse, 1.0), "cnt": ncnt, "floor_rel": nfloor_rel}}


def fit_layers(ref, rs, bank, late_fixed=None, early_on=True, threads=1, start=None):
    """Analysis by synthesis of the two-layer model against a reference decay statistic.
    ref = voice_stats() (or an M&E curve in the same format, without "nom"); rs = reference_sim(). late_fixed = per-band slopes (None
    where no tail) to fit only the late LEVELS with these slopes (M&E / blend rule); early_on=False skips the early layer.
    Cost of a candidate = squared error of the anchored curve (all taus of the layer) + of the nominal-end curve (when ref has "nom").
    start = per-band late levels (dB at T_REF_L_S) of an earlier, single-layer fit of the same stem (channel.estimate): when every
    active band has one, the first level search covers start +-12 dB (7 simulations) instead of -60 .. -20 dB (16).
    threads > 1 evaluates the independent candidates of a level search on that many threads (FFT and BLAS release the GIL).
    Returns dict(early=(levels, slopes), late=(levels, slopes), info_early, info_late, n_sims, final, valid)."""
    stat, se, cnt, floor_rel = ref["stat"], ref["se"], ref["cnt"], ref["floor_rel"]
    nom = ref.get("nom")
    dry_stat0, dry_se0 = rs["dry"]
    dry_se = np.where(dry_stat0 < floor_rel[None, :], 0.0, dry_se0)
    dry_stat = np.maximum(dry_stat0, floor_rel[None, :])
    valid = np.isfinite(stat) & np.isfinite(dry_stat) & (stat > (floor_rel + cm.FLOOR_MARGIN_DB)[None, :])
    noise = 2 * se ** 2
    if nom is not None:
        valid_n = np.isfinite(nom["stat"]) & (nom["stat"] > (nom["floor_rel"] + cm.FLOOR_MARGIN_DB)[None, :])
        noise_n = 2 * nom["se"] ** 2
    n_sims = [0]
    pool = ThreadPoolExecutor(threads) if threads > 1 else None
    e_lv, e_sl = [None] * NB, [None] * NB
    l_lv, l_sl = [None] * NB, [None] * NB

    def simulate(early, late):
        n_sims[0] += 1
        ev, nevc = sim_events(ref_through(rs, mono_ir(bank, early, late, cap_s=rs["n_total"] / SR)), rs["end_s"])
        sim = np.maximum(cm.decay_stat(ev)[0], floor_rel[None, :])  # below the stem's ambient nothing can be observed
        sim_n = None
        if nom is not None:
            sim_n = np.maximum(cm.decay_stat(nevc)[0], nom["floor_rel"][None, :])
        return sim, sim_n

    def zscore(v, b, base, base_se):
        return (stat[v, b] - base[v, b]) / np.sqrt(se[v, b] ** 2 + base_se[v, b] ** 2 + 1e-6)

    def cost_of(b, mask, sim, sim_n):
        v = valid[:, b] & mask & np.isfinite(sim[:, b])
        c = float(((stat[v, b] - sim[v, b]) ** 2).sum()) if v.any() else 0.0
        found = v.any()
        if nom is not None:
            vn = valid_n[:, b] & mask & np.isfinite(sim_n[:, b])
            if vn.any():
                c += float(((nom["stat"][vn, b] - sim_n[vn, b]) ** 2).sum())
                found = True
        return c if found else np.inf

    def level_search(active, lv, sl, mask, grid, make, closest=False, tol_sum=False):
        """Lowest level (per band; closest=True: the candidate nearest to the current value) whose error is within one point's
        statistical noise of the best one (tol_sum: all points' noise): where the anchored curves saturate the smallest consistent
        level is taken, where the nominal curve has sensitivity the minimum is resolved."""
        costs = np.full((len(grid), NB), np.inf)
        cands = []
        for g in grid:
            cand = list(lv)
            for b in active:
                cand[b] = lv[b] + g
            cands.append(cand)
        sims = list(pool.map(lambda c: simulate(*make(c)), cands)) if pool is not None else [simulate(*make(c)) for c in cands]
        for gi, (sim, sim_n) in enumerate(sims):
            for b in active:
                costs[gi, b] = cost_of(b, mask, sim, sim_n)
        for b in active:
            ns = list(noise[valid[:, b] & mask, b])
            if nom is not None:
                ns += list(noise_n[valid_n[:, b] & mask, b])
            tol = (float(np.sum(ns)) if tol_sum else float(np.mean(ns))) if ns else 0.0
            ok = np.where(costs[:, b] <= costs[:, b].min() + tol)[0]
            lv[b] = lv[b] + float(grid[ok[np.argmin(np.abs(grid[ok]))] if closest else ok[0]])

    def slope_update(active, lv, sl, mask, sim, sim_n, t_ref, lo, hi, min_pts=3):
        """One damped regression step of the slope on the residual of the anchored AND the nominal-end curve (stacked)."""
        for b in active:
            v = valid[:, b] & np.isfinite(sim[:, b]) & mask
            d, t = list(stat[v, b] - sim[v, b]), list(TAU[v])
            if nom is not None:
                vn = valid_n[:, b] & np.isfinite(sim_n[:, b]) & mask
                d += list(nom["stat"][vn, b] - sim_n[vn, b])
                t += list(TAU[vn])
            if len(d) >= min_pts:
                A = np.vstack([np.ones(len(d)), np.array(t) - t_ref]).T
                sl[b] = float(np.clip(sl[b] + 0.7 * np.clip(np.linalg.lstsq(A, np.array(d), rcond=None)[0][1], -30, 30), lo, hi))

    sl_lo_l, sl_hi_l = -60.0 / RT_MIN_S, -60.0 / RT_MAX_L_S
    sl_lo_e, sl_hi_e = -60.0 / RT_MIN_S, -60.0 / RT_MAX_E_S
    none_e = ([None] * NB, [None] * NB)
    grid0 = np.arange(-20, 41, 4.0)

    # ---- late layer
    if late_fixed is not None:
        active_l = [b for b in range(NB) if late_fixed[b] is not None]
        for b in active_l:
            l_sl[b] = float(np.clip(late_fixed[b], sl_lo_l, sl_hi_l))
            l_lv[b] = -60.0
    else:
        active_l = []
        for b in range(NB):
            v = valid[:, b] & LATE_TAUS
            if v.sum() < 2:
                continue
            z = zscore(v, b, dry_stat, dry_se)
            if (z > cm.Z_TAIL).sum() < 2:
                continue
            active_l.append(b)
            l_sl[b] = float(np.clip(np.polyfit(TAU[v], stat[v, b], 1)[0], sl_lo_l, sl_hi_l))
            l_lv[b] = -60.0
    mk_l = lambda lv: (none_e, (lv, l_sl))
    if active_l:
        if late_fixed is None and start is not None and all(start[b] is not None for b in active_l):
            for b in active_l:
                l_lv[b] = float(start[b])
            grid0 = np.arange(-12.0, 12.1, 4.0)
        level_search(active_l, l_lv, l_sl, LATE_TAUS, grid0, mk_l)
        if late_fixed is None:
            for step in (2.0, 1.0, 1.0):
                sim, sim_n = simulate(*mk_l(l_lv))
                slope_update(active_l, l_lv, l_sl, LATE_TAUS & (TAU >= 0.15), sim, sim_n, T_REF_L_S, sl_lo_l, sl_hi_l)
                level_search(active_l, l_lv, l_sl, LATE_TAUS, np.arange(-3 * step, 3 * step + 0.1, step), mk_l, closest=True)
        else:
            level_search(active_l, l_lv, l_sl, LATE_TAUS, np.arange(-3.0, 3.1, 1.0), mk_l, closest=True)

    # ---- early layer (late applied): diffuse-continuation prior, moved only where the statistics say so
    active_e = []
    if early_on and active_l:
        # Prior: in a diffuse field the energy envelope is ONE exponential from the first reflection on, so the early layer is the
        # late exponential continued back through 5..100 ms (same slope, level on its line at T_REF_E_S) rendered as discrete taps.
        # Free-decay statistics see an early layer only weakly, so only the LEVEL of every band is searched around the prior (+-16 dB)
        # and the candidate CLOSEST to the prior within the whole noise budget of the layer's points is kept (shrinkage).
        active_e = list(active_l)
        for b in active_e:
            e_sl[b] = float(np.clip(l_sl[b], sl_lo_e, sl_hi_e))
            e_lv[b] = l_lv[b] + l_sl[b] * (T_REF_E_S - T_REF_L_S)
        mk_e = lambda lv: ((lv, e_sl), (l_lv, l_sl))
        level_search(active_e, e_lv, e_sl, EARLY_TAUS, np.arange(-16, 16.1, 4.0), mk_e, closest=True, tol_sum=True)
        level_search(active_e, e_lv, e_sl, EARLY_TAUS, np.arange(-3.0, 3.1, 1.0), mk_e, closest=True, tol_sum=True)
        mk_l2 = lambda lv: ((e_lv, e_sl), (lv, l_sl))  # the early layer's leakage into the late taus: late levels once more
        level_search(active_l, l_lv, l_sl, LATE_TAUS, np.arange(-3.0, 3.1, 1.0), mk_l2, closest=True)

    # ---- final curves and per-band info
    final, final_n = simulate((e_lv, e_sl), (l_lv, l_sl))
    if pool is not None:
        pool.shutdown()

    def rms(b, mask):
        v = valid[:, b] & mask & np.isfinite(final[:, b])
        return round(float(np.sqrt(np.mean((stat[v, b] - final[v, b]) ** 2))), 2) if v.any() else None

    info_l = [{"band_hz": float(OCT[b]), "tail": b in active_l, "events": int(cnt[:, b].max()),
               **({"rt60_s": round(60.0 / -l_sl[b], 3), "level_db": round(l_lv[b], 1), "fit_rms_db": rms(b, LATE_TAUS)} if b in active_l else {})} for b in range(NB)]
    info_e = [{"band_hz": float(OCT[b]), "tail": b in active_e,
               **({"rt60_s": round(60.0 / -e_sl[b], 3), "level_db": round(e_lv[b], 1), "decay_db_s": round(e_sl[b], 1), "fit_rms_db": rms(b, EARLY_TAUS)} if b in active_e else {})} for b in range(NB)]
    return {"early": (e_lv, e_sl), "late": (l_lv, l_sl), "info_early": info_e, "info_late": info_l, "n_sims": n_sims[0], "final": final, "final_nom": final_n, "valid": valid}


def plausible_late(info):
    """Per band: late fit usable? -> (bool, reason or None)."""
    if not info.get("tail"):
        return False, "no tail"
    if info["rt60_s"] > IMPL_RT_S and info["level_db"] < IMPL_LEVEL_DB:
        return False, f"rt60 {info['rt60_s']} s at level {info['level_db']} dB"
    if info["fit_rms_db"] is not None and info["fit_rms_db"] > IMPL_FIT_RMS_DB:
        return False, f"fit rms {info['fit_rms_db']} dB"
    return True, None


def plausible_early(info):
    if not info.get("tail"):
        return False
    return info["fit_rms_db"] is None or info["fit_rms_db"] <= IMPL_FIT_RMS_DB


def band_median(vals, bands=RT_BANDS):
    v = [vals[b] for b in range(bands[0], bands[1] + 1) if vals[b] is not None and np.isfinite(vals[b])]
    return float(np.median(v)) if v else None


# ----------------------------------------------------------------------------- M&E events

def me_env(bg_mono, workers):
    """Octave-band envelope of the M&E + its floor + broadband level + live mask + the detection mask (whole clip, computed once)."""
    db = env_db_parallel(bg_mono, workers)
    live = db.max(1) > -120.0
    floor = np.percentile(db[live], 5, axis=0) if live.any() else np.full(NB, -200.0)
    bb = 10 * np.log10((10 ** (db / 10)).sum(1) + 1e-20)
    bb_floor = 10 * np.log10((10 ** (floor / 10)).sum() + 1e-20)
    hop = cm.ENV_HOP_S
    fall, plat = int(round(ME_FALL_S / hop)), int(round(ME_PLATEAU_S / hop))
    # event test per frame i (a = i - fall): the plateau bb[a-plat:a+1] is live, above the floor and flat, and bb falls by ME_FALL_DB
    nfr = len(bb)
    cand = np.zeros(nfr, bool)
    if nfr > plat + fall + 2:
        # i qualifies only if bb[i] <= bb[a] - 6 (bb[a] >= pre - 6 and bb[i] <= pre - 12): the rolling median is taken on those only
        i = np.arange(plat + fall, nfr)
        a = i - fall
        pre_ok = bb[i] <= bb[a] - (ME_FALL_DB - ME_PLATEAU_FLAT_DB)
        i, a = i[pre_ok], a[pre_ok]
        j = a - plat
        lv = np.lib.stride_tricks.sliding_window_view(live, plat + 1)[j].all(axis=1)
        win = np.lib.stride_tricks.sliding_window_view(bb, plat + 1)[j]  # bb[a - plat : a + 1]
        pre = np.median(win, axis=1)
        cand[i] = ((win.min(axis=1) >= bb_floor + ME_ABOVE_FLOOR_DB) & (bb[a] >= pre - ME_PLATEAU_FLAT_DB) & (bb[i] <= pre - ME_FALL_DB) & lv)
    return {"db": db, "live": live, "floor": floor, "bb": bb, "bb_floor": bb_floor, "cand": np.flatnonzero(cand)}


def me_events(env, t0, t1):
    """Decay events of the M&E inside [t0, t1].
    Returns dict(curves (events, taus, bands) in the format of cm.decay_curves, rt60 (events, bands) with NaN, n_detected, n_gated,
    times)."""
    db, bb, live, floor = env["db"], env["bb"], env["live"], env["floor"]
    hop = cm.ENV_HOP_S
    off = cm.ENV_WIN / 2 / SR
    nfr = len(bb)
    fall, plat = int(round(ME_FALL_S / hop)), int(round(ME_PLATEAU_S / hop))
    f_lo = max(int((t0 - off) / hop), plat + fall + 1)
    f_hi = min(int((t1 - off) / hop), nfr - 1 - int(ME_MEAS_S[1] / hop) - 1)
    curves, rts, times, pres = [], [], [], []
    n_det = n_gate = 0
    cand = env["cand"]
    pos = np.searchsorted(cand, f_lo)
    while pos < len(cand) and cand[pos] < f_hi:
        i = int(cand[pos])
        a = i - fall
        seg = bb[a - plat:a + 1]
        pre = float(np.median(seg))
        n_det += 1
        near = np.where(bb[i:i + fall + 1] >= pre - cm.ANCHOR_DROP_DB)[0]
        ia = i + int(near[-1]) if len(near) else i - 1
        if not live[ia + 2: ia + 2 + int(ME_GATE_S / hop)].all():
            n_gate += 1
            pos = np.searchsorted(cand, ia + plat)
            continue
        # end of the valid free decay: first rise of the broadband level over its running minimum
        seg_after = bb[ia:ia + int(ME_MEAS_S[1] / hop) + int(TAU[-1] / hop)]
        runmin = np.minimum.accumulate(seg_after)
        rise = np.where(seg_after > runmin + ME_RISE_DB)[0]
        end = ia + int(rise[0]) if len(rise) else ia + len(seg_after)
        p = np.percentile(db[max(ia - int(cm.PRE_WIN_S / hop), 0): ia + 1], 90, axis=0)
        okb = p > floor + 15.0
        cv = np.full((len(TAU), NB), np.nan)
        for j, tau in enumerate(TAU):
            k = ia + int(round(tau / hop))
            if k >= end or k >= nfr:
                break
            cv[j] = np.where(okb, db[k] - p, np.nan)
        rt = np.full(NB, np.nan)
        k0, k1 = ia + int(ME_MEAS_S[0] / hop), min(ia + int(ME_MEAS_S[1] / hop), end)
        for b in range(NB):
            if not okb[b]:
                continue
            x = db[k0:k1, b]
            low = np.where(x <= floor[b] + ME_END_ABOVE_FLOOR_DB)[0]
            if len(low):
                x = x[:low[0]]
            if len(x) * hop < ME_MIN_SPAN_S:
                continue
            t = np.arange(len(x)) * hop
            sl = float(np.polyfit(t, x, 1)[0])
            if sl < 0 and -sl * len(x) * hop >= ME_MIN_FALL_DB:
                rt[b] = float(np.clip(60.0 / -sl, cm.RT_MIN_S, RT_MAX_L_S))
        curves.append(cv)
        pres.append(np.where(okb, p, np.nan))
        rts.append(rt)
        times.append(ia * hop + off)
        pos = np.searchsorted(cand, ia + plat)
    return {"curves": np.array(curves) if curves else np.zeros((0, len(TAU), NB)), "rt60": np.array(rts) if rts else np.zeros((0, NB)),
            "n_detected": n_det, "n_gated": n_gate, "times": times, "floor": floor, "pre": np.array(pres) if pres else np.zeros((0, NB))}


def me_summary(res):
    """Per band low-quartile RT60 of the events, counts, scene-level median over RT_BANDS (bands with >= MIN_EVENTS events)."""
    rt = res["rt60"]
    by_band, cnt = [None] * NB, np.zeros(NB, int)
    for b in range(NB):
        v = rt[:, b][np.isfinite(rt[:, b])] if len(rt) else np.array([])
        cnt[b] = len(v)
        if len(v) >= MIN_EVENTS:
            by_band[b] = float(np.quantile(v, cm.DECAY_QUANTILE))
    used = int(np.isfinite(rt).any(1).sum()) if len(rt) else 0
    return {"by_band": by_band, "cnt": cnt, "rt60": band_median(by_band), "events_used": used,
            "n_detected": res["n_detected"], "n_gated": res["n_gated"]}


def me_reference(res):
    """The M&E events as a 'stem curve' in the format of voice_stats (for the level fit of the me / blend sources)."""
    ev = res["curves"]
    if not len(ev):
        return None
    stat, cnt = cm.decay_stat(ev)
    floor_rel = res["floor"] - _silent_nanmedian(res["pre"], 0)  # the bg's own floor re the events' plateau, as voice_stats does
    floor_rel = np.where(np.isfinite(floor_rel), floor_rel, -60.0)
    se = cm.decay_stat_se(ev)
    return {"stat": cm.minus_ambient(stat, floor_rel), "se": np.where(np.isfinite(se), se, 1.0), "cnt": cnt, "floor_rel": floor_rel}


# ----------------------------------------------------------------------------- inter-channel measures

def corr(a, b):
    n = min(len(a), len(b))
    a, b = a[:n].astype(np.float64), b[:n].astype(np.float64)
    d = math.sqrt(float((a * a).sum() * (b * b).sum()))
    return float((a * b).sum() / d) if d > 0 else None


def floor_rho(est):
    """Octave-band L/R coherence of the original mix's room tone: a room_tone.analyze() document (edges, rho) mapped to OCT_CENTRES
    (None -> None)."""
    if est is None:
        return None
    e = est["edges"]
    centres = np.sqrt(np.maximum(e[:-1], 50.0) * np.minimum(e[1:], SR / 2 * 0.98))
    return [round(float(v), 3) for v in np.interp(np.log2(OCT), np.log2(centres), est["rho"])]


# ----------------------------------------------------------------------------- one scene

def ratio_db(early, late):
    ee = float(np.sum(early.astype(np.float64) ** 2))
    el = float(np.sum(late.astype(np.float64) ** 2))
    if ee <= 0 or el <= 0:
        return None
    return round(10 * math.log10(ee / el), 1)


def fmt_params(levels, slopes):
    return [None if lv is None else round(float(lv), 1) for lv in levels], [None if s is None else round(float(s), 1) for s in slopes]


def decide(vinfo_l, vinfo_e, v_events, me, voice_dry):
    """Rule of step 4. Returns dict(source, agreement, ratio, late_slopes (per band or None), keep_early, reasons, voice_rt60 (raw),
    voice_rt60_ok, w)."""
    ok = [plausible_late(i) for i in vinfo_l]
    reasons = sorted({r for k, r in ok if r and r != "no tail"})
    raw_rt = band_median([i.get("rt60_s") for i in vinfo_l])
    good_rt = band_median([i["rt60_s"] if o[0] else None for i, o in zip(vinfo_l, ok)])
    any_good = any(o[0] for o in ok)
    me_ok = me is not None and me["rt60"] is not None and me["events_used"] >= MIN_EVENTS
    out = {"voice_rt60": raw_rt, "voice_rt60_ok": good_rt, "reasons": reasons, "ratio": None}
    me_slopes = None
    if me_ok:
        me_slopes = [-60.0 / (me["by_band"][b] if me["by_band"][b] is not None else me["rt60"]) for b in range(NB)]
    if any_good and good_rt is not None and me_ok:  # good bands outside RT_BANDS leave the median empty: the voice alone decides
        ratio = max(good_rt, me["rt60"]) / min(good_rt, me["rt60"])
        out["ratio"] = ratio
        if ratio <= AGREE_RATIO:
            out.update(source="voice", agreement="agree")
        else:
            wv, wm = float(v_events), float(me["events_used"])
            sl = []
            for b in range(NB):
                if not ok[b][0] and not me_ok:
                    sl.append(None)
                    continue
                rm = me["by_band"][b] if me["by_band"][b] is not None else me["rt60"]
                rv = vinfo_l[b]["rt60_s"] if ok[b][0] else None
                r = rm if rv is None else math.exp((wv * math.log(rv) + wm * math.log(rm)) / (wv + wm))
                sl.append(-60.0 / r)
            out.update(source="blend", agreement="disagree", late_slopes=sl, w=(wv, wm))
    elif any_good:
        out.update(source="voice", agreement="voice only")
    elif voice_dry:
        out.update(source="voice", agreement="voice dry")
    elif me_ok:
        out.update(source="me", agreement="me only", late_slopes=me_slopes)
    else:
        out.update(source="none", agreement="no evidence")
    out["keep_early"] = out["source"] in ("voice", "blend") and any(plausible_early(i) for i in vinfo_e)
    return out


def empty_record(sc, me):
    """Scene with fewer than 2 speech regions: dry, identity tone (analyze() lets it borrow the donor's room and fir)."""
    ident = np.zeros(EQ_TAPS, np.float32)
    ident[0] = 1.0
    none = np.zeros((2, 1), np.float32)
    rec = {"id": sc["id"], "start": sc["start"], "end": sc["end"], "source": "none", "agreement": "no evidence", "agreement_ratio": None, "borrowed_from": None,
           "note": "fewer than 2 speech regions: dry, identity tone", "me_rt60_s": None if me["rt60"] is None else round(me["rt60"], 3),
           "me_events": me["events_used"], "me_detail": {"detected": me["n_detected"], "gated": me["n_gated"], "per_band_events": [int(v) for v in me["cnt"]]},
           "me_rt60_by_band_s": [None if v is None else round(v, 3) for v in me["by_band"]], "voice_rt60_s": None, "voice_plausible": False,
           "voice_reasons": [], "voice_events": 0, "early_params": None, "late_params": None, "early_late_energy_db": None,
           "corr": {"early_lr": None, "late_lr": None, "early_late": None}, "rho_late": None, "n_sims": 0}
    return rec, ident, none, none


def build_room(bank, late_params, early_params, source_none=False):
    """Stereo early / late arrays from per-band parameters."""
    if source_none:
        return np.zeros((2, 1), np.float32), np.zeros((2, 1), np.float32)
    e_lv, e_sl = early_params
    l_lv, l_sl = late_params
    early = np.stack([bank.render_early(c, e_lv, e_sl) for c in (0, 1)]) if any(v is not None for v in e_lv) else np.zeros((2, 1))
    ln = late_length(l_sl) if any(v is not None for v in l_lv) else 0.0
    late = np.stack([bank.render_late(c, l_lv, l_sl, ln) for c in (0, 1)]) if ln > 0 else np.zeros((2, 1))
    return early.astype(np.float32), late.astype(np.float32)


def dub_ltas(early, late):
    """LTAS of the dry reference voice through delta + room (channel 0 of the layers): the dub side of the tone FIR."""
    if "tone" not in _REF:
        _REF["tone"] = core.reference().excerpt(TONE_REF_S)
    x, regs = _REF["tone"]
    h = np.zeros(max(early.shape[1], late.shape[1]), np.float32)
    h[0] = 1.0
    h[:early.shape[1]] += early[0]
    h[:late.shape[1]] += late[0]
    with threadpool_limits(1):
        return cm.ltas(cm.conv(x, h), regs)[0]


def tone_fir(p_stem, p_noise, p_dub):
    """channel_match.estimate_scene's tone FIR with the room applied: stem LTAS over the LTAS of the reference voice through
    delta + room (dub_ltas), noise-subtracted. p_stem / p_noise: the scene's cm.ltas / cm.noise_psd power spectra."""
    fir, _, _ = cm.eq_filter(p_stem, p_dub, SR, p_noise=p_noise)
    return fir.astype(np.float32)


def scene_record(sc, res, bank, threads=1):
    """Decide and render one scene from analyze_scene()'s result. Returns (record, early, late)."""
    fit, me, vs = res["fit"], res["me"], res["vs"]
    none_p = ([None] * NB, [None] * NB)
    rec = {"id": sc["id"], "start": sc["start"], "end": sc["end"], "me_rt60_s": None if me["rt60"] is None else round(me["rt60"], 3),
           "me_events": me["events_used"], "me_detail": {"detected": me["n_detected"], "gated": me["n_gated"], "per_band_events": [int(v) for v in me["cnt"]]},
           "me_rt60_by_band_s": [None if v is None else round(v, 3) for v in me["by_band"]]}
    if fit is None:  # voice evidence too thin
        d = decide([{"tail": False}] * NB, [{"tail": False}] * NB, 0, me, False)
        late_p = (list(none_p[0]), list(none_p[1]))
        early_p = none_p
        info_l, info_e = [{"band_hz": float(o), "tail": False} for o in OCT], [{"band_hz": float(o), "tail": False} for o in OCT]
        v_events = 0
    else:
        info_l, info_e = fit["info_late"], fit["info_early"]
        v_events = int(max(vs["cnt"][[int(np.argmin(abs(TAU - t))) for t in (0.1, 0.2, 0.4)]].max(), 0))
        voice_dry = not any(i["tail"] for i in info_l) and not any(i["tail"] for i in info_e)
        d = decide(info_l, info_e, v_events, me, voice_dry)
        late_p, early_p = None, None
    src = d["source"]
    ok = [plausible_late(i)[0] for i in info_l]
    n_sims = fit["n_sims"] if fit else 0
    if src == "voice":
        late_p = ([fit["late"][0][b] if ok[b] else None for b in range(NB)], [fit["late"][1][b] if ok[b] else None for b in range(NB)]) if fit else none_p
    elif src in ("blend", "me"):
        ref = vs if src == "blend" else me_reference(res["me_res"])
        if ref is None:
            src, d["source"], d["agreement"] = "none", "none", "no evidence"
            late_p = none_p
        else:
            f2 = fit_layers(ref, res["rs"], bank, late_fixed=d["late_slopes"], early_on=False, threads=threads)
            n_sims += f2["n_sims"]
            info_l = f2["info_late"]
            ok2 = [plausible_late(i)[0] for i in info_l]  # same garbage rule on the refit
            late_p = ([f2["late"][0][b] if ok2[b] else None for b in range(NB)], [f2["late"][1][b] if ok2[b] else None for b in range(NB)])
            d["reasons"] = sorted(set(d["reasons"]) | {plausible_late(i)[1] for i in info_l if i.get("tail") and not plausible_late(i)[0]})
    else:
        late_p = none_p
    if src in ("voice", "blend") and d["keep_early"]:
        eok = [plausible_early(i) for i in info_e]
        early_p = ([fit["early"][0][b] if eok[b] else None for b in range(NB)], [fit["early"][1][b] if eok[b] else None for b in range(NB)])
    else:
        early_p = none_p
    if src == "none" or (not any(v is not None for v in late_p[0]) and not any(v is not None for v in early_p[0])):
        early, late = build_room(bank, none_p, none_p, True)
        if src != "none" and src != "voice":
            src = "none"
        d["source"] = src
    else:
        early, late = build_room(bank, late_p, early_p)
    l_lv, l_sl = fmt_params(*late_p)
    e_lv, e_sl = fmt_params(*early_p)
    rec.update({
        "early_params": {"band_hz": [float(o) for o in OCT], "level_db": e_lv, "decay_db_s": e_sl,
                         "rt60_s": [None if s is None else round(60.0 / -s, 3) for s in early_p[1]],
                         "fit_rms_db": [i.get("fit_rms_db") if v is not None else None for i, v in zip(info_e, early_p[0])]},
        "late_params": {"band_hz": [float(o) for o in OCT], "levels_db": l_lv, "slopes_db_s": l_sl,
                        "rt60_by_band_s": [None if s is None else round(60.0 / -s, 3) for s in late_p[1]],
                        "fit_rms_db": [i.get("fit_rms_db") if v is not None else None for i, v in zip(info_l, late_p[0])]},
        "voice_rt60_s": None if d["voice_rt60"] is None else round(d["voice_rt60"], 3), "voice_plausible": d["voice_rt60_ok"] is not None,
        "voice_reasons": d["reasons"], "voice_events": v_events, "agreement": d["agreement"],
        "agreement_ratio": None if d["ratio"] is None else round(d["ratio"], 2), "source": d["source"], "borrowed_from": None,
        "rho_late": [round(float(v), 2) for v in bank.rho], "n_sims": n_sims})
    rec["note"] = note_for(rec, res)
    rec["early_late_energy_db"] = ratio_db(early[0], late[0]) if early.shape[1] > 1 and late.shape[1] > 1 else None
    rec["corr"] = {"early_lr": corr(early[0], early[1]) if early.shape[1] > 1 else None, "late_lr": corr(late[0], late[1]) if late.shape[1] > 1 else None,
                   "early_late": corr(early[0], late[0]) if early.shape[1] > 1 and late.shape[1] > 1 else None}
    for k in ("early_lr", "late_lr", "early_late"):
        if rec["corr"][k] is not None:
            rec["corr"][k] = round(rec["corr"][k], 3)
    return rec, early, late


def note_for(rec, res):
    src, ag = rec["source"], rec["agreement"]
    if res["fit"] is None:
        return "voice evidence too thin (< MIN_EVENTS events at 0.1-0.4 s)"
    base = {"voice": {"agree": "voice and M&E agree", "voice only": "no well-evidenced M&E estimate", "voice dry": "voice measured dry (no band distinguishable from the dry reference)"},
            "blend": {"disagree": "voice and M&E disagree by > 2x: log-RT60 blended by event counts, levels refitted"},
            "me": {"me only": "voice estimate implausible or absent: late tail from the M&E decays"},
            "none": {"no evidence": "no usable voice or M&E evidence: dry"}}
    n = base.get(src, {}).get(ag, "")
    if rec["voice_reasons"]:
        n += "; dropped voice bands: " + "; ".join(rec["voice_reasons"][:3])
    return n


def _stats_job(i):
    """Worker, phase 1: the cheap evidence of scene i of the shared state (M&E events, the stem's decay statistics, the stem's LTAS and
    noise spectra). Returns dict(few, me, me_res, vs, p_stem, p_noise); `few` = fewer than 2 speech regions."""
    g = _G
    sc = g["scenes"][i]
    hop = cm.ENV_HOP_S
    t0 = sc["start"]
    t1 = min(sc["end"], g["n"] / SR)
    f0 = int(round(t0 / hop))
    f1 = min(int(round(t1 / hop)), len(g["stem_db"]))
    tb = f0 * hop  # scene time origin on the frame grid
    me_res = me_events(g["me_env"], t0, t1)
    out = {"few": False, "me": me_summary(me_res), "me_res": me_res}
    sreg = [(a - tb, b - tb) for a, b in cm.window_regions(g["regions"], t0, t1)]
    if len(sreg) < 2:
        out["few"] = True
        return out
    i0, i1 = int(t0 * SR), min(int(t1 * SR), g["n"])
    sseg = g["stem"][i0:i1]
    with threadpool_limits(1), ThreadPoolExecutor(3) as ex:  # the three are independent and mostly FFT / sorting (GIL released)
        f_vs = ex.submit(voice_stats, g["stem_db"][f0:f1], sreg)
        f_ltas = ex.submit(cm.ltas, sseg, [(a - t0, b - t0) for a, b in cm.window_regions(g["regions"], t0, t1)], SR)
        f_noise = ex.submit(cm.noise_psd, sseg, SR)
        out["vs"], out["p_stem"], out["p_noise"] = f_vs.result(), f_ltas.result()[0], f_noise.result()
    return out


def _start_levels(sc):
    """Late levels at T_REF_L_S from the channel document's single-layer tail of the scene (level at its T_REF_S on the same
    exponential), None per band without a tail; None when the scene carries no tail."""
    lv, sl = sc.get("levels"), sc.get("slopes")
    if not lv or not sl:
        return None
    return [None if a is None or b is None else a + b * (T_REF_L_S - cm.T_REF_S) for a, b in zip(lv, sl)]


def _fit_job(args):
    """Worker, phase 2: fit, decision and rendering of a scene from its phase 1 evidence. Returns (record, early, late)."""
    i, st, threads = args
    g = _G
    res = {"me": st["me"], "me_res": st["me_res"], "rs": g["rs"], "vs": st["vs"]}
    with threadpool_limits(1):
        res["fit"] = None if not cm.enough_events(st["vs"]["cnt"]) else fit_layers(st["vs"], g["rs"], g["bank"], threads=threads, start=_start_levels(g["scenes"][i]))
        return scene_record(g["scenes"][i], res, g["bank"], threads)


# ----------------------------------------------------------------------------- clip level

def _dub_ltas_job(room):
    return dub_ltas(*room)


def analyze(clip, chan, regions, workers=8, tone=None):
    """See the module docstring."""
    rho = floor_rho(tone)
    scenes = chan["scenes"]
    stem = clip.speech
    _G.update(scenes=scenes, n=clip.n, stem=stem, regions=regions, bank=get_bank(rho), rs=reference_sim(),
              stem_db=env_db_parallel(stem, workers), me_env=me_env(clip.bg2.mean(0), workers))
    dub_ltas(np.zeros((2, 1), np.float32), np.zeros((2, 1), np.float32))  # loads the reference excerpt before the fork
    ctx = mp.get_context("fork")
    with ProcessPoolExecutor(max(1, min(workers, len(scenes))), mp_context=ctx) as ex:
        stats = list(ex.map(_stats_job, range(len(scenes))))
        todo = [i for i, st in enumerate(stats) if not st["few"]]
        n_fit = sum(1 for i in todo if cm.enough_events(stats[i]["vs"]["cnt"])) or 1
        threads = max(1, workers // min(workers, n_fit))  # the candidates of a level search use the cores the scene processes leave free
        fits = dict(zip(todo, ex.map(_fit_job, [(i, stats[i], threads) for i in todo])))
        ents = _donor_rule(scenes, stats, fits)
        rooms_ = {}  # unique rooms (by the scene that owns the arrays) -> LTAS of the reference through them
        for e in ents.values():
            if not e["few"]:
                rooms_.setdefault(e["room"], (e["early"], e["late"]))
        keys = list(rooms_)
        p_dub = dict(zip(keys, ex.map(_dub_ltas_job, [rooms_[k] for k in keys])))
    _G.clear()
    ident = np.zeros(EQ_TAPS, np.float32)
    ident[0] = 1.0
    for e in ents.values():
        e["fir"] = ident if e["few"] else tone_fir(e["p_stem"], e["p_noise"], p_dub[e["room"]])
    for sc in scenes:  # a scene with fewer than 2 speech regions that borrowed a room takes the donor's tone filter too
        e = ents[sc["id"]]
        if e["few"] and e["rec"]["borrowed_from"] is not None:
            e["fir"] = ents[e["rec"]["borrowed_from"]]["fir"]
    return {"sr": SR, "floor_coherence": None if rho is None else [float(v) for v in rho],
            "scenes": [dict(ents[sc["id"]]["rec"], _fir=ents[sc["id"]]["fir"], _early=ents[sc["id"]]["early"], _late=ents[sc["id"]]["late"]) for sc in scenes]}


def _donor_rule(scenes, stats, fits):
    """Entries {scene id: dict(rec, early, late, few, room (id of the scene that owns the room arrays), p_stem, p_noise)}.
    A scene with no evidence of its own (source none and < MIN_EVENTS voice events) takes the ROOM of its channel-document donor; a
    donor that has no evidence of its own either passes the request on to ITS donor (chains end at a scene with evidence)."""
    ents = {}
    for k, (sc, st) in enumerate(zip(scenes, stats)):
        i = sc["id"]
        if st["few"]:
            rec, _, none, _ = empty_record(sc, st["me"])
            ents[i] = {"rec": rec, "early": none, "late": none, "few": True, "room": i}
        else:
            rec, early, late = fits[k]
            ents[i] = {"rec": rec, "early": early, "late": late, "few": False, "room": i, "p_stem": st["p_stem"], "p_noise": st["p_noise"]}
    own = {i: not (e["rec"]["source"] == "none" and e["rec"].get("voice_events", 0) < MIN_EVENTS) for i, e in ents.items()}
    donor_id = {sc["id"]: sc.get("borrowed_from") for sc in scenes}
    base = {i: dict(e) for i, e in ents.items()}
    for sc in scenes:
        i = sc["id"]
        if own[i]:
            continue
        cur, seen = donor_id[i], {i}
        while cur is not None and cur in ents and not own[cur] and cur not in seen:
            seen.add(cur)
            cur = donor_id[cur]
        if cur is None or cur not in ents or not own[cur]:
            continue
        drec = base[cur]["rec"]
        e = ents[i]
        keep = {k: e["rec"][k] for k in ("me_rt60_s", "me_events", "me_detail", "me_rt60_by_band_s", "voice_events") if k in e["rec"]}
        e["rec"] = dict(drec, id=i, start=sc["start"], end=sc["end"], borrowed_from=cur, voice_rt60_s=None, voice_plausible=False, voice_reasons=[],
                        agreement="no evidence", agreement_ratio=None, n_sims=0, **keep,
                        note=f"no evidence of its own: room of scene {cur} ({drec['source']}: {drec['note']})")
        e["early"], e["late"], e["room"] = base[cur]["early"], base[cur]["late"], base[cur]["room"]
    return ents
