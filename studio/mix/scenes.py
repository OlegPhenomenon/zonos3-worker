"""Acoustic scenes: stretches over which the channel of the separated voice stem is constant, each with a signature.

The dub is passed through the channel (room, mic band, compressor) of the original voice, measured per scene (channel.py). This
module finds the scenes and gives each one a signature so a scene with too little speech can borrow the channel of the most
similar other scene. Port of the lab's `channel_scenes.py` without shot cuts and without speaker labels (the clip is analysed
before any parts exist: one speaker "all", no cuts).

API
    detect(clip, regions, workers=8) -> dict
        clip: core.Clip (mono `clip.speech` is analysed), regions: [(start_s, end_s)] nominal speech regions
        (core.speech_regions(clip.speech)). Result: {"scenes": [{id, start, end (s), speech_s, regions, signature (dB per band
        or None), speakers, n_atoms, signature_noise (dB), band_limits_hz, p_vs_prev}], "bands_hz", "noise_db",
        "candidate_boundaries", "merge_log", "params"}. Scenes tile [0, duration_s].
    signature_distance(a, b) -> dB (rms over bands); compare with sqrt(noise_a^2 + noise_b^2) (`signature_noise` of each scene).
    nearest_scene(scenes, i) -> (j, distance dB, distance / pair noise) or None

Method (unchanged from the lab)
1. Frames (4096 FFT, 2048 hop) -> power in 1/3-octave bands from 100 Hz up to Nyquist (24 bands at 48 kHz). Only frames inside
   the speech regions count. Long-term average speech spectrum LTASS_k = 10 log10(sum P_k / sum P_total) of a group of frames
   (power weighted, level drops out); signature = LTASS of the group - LTASS of the whole clip.
2. Candidate boundaries = audio change points: the stretch is split at the region gaps nearest to equal-speech positions every
   WINDOW_SPEECH_S speech seconds (atoms).
3. Noise model measured from the clip itself (split-half variance v_k, content-drift floor b_k, chi2 pair statistic with
   empirical-null inflation lambda(T)).
4. Agglomeration: atoms with too little speech are glued to the neighbour with more speech, then the adjacent pair with the largest
   p is merged until every pair has p < ALPHA / n_candidates (Bonferroni); each surviving boundary may move by at most one atom to
   the region gap that best splits the phrase signatures into two constant groups.
5. band_limits_hz per scene = outermost bands whose signature is not more than BAND_Z sigmas below the clip's own typical level.

Differences from the lab: analysis at 48 kHz (FFT 4096 / hop 2048 samples = 85 / 43 ms instead of 93 / 46 ms; 24 bands instead of
23), FFTs only for frames inside speech regions (the others are never read), no shot cuts / speaker labels.
"""
import math
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import scipy.fft as sfft

from . import core

NFFT, HOP = 4096, 2048  # analysis geometry (samples)
F0 = 100.0
BLOCK_S = 2.0  # speech chunk dealt alternately into the two halves of the noise estimate (>> syllable correlation time)
WINDOW_SPEECH_S = 8.0  # resolution of audio-only change points: atom size in speech seconds (noise ~ 1/sqrt(speech s))
ALPHA = 0.05  # family-wise false-split probability over all candidate boundaries (Bonferroni)
BAND_Z = 3.0  # band is lost when its signature is this many noise sigmas below 0 (one-sided p = 0.0013 ~ ALPHA / 23 bands)
CHI2_MEDIAN_1 = 0.4549  # median of chi2 with 1 dof
CHUNK = 128  # frames per FFT batch (one thread task)


# ----------------------------------------------------------------------------- features

def band_layout(sr=core.SR):
    """(centres, lower edges, upper edges) in Hz of the 1/3-octave bands from F0 up to Nyquist."""
    k = np.arange(0, 64)
    c = F0 * 2.0 ** (k / 3)
    c = c[c * 2 ** (1 / 6) <= sr / 2]
    return c, c * 2 ** (-1 / 6), c * 2 ** (1 / 6)


def _speech_mask(t, regions):
    """Frames (centre times t) inside a region, and the index of the region holding each (last start before t)."""
    starts = np.array([a for a, _ in regions] + [np.inf])
    ends = np.array([b for _, b in regions] + [-np.inf])
    ri = np.maximum(np.searchsorted(starts, t, side="right") - 1, 0)
    return (t >= starts[ri]) & (t < ends[ri]), ri, starts


def band_power(x, speech, sr, workers):
    """1/3-octave band power [frames, K] float64 of the 4096-FFT frames (hop 2048) of x; rows of frames outside `speech` stay 0."""
    c, lo, hi = band_layout(sr)
    f = np.fft.rfftfreq(NFFT, 1 / sr)
    B = ((f[:, None] >= lo[None]) & (f[:, None] < hi[None])).astype(np.float32)
    k0 = int(np.argmax(B.any(1)))  # bins below the first band are never read
    B = np.ascontiguousarray(B[k0:])
    n = len(speech)
    win = np.hanning(NFFT).astype(np.float32)
    P = np.zeros((n, len(c)), np.float64)
    idx = np.flatnonzero(speech)
    view = np.lib.stride_tricks.sliding_window_view(x, NFFT)[::HOP]

    def run(s0):
        sel = idx[s0:s0 + CHUNK]
        X = sfft.rfft(view[sel] * win, axis=1)[:, k0:]
        P[sel] = (X.real * X.real + X.imag * X.imag) @ B

    with ThreadPoolExecutor(max(1, workers)) as ex:  # FFT, gather and matmul release the GIL; chunks write disjoint rows
        list(ex.map(run, range(0, len(idx), CHUNK)))
    return P


class Frames:
    """Per-frame speech mask / block id and cumulative band power (any frame span in O(1)); one speaker."""

    def __init__(self, x, sr, regions, workers):
        self.sr = sr
        n = max((len(x) - NFFT) // HOP + 1, 0)
        self.t = (np.arange(n) * HOP + NFFT / 2) / sr
        self.speech, ri, starts = _speech_mask(self.t, regions)
        self.block = np.where(self.speech, ri * 1000 + ((self.t - starts[ri]) // BLOCK_S).astype(int), -1)
        self.P = band_power(x, self.speech, sr, workers)
        self.centres = band_layout(sr)[0]
        self.K = len(self.centres)
        self.cum = np.zeros((n + 1, self.K + 1))
        self.cum[1:, :-1] = np.cumsum(self.P, axis=0)
        self.cum[1:, -1] = np.cumsum(self.speech)
        self.ref = np.nan_to_num(self.feats(0, n)[0])
        self.cache = {}

    def span(self, a, b):
        return int(np.searchsorted(self.t, a)), int(np.searchsorted(self.t, b))

    def feats(self, i0, i1, sel=None):
        """Band levels [K] (dB re the total power) and the speech-frame count over frames i0:i1, optionally only where `sel`
        (bool over i0:i1). Levels are NaN when there is no speech."""
        if sel is None:
            S = self.cum[i1] - self.cum[i0]
        else:
            m = self.speech[i0:i1] & sel
            S = np.append(self.P[i0:i1][m].sum(axis=0), m.sum())
        out = np.full(self.K, np.nan)
        if S[-1] > 0:
            out = 10 * np.log10((S[:-1] + 1e-20) / (S[:-1].sum() + 1e-20))
        return out, S[-1]

    def sig(self, i0, i1, sel=None):
        """Signature of a frame span (band levels here - the clip-wide band levels): (vector [K] dB or None, speech seconds)."""
        if sel is None and (i0, i1) in self.cache:
            return self.cache[(i0, i1)]
        f, n = self.feats(i0, i1, sel)
        if n <= 0:
            return None, 0.0
        out = np.nan_to_num(f - self.ref), n * HOP / self.sr
        if sel is None:
            self.cache[(i0, i1)] = out
        return out


def signature_distance(a, b):
    """rms (dB) over the bands of the difference of two signatures."""
    d = np.asarray(a, float) - np.asarray(b, float)
    return float(np.sqrt(np.mean(d * d)))


# ----------------------------------------------------------------------------- noise model and test

class NullModel:
    """What two groups of the SAME channel look like: per-feature noise variance  v_k (1/Ta + 1/Tb) + 2 b_k  and the correlation
    rho of the features. v_k: split-half of each atom's own speech (median of d^2). b_k: floor that does not shrink with more
    speech (content drift between neighbouring atoms: median adjacent difference minus what v_k explains). The test statistic
    x = sum_k d_k^2 / var_k  ~  g chi2(nu), g = sum(rho^2)/K, nu = K/g (Satterthwaite; Wilson-Hilferty tail)."""

    def __init__(self, F, atoms):
        U, w = [], []
        for a, b in atoms:
            i0, i1 = F.span(a, b)
            blk = F.block[i0:i1]
            ids = sorted(set(blk[blk >= 0].tolist()))
            if len(ids) < 2:
                continue
            par = np.isin(blk, ids[::2])
            (h1, T1), (h2, T2) = F.sig(i0, i1, par), F.sig(i0, i1, ~par & (blk >= 0))
            if h1 is None or h2 is None:
                continue
            U.append((h1 - h2) / math.sqrt(1 / T1 + 1 / T2))
            w.append(min(T1, T2))
        self.n = len(U)
        self.ok = self.n >= 3
        if not self.ok:
            return
        U = np.array(U)[np.array(w) >= np.median(w)]  # the better-measured half of the atoms
        self.v = np.median(U * U, axis=0) / CHI2_MEDIAN_1  # per-feature variance of a signature for one speech second
        sg = [F.sig(*F.span(a, b)) for a, b in atoms]
        D, tau = [], []
        for (s1, T1), (s2, T2) in zip(sg[:-1], sg[1:]):
            if s1 is not None and s2 is not None:
                D.append(s1 - s2)
                tau.append(1 / T1 + 1 / T2)
        D, tau = np.array(D), np.array(tau)
        self.b = np.maximum(np.median(D * D, axis=0) / CHI2_MEDIAN_1 - self.v * np.median(tau), 0) / 2 if len(D) else 0 * self.v
        z = np.vstack([U / np.sqrt(self.v + 1e-12)] + [D / np.sqrt(self.var(tau)[:, :] + 1e-12)] * (len(D) > 0))
        rho = np.nan_to_num(np.corrcoef(np.clip(z, -3, 3).T))  # winsorised: real changes cannot drive the correlation
        self.nb = len(self.v)
        self.g = float(np.sum(rho ** 2) / self.nb)
        self.nu = self.nb / self.g
        self.s0_db = math.sqrt(float(np.mean(self.v)))
        # Empirical-null inflation lambda(T): the Gaussian 1/T scaling fails for short groups (few frames, heavy tails). For windows
        # of 0.5, 1, 2, ... speech seconds cut consecutively out of the clip's speech, the median statistic of adjacent window
        # pairs must match the chi2 median, and the 95th percentile when there are >= 200 pairs (>= 10 pairs in the tail; real
        # changes are a small minority of the pairs). lambda >= 1.
        sf = np.nonzero(F.speech)[0]
        sizes, lams = [], []
        for T in 0.5 * 2.0 ** np.arange(0, 12):
            n = max(int(round(T * F.sr / HOP)), 1)
            if len(sf) // n < 8:
                break
            nw = len(sf) // n
            i0 = sf[np.arange(nw) * n]
            i1 = sf[(np.arange(nw) + 1) * n - 1] + 1
            S = F.cum[i1] - F.cum[i0]  # [windows, K + 1]
            ok = S[:, -1] > 0
            lv = 10 * np.log10((S[:, :-1] + 1e-20) / (S[:, :-1].sum(1, keepdims=True) + 1e-20))
            sgs = np.nan_to_num(lv - F.ref)
            Ts = S[:, -1] * HOP / F.sr
            xs = [self.x(sgs[j], Ts[j], sgs[j + 1], Ts[j + 1]) for j in range(nw - 1) if ok[j] and ok[j + 1]]
            sizes.append(n * HOP / F.sr)
            q = lambda z: self.nu * (1 - 2 / (9 * self.nu) + z * math.sqrt(2 / (9 * self.nu))) ** 3  # chi2 quantile, z = normal score
            lams.append(max(1.0, float(np.median(xs)) / q(0), float(np.quantile(xs, 0.95)) / q(1.645) if len(xs) >= 200 else 1.0))
        self.lam_sizes, self.lams = sizes, np.maximum.accumulate(lams[::-1])[::-1] if lams else []

    def lam(self, T):
        """Inflation for two groups of about T speech seconds each (constant outside the measured window sizes)."""
        return float(np.interp(math.log(max(T, 1e-9)), np.log(self.lam_sizes), self.lams)) if len(self.lams) else 1.0

    def x(self, a, Ta, b, Tb):
        """Standardised squared distance, ~ chi2(nu) under the null (before inflation)."""
        return float(np.sum((a - b) ** 2 / self.var([1 / Ta + 1 / Tb])[0])) / self.g

    def var(self, tau):
        """Noise variance of a difference of two groups, per feature, for an array of 1/Ta+1/Tb."""
        return np.asarray(tau, float)[:, None] * self.v[None] + 2 * self.b[None]

    def noise(self, T):
        """rms (dB) of the signature error of one group with T speech seconds."""
        return math.sqrt(float(np.mean(self.v / max(T, 1e-9) + self.b)))

    def p(self, a, Ta, b, Tb):
        """p-value that two groups with signatures a, b (speech Ta, Tb s) share one channel (Wilson-Hilferty chi2 tail)."""
        x = self.x(a, Ta, b, Tb) / self.lam(2 / (1 / Ta + 1 / Tb))
        k = self.nu
        z = ((x / k) ** (1 / 3) - (1 - 2 / (9 * k))) / math.sqrt(2 / (9 * k))
        return 0.5 * math.erfc(z / math.sqrt(2))


# ----------------------------------------------------------------------------- scene detection

def _atoms(F, regions, duration):
    """Atom boundaries (s): 0, duration and, in the stretch, the region gaps nearest to equal-speech positions."""
    gaps = np.array([(regions[i][1] + regions[i + 1][0]) / 2 for i in range(len(regions) - 1)])
    out = {0.0, float(duration)}
    i0, i1 = 0, len(F.speech)
    T = F.speech[i0:i1].sum() * HOP / F.sr
    k = int(round(T / WINDOW_SPEECH_S))
    g = gaps[(gaps > 0) & (gaps < duration)]
    if k >= 2 and len(g):
        cs = np.cumsum(F.speech[i0:i1])
        for j in range(1, k):
            t = F.t[i0 + int(np.searchsorted(cs, cs[-1] * j / k))]
            out.add(float(g[np.argmin(np.abs(g - t))]))
    return sorted(out)


def detect(clip, regions, workers=8):
    """Acoustic scenes of the clip's speech stem, see the module docstring."""
    duration = clip.duration_s
    sr = core.SR
    regions = sorted((float(a), float(b)) for a, b in regions)
    F = Frames(clip.speech, sr, regions, workers)
    bnd = _atoms(F, regions, duration)
    n_cand = len(bnd) - 2
    atoms = list(zip(bnd[:-1], bnd[1:]))
    null = NullModel(F, atoms)
    log = []
    res = {"candidate_boundaries": n_cand, "merge_log": log, "bands_hz": [float(c) for c in F.centres]}
    p_crit = ALPHA / max(n_cand, 1)
    groups = [[i, i] for i in range(len(atoms))]  # [first atom, last atom]

    def sig(g):
        return F.sig(*F.span(atoms[g[0]][0], atoms[g[1]][1]))

    if not null.ok:
        log.append({"note": "fewer than 3 atoms with two speech halves: noise unmeasurable, one scene"})
        groups = [[0, len(atoms) - 1]]
        p_crit = None
    else:
        res["noise_db"] = null.s0_db
        # 1. groups with less speech than the smallest window the null model is calibrated on cannot be judged: glue to the
        #    neighbour with more speech
        t_min = null.lam_sizes[0] if null.lam_sizes else 0.0
        while len(groups) > 1:
            Ts = [sig(g)[1] for g in groups]
            z0 = [i for i, T in enumerate(Ts) if T < t_min]
            if not z0:
                break
            i = z0[0]
            j = i + 1 if i == 0 else (i - 1 if i == len(groups) - 1 else (i - 1 if Ts[i - 1] >= Ts[i + 1] else i + 1))
            log.append({"t": atoms[groups[max(i, j)][0]][0], "op": "attach_no_speech", "speech_s": round(Ts[i], 2), "into": "prev" if j < i else "next"})
            lo, hi = min(i, j), max(i, j)
            groups[lo:hi + 1] = [[groups[lo][0], groups[hi][1]]]
        sg = [sig(g) for g in groups]
        ps = [null.p(sg[i][0], sg[i][1], sg[i + 1][0], sg[i + 1][1]) for i in range(len(groups) - 1)]
        while len(groups) > 1:  # 2. merge the least significant adjacent pair until all differ significantly
            i = int(np.argmax(ps))
            p = ps[i]
            if p < p_crit:
                break
            t = atoms[groups[i + 1][0]][0]
            log.append({"t": round(t, 3), "op": "merge", "p": p, "cut": False,
                        "speech_s": [round(sg[i][1], 1), round(sg[i + 1][1], 1)]})
            groups[i:i + 2] = [[groups[i][0], groups[i + 1][1]]]
            sg[i:i + 2] = [sig(groups[i])]  # only the merged group's signature and its two pairs change
            ps[i:i + 2] = []
            if i > 0:
                ps[i - 1] = null.p(sg[i - 1][0], sg[i - 1][1], sg[i][0], sg[i][1])
            if i < len(groups) - 1:
                ps.insert(i, null.p(sg[i][0], sg[i][1], sg[i + 1][0], sg[i + 1][1]))
    # 3. every surviving boundary may move to the region gap that best splits the phrases between its neighbouring boundaries
    #    into two constant-signature groups, at most one atom away from where it is (least squares on per-phrase signatures in
    #    noise units, weights = speech seconds; phrases are pure, group power sums are not).
    edges = [atoms[g[0]][0] for g in groups] + [duration]
    if null.ok:
        ps = [F.sig(*F.span(a, b)) for a, b in regions]
        pT = np.array([t for _, t in ps])
        pS = np.array([s if s is not None else np.zeros(F.K) for s, _ in ps]) / np.sqrt(null.v + 1e-12)
        mid = np.array([(a + b) / 2 for a, b in regions])

        def sse(w, X):  # weighted within-group sum of squares of every prefix
            cw, cx, cq = np.cumsum(w), np.cumsum(w[:, None] * X, axis=0), np.cumsum(w * np.sum(X * X, axis=1))
            return cq - np.sum(cx * cx, axis=1) / cw

        for k in range(1, len(edges) - 1):
            idx = np.nonzero((mid >= edges[k - 1]) & (mid < edges[k + 1]) & (pT > 0))[0]
            if len(idx) < 2:
                continue
            w, X = pT[idx], pS[idx]
            cost = sse(w, X)[:-1] + sse(w[::-1], X[::-1])[:-1][::-1]  # split after phrase j (j = 0..n-2)
            ia = groups[k][0]
            wl, wr = atoms[max(ia - 1, 0)][0], atoms[min(ia + 1, len(atoms) - 1)][1]
            tg = np.array([(regions[idx[j]][1] + regions[idx[j + 1]][0]) / 2 for j in range(len(idx) - 1)])
            cost = np.where((tg >= wl) & (tg <= wr), cost, np.inf)
            cur = int(np.sum(mid[idx] < edges[k])) - 1
            j = int(np.argmin(cost))
            if j != cur and cost[j] < (cost[cur] if 0 <= cur < len(cost) else sse(w, X)[-1]) - 1e-9:
                t = (regions[idx[j]][1] + regions[idx[j + 1]][0]) / 2

                def p_at(t):  # evidence of a change at t between the neighbouring boundaries
                    (sa, Ta), (sb, Tb) = F.sig(*F.span(edges[k - 1], t)), F.sig(*F.span(t, edges[k + 1]))
                    return 1.0 if sa is None or sb is None else null.p(sa, Ta, sb, Tb)

                nv = int(np.isfinite(cost).sum())  # gaps searched: Bonferroni
                pn = p_at(t)
                if pn < p_at(edges[k]) and pn * nv < p_crit:  # only when the move strengthens the evidence and stays significant
                    log.append({"t": round(t, 3), "op": "refine", "from": round(edges[k], 3), "was_cut": False})
                    edges[k] = t
    # 4. report
    scenes = []
    for i, (a, b) in enumerate(zip(edges[:-1], edges[1:])):
        i0, i1 = F.span(a, b)
        s, T = F.sig(i0, i1)
        regs = [(max(ra, a), min(rb, b)) for ra, rb in regions if rb > a and ra < b]
        sc = {"id": i, "start": float(a), "end": float(b), "speech_s": float(T), "regions": regs,
              "signature": None if s is None else [float(v) for v in s], "speakers": {"all": float(T)} if T > 0 else {},
              "n_atoms": sum(1 for at in atoms if a - 1e-9 <= at[0] < b - 1e-9),
              "signature_noise": null.noise(T) if null.ok and T > 0 else None, "band_limits_hz": None}
        if null.ok and T > 0:  # outermost bands not lost: signature_k >= -BAND_Z * sigma_k (relative to the clip's own spectrum)
            kept = np.nonzero(s >= -BAND_Z * np.sqrt(null.v / T + null.b))[0]
            if len(kept):
                sc["band_limits_hz"] = [float(F.centres[kept[0]] * 2 ** (-1 / 6)), float(F.centres[kept[-1]] * 2 ** (1 / 6))]
        scenes.append(sc)
    for i, sc in enumerate(scenes):  # evidence for the boundary in front of each scene
        a, b = (scenes[i - 1]["signature"], sc["signature"]) if i else (None, None)
        sc["p_vs_prev"] = null.p(np.array(a), scenes[i - 1]["speech_s"], np.array(b), sc["speech_s"]) if null.ok and a and b else None
    res["scenes"] = scenes
    res["params"] = {"alpha": ALPHA, "p_crit_per_boundary": p_crit, "window_speech_s": WINDOW_SPEECH_S, "atoms": len(atoms),
                     "null_atoms": null.n, "null_nu": null.nu if null.ok else None,
                     "null_lambda": dict(zip([round(t, 2) for t in null.lam_sizes], [round(float(v), 2) for v in null.lams])) if null.ok else None,
                     "speakers": ["all"], "n_surviving_non_cut": len(edges) - 2, "n_surviving_cut": 0}
    return res


def nearest_scene(scenes, i):
    """Most similar other scene with a signature: (j, distance dB, distance / sqrt(noise_i^2 + noise_j^2)); None if there is none."""
    best = None
    for j, s in enumerate(scenes):
        if j == i or s["signature"] is None or scenes[i]["signature"] is None or s["signature_noise"] is None:
            continue
        d = signature_distance(scenes[i]["signature"], s["signature"])
        r = d / math.hypot(scenes[i]["signature_noise"], s["signature_noise"])
        if best is None or r < best[2]:
            best = (j, d, r)
    return best
