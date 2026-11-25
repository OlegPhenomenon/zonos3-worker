"""The voice channel of the `netflix` mix variant: location library, honest tone, two reverb layers, perspective tilt, dynamics, bed.

Port of the lab's `netflix_channel.py`. Where the `organic` variant copies every scene's channel as measured, this builds one
treatment per LOCATION and renders per PART:

* location (locations.py): scenes that cannot be told apart by their audio share one treatment. Tone = speech-weighted mean (dB) of
  the member scenes' tone filters (flat when the location has < locations.MIN_SPEECH_S of speech: too thin to fit a tone), room
  layers and dynamics = those of the member with the most speech.
* fidelity (defects.py): the scene's verdict decides what of the original is NOT copied: a defect low-pass / high-pass edge is relaxed
  (the tone filter is held flat over the band the dub has), a "compressor" that is really clipping is not applied, a deliberate futz
  (phone, TV, ...) is matched as measured.
* room (rooms.py): early reflections and the late tail as two decorrelated stereo layers, y_c = fir * (delta + early_c + late_c).
* perspective (perspective.py): per part the wet layers are scaled by the part's direct-to-reverberant departure
  (cue["wet_db"]) and the voice is tilted by its high-frequency departure from the speaker's norm (cue["tilt_db"]).
* dynamics (dynamics.py): the location's compressor / saturation per scene time range, per channel.

API (all arrays float32, SR = core.SR, time in seconds)
    build_plan(chan, rooms, defects, locs) -> dict   picklable plan: {"locations": {id: {"fir" (taps,), "early" (2, m), "late" (2, m),
        "wet" (2, m + taps - 1) = conv(fir, early + late) per channel, "dynamics", "rep", "futz", "flat_tone", "verdict", "scenes"}},
        "scene_location": {scene id: location id}, "scenes": [(id, start, end)], "bed": {scene id: None | {highpass_hz, gain_db}},
        "tail_s": longest tail of this plan, "report": [...]}
    render_part(plan, seg, t_mid_s, cue, workers=1) -> (2, m) float32, m = len(seg) + tail
    apply_dynamics_scenes(plan, out2, t0_s) -> (2, n) float32
    bed_window(bed, plan, t0_s) -> (2, n) float32
    TAIL_S: the longest tail any room can add after a part ends (s): a part that ended up to TAIL_S before a window starts still
        rings into it. plan["tail_s"] is the actual longest tail of the clip (<= TAIL_S).
"""
import math

import numpy as np
from scipy import fft as sfft
from scipy.signal import fftconvolve

from . import channel as cm
from . import core, dynamics, perspective, rooms
from .locations import MIN_SPEECH_S

SR = core.SR
NFFT = 16384
TAPS = cm.EQ_TAPS
BED_GAIN_MIN_DB = -6.0  # defects measures noise on the original MIX floor (audience, music bed included, the separated stem is clean):
# the bed is never attenuated by more than this, a studio audience is the programme, not a recording defect
TONE_REF_HZ = (300.0, 3000.0)  # the tone filter is normalised over this band (channel.eq_filter)
HP_TAPS = 4097
FADE_S = 0.25  # bed treatment cross-fade at the scene edges
MIN_DYN_SCENE_S = 1.0  # dynamics are not applied to scenes shorter than this
TAIL_S = rooms.LATE_BANK_S + TAPS / SR  # the longest late layer (the bank) + the tone filter


def fir_db(fir):
    """Magnitude (dB) of a filter on the rfft grid of NFFT."""
    return 20 * np.log10(np.abs(np.fft.rfft(fir, NFFT)) + 1e-9)


def minphase(mag_db, taps=TAPS):
    """Minimum-phase FIR of a magnitude curve (dB, rfft grid of NFFT) by the cepstral method, tail faded out like channel.eq_filter."""
    c = np.fft.irfft(mag_db / 20 * math.log(10), NFFT)
    w = np.zeros(NFFT)
    w[0] = 1
    w[1: NFFT // 2] = 2
    w[NFFT // 2] = 1
    h = np.fft.irfft(np.exp(np.fft.rfft(c * w)), NFFT)[:taps]
    fade = np.ones(taps)
    k = taps // 4
    fade[-k:] = 0.5 + 0.5 * np.cos(np.pi * np.arange(k) / k)
    return (h * fade).astype(np.float32)


def relax_edges(mag_db, hp_hz, lp_hz, edge_oct=0.25):
    """Hold the tone filter at the speech-band gain (median 300-3000 Hz = 0 dB after eq_filter) on the side(s) where the original's
    band edge is a recording defect: below hp_hz and/or above lp_hz the cut is removed (a boost is kept: only a cut is a defect to
    undo), with a 1/4-octave cosine blend at the edge."""
    f = np.fft.rfftfreq(NFFT, 1 / SR)
    ref = float(np.median(mag_db[(f >= TONE_REF_HZ[0]) & (f <= TONE_REF_HZ[1])]))
    out = mag_db.copy()
    lf = np.log2(np.maximum(f, 1.0))
    if hp_hz:
        t = np.clip((np.log2(hp_hz) + edge_oct - lf) / edge_oct, 0, 1)  # 1 below hp_hz / 2^edge, falling to 0 at hp_hz
        out = np.where(f < hp_hz * 2 ** edge_oct, np.maximum(out, ref * t + out * (1 - t)), out)
    if lp_hz:
        t = np.clip((lf - (np.log2(lp_hz) - edge_oct)) / edge_oct, 0, 1)
        out = np.where(f > lp_hz / 2 ** edge_oct, np.maximum(out, ref * t + out * (1 - t)), out)
    return out


def bed_treatment(b):
    return None if not b else dict(b, gain_db=max(b.get("gain_db", 0.0), BED_GAIN_MIN_DB))


def layer_sum(a, b):
    """Sum of two IRs of different lengths (the dry scenes have a 1-sample layer)."""
    out = np.zeros(max(len(a), len(b)), np.float32)
    out[:len(a)] += a
    out[:len(b)] += b
    return out


def build_plan(chan, rooms_doc, defects_doc, locs):
    """Location treatments + the per-scene mapping (module docstring). chan = channel document, rooms_doc = rooms.analyze(),
    defects_doc = defects.analyze(), locs = locations.cluster()."""
    defects = {s["id"]: s for s in defects_doc["scenes"]}
    scene_by_id = {s["id"]: s for s in chan["scenes"]}
    room_by_id = {s["id"]: s for s in rooms_doc["scenes"]}
    locations, report = {}, []
    for L in locs["locations"]:
        ids = L["scenes"]
        w = np.array([max(L["weights"][str(i)], 1e-3) for i in ids])
        rep = L["rep_scene"]
        mags = np.stack([fir_db(room_by_id[i]["_fir"]) for i in ids])
        # tone: speech-weighted mean in dB, each member normalised over the reference band first (they already are)
        mag = (mags * w[:, None]).sum(0) / w.sum()
        thin = L["speech_s"] < MIN_SPEECH_S
        if thin:  # a handful of seconds cannot fix a tone filter: flat tone
            mag = np.zeros_like(mag)
        spread = float(np.sqrt(((mags - mag) ** 2 * w[:, None]).sum() / (w.sum() * mags.shape[1])))
        trt = defects[rep]["treatment"]
        mag = relax_edges(mag, trt.get("relax_hp_hz"), trt.get("relax_lp_hz"))
        dyn = scene_by_id[rep].get("dynamics")
        if dyn and not trt.get("copy_dynamics", True):
            dyn = None
        fir = minphase(mag)
        early, late = room_by_id[rep]["_early"], room_by_id[rep]["_late"]
        wet = np.stack([fftconvolve(layer_sum(early[c], late[c]), fir) for c in range(2)]).astype(np.float32)  # once per location
        locations[L["id"]] = {"fir": fir, "early": early, "late": late, "wet": wet, "dynamics": dyn, "rep": rep, "futz": trt.get("futz"),
                              "flat_tone": thin, "verdict": defects[rep]["verdict"], "scenes": ids}
        report.append({"location": L["id"], "flat_tone": thin, "scenes": ids, "rep_scene": rep, "verdict": defects[rep]["verdict"], "futz": trt.get("futz"),
                       "relax_hp_hz": trt.get("relax_hp_hz"), "relax_lp_hz": trt.get("relax_lp_hz"), "tone_spread_db_rms": round(spread, 2),
                       "dynamics": None if not dyn else {k: dyn[k] for k in ("compressor", "saturation")}})
    tail = max(loc["wet"].shape[1] - 1 for loc in locations.values()) / SR
    return {"locations": locations, "scene_location": {int(s): l for s, l in locs["scene_to_location"].items()},
            "scenes": [(s["id"], s["start"], s["end"]) for s in chan["scenes"]],
            "bed": {sid: bed_treatment(d["treatment"].get("bed")) for sid, d in defects.items()}, "tail_s": tail, "report": report}


def location_at(plan, t):
    """The location treatment of the scene that contains original time t (the last scene beyond the end)."""
    for sid, a, b in plan["scenes"]:
        if a <= t < b:
            return plan["locations"][plan["scene_location"][sid]]
    return plan["locations"][plan["scene_location"][plan["scenes"][-1][0]]]


# ----------------------------------------------------------------------------- one part

def render_part(plan, seg, t_mid_s, cue, workers=1):
    """One dub part: the dry mono take `seg` (n,) at SR that belongs to original time t_mid_s -> (2, n + tail) float32. Tilt
    (cue["tilt_db"]), direct = fir * seg, wet = (early + late) * fir * seg scaled by 10 ** (cue["wet_db"] / 20), summed and
    normalised to the RMS of seg over the first len(seg) samples (the part keeps its own level; level is the mixer's job).
    No dynamics here."""
    x = np.asarray(seg, np.float32)
    n = len(x)
    if n < 2 or not x.any():
        return np.zeros((2, n), np.float32)
    loc = location_at(plan, t_mid_s)
    if cue.get("tilt_db"):
        x = perspective.tilt_filter(x, cue["tilt_db"], SR)
    wet = loc["wet"]
    m_wet = n + wet.shape[1] - 1
    m = max(n + len(loc["fir"]) - 1, m_wet)
    nfft = sfft.next_fast_len(m, real=True)
    spec = sfft.rfft(x, nfft, workers=workers)
    g = np.float32(10 ** (cue.get("wet_db", 0.0) / 20))
    h = sfft.rfft(wet, nfft, axis=1, workers=workers) * g + sfft.rfft(loc["fir"], nfft, workers=workers)[None, :]
    y = sfft.irfft(h * spec[None, :], nfft, axis=1, workers=workers)[:, :m].astype(np.float32)
    rin = float(np.sqrt(np.mean(x ** 2) + 1e-20))
    rout = float(np.sqrt(np.mean(y[:, :n] ** 2) + 1e-20))
    y *= np.float32(rin / rout)
    return y


# ----------------------------------------------------------------------------- dynamics of the scenes

def apply_dynamics_scenes(plan, out2, t0_s):
    """The location's compressor / saturation per scene time range, per channel, on the window out2 (2, n) that starts at absolute
    time t0_s. Scenes shorter than MIN_DYN_SCENE_S are left alone (decided on the whole scene, not on its part in the window). Each
    channel of the part of a scene inside the window is processed by dynamics.apply_dynamics (a pure local function; the window's
    pre-roll, dynamics.memory_s, warms its gain smoother up)."""
    out = np.array(out2, np.float32)
    w0 = int(round(t0_s * SR))
    n = out.shape[1]
    for sid, a, b in plan["scenes"]:
        dyn = plan["locations"][plan["scene_location"][sid]]["dynamics"]
        if not dyn or b - a < MIN_DYN_SCENE_S:
            continue
        lo, hi = max(int(a * SR) - w0, 0), min(int(b * SR) - w0, n)
        if hi <= lo:
            continue
        for c in range(out.shape[0]):
            out[c, lo:hi] = dynamics.apply_dynamics(out[c, lo:hi], dyn, SR)
    return out


# ----------------------------------------------------------------------------- bed

_HP = {}


def highpass_fir(hp_hz, taps=HP_TAPS):
    """Linear-phase FIR with the magnitude of a 4th-order Butterworth high-pass at hp_hz."""
    key = (hp_hz, taps)
    if key not in _HP:
        f = np.fft.rfftfreq(NFFT, 1 / SR)
        mag = 1 / np.sqrt(1 + (hp_hz / np.maximum(f, 1e-3)) ** 8)
        h = np.fft.fftshift(np.fft.irfft(mag, NFFT))
        mid = NFFT // 2
        _HP[key] = (h[mid - taps // 2: mid + taps // 2 + 1] * np.hanning(taps)).astype(np.float32)
    return _HP[key]


def bed_window(bed, plan, t0_s, fade_s=FADE_S):
    """The room-tone bed window (2, n) float32 starting at absolute time t0_s without the original's recording defects: in a scene whose
    treatment says so, the bed is high-passed (wind / handling rumble in the floor is not synthesised under the dub) and attenuated
    (heavy noise is not copied), with fade_s cross-fades at the scene edges. The filter is linear phase and delay compensated; the
    first / last 85 ms of the window are filter edge (the caller supplies a margin)."""
    bed = np.asarray(bed, np.float32)
    out = bed.copy()
    w0 = int(round(t0_s * SR))
    n = bed.shape[1]
    for sid, a, b in plan["scenes"]:
        t = plan["bed"].get(sid)
        if not t or not (t.get("highpass_hz") or t.get("gain_db")):
            continue
        a0, a1 = int(a * SR) - w0, int(b * SR) - w0  # the scene in window samples
        lo, hi = max(a0, 0), min(a1, n)
        if hi <= lo:
            continue
        seg = bed[:, lo:hi]
        proc = seg
        if t.get("highpass_hz"):
            h = highpass_fir(t["highpass_hz"])
            proc = np.stack([core.conv_same(c, h, len(h) // 2) for c in seg])
        proc = proc * np.float32(10 ** (t.get("gain_db", 0.0) / 20))
        k = min(int(fade_s * SR), (a1 - a0) // 2)
        p = np.arange(lo, hi)
        if k > 1:
            wgt = np.minimum(np.clip((p - a0) / (k - 1), 0, 1), np.clip((a1 - 1 - p) / (k - 1), 0, 1)).astype(np.float32)
        else:
            wgt = np.ones(hi - lo, np.float32)
        out[:, lo:hi] = seg * (1 - wgt) + proc * wgt
    return out
