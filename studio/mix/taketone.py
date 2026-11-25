"""The tone of the voice from the ACTUAL takes (lab: tone filter = long-term spectrum of the original speech stem in a scene over the
long-term spectrum of the dub track in the same scene, through the scene's room).

The product has no dub when the clip is analysed (rooms.analyze computes the scene filters against a dry reference voice, a prior) but
every rendered window brings takes. Each window adds the take LTAS of its parts to a per-scene running sum stored next to the analysis
(`analysis/dubstats.pkl`, one entry per rendered segment, so re-rendering a window does not count it twice). The estimate of a scene is
(sum of take spectra + PRIOR_FRAMES * reference spectrum) / (frames + PRIOR_FRAMES): the first window leans on the reference of the
voices' gender, later windows on the voices that really speak in that scene, and the estimate moves slowly because it is a cumulative
mean. The filters of the scene's location are rebuilt from it (netflix._location_tone, wet IR) for every window.
"""
import fcntl
import os
import pickle

import numpy as np
from scipy.signal import fftconvolve

from . import channel as cm
from . import core, netflix, rooms
from .core import SR

PRIOR_FRAMES = 150.0  # frames (hann 4096, hop 2048: 1 frame = 43 ms of speech) of the reference that count as evidence (~6 s)
STORE = "dubstats.pkl"
TAKES_MAX_BOOST_DB = 12.0  # a filter measured against the voices that really speak needs no reference-mismatch cap (netflix.MAX_TONE_BOOST_DB guards the prior)
GENDERS = (None,) + core.GENDERS


def prepare(clip, chan, rooms_doc, defects_doc, locs, regions):
    """What the per-window tone needs from the analysis: per scene with a room of its own the stem LTAS, the noise spectrum and the room's
    effect on a long-term spectrum (reference through the room over the dry reference); per location its scenes, weights and treatment."""
    refs = {}
    for g in GENDERS:
        x, regs = core.reference(g).excerpt(rooms.TONE_REF_S)
        refs[g] = cm.ltas(x, regs)[0]
    scenes = {}
    for sc in rooms_doc["scenes"]:
        i = sc["id"]
        chan_sc = next(s for s in chan["scenes"] if s["id"] == i)
        t0, t1 = chan_sc["start"], min(chan_sc["end"], clip.duration_s)
        sreg = [(a - t0, b - t0) for a, b in cm.window_regions(regions, t0, t1)]
        if len(sreg) < 2:
            continue  # a scene with fewer than 2 regions keeps its static (borrowed or identity) filter
        seg = clip.speech[int(t0 * SR):int(t1 * SR)]
        p_room = rooms.dub_ltas(sc["_early"], sc["_late"], None)
        scenes[i] = {"p_stem": cm.ltas(seg, sreg)[0], "p_noise": cm.noise_psd(seg), "room_gain": p_room / np.maximum(refs[None], 1e-20)}
    defects = {s["id"]: s for s in defects_doc["scenes"]}
    locations = {}
    for L in locs["locations"]:
        locations[L["id"]] = {"scenes": L["scenes"], "w": np.array([max(L["weights"][str(i)], 1e-3) for i in L["scenes"]]),
                              "thin": L["speech_s"] < netflix.MIN_SPEECH_S, "trt": defects[L["rep_scene"]]["treatment"],
                              "static": {i: next(s["_fir"] for s in rooms_doc["scenes"] if s["id"] == i) for i in L["scenes"]}}
    return {"scenes": scenes, "locations": locations, "refs": refs, "bounds": [(s["id"], s["start"], s["end"]) for s in chan["scenes"]]}


def take_ltas(x):
    """(summed power spectrum, frame count) of a dry take."""
    p, cnt = cm.ltas(x, [(0.0, len(x) / SR)])
    return p * cnt, cnt


def _update_store(stems_dir, segment, contribution):
    """Atomically store this segment's contribution and return {segment: contribution} of every stored segment."""
    path = os.path.join(stems_dir, "analysis", STORE)
    with open(path + ".lock", "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        store = pickle.load(open(path, "rb")) if os.path.exists(path) else {}
        store[segment] = contribution
        tmp = path + ".tmp"
        with open(tmp, "wb") as f:
            pickle.dump(store, f, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(tmp, path)
    return store


def contribution(src, parts):
    """{scene id: {"S", "cnt", "g": {gender: frames}}} of the parts of a window: [(original mid time s, dry take, gender)]."""
    out = {}
    for t_mid, x, gender in parts:
        sid = next((s for s, a, b in src["bounds"] if a <= t_mid < b), src["bounds"][-1][0])  # the last scene beyond the end, like netflix.location_at
        if len(x) < 4096:
            continue
        S, cnt = take_ltas(x)
        c = out.setdefault(sid, {"S": 0.0, "cnt": 0, "g": {}})
        c["S"] = c["S"] + S
        c["cnt"] += cnt
        c["g"][gender if gender in core.GENDERS else None] = c["g"].get(gender if gender in core.GENDERS else None, 0) + cnt
    return out


def window_plan(plan, src, store):
    """A copy of `plan` whose locations carry the tone filters (and wet IRs) estimated from the takes of every stored window."""
    total = {}
    for contrib in store.values():
        for sid, c in contrib.items():
            t = total.setdefault(sid, {"S": 0.0, "cnt": 0, "g": {}})
            t["S"] = t["S"] + c["S"]
            t["cnt"] += c["cnt"]
            for g, n in c["g"].items():
                t["g"][g] = t["g"].get(g, 0) + n
    locations = dict(plan["locations"])
    for lid, L in src["locations"].items():
        firs, changed = [], False
        for i in L["scenes"]:
            sc, t = src["scenes"].get(i), total.get(i)
            if sc is None or t is None or t["cnt"] == 0:
                firs.append(L["static"][i])
                continue
            g = max(t["g"], key=t["g"].get)  # the gender that speaks most in the scene gives the prior
            p_take = (t["S"] + PRIOR_FRAMES * src["refs"][g]) / (t["cnt"] + PRIOR_FRAMES)
            firs.append(rooms.tone_fir(sc["p_stem"], sc["p_noise"], p_take * sc["room_gain"]))
            changed = True
        if not changed:
            continue
        fir, _ = netflix._location_tone(firs, L["w"], L["thin"], L["trt"], TAKES_MAX_BOOST_DB)
        rep = plan["locations"][lid]
        early, late = rep["early"], rep["late"]
        wet = np.stack([fftconvolve(netflix.layer_sum(early[c], late[c]), fir) for c in range(2)]).astype(np.float32)
        locations[lid] = {**rep, "fir": fir, "wet": wet, "by_gender": {}}
    return {**plan, "locations": locations}
