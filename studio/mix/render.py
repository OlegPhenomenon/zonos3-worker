"""One window of the studio track: the lab's `netflix` mix (dub_mix.render_mix) for a 30 s window, from the clip's analysis.

A window is rendered from its parts alone (no whole-file state): each take is stretched by PSOLA, followed to the original voice's
level, run through the location's channel (tone, early + late reverb, perspective tilt), the scene dynamics, followed to the level
again, panned, mixed with the M&E (ducked where the dub would lose the original's voice-to-background ratio), the non-verbal sounds
and the room tone, and mastered. Everything is computed on the window plus ROLL_S of context on both sides (reverb tails of parts
that ended before the window are rung out because the caller passes those parts; limiter, dynamics and level memories are shorter
than the roll), the context is cropped, so adjacent windows join.
"""
import os
import time
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp

import numpy as np

from . import levels, master, netflix, perspective, room_tone, take
from .core import SR, Clip, frame_db, frame_power, read_range, runs

ROLL_S = 2.0
NONVERBAL_GAIN = 0.8
NONVERBAL_RANGE_S = (0.25, 1.5)  # a speech-activity run with no part on it that long is a laugh / breath, shorter is a speck, longer is
# speech the transcript missed (it must not leak back in the original language)
NONVERBAL_EXCLUDE_S = 0.12  # margin around every part
NONVERBAL_FADE_S = 0.03
BRIDGE_S = 0.2
VOICE_LIMIT_RELEASE_MS = 80.0
PSOLA_WORKERS = 4
MAX_PART_S = 20.0  # longest dub part: parts that started earlier than this plus the tail before the window cannot reach it

_POOL = []


def _pool():
    if not _POOL:
        _POOL.append(ProcessPoolExecutor(PSOLA_WORKERS, mp_context=mp.get_context("fork")))
    return _POOL[0]


class NotPrepared(Exception):
    """No analysis for this stems_dir (the worker answers 409 not_prepared and the backend re-runs /prepare)."""


_DOCS = {}


def load_docs(stems_dir):
    path = Clip(stems_dir).analysis_dir / "render.pkl"
    if not path.exists():
        raise NotPrepared(str(stems_dir))
    key = (str(path), path.stat().st_mtime_ns)
    if key not in _DOCS:
        _DOCS.clear()
        _DOCS[key] = Clip(stems_dir).load_doc("render")
    return _DOCS[key]


def _slice(x, pos, n):
    """x placed at sample `pos` of an n-sample line: (x part inside, start index in the line) or (None, 0) when outside."""
    a, b = max(pos, 0), min(pos + x.shape[-1], n)
    return (x[..., a - pos:b - pos], a) if b > a else (None, 0)


def _add(dst, x, pos):
    seg, a = _slice(x, pos, dst.shape[-1])
    if seg is not None:
        dst[..., a:a + seg.shape[-1]] += seg


def _nonverbal(speech2, parts, thr_db, n):
    """Sounds on the speech stem that carry no words: speech activity outside every part (+- margin), runs of NONVERBAL_RANGE_S."""
    nf = n // (SR // 100)
    on = frame_db(0.5 * (speech2[0] + speech2[1])) > thr_db
    out = on.copy()
    for a, b in runs(~on):  # bridge short gaps like spans.activity
        if b - a < int(BRIDGE_S * 100) and a > 0 and b < len(on):
            out[a:b] = True
    for p in parts:
        out[max(0, int((p["o0"] - NONVERBAL_EXCLUDE_S) * 100)):max(0, int((p["o1"] + NONVERBAL_EXCLUDE_S) * 100)) + 1] = False
    keep = np.zeros(nf, bool)
    for a, b in runs(out):
        if NONVERBAL_RANGE_S[0] * 100 <= b - a <= NONVERBAL_RANGE_S[1] * 100:
            keep[a:b] = True
    if not keep.any():
        return None
    env = np.zeros(n, np.float32)
    env[: nf * (SR // 100)] = np.repeat(keep.astype(np.float32), SR // 100)
    w = int(NONVERBAL_FADE_S * SR)
    c = np.concatenate([[0.0], np.cumsum(env, dtype=np.float64)])
    sm = ((c[w:] - c[:-w]) / w).astype(np.float32)
    env = np.concatenate([sm, np.zeros(n - len(sm), np.float32)])
    return speech2 * (env * np.float32(NONVERBAL_GAIN))


def render_pcm(stems_dir, segment, segment_ms, parts):
    """The window as (2, n) float32 at SR (n = the window's length: segment_ms, shorter for the last window) and a timings dict."""
    t_start = time.perf_counter()
    docs = load_docs(stems_dir)
    meta, plan = docs["meta"], docs["plan"]
    clip = Clip(stems_dir)
    i0 = int(round(segment * segment_ms * SR / 1000))
    i1 = min(int(round((segment + 1) * segment_ms * SR / 1000)), meta["n"])
    if i1 <= i0:
        raise ValueError("segment beyond the end of the clip")
    roll = int(ROLL_S * SR)
    w0, n = i0 - roll, i1 - i0 + 2 * roll
    t0s = w0 / SR
    tim = {}

    # --- takes: trim + PSOLA (parallel processes), placement relative to the window origin
    t = time.perf_counter()
    reach = plan["tail_s"]
    parts = [p for p in parts if p["placeStartMs"] / 1000 - t0s < n / SR and p["placeStartMs"] / 1000 + MAX_PART_S + reach > t0s]
    jobs = [(p["take"], p["tempo"]) for p in parts]
    xs = list(_pool().map(take.prepare, *zip(*jobs))) if len(jobs) > 1 else [take.prepare(*j) for j in jobs]
    tim["takes"] = time.perf_counter() - t
    lp = []
    for p, x in zip(parts, xs):
        d0 = p["placeStartMs"] / 1000
        o0 = p["startMs"] / 1000
        o1 = max(p["endMs"] / 1000, o0 + 0.2)
        d1 = max(d0 + 0.2, d0 + len(x) / SR)
        lp.append({"o0": o0 - t0s, "o1": o1 - t0s, "d0": d0 - t0s, "d1": d1 - t0s, "err": d0 + len(x) / SR - p["endMs"] / 1000,
                   "pos": int(round((d0 - t0s) * SR)), "x": x, "speaker": p["speaker"], "o0_abs": o0, "o1_abs": o1})
    lp = [q for q in lp if q["pos"] + len(q["x"]) + int(reach * SR) > 0 and q["pos"] < n]

    # --- the original of this window
    t = time.perf_counter()
    stem2 = read_range(clip.dir / "speech.flac", w0, w0 + n, 2)
    bg2 = read_range(clip.dir / "bg.flac", w0, w0 + n, 2)
    p_orig = frame_power(stem2).mean(0)
    p_bg = frame_power(bg2).mean(0)
    tim["read"] = time.perf_counter() - t

    # --- level 1: the dry dub followed to the original voice
    t = time.perf_counter()
    dub = np.zeros(n, np.float32)
    for q in lp:
        _add(dub, q["x"], q["pos"])
    g1 = levels.curve(lp, p_orig, frame_power(dub)[0], n / SR)
    t_grid = np.arange(len(g1)) * (SR // 100) + (SR // 200)
    gain1 = (10 ** (np.interp(np.arange(n), t_grid, g1) / 20)).astype(np.float32)

    # --- channel per part: perspective cue, tilt, location room
    voice = np.zeros((2, n), np.float32)
    order = sorted(q["o0_abs"] for q in lp)
    for q in lp:
        x = q["x"] * gain1[np.clip(np.arange(q["pos"], q["pos"] + len(q["x"])), 0, n - 1)]
        later = [s for s in order if s > q["o0_abs"] + 1e-6]
        cue = perspective.part_cue(clip, docs["persp"], q["o0_abs"], q["o1_abs"], later[0] if later else None)
        q["cue"] = cue
        _add(voice, netflix.render_part(plan, x, 0.5 * (q["o0_abs"] + q["o1_abs"]), cue), q["pos"])
    tim["channel"] = time.perf_counter() - t

    # --- dynamics of the scenes, level 2, perspective pan
    t = time.perf_counter()
    voice = netflix.apply_dynamics_scenes(plan, voice, t0s)
    g2 = levels.curve(lp, p_orig, frame_power(voice).mean(0), n / SR)
    voice = levels.apply(voice, g2)
    gl, gr = perspective.pan_gains(n, [{"d0": q["d0"], "d1": q["d1"], "o0": q["o0"], "o1": q["o1"], "pan_db_track": q["cue"].get("pan_db_track")}
                                       for q in lp])
    voice = voice * np.stack([gl, gr])
    voice, _ = master.limit_peaks(voice, meta["voice_peak_db"], VOICE_LIMIT_RELEASE_MS)
    tim["voice"] = time.perf_counter() - t

    # --- mix: M&E with dialogue focus + voice + non-verbal + room tone
    t = time.perf_counter()
    g_duck = master.duck_curve(p_orig, p_bg, voice, bg2, lp)
    mixed = levels.apply(bg2, -g_duck) + voice
    nv = _nonverbal(stem2, lp, meta["activity_thr_db"], n)
    if nv is not None:
        mixed += nv
    mixed += netflix.bed_window(room_tone.bed_window(docs["tone"], w0, n), plan, t0s)
    tim["mix"] = time.perf_counter() - t

    t = time.perf_counter()
    out = master.finish(mixed, docs["master"], roll, roll)
    tim["master"] = time.perf_counter() - t
    tim["total"] = time.perf_counter() - t_start
    return out[:, : i1 - i0], tim
