"""Whole-clip analysis, done once per clip when the stems are written (the lab does this per clip in 5-8 min; see each stage module).

Stages (modules): speech activity (spans) -> acoustic scenes (scenes) -> per-scene channel with dynamics (channel, dynamics) ->
reverb layers (rooms), recording defects (defects), locations (locations) -> per-location treatments (netflix.build_plan) ->
room-tone floor (room_tone), perspective norms (perspective), mastering reference (master). The result is one document,
`analysis/render.pkl`, that `render.render_pcm` reads.
"""
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from . import channel, defects, locations, master, netflix, perspective, room_tone, rooms, scenes, spans
from .core import SR, Clip, speech_regions

WORKERS = 8


def analyze(stems_dir, workers=WORKERS):
    """Analyse `stems_dir` (mix.flac, speech.flac, bg.flac) and write `stems_dir/analysis/render.pkl`. Returns seconds per stage."""
    clip = Clip(stems_dir)
    tim, t = {}, time.perf_counter()

    def lap(name):
        nonlocal t
        now = time.perf_counter()
        tim[name] = round(now - t, 3)
        t = now

    with ThreadPoolExecutor(3) as ex:  # libsndfile releases the GIL
        list(ex.map(lambda a: getattr(clip, a), ("mix2", "speech2", "bg2")))
    lap("load")
    regions = speech_regions(clip.speech)
    mask, thr = spans.activity(clip.speech)
    lap("activity")
    # Independent stages run side by side (numpy/scipy release the GIL, the heavy ones fork their own pools): the room-tone floor and
    # the mastering reference need only the stems; perspective only the scene boundaries; rooms need the channel and the room tone;
    # defects need the channel.
    with ThreadPoolExecutor(5) as ex:
        f_tone = ex.submit(room_tone.analyze, clip, regions, workers)
        f_master = ex.submit(master.analyze, clip, regions)
        scene_doc = scenes.detect(clip, regions, workers)
        lap("scenes")
        f_persp = ex.submit(perspective.analyze, clip, scene_doc, regions, workers)
        chan = channel.estimate(clip, scene_doc, regions, workers)
        lap("channel")
        f_defects = ex.submit(defects.analyze, clip, chan, regions, workers)
        tone = f_tone.result()
        rooms_doc = rooms.analyze(clip, chan, regions, workers, tone=tone)
        lap("rooms")
        defects_doc = f_defects.result()
        persp, mast = f_persp.result(), f_master.result()
        lap("defects_persp_master")
    locs = locations.cluster(scene_doc, chan)
    plan = netflix.build_plan(chan, rooms_doc, defects_doc, locs)
    lap("plan")
    voice = np.zeros(clip.n, bool)
    for a, b in regions:
        voice[int(a * SR):int(b * SR)] = True
    peak = float(np.abs(clip.speech2[:, voice]).max()) if voice.any() else 1.0
    meta = {"n": clip.n, "duration_s": clip.duration_s, "activity_thr_db": thr, "spans": spans.spans_of(mask),
            "voice_peak_db": float(20 * np.log10(peak + 1e-12))}
    clip.save_doc("render", {"meta": meta, "plan": plan, "tone": tone, "persp": persp, "master": mast})
    lap("save")
    tim["total"] = round(sum(tim.values()), 3)
    tim["mixTailMs"] = int(np.ceil(plan["tail_s"] * 1000))
    return tim
