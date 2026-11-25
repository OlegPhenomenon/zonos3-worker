"""Python API of the studio mix, used by the studio-prep worker server (see docs/architecture/studio-dub-realtime.md).

`stems_dir` layout (written by the worker after separation): `mix.flac` (original, stereo 48 kHz), `speech.flac`, `bg.flac`
(dialogue / music+effects stems, same length and rate). `analyze` adds `analysis/`.
"""
from . import analysis, cues, render, spans
from .encode import encode_opus
from .render import NotPrepared, load_docs

__all__ = ["NotPrepared", "speech_spans", "analyze", "part_cues", "render_segment"]


def speech_spans(stems_dir):
    """[(start_ms, end_ms)] of the speech activity of speech.flac (from the analysis when it exists)."""
    try:
        return [tuple(s) for s in load_docs(stems_dir)["meta"]["spans"]]
    except NotPrepared:
        return spans.speech_spans(stems_dir)


def analyze(stems_dir):
    """Whole-file channel analysis; writes stems_dir/analysis/render.pkl and returns the seconds spent per stage."""
    return analysis.analyze(stems_dir)


def part_cues(stems_dir, parts):
    """parts: [{id, startMs, endMs, speaker}] -> {"parts": {id: {pitch_st, range_st, loud_db, rate_sps, clip_ogg: bytes}},
    "speakers": {speaker: ogg/opus bytes <= 12 s}}. The clip is exactly [startMs, endMs] of speech.flac, mono 16 kHz ogg/opus.
    rate_sps is an acoustic syllable-nuclei rate (there is no text here), only its ratio to the speaker's median is meaningful.
    Parts of speaker "gap" only get a clip. Raises NotPrepared when the clip was not analysed."""
    return cues.part_cues(stems_dir, parts, load_docs(stems_dir)["meta"]["activity_thr_db"])


def render_segment(stems_dir, segment, segment_ms, parts, nonverbal=()):
    """parts: [{id, startMs, endMs, speaker, tempo, placeStartMs, take: WAV bytes, gender?: "male"|"female"|None}] (original slot, PSOLA tempo, placement
    of the stretched take, gender of the voice: its tone filter is measured against that gender's dry reference). nonverbal: [[startMs, endMs]] of
    the clip where the original's speech stem carries laughs / breaths and no words (decided by the dub pipeline); only those come back from the
    original voice. Returns Ogg Opus bytes (audio/ogg, encode.py), stereo 48 kHz, decoding to exactly the window [segment * segment_ms, +segment_ms) of the clip
    (the last window shorter). Parts whose reverb tail reaches into the window must be included (analyze() returns their reach as "mixTailMs"); raises
    NotPrepared when the clip was not analysed."""
    pcm, pre, post, _ = render.render_pcm(stems_dir, segment, segment_ms, parts, context=True, nonverbal=nonverbal)
    return encode_opus(pcm, pre, post)
