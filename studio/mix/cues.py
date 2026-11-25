"""How each part is delivered in the original, measured on the separated speech stem (lab `emotion_cues.py`, `speaker_clips.py`).

Per part: median pitch and pitch range (semitones re 100 Hz), loudness (dB over the frames the speech activity calls active), a
speech-rate measure and the part's audio clip (ogg/opus) for the audio-LLM. The backend turns these into deviations from each
speaker's own baseline (median over the speaker's parts), so only consistency matters, not absolute values. Per speaker: up to
12 s of clean (non-overlapping) speech for the gender check.
"""
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import parselmouth
from scipy.signal import find_peaks, resample_poly

from .core import SR, FR, read_range, frame_db
from .spans import activity

CUE_SR = 16000
MIN_VOICED_FRAMES = 20  # 0.2 s of voiced speech: fewer and the median pitch is noise
MIN_ACTIVE_FRAMES = 10
SPEAKER_CLIP_S = 12.0
SPEAKER_MIN_PART_S = 0.4
NUCLEUS_DIP_DB = 2.0  # a syllable nucleus is an intensity peak that stands this far above the dip before it
NUCLEUS_RANGE_DB = 25.0  # ... and is within this of the part's loudest voiced frame (de Jong & Wempe, Praat syllable nuclei)


def mono16(x48):
    return resample_poly(x48, 1, SR // CUE_SR).astype(np.float32)


def opus(x16):
    """Mono 16 kHz float -> ogg/opus bytes (ffmpeg libopus, 24 kbit/s voip: ~30 ms per clip; libsndfile's opus is 10x slower)."""
    return subprocess.run(["ffmpeg", "-v", "error", "-f", "f32le", "-ar", str(CUE_SR), "-ac", "1", "-i", "-", "-c:a", "libopus", "-b:a", "24k",
                           "-application", "voip", "-f", "ogg", "-"], input=np.clip(x16, -1, 1).astype(np.float32).tobytes(),
                          check=True, capture_output=True).stdout


def measure(x48, thr_db):
    """pitch_st, range_st, loud_db, rate_sps of one part (mono 48 kHz)."""
    out = {"pitch_st": None, "range_st": None, "loud_db": None, "rate_sps": None}
    if len(x48) < 3 * FR:
        return out
    db = frame_db(x48)
    act = db > thr_db
    if act.sum() >= MIN_ACTIVE_FRAMES:
        out["loud_db"] = round(float(10 * np.log10(np.mean(10 ** (db[act] / 10)))), 2)
    snd = parselmouth.Sound(mono16(x48).astype(np.float64), CUE_SR)
    f0 = snd.to_pitch_ac(time_step=0.01, pitch_floor=70, pitch_ceiling=450).selected_array["frequency"]
    v = f0[f0 > 0]
    if len(v) >= MIN_VOICED_FRAMES:
        st = 12 * np.log2(v / 100.0)
        out["pitch_st"] = round(float(np.median(st)), 2)
        out["range_st"] = round(float(np.percentile(st, 90) - np.percentile(st, 10)), 2)
    inten = snd.to_intensity(minimum_pitch=70, time_step=0.01).values[0]
    k = min(len(inten), len(f0))
    if k > 10 and (f0[:k] > 0).any():
        inten, voiced = inten[:k], f0[:k] > 0
        peaks, _ = find_peaks(inten, prominence=NUCLEUS_DIP_DB)
        peaks = [p for p in peaks if voiced[p] and inten[p] >= inten[voiced].max() - NUCLEUS_RANGE_DB]
        out["rate_sps"] = round(len(peaks) / (len(x48) / SR), 2)
    return out


def speaker_clips(parts, speech_path):
    """{speaker: ogg bytes (<= 12 s)}: clean parts (>= 0.4 s, not overlapped by another speaker's part) joined by 0.2 s gaps."""
    clips = {}
    gap = np.zeros(int(0.2 * CUE_SR), np.float32)
    for spk in sorted({p["speaker"] for p in parts}):
        chunks, total = [], 0.0
        for p in parts:
            if p["speaker"] != spk or p["speaker"] == "gap":
                continue
            s, e = p["startMs"] / 1000, p["endMs"] / 1000
            if e - s < SPEAKER_MIN_PART_S or any(o["speaker"] != spk and o["startMs"] / 1000 < e and o["endMs"] / 1000 > s for o in parts):
                continue
            take = min(e - s, SPEAKER_CLIP_S - total)
            x = read_range(speech_path, int(s * SR), int((s + take) * SR), 1)[0]
            chunks += [mono16(x), gap]
            total += take
            if total >= SPEAKER_CLIP_S:
                break
        if chunks:
            clips[spk] = opus(np.concatenate(chunks))
    return clips


def part_cues(stems_dir, parts, thr_db):
    """See studio/mix/api.py. thr_db: activity threshold of the clip (spans.activity), from the analysis."""
    speech_path = Path(stems_dir) / "speech.flac"

    def one(p):
        x = read_range(speech_path, int(p["startMs"] * SR / 1000), int(p["endMs"] * SR / 1000), 1)[0]
        m = measure(x, thr_db)
        m["clip_ogg"] = opus(mono16(x))
        return p["id"], m

    with ThreadPoolExecutor(4) as ex:
        out = dict(ex.map(one, parts))
    return {"parts": out, "speakers": speaker_clips([p for p in parts if p["speaker"] != "gap"], speech_path)}
