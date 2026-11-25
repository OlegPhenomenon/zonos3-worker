"""Speech activity of the separated speech stem (lab `sync_metrics.activity`): 10 ms frames within FLOOR_DB of the 95th percentile
level, gaps shorter than BRIDGE_S bridged."""
import numpy as np

from .core import SR, frame_db, runs, load_audio

FLOOR_DB = 30.0
BRIDGE_S = 0.2
FPS = 100


def activity(x, floor_db=FLOOR_DB, bridge_s=BRIDGE_S):
    """Boolean mask per 10 ms frame of a mono stem at SR. Returns (mask, threshold_db) where threshold_db is the frame level
    (10*log10 power, see core.frame_db) above which a frame is active."""
    db = frame_db(x, SR)
    thr = float(np.percentile(db, 95) - floor_db) if len(db) else -120.0
    on = db > thr
    out = on.copy()
    bridge = int(bridge_s * FPS)
    for a, b in runs(~on):
        if b - a < bridge and a > 0 and b < len(on):
            out[a:b] = True
    return out, thr


def speech_spans(stems_dir):
    """[(start_ms, end_ms)] of the speech activity of `speech.flac`."""
    from pathlib import Path
    x = load_audio(Path(stems_dir) / "speech.flac", 1, SR)[0]
    return spans_of(activity(x)[0])


def spans_of(mask):
    return [(int(a * 1000 / FPS), int(b * 1000 / FPS)) for a, b in runs(mask)]
