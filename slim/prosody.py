"""Optional prosody controls of a /tts request -> zonos3-server /v2/generate fields.

No prosody field in the request = no change to the request body = exactly the old output.

The server (Zyphra/ZONOS3-Mini, servers/ggml) has no `emotion` / `pitch_std` parameter. What it has:
  * `controls` {"<set>.<direction>": dose}: steering directions of /app/models/steer.gguf added to the model's hidden
    state (emotions, mood axes, pitch, pitch range, tempo, whisper, ...). Ranges are the studio's slider ranges
    (studio_bundle_mect60k_x10.json, ui.controls); the doses are raw, 0 = off.
  * `condition_values.speaking_rate` (UTF-8 bytes of text per second of speech), the condition row the worker pins at
    22.0 for uniform chunks.
  * `steer_hold`: the server's "steer then hold" fade needs a `steer_release` block in steer.gguf; the published
    steer.gguf has none, so controls are always a constant push and the server answers 400 to steer_hold < 1.
    The worker accepts `steer_hold: 1` (a no-op, so callers may state it) and refuses anything lower.
  * `controls_cfg_scale` cannot be combined with the speaker CFG the worker always uses (server 400): not offered.
"""
import math

RATE_RANGE = (8.0, 30.0)         # the checkpoint's rate buckets' edges run 6.5..29.3 bytes/s

# emotion name -> (control id, min dose, max dose, dose used when the caller sends just the name)
EMOTIONS = {
    'angry': ('emotions.angry', 0.0, 6.0, 4.0),
    'happy': ('emotions.happy', 0.0, 6.0, 4.0),
    'sad': ('emotions.sad', 0.0, 8.0, 6.0),
    'calm': ('emotions.serene', 0.0, 8.0, 4.0),
    'scared': ('emotions.scared', 0.0, 14.0, 12.0),
    'confused': ('emotions.confused', 0.0, 4.0, 3.0),
    'valence': ('mood.valence', -6.0, 6.0, None),   # negative = unpleasant; no name-only form
    'arousal': ('mood.arousal', -6.0, 6.0, None),   # negative = low energy
}
# every other steering direction a request may set: control id -> (min dose, max dose). Doses beyond are clamped.
CONTROLS = {
    **{cid: (lo, hi) for cid, lo, hi, _ in EMOTIONS.values()},
    'identity.gender': (-3.0, 3.0), 'identity.age': (-3.0, 3.0),
    'pitch.pitch': (-3.0, 3.0), 'pitch.pitch_range': (-3.0, 3.0),
    'texture.deep': (0.0, 6.0), 'texture.shrill': (0.0, 6.0), 'texture.husky': (0.0, 6.0),
    'texture.raspy': (0.0, 6.0), 'texture.soft': (0.0, 6.0), 'texture.authoritative': (0.0, 6.0),
    'room.clean': (-4.0, 6.0),
    'delivery.whisper': (0.0, 3.0), 'delivery.loud': (0.0, 4.0), 'delivery.tempo': (-3.0, 3.0),
    'delivery.conversational': (-5.0, 5.0), 'disfluency.disfluent': (-8.0, 20.0),
    'expressive.expressive': (-16.0, 16.0),
}
FIELDS = {'emotion', 'controls', 'speaking_rate', 'steer_hold'}


def _number(name: str, value) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f'{name} must be a finite number, got {value!r}')
    return float(value)


def _clamp(value: float, lo: float, hi: float) -> float:
    return min(max(value, lo), hi)


def parse(data: dict) -> tuple[dict, dict]:
    """-> (fields to merge into the /v2/generate body, controls applied after clamping). Raises ValueError.
    The returned `speaking_rate` is not a /v2/generate field: the caller puts it into condition_values."""
    body, applied = {}, {}
    emotion = data.get('emotion')
    if isinstance(emotion, str):
        emotion = {emotion: None}
    if emotion:
        if not isinstance(emotion, dict):
            raise ValueError('emotion must be a name or an object {name: dose}')
        for name, dose in emotion.items():
            if name not in EMOTIONS:
                raise ValueError(f'unknown emotion {name!r}; known: {", ".join(EMOTIONS)}')
            cid, lo, hi, default = EMOTIONS[name]
            if dose is None:
                if default is None:
                    raise ValueError(f'emotion {name!r} needs a dose in [{lo:g}, {hi:g}]')
                dose = default
            applied[cid] = _clamp(_number(f'emotion.{name}', dose), lo, hi)
    controls = data.get('controls')
    if controls:
        if not isinstance(controls, dict):
            raise ValueError('controls must be an object {"<set>.<direction>": dose}')
        for cid, dose in controls.items():
            if cid not in CONTROLS:
                raise ValueError(f'unknown control {cid!r}; known: {", ".join(CONTROLS)}')
            lo, hi = CONTROLS[cid]
            applied[cid] = _clamp(_number(f'controls.{cid}', dose), lo, hi)
    if any(dose != 0 for dose in applied.values()):
        body['controls'] = applied
    else:
        applied = {}
    if data.get('speaking_rate') is not None:
        body['speaking_rate'] = _clamp(_number('speaking_rate', data['speaking_rate']), *RATE_RANGE)
    if data.get('steer_hold') is not None and _number('steer_hold', data['steer_hold']) < 1.0:
        raise ValueError('steer_hold < 1 is not available: the steering bundle has no steer_release (controls are '
                         'a constant push); use a lower dose instead')
    return body, applied
