"""Shared by the RunPod handler and the Vast PyWorker: run the local zonos3-server and call it.
Voices: /app/voices/<language>/<voice>.ogg (bank: the same Gemini voice name in every language, cloned one-shot), or
an inline reference WAV. A voice must come from a clip in the target language: cross-language clones drift and truncate."""
import base64, ctypes, io, re, subprocess, time, wave
from pathlib import Path

import requests

URL = 'http://127.0.0.1:1919'
VOICES = Path('/app/voices')
MAX_ATTEMPTS = 3          # Zonos sometimes stops early; a different seed fixes it
MIN_CHARS_PER_S = 25.0    # shortest plausible speech for the text: fewer seconds than chars/25 = truncated
load_s = None
TUNABLE = {'speaker_cfg_scale', 'temperature', 'topk', 'top_p', 'min_p', 'condition_values', 'repetition_penalty',
           'repetition_window', 'silence_prefill_frames'}
gpu = None


def start_server():
    global load_s, gpu
    started = time.monotonic()
    gpu = subprocess.run(['nvidia-smi', '--query-gpu=name,driver_version', '--format=csv,noheader'],
                         capture_output=True, text=True).stdout.strip()
    server = subprocess.Popen(['/app/build/zonos3-server', '/app/models/backbone-ptq.gguf', '--gpu',
                               '--zac', '/app/models/zac-keep2.gguf', '--spk-encoder', '/app/models/spk-encoder.gguf',
                               '--port', '1919'])
    while True:
        if server.poll() is not None:
            raise RuntimeError('Zonos server exited during startup')
        try:
            if requests.get(URL + '/v2/capabilities', timeout=2).ok:
                break
        except requests.RequestException:
            pass
        if time.monotonic() - started > 300:
            raise RuntimeError('Zonos startup timeout')
        time.sleep(0.2)
    load_s = time.monotonic() - started


def _duration(wav_bytes: bytes) -> float:
    with wave.open(io.BytesIO(wav_bytes)) as w:
        return w.getnframes() / w.getframerate()


# Latin words inside non-Latin-script text: the server's text normalizer (zyphra-textnorm, NeMo grammars) spells
# every all-Latin token letter by letter ("Python" -> "пи уай ти эйч оу эн", ru/whitelist.rs), while the model reads
# the raw word as a word (Gemini transcript: "Пайтон", "Докер"; "API" -> "эй-пи-ай"). So for these languages the
# worker normalizes the text itself with each Latin word wrapped as a pronunciation group (U+E000 … U+E001, which
# the normalizer keeps verbatim), unwraps, and sends it with the server's normalization off. Numbers and dates are
# still verbalized by the same grammar. libzyphra_textnorm.so = the server's own static library, linked shared.
NON_LATIN_SCRIPT = {'ru': 'ru', 'uk': 'uk', 'bg': 'bg', 'el': 'el', 'ka': 'ka', 'he': 'he', 'ar': 'ar', 'ur': 'ur',
                    'hi': 'hi', 'mr': 'mr', 'bn': 'bn', 'gu': 'gu', 'pa': 'pa', 'or': 'or', 'kn': 'kn', 'ml': 'ml',
                    'ta': 'ta', 'te': 'te', 'th': 'th', 'ja': 'ja', 'ko': 'ko', 'cmn': 'zh'}  # code -> nemo grammar
LATIN_WORD = re.compile(r"[A-Za-z][A-Za-z0-9]*(?:[.+#'-][A-Za-z0-9+#]+)*\+*")
_tn_lib = ctypes.CDLL('/app/lib/libzyphra_textnorm.so')
_tn_lib.zyphra_textnorm_new.restype = ctypes.c_void_p
_tn_lib.zyphra_textnorm_new.argtypes = [ctypes.c_char_p, ctypes.c_size_t]
_tn_lib.zyphra_textnorm_normalize.restype = ctypes.c_void_p
_tn_lib.zyphra_textnorm_normalize.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_size_t, ctypes.c_char_p, ctypes.c_size_t]
_tn_lib.zyphra_textnorm_free.argtypes = [ctypes.c_void_p]
_tn = _tn_lib.zyphra_textnorm_new(None, 0)


def keep_latin_words(text: str, nemo: str) -> str:
    wrapped = LATIN_WORD.sub(lambda m: '\ue000' + m.group() + '\ue001', text).encode()
    out = _tn_lib.zyphra_textnorm_normalize(_tn, wrapped, len(wrapped), nemo.encode(), len(nemo))
    try:
        return ctypes.string_at(out).decode().replace('\ue000', '').replace('\ue001', '')
    finally:
        _tn_lib.zyphra_textnorm_free(out)


_speakers = {}   # (language, voice) -> speaker id registered on this server; encoded once (~0.15 s on GPU)
SESSION = {'X-TTS-Session-ID': 'bank'}


def _speaker_id(language: str, voice: str) -> str:
    key = (language, voice)
    if key not in _speakers:
        clip = VOICES / language / f'{voice}.ogg'
        if not clip.is_file():
            raise ValueError(f'no voice {voice!r} for language {language!r}')
        r = requests.post(URL + '/v2/speakers', headers=SESSION, timeout=60,
                          json={'label': f'{language}/{voice}', 'speaker_audio_base64': base64.b64encode(clip.read_bytes()).decode()})
        if not r.ok:
            raise RuntimeError(f'zonos3-server HTTP {r.status_code}: {r.text[:300]}')
        _speakers[key] = r.json()['id']
    return _speakers[key]


def generate(data: dict) -> dict:
    text = data['text']
    language = data.get('language', 'ru')
    # Uniform chunks: reference CFG 1.5 + fixed loudness (CPU A/B 2026-11-16: speaker cosine 0.795 -> 0.849, loudness
    # spread 2.1 -> 0.5 dB SD). temperature 0.5 cut the pitch spread between chunks 6.1 -> 3.5 semitones, and a fixed
    # speaking_rate keeps every chunk at one pace; the client speeds speech up itself (atempo). Both chosen by ear
    # on whole lectures (2026-11-17): model-side pace changes hurt quality.
    body = {'text': text, 'language': language, 'stream': False, 'format': 'wav', 'speaker_mode': 'embedding',
            'temperature': 0.5, 'condition_values': {'loudness': -20.0, 'speaking_rate': 22.0}}
    nemo = NON_LATIN_SCRIPT.get(language)
    if nemo and LATIN_WORD.search(text):
        body.update(text=keep_latin_words(text, nemo), text_normalization=False)
    if data.get('voice'):
        body['speaker_embedding_id'] = _speaker_id(language, data['voice'])
    elif data.get('reference_wav_base64'):
        body['speaker_audio_base64'] = data['reference_wav_base64']
    if 'speaker_embedding_id' in body or 'speaker_audio_base64' in body:
        body['speaker_cfg_scale'] = 1.5  # the server refuses speaker CFG without a speaker (HTTP 400)
    # Tuning without an image rebuild: the caller (the proxy's params.json) may override these sampling fields.
    body.update({k: v for k, v in (data.get('params') or {}).items() if k in TUNABLE})
    seed = int(data.get('seed', 42))
    t = time.monotonic()
    for attempt in range(MAX_ATTEMPTS):
        r = requests.post(URL + '/v2/generate', headers=SESSION, json={**body, 'seed': seed + attempt}, timeout=600)
        if not r.ok:
            raise RuntimeError(f'zonos3-server HTTP {r.status_code}: {r.text[:300]}')
        audio_s = _duration(r.content)
        if audio_s >= len(text) / MIN_CHARS_PER_S:
            break
    return {'wav_base64': base64.b64encode(r.content).decode(), 'audio_s': audio_s, 'attempts': attempt + 1,
            'generation_s': time.monotonic() - t, 'model_load_s': load_s, 'gpu': gpu,
            'text_sent': body['text'], 'text_normalization': body.get('text_normalization', True)}
