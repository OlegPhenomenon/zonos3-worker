"""Shared by the RunPod handler and the Vast PyWorker: run the local zonos3-server and call it.
Voices: /app/voices/<language>/<voice>.ogg (bank: the same Gemini voice name in every language, cloned one-shot), or
an inline reference WAV. A voice must come from a clip in the target language: cross-language clones drift and truncate."""
import base64, io, subprocess, time, wave
from pathlib import Path

import requests

URL = 'http://127.0.0.1:1919'
VOICES = Path('/app/voices')
MAX_ATTEMPTS = 3          # Zonos sometimes stops early; a different seed fixes it
MIN_CHARS_PER_S = 25.0    # shortest plausible speech for the text: fewer seconds than chars/25 = truncated
load_s = None
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
    body = {'text': text, 'language': language, 'stream': False, 'format': 'wav', 'speaker_mode': 'embedding'}
    if data.get('voice'):
        body['speaker_embedding_id'] = _speaker_id(language, data['voice'])
    elif data.get('reference_wav_base64'):
        body['speaker_audio_base64'] = data['reference_wav_base64']
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
            'generation_s': time.monotonic() - t, 'model_load_s': load_s, 'gpu': gpu}
