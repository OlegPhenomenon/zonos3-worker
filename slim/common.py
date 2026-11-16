"""Shared by the RunPod handler and the Vast PyWorker: run the local zonos3-server and call it.
Voices: /app/voices/<id>.embedding.npy (precomputed speaker embeddings) or an inline reference WAV."""
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


def generate(data: dict) -> dict:
    text = data['text']
    body = {'text': text, 'language': data.get('language', 'ru'), 'stream': False, 'format': 'wav',
            'speaker_mode': 'embedding'}
    if data.get('voice'):
        body['speaker_embedding_base64'] = base64.b64encode((VOICES / f"{data['voice']}.embedding.npy").read_bytes()).decode()
    elif data.get('reference_wav_base64'):
        body['speaker_audio_base64'] = data['reference_wav_base64']
    seed = int(data.get('seed', 42))
    t = time.monotonic()
    for attempt in range(MAX_ATTEMPTS):
        r = requests.post(URL + '/v2/generate', json={**body, 'seed': seed + attempt}, timeout=600)
        r.raise_for_status()
        audio_s = _duration(r.content)
        if audio_s >= len(text) / MIN_CHARS_PER_S:
            break
    return {'wav_base64': base64.b64encode(r.content).decode(), 'audio_s': audio_s, 'attempts': attempt + 1,
            'generation_s': time.monotonic() - t, 'model_load_s': load_s, 'gpu': gpu}
