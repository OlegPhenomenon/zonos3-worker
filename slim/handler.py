import base64, os, subprocess, time, uuid
import requests, runpod

boot_id = str(uuid.uuid4())
started = time.monotonic()
server = subprocess.Popen(['/app/build/zonos3-server', '/app/models/backbone-ptq.gguf', '--gpu',
                           '--zac', '/app/models/zac-keep2.gguf', '--spk-encoder', '/app/models/spk-encoder.gguf',
                           '--port', '1919'])
while True:
    if server.poll() is not None:
        raise RuntimeError('Zonos server exited during startup')
    try:
        if requests.get('http://127.0.0.1:1919/v2/capabilities', timeout=2).ok:
            break
    except requests.RequestException:
        pass
    if time.monotonic() - started > 300:
        raise RuntimeError('Zonos startup timeout')
    time.sleep(0.2)
load_s = time.monotonic() - started


def handler(job):
    data = job['input']
    t = time.monotonic()
    body = {'text': data['text'], 'language': data.get('language', 'ru'), 'stream': False, 'format': 'wav',
            'seed': data.get('seed', 42), 'speaker_mode': 'embedding'}
    if data.get('reference_wav_base64'):
        body['speaker_audio_base64'] = data['reference_wav_base64']
    r = requests.post('http://127.0.0.1:1919/v2/generate', json=body, timeout=1200)
    r.raise_for_status()
    return {'wav_base64': base64.b64encode(r.content).decode(), 'generation_s': time.monotonic() - t,
            'model_load_s': load_s, 'boot_id': boot_id}


runpod.serverless.start({'handler': handler})
