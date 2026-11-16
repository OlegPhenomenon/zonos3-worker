import base64, os, subprocess, threading, time, uuid
import requests, runpod

boot_id = str(uuid.uuid4())
started = time.monotonic()
server = subprocess.Popen(['/app/build/zonos3-server','/app/models/backbone-f16.gguf','--gpu','--zac','/app/models/zac.gguf','--zac-encoder','/app/models/zac-encoder.gguf','--ssl-teacher','/app/models/ssl-teacher.gguf','--port','1919'], stdout=None, stderr=None)
while True:
    if server.poll() is not None:
        raise RuntimeError('Zonos server exited during startup')
    try:
        if requests.get('http://127.0.0.1:1919/v2/capabilities', timeout=2).ok: break
    except requests.RequestException: pass
    if time.monotonic()-started > 600: raise RuntimeError('Zonos startup timeout')
    time.sleep(1)
load_s = time.monotonic()-started

def handler(job):
    data=job['input']
    t=time.monotonic()
    request={'text':data['text'], 'speaker_audio_base64':data['reference_wav_base64'], 'speaker_mode':'reference','decoder':'full_mr','language':data['language'],'stream':False,'format':'wav','seed':42}
    samples=[]; done=threading.Event()
    def sample():
        while not done.is_set():
            try: samples.append(int(subprocess.check_output(['nvidia-smi','--query-gpu=memory.used','--format=csv,noheader,nounits'], text=True).splitlines()[0]))
            except Exception: pass
            done.wait(.25)
    thread=threading.Thread(target=sample,daemon=True);thread.start()
    try: r=requests.post('http://127.0.0.1:1919/v2/generate',json=request,timeout=1200)
    finally: done.set();thread.join()
    r.raise_for_status()
    gpu=subprocess.check_output(['nvidia-smi','--query-gpu=name,memory.used,memory.total','--format=csv,noheader,nounits'],text=True).strip()
    return {'wav_base64':base64.b64encode(r.content).decode(), 'generation_s':time.monotonic()-t, 'model_load_s':load_s, 'boot_id':boot_id, 'peak_vram_sampled_mib':max(samples) if samples else None, 'gpu':gpu, 'decoder':r.headers.get('X-Decoder')}
runpod.serverless.start({'handler':handler})
