"""Test-only HTTPS-fronted proxy (Caddy terminates TLS) from the browser extension to the Vast endpoint `zonos3-tts`.
Why: the Vast worker's certificate is signed by Vast's own root, which browsers reject, and the endpoint key
must not ship inside the extension.
POST /tts {text, language, voice, seed?} with X-Zonos-Token -> audio/wav; every request is logged to requests.jsonl.
GET /status -> worker state + wake estimate (CORS open: read by the content script; does not wake the worker).
GET /stats -> totals. Env: ZONOS_ENDPOINT_KEY, ZONOS_PROXY_TOKEN."""
import base64, json, os, threading, time, urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import requests

ENDPOINT = 'zonos3-tts'
ENDPOINT_ID = 44925
KEY = os.environ['ZONOS_ENDPOINT_KEY']
TOKEN = os.environ['ZONOS_PROXY_TOKEN']
HERE = Path(__file__).resolve().parent
LOG = HERE / 'requests.jsonl'
CA = HERE / 'jvastai_root.cer'
if not CA.exists():
    CA.write_bytes(urllib.request.urlopen('https://console.vast.ai/static/jvastai_root.cer').read())
ZONOS = set('en_us en_gb fr_fr de es it pt_br ja cmn ko ar bn bg cs da nl fi ka el gu he hi hu id kn ms ml mr no or pl pa '
            'ro ru sk sv tl ta te th tr uk ur vi'.split())
ALIASES = {'en': 'en_us', 'fr': 'fr_fr', 'pt': 'pt_br', 'zh': 'cmn', 'nb': 'no', 'nn': 'no', 'fil': 'tl', 'iw': 'he'}
lock = threading.Lock()
# Measured wake times (probes.jsonl, 2026-11-16): a stopped worker whose host already holds the image ~17-19 s;
# a host that must pull the 3.6 GB image first ~72-93 s.
WAKE_CACHED_S, WAKE_FRESH_S = 20, 90
AWAKE = {'idle', 'running', 'busy'}


def worker_status() -> dict:
    """Read-only: get_endpoint_workers does not route a request, so it never wakes a worker."""
    workers = requests.post('https://run.vast.ai/get_endpoint_workers/', json={'id': ENDPOINT_ID, 'api_key': KEY},
                            timeout=10).json()
    if not isinstance(workers, list):  # seen once mid-wake: Vast answered with a non-list body
        raise ValueError(f'unexpected Vast answer: {str(workers)[:120]}')
    if not workers:
        return {'state': 'none', 'eta_s': WAKE_FRESH_S, 'gpu': None, 'vast_status': None}
    w = next((w for w in workers if w['status'] in AWAKE), None) or next(
        (w for w in workers if w['status'] != 'stopped'), None) or workers[0]
    state = 'awake' if w['status'] in AWAKE else 'asleep' if w['status'] == 'stopped' else 'waking'
    eta = 0 if state == 'awake' else WAKE_CACHED_S if w.get('ready_ever') else WAKE_FRESH_S
    return {'state': state, 'eta_s': eta, 'gpu': w.get('gpu_name'), 'vast_status': w['status']}


def zonos_language(lang: str) -> str:
    lang = lang.lower().replace('-', '_')
    if lang in ZONOS:
        return lang
    base = lang.split('_')[0]
    code = ALIASES.get(base, base)
    if code not in ZONOS:
        raise ValueError(f'language {lang!r} is not supported by Zonos')
    return code


def synthesize(payload: dict) -> tuple[bytes, dict]:
    t0 = time.monotonic(); idx = 0
    while True:  # a cold worker answers without a url until it is ready
        g = requests.post('https://run.vast.ai/route/', params={'api_key': KEY}, headers={'Authorization': f'Bearer {KEY}'},
                          json={'endpoint': ENDPOINT, 'api_key': KEY, 'cost': 100, 'request_idx': idx, 'replay_timeout': 60.0},
                          timeout=10).json()
        idx = g.get('request_idx') or idx
        if g.get('url'):
            break
        if time.monotonic() - t0 > 300:
            raise TimeoutError(f"no Vast worker within 300 s: {g.get('status')}")
        time.sleep(1)
    route_s = time.monotonic() - t0
    r = requests.post(g['url'] + '/tts', params={'api_key': KEY}, headers={'Authorization': f'Bearer {KEY}'},
                      json={'auth_data': g, 'session_id': None, 'payload': payload}, timeout=300, verify=str(CA))
    r.raise_for_status()
    res = r.json(); res = res.get('result', res)
    wav = base64.b64decode(res.pop('wav_base64'))
    return wav, {'route_s': round(route_s, 2), 'request_s': round(time.monotonic() - t0, 2), 'worker': g['url'], **res}


class Handler(BaseHTTPRequestHandler):
    def reply(self, code: int, body: bytes, ctype='application/json', headers=None):
        self.send_response(code)
        self.send_header('Content-Type', ctype); self.send_header('Content-Length', str(len(body)))
        for k, v in (headers or {}).items():
            self.send_header(k, str(v))
        self.end_headers(); self.wfile.write(body)

    def do_GET(self):
        if self.path == '/status':
            try:
                body = json.dumps(worker_status())
            except Exception as e:  # noqa: BLE001 — status is advisory; the caller shows nothing on error
                return self.reply(502, json.dumps({'error': f'{type(e).__name__}: {e}'[:200]}).encode(),
                                  headers={'Access-Control-Allow-Origin': '*'})
            return self.reply(200, body.encode(), headers={'Access-Control-Allow-Origin': '*', 'Cache-Control': 'no-store'})
        if self.path != '/stats':
            return self.reply(404, b'{}')
        rows = [json.loads(line) for line in LOG.read_text().splitlines()] if LOG.exists() else []
        ok = [r for r in rows if 'audio_s' in r]
        stats = {'requests': len(rows), 'failed': len(rows) - len(ok),
                 'audio_s': round(sum(r['audio_s'] for r in ok), 1),
                 'generation_s': round(sum(r['generation_s'] for r in ok), 1),
                 'chars': sum(r['chars'] for r in rows), 'first': rows[0]['utc'] if rows else None,
                 'last': rows[-1]['utc'] if rows else None}
        self.reply(200, json.dumps(stats).encode())

    def do_POST(self):
        if self.path != '/tts' or self.headers.get('X-Zonos-Token') != TOKEN:
            return self.reply(403, b'{"error":"forbidden"}')
        row = {'utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}
        try:
            data = json.loads(self.rfile.read(int(self.headers.get('Content-Length', 0))))
            text = str(data['text']).strip()
            payload = {'text': text, 'language': zonos_language(str(data.get('language', 'ru'))),
                       'voice': str(data['voice']), 'seed': int(data.get('seed', 42))}
            row.update(chars=len(text), language=payload['language'], voice=payload['voice'])
            wav, meta = synthesize(payload)
            row.update(meta)
            self.reply(200, wav, 'audio/wav', {'X-Audio-S': meta['audio_s'], 'X-Generation-S': round(meta['generation_s'], 3),
                                               'X-Attempts': meta['attempts']})
        except Exception as e:  # noqa: BLE001 — every failure goes back to the caller and into the log
            row.setdefault('chars', 0); row['error'] = f'{type(e).__name__}: {e}'[:300]
            self.reply(400 if isinstance(e, (ValueError, KeyError)) else 502, json.dumps({'error': row['error']}).encode())
        with lock, open(LOG, 'a') as f:
            f.write(json.dumps(row, ensure_ascii=False) + '\n')

    def log_message(self, *args):
        pass


if __name__ == '__main__':
    ThreadingHTTPServer(('127.0.0.1', 8090), Handler).serve_forever()
