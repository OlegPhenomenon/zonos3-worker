"""Production Zonos gateway for the browser extensions (Udemy first): license check and per-license hour quota via
the Rails internal studio-voice API, synthesis on the Vast endpoint `zonos3-tts`, WAV back to the extension.
Runs on Contabo behind Caddy (https://zonos.mylittleindiestudio.com).

POST /tts   {text, language, voice} + `Authorization: Bearer <license key>` + `X-Fingerprint`
            -> audio/wav, X-Audio-S, X-Remaining-S.  401 invalid_license, 402 quota_exhausted {resets_at},
            403 not_entitled, 422 unsupported_language / invalid_request, 429 too_many_in_flight, 503 waking {eta_s},
            502 upstream_failed.
GET /status -> {state: awake|asleep|waking|none, eta_s, gpu}; public, never wakes the GPU.

Quota: the gateway holds one Rails lease (up to 120 s of audio, 5 min) per license+fingerprint and spends it phrase by
phrase; a phrase that needs more than the lease has left settles it and opens a new one. A lease idle for 60 s is
settled with the seconds actually used. The last phrase may overrun its lease by a few seconds; that is forgiven.
Audio is not stored. Env: ZONOS_ENDPOINT_KEY, RAILS_STUDIO_VOICE_URL, METERING_SHARED_SECRET, [GATEWAY_PORT=8093].
"""
import base64, json, logging, os, threading, time, urllib.request, uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import requests

ENDPOINT, ENDPOINT_ID = 'zonos3-tts', 44925
KEY = os.environ['ZONOS_ENDPOINT_KEY']
RAILS = os.environ['RAILS_STUDIO_VOICE_URL'].rstrip('/')    # https://mylittleindiestudio.com/api/internal/studio_voice
METER_SECRET = os.environ['METERING_SHARED_SECRET']
PORT = int(os.environ.get('GATEWAY_PORT', '8093'))
CA = Path(__file__).resolve().parent / 'jvastai_root.cer'   # Vast signs worker certificates with its own root
if not CA.exists():
    CA.write_bytes(urllib.request.urlopen('https://console.vast.ai/static/jvastai_root.cer').read())

ZONOS = set('en_us en_gb fr_fr de es it pt_br ja cmn ko ar bn bg cs da nl fi ka el gu he hi hu id kn ms ml mr no or pl pa '
            'ro ru sk sv tl ta te th tr uk ur vi'.split())
ALIASES = {'en': 'en_us', 'fr': 'fr_fr', 'pt': 'pt_br', 'zh': 'cmn', 'nb': 'no', 'nn': 'no', 'fil': 'tl', 'iw': 'he'}
VOICES = {'Kore', 'Aoede', 'Callirrhoe', 'Despina', 'Sulafat', 'Charon', 'Iapetus', 'Algieba', 'Schedar', 'Sadaltager'}
MAX_CHARS = 400            # tested up to ~170; the extension splits longer text into sentences
MAX_IN_FLIGHT = 3          # per license: the extension keeps up to 3 phrases in flight
WAKE_WAIT_S = 110          # route waits this long for a sleeping worker, then 503 waking (client retries)
CHARS_PER_AUDIO_S = 13.7   # measured over 1701 phrases at speaking_rate 22
LEASE_IDLE_S = 60
WAKE_CACHED_S, WAKE_FRESH_S = 20, 90
AWAKE = {'idle', 'running', 'busy'}
log = logging.getLogger('zonos-gateway')


class Reject(Exception):
    def __init__(self, status: int, body: dict):
        super().__init__(body.get('error'))
        self.status, self.body = status, body


def zonos_language(lang: str) -> str:
    lang = lang.lower().replace('-', '_')
    if lang in ZONOS:
        return lang
    base = lang.split('_')[0]
    code = ALIASES.get(base, base)
    if code not in ZONOS:
        raise Reject(422, {'error': 'unsupported_language'})
    return code


# ---- quota (Rails leases) -------------------------------------------------------------------------------------------
def rails(path: str, body: dict) -> dict:
    r = requests.post(f'{RAILS}/{path}', json=body, headers={'X-Meter-Secret': METER_SECRET}, timeout=15)
    if r.status_code in (401, 402, 403):
        raise Reject(r.status_code, r.json())
    r.raise_for_status()
    return r.json()


class Wallet:
    """Lease of one license+fingerprint. All fields are guarded by `lock`."""

    def __init__(self, license_key: str, fingerprint: str):
        self.license_key, self.fingerprint = license_key, fingerprint
        self.lock = threading.Lock()
        self.lease = None          # {'id', 'granted_ms', 'expires': monotonic}
        self.used_ms = 0
        self.usage: list[dict] = []
        self.remaining_s = 0
        self.in_flight = 0
        self.last_used = time.monotonic()

    def _settle(self) -> None:
        if self.lease:
            res = rails(f"leases/{self.lease['id']}/settle", {'used_ms': min(self.used_ms, self.lease['granted_ms']),
                                                              'final': True, 'usage': self.usage})
            self.remaining_s = res.get('remaining_seconds', self.remaining_s)
        self.lease, self.used_ms, self.usage = None, 0, []

    def reserve(self, need_ms: int) -> None:
        """Make sure the current lease can cover a phrase of ~need_ms (or open a new one)."""
        with self.lock:
            lease = self.lease
            if lease and (lease['granted_ms'] - self.used_ms >= need_ms) and lease['expires'] - time.monotonic() > 30:
                return
            self._settle()
            res = rails('leases', {'license_key': self.license_key, 'fingerprint': self.fingerprint,
                                   'extension': 'udemy', 'request_id': uuid.uuid4().hex})
            self.lease = {'id': res['lease_id'], 'granted_ms': res['granted_ms'], 'expires': time.monotonic() + 270}
            self.remaining_s = res.get('remaining_seconds', 0)

    def spend(self, audio_ms: int, row: dict) -> int:
        """Book a finished phrase; returns seconds left: Rails' balance after the lease plus what the lease has left."""
        with self.lock:
            self.used_ms += audio_ms
            self.usage.append(row)
            self.last_used = time.monotonic()
            left_in_lease = (self.lease['granted_ms'] - self.used_ms) if self.lease else 0
            return max(0, self.remaining_s + left_in_lease // 1000)

    def settle_if_idle(self) -> None:
        with self.lock:
            if self.lease and self.in_flight == 0 and time.monotonic() - self.last_used > LEASE_IDLE_S:
                self._settle()


wallets: dict[tuple[str, str], Wallet] = {}
wallets_lock = threading.Lock()


def wallet(license_key: str, fingerprint: str) -> Wallet:
    with wallets_lock:
        return wallets.setdefault((license_key, fingerprint), Wallet(license_key, fingerprint))


def sweeper() -> None:
    while True:
        time.sleep(15)
        with wallets_lock:
            items = list(wallets.items())
        for key, w in items:
            try:
                w.settle_if_idle()
            except Exception as e:  # noqa: BLE001 — Rails charges an unsettled lease in full when it expires
                log.warning('settle failed: %s', e)
            with wallets_lock:  # forget wallets with nothing open; they are rebuilt on the next request
                if w.lease is None and w.in_flight == 0 and time.monotonic() - w.last_used > 600:
                    wallets.pop(key, None)


# ---- Vast -----------------------------------------------------------------------------------------------------------
def worker_status() -> dict:
    """Read-only: get_endpoint_workers does not route a request, so it never wakes a worker."""
    workers = requests.post('https://run.vast.ai/get_endpoint_workers/', json={'id': ENDPOINT_ID, 'api_key': KEY},
                            timeout=10).json()
    if not isinstance(workers, list):
        raise ValueError(f'unexpected Vast answer: {str(workers)[:120]}')
    if not workers:
        return {'state': 'none', 'eta_s': WAKE_FRESH_S, 'gpu': None}
    w = next((w for w in workers if w['status'] in AWAKE), None) or next(
        (w for w in workers if w['status'] != 'stopped'), None) or workers[0]
    state = 'awake' if w['status'] in AWAKE else 'asleep' if w['status'] == 'stopped' else 'waking'
    eta = 0 if state == 'awake' else WAKE_CACHED_S if w.get('ready_ever') else WAKE_FRESH_S
    return {'state': state, 'eta_s': eta, 'gpu': w.get('gpu_name')}


def synthesize(payload: dict) -> dict:
    t0, idx = time.monotonic(), 0
    while True:  # a sleeping worker answers without a url until it is ready
        g = requests.post('https://run.vast.ai/route/', params={'api_key': KEY}, headers={'Authorization': f'Bearer {KEY}'},
                          json={'endpoint': ENDPOINT, 'api_key': KEY, 'cost': 100, 'request_idx': idx, 'replay_timeout': 60.0},
                          timeout=10).json()
        idx = g.get('request_idx') or idx
        if g.get('url'):
            break
        if time.monotonic() - t0 > WAKE_WAIT_S:
            raise Reject(503, {'error': 'waking', 'eta_s': WAKE_CACHED_S})
        time.sleep(1)
    for attempt in range(2):  # one retry on a worker failure, through a fresh route
        try:
            r = requests.post(g['url'] + '/tts', params={'api_key': KEY}, headers={'Authorization': f'Bearer {KEY}'},
                              json={'auth_data': g, 'session_id': None, 'payload': payload}, timeout=120, verify=str(CA))
            r.raise_for_status()
            res = r.json()
            return {**res.get('result', res), 'worker': g['url'], 'request_s': time.monotonic() - t0}
        except requests.RequestException:
            if attempt:
                raise
            g = requests.post('https://run.vast.ai/route/', params={'api_key': KEY}, headers={'Authorization': f'Bearer {KEY}'},
                              json={'endpoint': ENDPOINT, 'api_key': KEY, 'cost': 100, 'request_idx': 0, 'replay_timeout': 60.0},
                              timeout=10).json()
            if not g.get('url'):
                raise


# ---- HTTP -----------------------------------------------------------------------------------------------------------
class Handler(BaseHTTPRequestHandler):
    def reply(self, code: int, body: bytes, ctype='application/json', headers=None):
        self.send_response(code)
        self.send_header('Content-Type', ctype); self.send_header('Content-Length', str(len(body)))
        self.cors()
        for k, v in (headers or {}).items():
            self.send_header(k, str(v))
        self.end_headers(); self.wfile.write(body)

    def cors(self):
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Expose-Headers', 'X-Audio-S, X-Remaining-S')

    def do_OPTIONS(self):
        self.send_response(204)
        self.cors()
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Authorization, X-Fingerprint, Content-Type')
        self.send_header('Access-Control-Max-Age', '86400')
        self.end_headers()

    def do_GET(self):
        if self.path != '/status':
            return self.reply(404, b'{"error":"not_found"}')
        try:
            return self.reply(200, json.dumps(worker_status()).encode(), headers={'Cache-Control': 'no-store'})
        except Exception as e:  # noqa: BLE001 — advisory; the client shows nothing on error
            return self.reply(502, json.dumps({'error': f'{type(e).__name__}'}).encode())

    def do_POST(self):
        if self.path != '/tts':
            return self.reply(404, b'{"error":"not_found"}')
        w, started = None, time.monotonic()
        try:
            auth = self.headers.get('Authorization', '')
            license_key = auth[7:].strip() if auth.startswith('Bearer ') else ''
            fingerprint = (self.headers.get('X-Fingerprint') or '').strip()
            if not license_key or not 8 <= len(fingerprint) <= 128:
                raise Reject(401, {'error': 'invalid_license'})
            try:
                data = json.loads(self.rfile.read(min(int(self.headers.get('Content-Length', 0)), 65536)))
                text, voice = str(data['text']).strip(), str(data['voice'])
                language = zonos_language(str(data['language']))
            except (ValueError, KeyError, TypeError):
                raise Reject(422, {'error': 'invalid_request'})
            if not 0 < len(text) <= MAX_CHARS or voice not in VOICES:
                raise Reject(422, {'error': 'invalid_request'})
            w = wallet(license_key, fingerprint)
            with w.lock:
                if w.in_flight >= MAX_IN_FLIGHT:
                    w = None
                    raise Reject(429, {'error': 'too_many_in_flight'})
                w.in_flight += 1
                w.last_used = time.monotonic()  # keeps the sweeper from dropping a wallet in use
            w.reserve(int(len(text) / CHARS_PER_AUDIO_S * 1000))
            res = synthesize({'text': text, 'language': language, 'voice': voice})
            wav = base64.b64decode(res['wav_base64'])
            audio_ms = int(res['audio_s'] * 1000)
            left = w.spend(audio_ms, {'language': language, 'voice': voice, 'chars': len(text), 'audio_ms': audio_ms,
                                      'generation_ms': int(res['generation_s'] * 1000), 'gpu': res.get('gpu')})
            self.reply(200, wav, 'audio/wav', {'X-Audio-S': round(res['audio_s'], 3), 'X-Remaining-S': left})
            log.info(json.dumps({'ok': 1, 'lang': language, 'chars': len(text), 'audio_s': round(res['audio_s'], 2),
                                 'gen_s': round(res['generation_s'], 2), 'req_s': round(time.monotonic() - started, 2),
                                 'gpu': (res.get('gpu') or '')[:24], 'attempts': res.get('attempts')}))
        except (BrokenPipeError, ConnectionResetError):
            return  # the client went away (aborted phrase); the audio is still charged, it was generated
        except Reject as e:
            self.reply(e.status, json.dumps(e.body).encode())
        except Exception as e:  # noqa: BLE001 — any upstream failure is a 502 for the client, which retries
            log.warning('tts failed: %s: %s', type(e).__name__, str(e)[:200])
            self.reply(502, b'{"error":"upstream_failed"}')
        finally:
            if w:
                with w.lock:
                    w.in_flight -= 1

    def log_message(self, *args):
        pass


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    threading.Thread(target=sweeper, daemon=True).start()
    ThreadingHTTPServer(('127.0.0.1', PORT), Handler).serve_forever()
