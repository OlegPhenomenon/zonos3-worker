"""Load test through the proxy: C simultaneous request streams, real lecture chunks, per-request latency and
audio throughput, grouped by the Vast worker that served each request. Run on the proxy box:
  python3 loadtest.py 1 2 4 8 12        (levels = parallel request streams; ~45 s each)
A viewer of the extension keeps up to 3 requests in flight and consumes ~1.3 s of audio per second (atempo)."""
import json, statistics, sys, threading, time
from pathlib import Path

import requests

PROXY = 'http://127.0.0.1:8090/tts'
TOKEN = next(l.split('=', 1)[1].strip() for l in open('/opt/zonos-proxy/env') if l.startswith('ZONOS_PROXY_TOKEN='))
TEXTS = [r['text_sent'] for r in map(json.loads, open('/opt/zonos-proxy/requests.jsonl'))
         if r.get('language') == 'ru' and 40 <= len(r.get('text_sent', '')) <= 160][-200:]
LEVEL_S = 45


def stream(i, stop, out):
    k = i
    while time.monotonic() < stop:
        text = TEXTS[k % len(TEXTS)]; k += 7
        t = time.monotonic()
        try:
            r = requests.post(PROXY, headers={'X-Zonos-Token': TOKEN}, timeout=300,
                              json={'text': text, 'language': 'ru', 'voice': 'Schedar'})
            out.append({'ok': r.ok, 'lat': time.monotonic() - t, 'audio': float(r.headers.get('X-Audio-S', 0)),
                        'gen': float(r.headers.get('X-Generation-S', 0)), 'end': time.monotonic()})
        except requests.RequestException as e:
            out.append({'ok': False, 'lat': time.monotonic() - t, 'audio': 0, 'gen': 0, 'end': time.monotonic(), 'err': str(e)})


def pct(xs, p):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(p * len(xs)))] if xs else float('nan')


before = Path('/opt/zonos-proxy/requests.jsonl').stat().st_size
for c in map(int, sys.argv[1:]):
    out, t0 = [], time.monotonic()
    threads = [threading.Thread(target=stream, args=(i, t0 + LEVEL_S, out)) for i in range(c)]
    [t.start() for t in threads]; [t.join() for t in threads]
    wall = max(r['end'] for r in out) - t0
    ok = [r for r in out if r['ok']]
    lat, gen = [r['lat'] for r in ok], [r['gen'] for r in ok]
    tput = sum(r['audio'] for r in ok) / wall
    print(json.dumps({'streams': c, 'requests': len(out), 'failed': len(out) - len(ok),
                      'lat_p50': round(pct(lat, .5), 2), 'lat_p90': round(pct(lat, .9), 2), 'lat_max': round(max(lat), 2),
                      'gen_p50': round(pct(gen, .5), 2), 'gen_p90': round(pct(gen, .9), 2),
                      'audio_s_per_s': round(tput, 2), 'viewers_at_1.3x': round(tput / 1.3, 1)}), flush=True)
rows = [json.loads(l) for l in open('/opt/zonos-proxy/requests.jsonl', 'rb').read()[before:].splitlines() if l.strip()]
workers = {}
for r in rows:
    w = workers.setdefault(r.get('worker'), {'n': 0, 'gpu': (r.get('gpu') or '')[:24], 'first': r['utc']})
    w['n'] += 1
print(json.dumps(workers, indent=1))
