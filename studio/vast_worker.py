"""Vast.ai Serverless mode, the studio-prep worker (docs/architecture/studio-dub-realtime.md): the Vast PyWorker (vastai SDK)
on WORKER_PORT (3000, TLS with the Vast-signed instance certificate), remote dispatch like slim/vast_worker.py.

Routes (payload -> answer, all JSON; the SDK wraps the answer as {"result": answer}):
  /prepare {audioUrl, audioHash}        -> {"pending": true} while the file is being prepared in the background (ask again),
                                           then {audioHash, durationMs, speechSpans, speechStem, timings, cached}
  /cues    {audioHash, parts}           -> {audioHash, parts: {id: {pitch_st, range_st, loud_db, rate_sps, clip_ogg}}, speakers: {spk: b64}}
  /mix     {audioHash, segment, segmentMs, parts, nonverbal?: [[startMs, endMs]]} -> {segment, audio, timings}
Binary values are base64. A failure is an answer {"error": "<code>", "detail": "..."} (the SDK turns an exception into a bare 500, and
a remote-dispatch answer is always HTTP 200). Codes: bad_request, not_prepared (unknown audioHash: run /prepare), mix_unavailable
(studio.mix.api is missing), too_long, download_failed, prepare_failed, mix_failed.

Disk cache: CACHE_DIR/<audioHash>/ {mix.flac, speech.flac, bg.flac, speech.opus, prepare.json, analysis/}, least recently used evicted
above STUDIO_CACHE_GB (default 40). The directory appears atomically (renamed from <hash>.tmp) only after a complete prepare.
"""
import asyncio, base64, contextlib, ipaddress, json, logging, os, re, shutil, subprocess, threading, time, uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests

log = logging.getLogger('studio.vast')
SDK_ENV_DEFAULTS = {'REPORT_ADDR': 'https://run.vast.ai', 'WORKER_PORT': '3000', 'USE_SSL': 'true'}
SIGN_CERT_URL = 'https://console.vast.ai/api/v0/sign_cert/'
CERT_FILE, KEY_FILE = Path('/etc/instance.crt'), Path('/etc/instance.key')

CACHE_DIR = Path(os.environ.get('STUDIO_CACHE_DIR', '/var/studio-cache'))
CACHE_BYTES = int(float(os.environ.get('STUDIO_CACHE_GB', '40')) * 1e9)
MODEL = os.environ.get('STUDIO_MODEL', '/app/models/bandit_speech.onnx')
MAX_AUDIO_S = float(os.environ.get('STUDIO_MAX_AUDIO_MIN', '180')) * 60  # RAM: the separator holds ~4 float32 copies of the stereo mix
DOWNLOAD_TIMEOUT_S = int(os.environ.get('STUDIO_DOWNLOAD_TIMEOUT_S', '300'))
HASH_RE = re.compile(r'^[A-Za-z0-9_-]{6,128}$')
BENCH_URL = 'bench:'  # the Vast startup benchmark: the first minute of the mix library's dry speech reference, no download

try:
    from studio.mix import api as mix_api
    from studio.mix.api import NotPrepared
    MIX_ERROR = None
except Exception as e:  # missing or broken mix library: /prepare, /cues and /mix answer mix_unavailable, the worker stays up
    mix_api, MIX_ERROR = None, repr(e)

    class NotPrepared(Exception):
        pass

_sep = None
_gpu = threading.Lock()      # one separation at a time: the model owns the card
_cpu = ThreadPoolExecutor(max_workers=min(os.cpu_count() or 2, 4), thread_name_prefix='studio-cpu')
_prepare_jobs: dict[str, asyncio.Future] = {}  # audioHash -> preparation running in the background (/prepare answers pending)
_active: dict[str, int] = {}  # audioHash -> requests using its directory right now (never evicted)
_cache_lock = threading.Lock()


class Fail(Exception):
    def __init__(self, code, detail=''):
        super().__init__(f'{code}: {detail}')
        self.code, self.detail = code, detail


def answer_error(e: Fail) -> dict:
    return {'error': e.code, 'detail': str(e.detail)[:500]}


# ---------------------------------------------------------------- environment, certificate (as slim/vast_worker.py)
def prepare_env() -> str:
    for k, v in SDK_ENV_DEFAULTS.items():
        os.environ.setdefault(k, v)
    port = os.environ['WORKER_PORT']
    missing = [k for k in ('CONTAINER_ID', 'PUBLIC_IPADDR', f'VAST_TCP_PORT_{port}') if not os.environ.get(k)]
    if missing:
        raise SystemExit(f"missing env: {', '.join(missing)} (set by Vast for a container started with -p {port}:{port})")
    return os.environ['CONTAINER_ID']


def sign_certificate(container_id: str) -> None:
    from cryptography import x509
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import rsa
    from cryptography.x509.oid import NameOID

    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    csr = (x509.CertificateSigningRequestBuilder()
           .subject_name(x509.Name([x509.NameAttribute(NameOID.COUNTRY_NAME, 'US'),
                                    x509.NameAttribute(NameOID.STATE_OR_PROVINCE_NAME, 'CA'),
                                    x509.NameAttribute(NameOID.COMMON_NAME, 'pyworker.vast.ai')]))
           .add_extension(x509.BasicConstraints(ca=False, path_length=None), critical=False)
           .add_extension(x509.KeyUsage(digital_signature=True, content_commitment=True, key_encipherment=True,
                                        data_encipherment=False, key_agreement=False, key_cert_sign=False,
                                        crl_sign=False, encipher_only=False, decipher_only=False), critical=False)
           .add_extension(x509.SubjectAlternativeName([x509.IPAddress(ipaddress.ip_address('0.0.0.0'))]),
                          critical=False)
           .sign(key, hashes.SHA256()))
    pem = csr.public_bytes(serialization.Encoding.PEM)
    for attempt in range(5):
        try:
            r = requests.post(SIGN_CERT_URL, params={'instance_id': container_id}, data=pem,
                              headers={'Content-Type': 'application/octet-stream'}, timeout=(10, 60))
            if r.ok:
                break
            if r.status_code == 403:  # a refusal, not a transient error: no point retrying
                raise SystemExit(f'could not get the instance TLS certificate signed: HTTP 403 {r.text[:120]}')
            detail = f'HTTP {r.status_code}'
        except requests.RequestException as e:
            detail = str(e)
        if attempt == 4:
            raise SystemExit(f'could not get the instance TLS certificate signed: {detail}')
        time.sleep(2 ** (attempt + 1))
    KEY_FILE.touch(mode=0o600)
    KEY_FILE.write_bytes(key.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8,
                                           serialization.NoEncryption()))
    CERT_FILE.write_bytes(r.content)


# ---------------------------------------------------------------- cache
def stems_dir(audio_hash: str) -> Path:
    return CACHE_DIR / audio_hash


def dir_bytes(p: Path) -> int:
    return sum(f.stat().st_size for f in p.rglob('*') if f.is_file())


def evict(keep: str | None = None) -> None:
    """Delete least recently used prepared directories until the cache fits CACHE_BYTES (the one being prepared or used stays)."""
    with _cache_lock:
        dirs = [d for d in CACHE_DIR.iterdir() if d.is_dir() and not d.name.endswith('.tmp')]
        sizes = {d: dir_bytes(d) for d in dirs}
        total = sum(sizes.values())
        for d in sorted(dirs, key=lambda d: d.stat().st_mtime):
            if total <= CACHE_BYTES:
                break
            if d.name == keep or _active.get(d.name):
                continue
            shutil.rmtree(d, ignore_errors=True)
            total -= sizes[d]
            log.info('evicted %s (%.0f MB)', d.name, sizes[d] / 1e6)


@contextlib.contextmanager
def using(audio_hash: str):
    """Mark the directory in use (no eviction) and refresh its LRU time."""
    with _cache_lock:
        _active[audio_hash] = _active.get(audio_hash, 0) + 1
    try:
        with contextlib.suppress(OSError):
            os.utime(stems_dir(audio_hash))
        yield
    finally:
        with _cache_lock:
            _active[audio_hash] -= 1
            if not _active[audio_hash]:
                del _active[audio_hash]


def need_hash(p: dict) -> str:
    h = p.get('audioHash')
    if not isinstance(h, str) or not HASH_RE.match(h):
        raise Fail('bad_request', 'audioHash must be 6..128 chars of [A-Za-z0-9_-]')
    return h


def need_prepared(p: dict) -> Path:
    if mix_api is None:
        raise Fail('mix_unavailable', MIX_ERROR)
    d = stems_dir(need_hash(p))
    if not (d / 'prepare.json').exists():
        raise Fail('not_prepared', p['audioHash'])
    return d


def need_parts(p: dict) -> list:
    parts = p.get('parts')
    if not isinstance(parts, list):
        raise Fail('bad_request', 'parts must be a list')
    return parts


# ---------------------------------------------------------------- /prepare
def run(cmd: list[str], what: str, timeout: float) -> None:
    r = subprocess.run(cmd, capture_output=True, timeout=timeout)
    if r.returncode:
        raise Fail(what, r.stderr.decode(errors='replace')[-400:])


def download(url: str, dst: Path) -> None:
    """Decode the audio at `url` to stereo 48 kHz FLAC. Only http(s) is followed; the reference clip for the startup benchmark is local."""
    if url.startswith(BENCH_URL):
        ref = Path(__file__).parent / 'mix' / 'assets' / 'dry_reference.flac'
        cmd = ['ffmpeg', '-nostdin', '-v', 'error', '-y', '-t', '60', '-i', str(ref), '-ac', '2', '-ar', '48000', '-c:a', 'flac', str(dst)]
    elif url.startswith(('https://', 'http://')):
        cmd = ['ffmpeg', '-nostdin', '-v', 'error', '-y', '-protocol_whitelist', 'http,https,tcp,tls,crypto',
               '-reconnect', '1', '-reconnect_streamed', '1', '-reconnect_delay_max', '5', '-rw_timeout', '30000000',
               '-i', url, '-vn', '-ac', '2', '-ar', '48000', '-c:a', 'flac', str(dst)]
    else:
        raise Fail('bad_request', 'audioUrl must be http(s)')
    try:
        run(cmd, 'download_failed', DOWNLOAD_TIMEOUT_S)
    except subprocess.TimeoutExpired:
        raise Fail('download_failed', f'timeout after {DOWNLOAD_TIMEOUT_S} s')


def duration_ms(flac: Path) -> int:
    import soundfile as sf
    i = sf.info(str(flac))
    assert i.samplerate == 48000 and i.channels == 2, (i.samplerate, i.channels)
    return round(i.frames * 1000 / i.samplerate)


def encode_stem(speech_flac: Path, dst: Path) -> None:
    """Speech stem for transcription: mono ogg/opus 48 kbit/s."""
    run(['ffmpeg', '-nostdin', '-v', 'error', '-y', '-i', str(speech_flac), '-ac', '1', '-c:a', 'libopus', '-b:a', '48k', '-vbr', 'on',
         str(dst)], 'prepare_failed', 600)


def prepare_blocking(url: str, audio_hash: str) -> dict:
    final = stems_dir(audio_hash)
    tmp = CACHE_DIR / f'{audio_hash}.tmp'
    if (final / 'prepare.json').exists():
        return answer_prepared(final, cached=True)
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    t = {}
    t0 = time.time()
    try:
        s = time.time()
        download(url, tmp / 'mix.flac')
        t['download_s'] = round(time.time() - s, 2)
        dur = duration_ms(tmp / 'mix.flac')
        if dur / 1000 > MAX_AUDIO_S:
            raise Fail('too_long', f'{dur / 60000:.0f} min > {MAX_AUDIO_S / 60:.0f} min')
        evict(keep=audio_hash)
        s = time.time()
        with _gpu:
            t['gpu_wait_s'] = round(time.time() - s, 2)
            s = time.time()
            from . import sep
            sep.separate_file(tmp / 'mix.flac', tmp, _sep)
        t['separate_s'] = round(time.time() - s, 2)
        s = time.time()
        th = threading.Thread(target=lambda: encode_stem(tmp / 'speech.flac', tmp / 'speech.opus'))
        th.start()  # the opus encode runs beside the analysis
        spans = [[int(a), int(b)] for a, b in mix_api.speech_spans(tmp)]
        t['spans_s'] = round(time.time() - s, 2)
        s = time.time()
        stage = mix_api.analyze(tmp)
        t['analyze_s'] = round(time.time() - s, 2)
        th.join()
        if not (tmp / 'speech.opus').exists():
            raise Fail('prepare_failed', 'speech stem encode failed')
        t['analyze_stages'] = stage
        t['total_s'] = round(time.time() - t0, 2)
        (tmp / 'prepare.json').write_text(json.dumps({'durationMs': dur, 'speechSpans': spans, 'timings': t}))
        shutil.rmtree(final, ignore_errors=True)
        tmp.rename(final)
    except Fail:
        shutil.rmtree(tmp, ignore_errors=True)
        raise
    except Exception as e:
        shutil.rmtree(tmp, ignore_errors=True)
        log.exception('prepare failed')
        raise Fail('prepare_failed', repr(e))
    return answer_prepared(final, cached=False)


def answer_prepared(d: Path, cached: bool) -> dict:
    meta = json.loads((d / 'prepare.json').read_text())
    return {'audioHash': d.name, 'durationMs': meta['durationMs'], 'speechSpans': meta['speechSpans'],
            'speechStem': base64.b64encode((d / 'speech.opus').read_bytes()).decode(), 'timings': meta['timings'], 'cached': cached}


def prepare_in_use(url: str, audio_hash: str) -> dict:
    with using(audio_hash):
        return prepare_blocking(url, audio_hash)


async def prepare(**p) -> dict:
    """Starts the preparation of a new audio file in the background and answers {"pending": true} at once; the caller asks
    again (same worker) until the answer is the prepared file. A 10-min file takes about a minute, longer than a proxied
    request may stay open, so the call itself never waits for it."""
    try:
        if mix_api is None:
            raise Fail('mix_unavailable', MIX_ERROR)
        h = need_hash(p)
        url = p.get('audioUrl')
        if not isinstance(url, str) or not url:
            raise Fail('bad_request', 'audioUrl required')
        final = stems_dir(h)
        if (final / 'prepare.json').exists():
            return answer_prepared(final, cached=True)
        job = _prepare_jobs.get(h)
        if job is None:
            job = _prepare_jobs[h] = asyncio.get_running_loop().run_in_executor(_cpu, prepare_in_use, url, h)
        if p.get('wait'):  # the PyWorker benchmark measures the whole preparation
            await asyncio.wait({job})
        if not job.done():
            return {'audioHash': h, 'pending': True}
        _prepare_jobs.pop(h, None)
        return job.result()
    except Fail as e:
        return answer_error(e)


# ---------------------------------------------------------------- /cues, /mix
def cues_blocking(p: dict) -> dict:
    d = need_prepared(p)
    parts = need_parts(p)
    with using(d.name):
        try:
            r = mix_api.part_cues(d, parts)
        except NotPrepared as e:
            raise Fail('not_prepared', e)
        except Exception as e:
            log.exception('cues failed')
            raise Fail('mix_failed', repr(e))
    b64 = lambda b: base64.b64encode(b).decode()
    return {'audioHash': d.name,
            'parts': {k: {**{f: x for f, x in v.items() if f != 'clip_ogg'}, 'clip_ogg': b64(v['clip_ogg'])} for k, v in r['parts'].items()},
            'speakers': {k: b64(v) for k, v in r['speakers'].items()}}


async def cues(**p) -> dict:
    try:
        return await asyncio.get_running_loop().run_in_executor(_cpu, cues_blocking, p)
    except Fail as e:
        return answer_error(e)


def mix_blocking(p: dict) -> dict:
    d = need_prepared(p)
    parts = need_parts(p)
    try:
        segment, segment_ms = int(p['segment']), int(p['segmentMs'])
        parts = [{**x, 'take': base64.b64decode(x['take'])} for x in parts]
        nonverbal = [(int(a), int(b)) for a, b in p.get('nonverbal') or []]
    except (KeyError, TypeError, ValueError) as e:
        raise Fail('bad_request', repr(e))
    s = time.time()
    with using(d.name):
        try:
            m4a = mix_api.render_segment(d, segment, segment_ms, parts, nonverbal)
        except NotPrepared as e:
            raise Fail('not_prepared', e)
        except Exception as e:
            log.exception('mix failed')
            raise Fail('mix_failed', repr(e))
    return {'segment': segment, 'audio': base64.b64encode(m4a).decode(), 'timings': {'render_s': round(time.time() - s, 2)}}


async def mix(**p) -> dict:
    try:
        return await asyncio.get_running_loop().run_in_executor(_cpu, mix_blocking, p)
    except Fail as e:
        return answer_error(e)


# ---------------------------------------------------------------- lifecycle
def load_separator():
    global _sep
    from . import sep
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    for d in CACHE_DIR.glob('*.tmp'):
        shutil.rmtree(d, ignore_errors=True)
    t = time.time()
    _sep = sep.Separator(MODEL)
    if 'CUDAExecutionProvider' not in _sep.sess.get_providers():  # onnxruntime falls back to CPU silently; that is ~50x slower
        raise RuntimeError(f'CUDA provider not active: {_sep.sess.get_providers()}')
    import numpy as np
    _sep.speech(np.zeros((2, 10 * 48000), np.float32))  # cuDNN algorithm selection and allocator warm-up
    log.info('separator ready in %.1f s (%s)', time.time() - t, os.path.basename(MODEL))
    if mix_api is not None:  # numba kernels, the analysis process pool and the reference signal, once (3-4 s), not on the first /prepare
        t = time.time()
        from studio.mix import channel
        channel.prepare()
        log.info('mix library ready in %.1f s', time.time() - t)


@contextlib.asynccontextmanager
async def lifecycle():
    await asyncio.get_running_loop().run_in_executor(None, load_separator)
    yield


def main():
    logging.basicConfig(level=logging.INFO)
    container_id = prepare_env()
    if os.environ['USE_SSL'] == 'true':
        try:
            sign_certificate(container_id)
        except SystemExit as e:
            log.warning('%s; serving plain HTTP', e)  # as in slim/vast_worker.py
            os.environ['USE_SSL'] = 'false'
    from vastai.serverless.server.worker import BenchmarkConfig, HandlerConfig, Worker, WorkerConfig
    # Workload is seconds of video: /prepare costs the file length (the backend may pass durationMs, else a 10 min typical file),
    # a /mix window or /cues call a fixed slice. Only /prepare is benchmarked (the others need a prepared file); a fresh audioHash
    # per run so the disk cache never answers the benchmark.
    bench = BenchmarkConfig(generator=lambda: {'audioUrl': BENCH_URL, 'audioHash': f'bench-{uuid.uuid4().hex[:12]}', 'durationMs': 60000, 'wait': True},
                            runs=2, concurrency=1, do_warmup=False)
    common = dict(is_remote_dispatch=True, allow_parallel_requests=True, max_queue_time=900.0)
    Worker(WorkerConfig(lifecycle=lifecycle(), handlers=[
        HandlerConfig(route='/prepare', remote_function=prepare, benchmark_config=bench,
                      workload_calculator=lambda p: float(p.get('durationMs') or 600000) / 1000, **common),
        HandlerConfig(route='/cues', remote_function=cues, workload_calculator=lambda p: 5.0, **common),
        HandlerConfig(route='/mix', remote_function=mix, workload_calculator=lambda p: 20.0, **common)])).run()


if __name__ == '__main__':
    main()
