"""Vast.ai Serverless mode, the studio-prep worker (docs/architecture/studio-dub-realtime.md): the Vast PyWorker (vastai SDK)
on WORKER_PORT (3000, TLS with the Vast-signed instance certificate), remote dispatch like slim/vast_worker.py. The GPU and CPU
heavy steps of the studio dub: it separates the dialogue from the rest of a video's sound, recognizes how each line is spoken
(emotion2vec+ large, ONNX on the GPU) and mixes each dubbed piece with the draft's mix (the lab's mix_clip.py: dub_mix.py netflix
and its analyses) on its own stems. The lab code (tool/studio_dub_lab of mobile-universal-dubber) is copied into the image at
build time and runs unchanged; the studio runner on the prod host does the rest (transcription, translation, Zonos, checks).

Routes (payload -> answer, all JSON; the SDK wraps the answer as {"result": answer}):
  /prepare   {audioUrl, audioHash}  -> {"pending": true} while the file is being separated in the background (ask again),
                                       then {audioHash, durationMs, speechStem, timings, cached}
  /ser       {clips: {id: flac}}    -> {probs: {id: {label: p}}}  (emotion2vec+ large; audioHash not needed)
  /mix_piece {audioHash, startS, rate, frames, dub: flac, nonverbal: flac, summary: json text}
                                    -> {mix: flac (stereo, `rate`, `frames` samples), timings}: the piece of the video from startS,
                                       its stems and original sound cut from the cache, mixed by mix_clip.py netflix
Binary values are base64. A failure is an answer {"error": "<code>", "detail": "..."} (the SDK turns an exception into a bare 500, and
a remote-dispatch answer is always HTTP 200). Codes: bad_request, not_prepared (unknown audioHash: run /prepare), too_long,
download_failed, prepare_failed, ser_failed, mix_failed.

Disk cache: CACHE_DIR/<audioHash>/ {mix.flac, speech.flac, bg.flac, speech.opus, prepare.json}, least recently used evicted
above STUDIO_CACHE_GB (default 40). The directory appears atomically (renamed from <hash>.tmp) only after a complete prepare.
"""
import asyncio, base64, contextlib, ipaddress, json, logging, math, os, re, shutil, subprocess, sys, tempfile, threading, time, uuid
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
BENCH_URL = 'bench:'  # the Vast startup benchmark: the first minute of a bundled clip, no download
MAX_PIECE_S = 300     # longest piece /mix_piece mixes
LAB = Path(os.environ.get('STUDIO_LAB', '/app/lab'))  # the lab code (mix_clip.py, emotion_ser.py)

_sep = None
_ser = None
_gpu = threading.Lock()      # one separation at a time: the model owns the card
_cpu = ThreadPoolExecutor(max_workers=min(os.cpu_count() or 2, 4), thread_name_prefix='studio-cpu')
_prepare_jobs: dict[str, asyncio.Future] = {}  # audioHash -> preparation running in the background (/prepare answers pending)
_prepare_slot = threading.Lock()
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
    d = stems_dir(need_hash(p))
    if not (d / 'prepare.json').exists():
        raise Fail('not_prepared', p['audioHash'])
    return d


# ---------------------------------------------------------------- /prepare
def run(cmd: list[str], what: str, timeout: float) -> None:
    r = subprocess.run(cmd, capture_output=True, timeout=timeout)
    if r.returncode:
        raise Fail(what, r.stderr.decode(errors='replace')[-400:])


def download(url: str, dst: Path) -> None:
    """Decode the audio at `url` to stereo 48 kHz FLAC. Only http(s) is followed; the reference clip for the startup benchmark is local."""
    if url.startswith(BENCH_URL):
        ref = Path(__file__).parent / 'assets' / 'bench.flac'
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
        encode_stem(tmp / 'speech.flac', tmp / 'speech.opus')
        t['encode_s'] = round(time.time() - s, 2)
        t['total_s'] = round(time.time() - t0, 2)
        (tmp / 'prepare.json').write_text(json.dumps({'durationMs': dur, 'timings': t}))
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
    return {'audioHash': d.name, 'durationMs': meta['durationMs'],
            'speechStem': base64.b64encode((d / 'speech.opus').read_bytes()).decode(), 'timings': meta['timings'], 'cached': cached}


def prepare_in_use(url: str, audio_hash: str) -> dict:
    with _prepare_slot, using(audio_hash):  # one preparation at a time
        return prepare_blocking(url, audio_hash)


async def prepare(**p) -> dict:
    """Starts the separation of a new audio file in the background and answers {"pending": true} at once; the caller asks
    again (same worker) until the answer is the prepared file. A 10-min file takes about 40 s, longer than a proxied
    request may stay open, so the call itself never waits for it."""
    try:
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


# ---------------------------------------------------------------- /ser
def decode_b64(value, what):
    try:
        return base64.b64decode(value, validate=True)
    except Exception as e:
        raise Fail('bad_request', f'{what}: {e!r}')


def ser_blocking(p: dict) -> dict:
    clips = p.get('clips')
    if not isinstance(clips, dict) or not clips:
        raise Fail('bad_request', 'clips must be {id: base64 audio}')
    tmp = Path(tempfile.mkdtemp(dir=CACHE_DIR, prefix='ser-'))
    try:
        out = {}
        for cid, b64 in clips.items():
            path = tmp / f'{len(out)}.flac'
            path.write_bytes(decode_b64(b64, cid))
            out[str(cid)] = _ser.probs(str(path))
        return {'probs': out}
    except Fail:
        raise
    except Exception as e:
        log.exception('ser failed')
        raise Fail('ser_failed', repr(e))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


async def ser(**p) -> dict:
    try:
        return await asyncio.get_running_loop().run_in_executor(_cpu, ser_blocking, p)
    except Fail as e:
        return answer_error(e)


# ---------------------------------------------------------------- /mix_piece
def cut(src: Path, start_s: float, rate: int, frames: int, channels: int, dst: Path) -> None:
    """`frames` samples at `rate` of src from start_s (resampled first, then padded / trimmed to the exact length)."""
    run(['ffmpeg', '-nostdin', '-v', 'error', '-y', '-ss', f'{start_s:.6f}', '-i', str(src), '-ac', str(channels),
         '-af', f'aresample={rate},apad=whole_len={frames},atrim=end_sample={frames}', str(dst)], 'mix_failed', 300)


def mix_piece_blocking(p: dict) -> dict:
    d = need_prepared(p)
    try:
        start_s, rate, frames = float(p['startS']), int(p['rate']), int(p['frames'])
        summary = json.loads(p['summary'])
    except (KeyError, TypeError, ValueError) as e:
        raise Fail('bad_request', repr(e))
    if start_s < 0 or rate not in (44100, 48000) or not 0 < frames <= MAX_PIECE_S * rate or 'report' not in summary:
        raise Fail('bad_request', f'need startS >= 0, rate 44100|48000, 0 < frames <= {MAX_PIECE_S} s, a summary with report')
    t0 = time.time()
    with using(d.name), _mix_slots:
        tmp = Path(tempfile.mkdtemp(dir=CACHE_DIR, prefix='piece-'))
        try:
            clip = tmp / 'data' / 'exp' / 'piece'
            (clip / 'iso').mkdir(parents=True)
            cut(d / 'mix.flac', start_s, rate, frames, 2, clip / 'mix.wav')
            cut(d / 'speech.flac', start_s, rate, frames, 2, clip / 'bandit_speech.flac')
            cut(d / 'bg.flac', start_s, rate, frames, 2, clip / 'bandit_bg.flac')
            for name, field in (('dub_track.wav', 'dub'), ('nonverbal.wav', 'nonverbal')):
                src = tmp / f'{field}.flac'
                src.write_bytes(decode_b64(p.get(field, ''), field))
                run(['ffmpeg', '-nostdin', '-v', 'error', '-y', '-i', str(src), '-ac', '1', '-ar', str(rate), str(clip / 'iso' / name)],
                    'bad_request', 120)
            (clip / 'iso' / 'summary.json').write_text(json.dumps(summary))
            t_in = time.time() - t0
            env = {**os.environ, 'LAB_DATA': str(tmp / 'data')}
            r = subprocess.run(['python3', str(LAB / 'mix_clip.py'), 'piece', 'netflix'], cwd=str(LAB), env=env,
                               capture_output=True, text=True, timeout=1800)
            if r.returncode:
                raise Fail('mix_failed', r.stderr[-600:])
            stages = [l for l in r.stderr.splitlines() if l.startswith('[mix ')]
            flac = tmp / 'mix.flac'
            run(['ffmpeg', '-nostdin', '-v', 'error', '-y', '-i', str(clip / 'iso' / 'mix_netflix.wav'), '-c:a', 'flac', str(flac)],
                'mix_failed', 120)
            return {'mix': base64.b64encode(flac.read_bytes()).decode(),
                    'timings': {'inputs_s': round(t_in, 2), 'total_s': round(time.time() - t0, 2), 'stages': stages}}
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


async def mix_piece(**p) -> dict:
    try:
        return await asyncio.get_running_loop().run_in_executor(_cpu, mix_piece_blocking, p)
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
    noise = (np.random.default_rng(0).standard_normal((2, 10 * 48000)) * 0.05).astype(np.float32)
    out = _sep.speech(noise)  # also the cuDNN algorithm selection and allocator warm-up
    if not np.isfinite(out).all():  # seen on an RTX 4070 (sm_89, driver 580): the fp16 model returns NaN on CUDA only
        raise RuntimeError(f'separator returns non-finite output on this GPU ({np.isnan(out).mean():.0%} NaN)')
    log.info('separator ready in %.1f s (%s)', time.time() - t, os.path.basename(MODEL))
    global _ser
    t = time.time()
    sys.path.insert(0, str(LAB))
    from emotion_ser import Ser
    _ser = Ser()  # E2V_ONNX: the emotion2vec+ large ONNX model in the image
    probs = _ser.probs_pcm(noise[0, :16000 * 3])
    # ONNX Runtime falls back to the CPU on a failing CUDA node (seen without the cuDNN engine libraries) and keeps answering:
    # checked after the first run, so a worker never serves the emotion model from the CPU.
    if _ser.sess.get_providers()[0] != 'CUDAExecutionProvider':
        raise RuntimeError(f'emotion model: not on CUDA after the first run: {_ser.sess.get_providers()}')
    if not all(math.isfinite(v) for v in probs.values()):
        raise RuntimeError('emotion model returns non-finite output on this GPU')
    log.info('emotion model ready in %.1f s', time.time() - t)


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
    # Workload is seconds of video: /prepare costs the file length (the backend passes durationMs; its polls pass 1 ms),
    # /mix_piece the piece, /ser a second per clip. Only /prepare is benchmarked; a fresh audioHash per run so the disk cache never
    # answers it.
    bench = BenchmarkConfig(generator=lambda: {'audioUrl': BENCH_URL, 'audioHash': f'bench-{uuid.uuid4().hex[:12]}', 'durationMs': 60000, 'wait': True},
                            runs=2, concurrency=1, do_warmup=False)
    common = dict(is_remote_dispatch=True, allow_parallel_requests=True, max_queue_time=900.0)
    Worker(WorkerConfig(lifecycle=lifecycle(), handlers=[
        HandlerConfig(route='/prepare', remote_function=prepare, benchmark_config=bench,
                      workload_calculator=lambda p: float(p.get('durationMs') or 600000) / 1000, **common),
        HandlerConfig(route='/ser', remote_function=ser, workload_calculator=lambda p: float(len(p.get('clips') or {}) or 1), **common),
        HandlerConfig(route='/mix_piece', remote_function=mix_piece,
                      workload_calculator=lambda p: float(p.get('frames') or 0) / float(p.get('rate') or 44100), **common)])).run()


if __name__ == '__main__':
    main()
