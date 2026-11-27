"""Vast.ai Serverless mode, the studio-prep worker (docs/architecture/studio-dub-realtime.md): the Vast PyWorker (vastai SDK)
on WORKER_PORT (3000, TLS with the Vast-signed instance certificate), remote dispatch like slim/vast_worker.py. The GPU and CPU
heavy steps of the studio dub: it separates the dialogue from the rest of a video's sound, recognizes how each line is spoken
(emotion2vec+ large, ONNX on the GPU) and mixes each dubbed piece with the draft's mix (the lab's mix_clip.py: dub_mix.py netflix
and its analyses) on its own stems. The lab code (tool/studio_dub_lab of mobile-universal-dubber) is copied into the image at
build time and runs unchanged; the studio runner on the prod host does the rest (transcription, translation, Zonos, checks).

Routes (payload -> answer, all JSON; the SDK wraps the answer as {"result": answer}):
  /prepare   {audioUrl, audioHash, haveHead?}
                                    -> {"pending": true} while the file is being separated in the background (ask again); once the
                                       head (the first HEAD_S of a longer file) is separated, the pending answer also carries
                                       {durationMs, headS, speechHead, backgroundHead} (not when the caller says haveHead), then
                                       {audioHash, durationMs, speechStem, backgroundStem, timings, cached}; the background
                                       (mix - speech, stereo ogg/opus 96 kbit/s) is what the per-cue path plays under its lines
  /ser       {clips: {id: flac}}    -> {probs: {id: {label: p}}}  (emotion2vec+ large; audioHash not needed)
  /mix_plan  {audioHash, rows: [{start, end, speaker}], voices: {speaker: voice}, samples: {voice: flac}, untilS?, windows?}
                                    -> {"pending": true} | {"ready": true}: the draft's mix analyses (the lab's mix_plan.py build:
                                       channel, defects, rooms, locations, perspective) of the whole file's stems (else the
                                       head's), measured once in the background against a reference dub of the voices; asked
                                       again for the whole file once it is separated. A stream video (<job>_video) names its
                                       windows instead of a file ([{audioHash: <job>_w<k>, fromS, toS}]: each window's own range,
                                       prepared here): their stems are joined into its stems dir first (join_windows)
  /mix_piece {audioHash, startS, rate, frames, dub: flac, nonverbal: flac, summary: json text}
                                    -> {mix: flac (stereo, `rate`, `frames` samples), timings}: the piece of the video from startS,
                                       its stems and original sound cut from the cache (from the head's while the whole file is
                                       still being separated, for a piece inside the head), mixed by mix_clip.py netflix on the
                                       plan's analyses (cut to the piece: mix_plan.py slice; a plan still being measured is
                                       waited for up to PLAN_WAIT_S), else on the piece's own
  /mix_cue   {audioHash, startS, endS, dub: audio, netflix?, planHash?, planOffsetS?, dubStartS?, tailEndS?, planWaitS?}
                                    -> {mix: flac (stereo 44.1 kHz), netflix: "video" | false, timings}: one dub line mixed against
                                       the original voice of its slot. With netflix and the plan of planHash (a stream video's;
                                       ready, or ready within planWaitS <= LINE_PLAN_WAIT_S; planOffsetS: where the window
                                       starts in the plan's time): the draft's netflix voice chain (the lab's mix_line.py: level,
                                       the location's tone and room, dynamics, perspective; the take placed at dubStartS, its
                                       room tail kept up to tailEndS), ~0.2 s; else the dry treatment (the lab's mix_cue.py:
                                       level, tone, balance; no room, the take's length), tens of milliseconds
Binary values are base64. A failure is an answer {"error": "<code>", "detail": "..."} (the SDK turns an exception into a bare 500, and
a remote-dispatch answer is always HTTP 200). Codes: bad_request, not_prepared (unknown audioHash: run /prepare), too_long,
download_failed, prepare_failed, ser_failed, mix_failed.

Disk cache: CACHE_DIR/<audioHash>/ {mix.flac, speech.flac, bg.flac, speech.opus, bg.opus, prepare.json, plan/}, least recently used evicted
above STUDIO_CACHE_GB (default 40). The directory appears atomically (renamed from <hash>.tmp) only after a complete prepare.
CACHE_DIR/<audioHash>.head/ holds the same files for the first HEAD_S + HEAD_PAD_S of a long file, separated first: the separator
works in 8 s chunks with no state across them, so up to HEAD_S + HEAD_PAD_S - 8 s the head's stems are the whole file's.
"""
import asyncio, base64, contextlib, io, ipaddress, json, logging, math, os, re, shutil, subprocess, sys, tempfile, threading, time, uuid
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
HEAD_S = 120          # a longer file's first HEAD_S are separated (and answered) first: the runner starts dubbing on them
HEAD_PAD_S = 16       # separated past HEAD_S: the stems of the last chunks are the whole file's only once the next chunks exist

_sep = None
_ser = None
_gpu = threading.Lock()      # one separation at a time: the model owns the card
# Request threads: mixes waiting for a slot hold one each, so there are more threads than slots (the emotion model and /prepare
# polls never queue behind waiting mixes).
_cpu = ThreadPoolExecutor(max_workers=12, thread_name_prefix='studio-cpu')
_prepare_jobs: dict[str, asyncio.Future] = {}  # audioHash -> preparation running in the background (/prepare answers pending)
_prepare_slot = threading.Lock()
_plan_jobs: dict[str, 'concurrent.futures.Future'] = {}  # stems dir -> mix plan being measured
_plan_lock = threading.Lock()
PLAN_WAIT_S = 90  # a piece waits this long for a plan being measured, then is measured on its own (the lab's way)


class EarliestFirst:
    """`slots` mixes at once; a waiting mix of an earlier piece (lower startS) goes before a later one, whatever came first:
    the runner dubs pieces side by side, and the windows playback needs next come from the earliest piece."""

    def __init__(self, slots):
        self.free, self.waiting, self.cond = slots, [], threading.Condition()

    @contextlib.contextmanager
    def slot(self, priority):
        with self.cond:
            entry = (priority, uuid.uuid4().hex)
            self.waiting.append(entry)
            while not (self.free and min(self.waiting) == entry):
                self.cond.wait()
            self.waiting.remove(entry)
            self.free -= 1
            self.cond.notify_all()
        try:
            yield
        finally:
            with self.cond:
                self.free += 1
                self.cond.notify_all()


# Pieces mixed at once: each mix runs its analyses side by side on half the cores (fork_pool.py) and its WPE dereverberation on 6
# processes; the pool takes hosts with >= 12 cores.
_mix_slots = EarliestFirst(int(os.environ.get('STUDIO_MIXES', '2')))
UNITS_PER_WORKER_S = 17.0  # /prepare benchmark: ~17 seconds of video separated per second of worker time
SER_S_PER_CLIP = 0.3
MIX_CUE_S = 0.1  # a line's mix (mix_cue.py): tens of milliseconds
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


def head_dir(audio_hash: str) -> Path:
    return CACHE_DIR / f'{audio_hash}.head'


def stems_for(p: dict, end_s: float) -> Path:
    """The directory whose stems a request up to end_s seconds of the video uses: the whole file's once prepared, else the
    head's when end_s lies in its exact part (HEAD_S + HEAD_PAD_S - 8 s), else not_prepared (the caller runs /prepare)."""
    h = need_hash(p)
    d = stems_dir(h)
    if (d / 'prepare.json').exists():
        return d
    hd = head_dir(h)
    if (hd / 'head.json').exists() and end_s <= HEAD_S + HEAD_PAD_S - 8:
        return hd
    raise Fail('not_prepared', p['audioHash'])


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


def background(d: Path) -> str:
    """The background stem (mix - speech) for the per-cue path, where the phone plays it under the dub lines: stereo ogg/opus
    96 kbit/s, base64; encoded once per stems dir."""
    dst = d / 'bg.opus'
    if not dst.exists():
        run(['ffmpeg', '-nostdin', '-v', 'error', '-y', '-i', str(d / 'bg.flac'), '-ac', '2', '-c:a', 'libopus', '-b:a', '96k',
             '-vbr', 'on', str(d / 'bg.tmp.opus')], 'prepare_failed', 600)
        (d / 'bg.tmp.opus').rename(dst)
    return base64.b64encode(dst.read_bytes()).decode()


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
        t['gpu_wait_s'] = 0.0
        from . import sep
        if dur / 1000 > HEAD_S + 2 * HEAD_PAD_S:
            # The head first (a few seconds on the card): the runner transcribes and dubs it while the rest is separated.
            head = head_dir(audio_hash)
            shutil.rmtree(head, ignore_errors=True)
            head.mkdir()
            run(['ffmpeg', '-nostdin', '-v', 'error', '-y', '-t', str(HEAD_S + HEAD_PAD_S), '-i', str(tmp / 'mix.flac'), '-c:a', 'flac',
                 str(head / 'mix.flac')], 'prepare_failed', 120)
            s = time.time()
            with _gpu:
                t['gpu_wait_s'] += round(time.time() - s, 2)
                s = time.time()
                sep.separate_file(head / 'mix.flac', head, _sep)
            encode_stem(head / 'speech.flac', head / 'speech.opus')
            background(head)
            (head / 'head.json').write_text(json.dumps({'durationMs': dur, 'headS': HEAD_S}))
            t['head_s'] = round(time.time() - t0, 2)
        s = time.time()
        with _gpu:
            t['gpu_wait_s'] += round(time.time() - s, 2)
            s = time.time()
            sep.separate_file(tmp / 'mix.flac', tmp, _sep)
        t['separate_s'] = round(time.time() - s, 2)
        s = time.time()
        encode_stem(tmp / 'speech.flac', tmp / 'speech.opus')
        background(tmp)
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
            'speechStem': base64.b64encode((d / 'speech.opus').read_bytes()).decode(), 'backgroundStem': background(d),
            'timings': meta['timings'], 'cached': cached}


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
            hd = head_dir(h)
            if not p.get('haveHead') and (hd / 'head.json').exists():
                meta = json.loads((hd / 'head.json').read_text())
                return {'audioHash': h, 'pending': True, 'durationMs': meta['durationMs'], 'headS': meta['headS'],
                        'speechHead': base64.b64encode((hd / 'speech.opus').read_bytes()).decode(), 'backgroundHead': background(hd)}
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


# ---------------------------------------------------------------- /mix_plan
# A plan is measured on a stems dir: on all of it ('plan') or, asked with untilS, on its first untilS seconds ('plan_start': the
# video's first pieces get theirs in a fraction of the time). A stream window's stems dir (<job>_w<k>) gets its own plan, which
# its lines' /mix_cue renders against. PLAN_RUNS measurements at a time (each forks its own pool on half the cores), in the
# order asked: the windows of several viewers do not queue behind one another.
PLANS = ('plan', 'plan_start')
_plan_run = threading.BoundedSemaphore(int(os.environ.get('STUDIO_PLAN_RUNS', '2')))
_plan_until: dict[str, float] = {}  # plan being measured -> seconds it will cover


def plan_data(d: Path, name: str) -> Path:
    return d / name / 'data'


def plan_covers(d: Path, name: str) -> float:
    """Seconds from the start a ready plan covers (0: not ready)."""
    f = plan_data(d, name) / 'exp' / 'plan' / 'iso' / 'netflix' / 'plan.json'
    return json.loads(f.read_text())['duration_s'] if f.exists() else 0.0


def to_wav(src: Path, dst: Path, channels: int, what: str, until_s: float | None = None) -> None:
    run(['ffmpeg', '-nostdin', '-v', 'error', '-y', '-i', str(src)] + (['-t', f'{until_s:.3f}'] if until_s else []) +
        ['-ac', str(channels), '-ar', '44100', str(dst)], what, 300)


def plan_blocking(d: Path, name: str, until_s: float | None, p: dict) -> None:
    with _plan_run:
        tmp = d / f'{name}.tmp'
        shutil.rmtree(tmp, ignore_errors=True)
        clip = tmp / 'data' / 'exp' / 'plan'
        clip.mkdir(parents=True)
        t0 = time.time()
        try:
            with using(d.name):
                to_wav(d / 'mix.flac', clip / 'mix.wav', 2, 'mix_failed', until_s)
                to_wav(d / 'speech.flac', clip / 'bandit_speech.flac', 2, 'mix_failed', until_s)
                to_wav(d / 'bg.flac', clip / 'bandit_bg.flac', 2, 'mix_failed', until_s)
                samples = tmp / 'samples'
                samples.mkdir()
                for voice, b64 in p['samples'].items():
                    (samples / f'{voice}.flac').write_bytes(decode_b64(b64, voice))
                    to_wav(samples / f'{voice}.flac', samples / f'{voice}.wav', 1, 'bad_request')
                (tmp / 'rows.json').write_text(json.dumps(p['rows']))
                (tmp / 'voices.json').write_text(json.dumps(p['voices']))
                env = {**os.environ, 'LAB_DATA': str(tmp / 'data')}
                r = subprocess.run(['python3', str(LAB / 'mix_plan.py'), 'build', str(tmp / 'data'), str(tmp / 'rows.json'),
                                    str(tmp / 'voices.json'), str(samples)], cwd=str(LAB), env=env, capture_output=True, text=True, timeout=1800)
                if r.returncode:
                    raise Fail('mix_failed', r.stderr[-600:])
                shutil.rmtree(d / name, ignore_errors=True)
                tmp.rename(d / name)
            log.info('mix %s %s: %s in %.1f s', name, d.name, r.stdout.strip()[-200:], time.time() - t0)
        except Exception:
            shutil.rmtree(tmp, ignore_errors=True)
            log.exception('mix %s %s failed', name, d.name)
            raise


def plan_dir_for(p: dict) -> Path:
    """The stems a plan is measured on: the whole file's once separated, else the head's."""
    h = need_hash(p)
    for d, marker in ((stems_dir(h), 'prepare.json'), (head_dir(h), 'head.json')):
        if (d / marker).exists():
            return d
    raise Fail('not_prepared', h)


_join_lock = threading.Lock()


def join_windows(p: dict) -> None:
    """A stream video's plan stems (CACHE_DIR/<audioHash>: mix, speech, bg) joined from its windows' stems, prepared on this
    worker: `windows` [{audioHash, fromS, toS}] in order, each window's own range in its own seconds. Written once and
    atomically like a prepare (prepare.json last), so /mix_plan measures it as any prepared file."""
    import soundfile as sf
    h = need_hash(p)
    final = stems_dir(h)
    wins = p.get('windows')
    if not isinstance(wins, list) or not wins:
        raise Fail('bad_request', 'windows must be a non-empty list')
    with _join_lock:
        if (final / 'prepare.json').exists():
            return
        parts = []
        for w in wins:
            wd = stems_dir(need_hash(w))
            try:
                a, b = float(w['fromS']), float(w['toS'])
            except (KeyError, TypeError, ValueError) as e:
                raise Fail('bad_request', f'windows: {e!r}')
            if not (wd / 'prepare.json').exists():
                raise Fail('not_prepared', w['audioHash'])
            if not 0 <= a < b:
                raise Fail('bad_request', 'windows: need 0 <= fromS < toS')
            parts.append((wd, a, b))
        tmp = CACHE_DIR / f'{h}.tmp'
        shutil.rmtree(tmp, ignore_errors=True)
        tmp.mkdir(parents=True)
        try:
            frames, rate = 0, None
            for name in ('mix.flac', 'speech.flac', 'bg.flac'):
                out, n = None, 0
                for wd, a, b in parts:
                    with using(wd.name):
                        info = sf.info(str(wd / name))
                        x, rate = sf.read(str(wd / name), start=int(a * info.samplerate), stop=int(b * info.samplerate),
                                          dtype='float32', always_2d=True)
                    if out is None:
                        out = sf.SoundFile(str(tmp / name), 'w', samplerate=rate, channels=x.shape[1], format='FLAC',
                                           subtype=info.subtype)
                    out.write(x)
                    n += len(x)
                out.close()
                frames = n
            (tmp / 'prepare.json').write_text(json.dumps({'durationMs': round(frames * 1000 / rate), 'joined': wins}))
            shutil.rmtree(final, ignore_errors=True)
            tmp.rename(final)
        except Fail:
            shutil.rmtree(tmp, ignore_errors=True)
            raise
        except Exception as e:
            shutil.rmtree(tmp, ignore_errors=True)
            log.exception('join %s failed', h)
            raise Fail('mix_failed', repr(e))


async def mix_plan(**p) -> dict:
    try:
        if p.get('windows') is not None:
            await asyncio.get_running_loop().run_in_executor(_cpu, join_windows, p)
        d = plan_dir_for(p)
        if not isinstance(p.get('rows'), list) or not isinstance(p.get('voices'), dict) or not isinstance(p.get('samples'), dict):
            raise Fail('bad_request', 'rows, voices and samples required')
        until_s = p.get('untilS')
        if until_s is not None and (not isinstance(until_s, (int, float)) or until_s <= 0):
            raise Fail('bad_request', 'untilS must be a positive number')
        name = 'plan_start' if until_s else 'plan'
        if plan_covers(d, name):
            return {'ready': True}
        with _plan_lock:
            job = _plan_jobs.get(f'{d}/{name}')
            if job is None or (job.done() and job.exception() is not None):
                job = _plan_jobs[f'{d}/{name}'] = _cpu.submit(plan_blocking, d, name, until_s, p)
                _plan_until[f'{d}/{name}'] = float(until_s or 'inf')
        return {'pending': True}
    except Fail as e:
        return answer_error(e)


def plan_for(d: Path, p: dict, end_s: float) -> Path | None:
    """The plan a piece up to end_s uses: a ready one that covers it, the whole one first (of its stems dir, else of the head's
    for a piece inside the head); else one being measured that will cover it, waited for up to PLAN_WAIT_S; else none (the piece
    is measured on its own)."""
    dirs = [d]
    hd = head_dir(need_hash(p))
    if hd != d and end_s <= HEAD_S + HEAD_PAD_S - 8:
        dirs.append(hd)
    plans = [(x, name) for x in dirs for name in PLANS]
    for x, name in plans:
        if plan_covers(x, name) >= end_s - 0.5:
            return plan_data(x, name)
    for x, name in reversed(plans):  # the start's plan (sooner) before the whole one
        with _plan_lock:
            job = _plan_jobs.get(f'{x}/{name}')
            until_s = _plan_until.get(f'{x}/{name}', 0.0)
        if job is not None and not job.done() and until_s >= end_s - 0.5:
            with contextlib.suppress(Exception):
                job.result(timeout=PLAN_WAIT_S)
            if plan_covers(x, name) >= end_s - 0.5:
                return plan_data(x, name)
    return None


# ---------------------------------------------------------------- /mix_piece
def cut(src: Path, start_s: float, rate: int, frames: int, channels: int, dst: Path) -> None:
    """`frames` samples at `rate` of src from start_s (resampled first, then padded / trimmed to the exact length)."""
    run(['ffmpeg', '-nostdin', '-v', 'error', '-y', '-ss', f'{start_s:.6f}', '-i', str(src), '-ac', str(channels),
         '-af', f'aresample={rate},apad=whole_len={frames},atrim=end_sample={frames}', str(dst)], 'mix_failed', 300)


def mix_piece_blocking(p: dict) -> dict:
    try:
        start_s, rate, frames = float(p['startS']), int(p['rate']), int(p['frames'])
        summary = json.loads(p['summary'])
    except (KeyError, TypeError, ValueError) as e:
        raise Fail('bad_request', repr(e))
    if start_s < 0 or rate not in (44100, 48000) or not 0 < frames <= MAX_PIECE_S * rate or 'report' not in summary:
        raise Fail('bad_request', f'need startS >= 0, rate 44100|48000, 0 < frames <= {MAX_PIECE_S} s, a summary with report')
    d = stems_for(p, start_s + frames / rate)
    plan = plan_for(d, p, start_s + frames / rate)
    t0 = time.time()
    with using(d.name), _mix_slots.slot(start_s):
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
            env = {**os.environ, 'LAB_DATA': str(tmp / 'data')}
            if plan is not None:
                run(['python3', str(LAB / 'mix_plan.py'), 'slice', str(plan), str(tmp / 'data'), 'piece', f'{start_s:.6f}'], 'mix_failed', 120)
            t_in = time.time() - t0
            r = subprocess.run(['python3', str(LAB / 'mix_clip.py'), 'piece', 'netflix'], cwd=str(LAB), env=env,
                               capture_output=True, text=True, timeout=1800)
            if r.returncode:
                raise Fail('mix_failed', r.stderr[-600:])
            stages = [l for l in r.stderr.splitlines() if l.startswith('[mix ')]
            flac = tmp / 'mix.flac'
            run(['ffmpeg', '-nostdin', '-v', 'error', '-y', '-i', str(clip / 'iso' / 'mix_netflix.wav'), '-c:a', 'flac', str(flac)],
                'mix_failed', 120)
            return {'mix': base64.b64encode(flac.read_bytes()).decode(),
                    'timings': {'inputs_s': round(t_in, 2), 'total_s': round(time.time() - t0, 2), 'stages': stages, 'plan': plan is not None}}
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


async def mix_piece(**p) -> dict:
    try:
        return await asyncio.get_running_loop().run_in_executor(_cpu, mix_piece_blocking, p)
    except Fail as e:
        return answer_error(e)


# ---------------------------------------------------------------- /mix_cue
LINE_PLAN_WAIT_S = 20  # longest a line waits for its video's plan still being measured (the caller's planWaitS, capped)
LINE_PLANS = 12        # loaded (plan, window) pairs kept in memory (a few windows of a few viewers)
_line_plans: dict[tuple[str, str, float], object] = {}
_line_plans_lock = threading.Lock()


def ready_plan(d: Path) -> Path | None:
    """A ready plan of these stems: the whole file's, else the one of its start (untilS)."""
    return next((plan_data(d, name) for name in PLANS if plan_covers(d, name)), None)


def line_plan(d: Path, plan_hash: str, offset_s: float, wait_s: float):
    """The lab's mix_line.Plan for a line of the stream window d: the plan of the video (/mix_plan of plan_hash, its windows'
    stems joined: the draft measured its room on the whole clip, a window's first 15-30 s gave none), read in the window's
    time (offset_s: the window's start in the plan's time) against the window's own stem; ready, or ready within wait_s;
    else None (the line is mixed dry)."""
    pd = stems_dir(plan_hash)
    path = ready_plan(pd)
    if path is None and wait_s > 0:
        with _plan_lock:
            jobs = [j for name in PLANS if (j := _plan_jobs.get(f'{pd}/{name}')) is not None and not j.done()]
        if jobs:
            with contextlib.suppress(Exception):
                jobs[0].result(timeout=wait_s)
            path = ready_plan(pd)
    if path is None:
        return None
    key = (str(path), str(d), offset_s)
    with _line_plans_lock:
        plan = _line_plans.get(key)
    if plan is None:
        import mix_line
        with using(pd.name):
            plan = mix_line.Plan(str(path), str(d / 'speech.flac'), offset_s)
        with _line_plans_lock:
            _line_plans[key] = plan
            while len(_line_plans) > LINE_PLANS:
                _line_plans.pop(next(iter(_line_plans)))
    return plan


def mix_cue_blocking(p: dict) -> dict:
    """One dub line mixed against the original line in its slot. With `netflix` and a plan of the stems (/mix_plan, a stream
    window's): the draft's netflix voice chain (the lab's mix_line.py: level, the location's tone and room, dynamics,
    perspective; the take placed at dubStartS, its room tail kept up to tailEndS); else the dry treatment (the lab's
    mix_cue.py: level, tone, balance; no room)."""
    import numpy as np
    import soundfile as sf
    from scipy.signal import resample_poly
    try:
        start_s, end_s = float(p['startS']), float(p['endS'])
        dub_start = float(p.get('dubStartS', start_s))
        tail_end = None if p.get('tailEndS') is None else float(p['tailEndS'])
        plan_wait = min(max(float(p.get('planWaitS') or 0), 0.0), LINE_PLAN_WAIT_S)
    except (KeyError, TypeError, ValueError) as e:
        raise Fail('bad_request', repr(e))
    if start_s < 0 or not 0 < end_s - start_s <= 60 or dub_start < 0 or (tail_end is not None and tail_end <= dub_start):
        raise Fail('bad_request', 'need 0 <= startS < endS <= startS + 60, dubStartS >= 0, tailEndS > dubStartS')
    import mix_cue as lab_mix
    t0 = time.time()
    try:
        dub, rate = sf.read(io.BytesIO(decode_b64(p.get('dub', ''), 'dub')), dtype='float32', always_2d=True)
    except Fail:
        raise
    except Exception as e:
        raise Fail('bad_request', f'dub: {e!r}')
    dub = dub.mean(1)
    if rate != lab_mix.SR:
        dub = resample_poly(dub, lab_mix.SR, rate).astype(np.float32)
    take_s = len(dub) / lab_mix.SR
    d = stems_for(p, max(end_s, dub_start + take_s))  # the context is cut to what the stems hold
    if p.get('netflix') and p.get('planHash') is not None:
        try:
            plan_hash, offset_s = need_hash({'audioHash': p['planHash']}), float(p.get('planOffsetS') or 0.0)
        except (TypeError, ValueError) as e:
            raise Fail('bad_request', repr(e))
        t_plan = time.time()
        with using(d.name):  # the window's stem is read when the plan is first loaded for it
            plan = line_plan(d, plan_hash, offset_s, plan_wait)
        waited = time.time() - t_plan
        if plan is not None:
            import mix_line
            out = mix_line.render(plan, start_s, end_s, dub, dub_start, tail_end)
            buf = io.BytesIO()
            sf.write(buf, out.T, lab_mix.SR, format='FLAC', subtype='PCM_16')
            return {'mix': base64.b64encode(buf.getvalue()).decode(), 'netflix': 'video',
                    'timings': {'total_s': round(time.time() - t0, 3), 'plan_wait_s': round(waited, 2)}}
    half = max(take_s, end_s - start_s, lab_mix.CONTEXT_S) / 2 + 1
    mid = start_s + (end_s - start_s) / 2
    with using(d.name):
        info = sf.info(str(d / 'speech.flac'))
        a = max(0, int((mid - half) * info.samplerate))
        b = min(info.frames, int((mid + half) * info.samplerate))
        stem, _ = sf.read(str(d / 'speech.flac'), start=a, stop=b, dtype='float32', always_2d=True)
    stem = resample_poly(stem.T, 147, 160, axis=1).astype(np.float32) if info.samplerate == 48000 else stem.T
    offset = a / info.samplerate
    slot, context = lab_mix.window(stem, start_s - offset, max(end_s, start_s + take_s) - offset)
    out = lab_mix.mix_cue(slot, context, dub)
    buf = io.BytesIO()
    sf.write(buf, out.T, lab_mix.SR, format='FLAC', subtype='PCM_16')
    return {'mix': base64.b64encode(buf.getvalue()).decode(), 'netflix': False, 'timings': {'total_s': round(time.time() - t0, 3)}}


async def mix_cue(**p) -> dict:
    try:
        return await asyncio.get_running_loop().run_in_executor(_cpu, mix_cue_blocking, p)
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
    # Workload units are what the /prepare benchmark measures: seconds of video separated, ~17 per second of worker time. The
    # other routes are priced in the same units by the worker time they take (measured: a piece's mix ~1 s per second of the
    # piece, the emotion model ~0.3 s per clip), so the autoscaler sees a mixing worker as busy and never stops it mid-request.
    # /prepare's polls pass durationMs 1. A fresh audioHash per benchmark run so the disk cache never answers it.
    bench = BenchmarkConfig(generator=lambda: {'audioUrl': BENCH_URL, 'audioHash': f'bench-{uuid.uuid4().hex[:12]}', 'durationMs': 60000, 'wait': True},
                            runs=2, concurrency=1, do_warmup=False)
    common = dict(is_remote_dispatch=True, allow_parallel_requests=True, max_queue_time=900.0)
    Worker(WorkerConfig(lifecycle=lifecycle(), handlers=[
        HandlerConfig(route='/prepare', remote_function=prepare, benchmark_config=bench,
                      workload_calculator=lambda p: float(p.get('durationMs') or 600000) / 1000, **common),
        HandlerConfig(route='/ser', remote_function=ser,
                      workload_calculator=lambda p: UNITS_PER_WORKER_S * SER_S_PER_CLIP * max(1, len(p.get('clips') or {})), **common),
        HandlerConfig(route='/mix_plan', remote_function=mix_plan, workload_calculator=lambda p: UNITS_PER_WORKER_S, **common),
        HandlerConfig(route='/mix_piece', remote_function=mix_piece,
                      workload_calculator=lambda p: UNITS_PER_WORKER_S * float(p.get('frames') or 0) / float(p.get('rate') or 44100), **common),
        HandlerConfig(route='/mix_cue', remote_function=mix_cue, workload_calculator=lambda p: UNITS_PER_WORKER_S * MIX_CUE_S, **common)])).run()


if __name__ == '__main__':
    main()
