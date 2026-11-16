"""Vast.ai Serverless mode: the Vast PyWorker (vastai SDK) on WORKER_PORT (3000, TLS with the Vast-signed instance
certificate) in front of the local zonos3-server. Env setup and certificate signing are copied from
talking_photos_gpu/tpgpu/vast.py (proven on Vast Serverless).

Route /tts, payload {"text", "language"?, "voice"? (bank id), "reference_wav_base64"?, "seed"?}
-> {"wav_base64", "audio_s", "generation_s"}. One request at a time: the model owns the GPU.
"""
import asyncio, base64, contextlib, ipaddress, logging, os, time
from pathlib import Path

import requests

import common

log = logging.getLogger('zonos3.vast')
SDK_ENV_DEFAULTS = {'REPORT_ADDR': 'https://run.vast.ai', 'WORKER_PORT': '3000', 'USE_SSL': 'true'}
SIGN_CERT_URL = 'https://console.vast.ai/api/v0/sign_cert/'
CERT_FILE, KEY_FILE = Path('/etc/instance.crt'), Path('/etc/instance.key')
BENCH = {'text': 'Привет! Сегодня мы разберём, как устроен этот курс.', 'language': 'ru'}


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


@contextlib.asynccontextmanager
async def lifecycle():
    await asyncio.get_running_loop().run_in_executor(None, common.start_server)
    yield


async def tts(**payload) -> dict:
    return await asyncio.get_running_loop().run_in_executor(None, common.generate, payload)


def main():
    logging.basicConfig(level=logging.INFO)
    container_id = prepare_env()
    if os.environ['USE_SSL'] == 'true':
        try:
            sign_certificate(container_id)
        except SystemExit as e:
            # Some hosts reach Vast from another IP than their public one; Vast then refuses to sign ("Caller is not
            # the instance machine", HTTP 403) and the worker used to exit, so Vast cycled hosts for 15+ minutes.
            log.warning('%s; serving plain HTTP', e)
            os.environ['USE_SSL'] = 'false'
    from vastai.serverless.server.worker import BenchmarkConfig, HandlerConfig, Worker, WorkerConfig
    Worker(WorkerConfig(lifecycle=lifecycle(), handlers=[HandlerConfig(
        route='/tts', is_remote_dispatch=True, remote_function=tts, allow_parallel_requests=False,
        max_queue_time=600.0, workload_calculator=lambda p: float(len(p.get('text', ''))),
        benchmark_config=BenchmarkConfig(dataset=[BENCH], runs=1, concurrency=1, do_warmup=False))])).run()


if __name__ == '__main__':
    main()
