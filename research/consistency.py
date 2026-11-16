"""Chunk-to-chunk consistency of one Zonos voice under different generation settings (run on Contabo, CPU server :1919).
Per take: speaker cosine to the reference clip (same spk-encoder the model conditions on), integrated loudness (LUFS),
characters per second. Lower spread across sentences = more uniform dub. Also a numbers check for text_normalization.
Writes takes into ./cons/<config>/<i>.wav and prints a summary table as JSON lines."""
import base64, json, statistics as st, subprocess, sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import requests

U = 'http://127.0.0.1:1919'
H = {'X-TTS-Session-ID': 'cons'}
REF = Path('/root/charon_ru.ogg')
SENTENCES = [
    'Привет! В этом уроке мы разберём, как устроен проект.',
    'Сначала создадим новую папку и откроем её в редакторе.',
    'Обратите внимание: этот шаг часто пропускают, и потом возникают ошибки.',
    'Хорошо.',
    'Теперь давайте посмотрим на результат и сравним его с тем, что мы получили в прошлый раз, когда настраивали сервер.',
    'Если что-то пошло не так, вернитесь к предыдущему видео.',
]
CONFIGS = {
    'A_default': {},
    'B_temp05': {'temperature': 0.5},
    'C_cond': {'condition_values': {'loudness': -20.0, 'speaking_rate': 14.5}},
    'D_cfg15': {'speaker_cfg_scale': 1.5},
    'E_all': {'temperature': 0.5, 'speaker_cfg_scale': 1.5, 'condition_values': {'loudness': -20.0, 'speaking_rate': 14.5}},
}


def register(audio: bytes, label: str) -> str:
    r = requests.post(U + '/v2/speakers', headers=H, json={'label': label, 'speaker_audio_base64': base64.b64encode(audio).decode()}, timeout=120)
    r.raise_for_status()
    return r.json()['id']


def embedding(sid: str) -> np.ndarray:
    v = np.asarray(requests.get(f'{U}/v2/speakers/{sid}/embedding', headers=H, timeout=60).json()['embedding'], dtype=np.float32)
    return v / np.linalg.norm(v)


def lufs(path: Path) -> float:
    out = subprocess.run(['ffmpeg', '-nostdin', '-hide_banner', '-i', str(path), '-af', 'ebur128', '-f', 'null', '-'],
                         capture_output=True, text=True).stderr
    return float(out.rsplit('I:', 1)[1].split('LUFS')[0])


def take(cfg: str, i: int, text: str, ref_id: str, ref_emb: np.ndarray) -> dict:
    body = {'text': text, 'language': 'ru', 'stream': False, 'format': 'wav', 'speaker_mode': 'embedding',
            'speaker_embedding_id': ref_id, 'seed': 42, **CONFIGS[cfg]}
    r = requests.post(U + '/v2/generate', headers=H, json=body, timeout=900)
    if not r.ok:
        return {'cfg': cfg, 'i': i, 'error': r.text[:200]}
    path = Path('cons') / cfg / f'{i}.wav'
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(r.content)
    seconds = (len(r.content) - 44) / 2 / 44100
    cos = float(embedding(register(r.content, f'{cfg}{i}')) @ ref_emb)
    return {'cfg': cfg, 'i': i, 'cos': round(cos, 3), 'lufs': round(lufs(path), 1), 'cps': round(len(text) / seconds, 1), 's': round(seconds, 2)}


def main():
    ref_id = register(REF.read_bytes(), 'ref')
    ref_emb = embedding(ref_id)
    jobs = [(c, i, s) for c in CONFIGS for i, s in enumerate(SENTENCES)]
    with ThreadPoolExecutor(3) as ex:
        rows = list(ex.map(lambda j: take(*j, ref_id, ref_emb), jobs))
    Path('cons/rows.json').write_text(json.dumps(rows, ensure_ascii=False, indent=1))
    for c in CONFIGS:
        ok = [r for r in rows if r['cfg'] == c and 'cos' in r]
        multi = [r for r in ok if r['i'] != 3]  # one-word take excluded from rate spread
        print(json.dumps({'cfg': c, 'n': len(ok), 'errors': sum(1 for r in rows if r['cfg'] == c and 'error' in r),
                          'cos_mean': round(st.mean(r['cos'] for r in ok), 3), 'cos_min': min(r['cos'] for r in ok),
                          'lufs_sd': round(st.pstdev(r['lufs'] for r in ok), 2),
                          'lufs_range': [min(r['lufs'] for r in ok), max(r['lufs'] for r in ok)],
                          'cps_sd': round(st.pstdev(r['cps'] for r in multi), 2)}), flush=True)


if __name__ == '__main__':
    sys.exit(main())
