#!/usr/bin/env bash
# Pre-deploy gate: run the worker's real code (slim/common.py + the Vast startup benchmark payload) against the CPU
# zonos3-server on the Contabo box (same upstream commit as the CUDA release) BEFORE building/pushing an image.
# A worker that fails its startup benchmark is killed by Vast, which then churns through fresh hosts that each pull
# the full 3.6 GB image. Usage: bash scripts/smoke_cpu.sh   (needs ssh root@161.97.163.85)
set -euo pipefail
cd "$(dirname "$0")/.."
H=root@161.97.163.85
ssh "$H" 'mkdir -p /root/smoke/voices/ru /root/smoke/lib && cp -n /root/tn/libzyphra_textnorm.so /root/smoke/lib/'
scp -q slim/common.py slim/vast_worker.py "$H":/root/smoke/
scp -q slim/voices/ru/Charon.ogg slim/voices/ru/Kore.ogg "$H":/root/smoke/voices/ru/
ssh "$H" 'cd /root/smoke && sed -i "s#/app/lib/#/root/smoke/lib/#; s#Path(\x27/app/voices\x27)#Path(\x27/root/smoke/voices\x27)#" common.py
if ! curl -s -m 2 localhost:1919/v2/health | grep -q ok; then
  (cd /root/ZONOS3-Mini/servers/ggml && nohup build/zonos3-server gguf/backbone-ptq.gguf --zac gguf/zac-keep2.gguf \
     --spk-encoder gguf/spk-encoder.gguf --port 1919 > /root/zsrv.log 2>&1 &)
  for i in $(seq 60); do curl -s -m 2 localhost:1919/v2/health | grep -q ok && break; sleep 2; done
fi
python3 - <<EOF
import ast, json, common
bench = ast.literal_eval(next(l for l in open("vast_worker.py") if l.startswith("BENCH =")).split("=", 1)[1].strip())
cases = {"startup benchmark": bench,
         "latin words": {"text": "Откройте Python и запустите Docker через API.", "language": "ru", "voice": "Kore"}}
for name, payload in cases.items():
    r = common.generate(payload); r.pop("wav_base64")
    assert r["audio_s"] > 0.5, (name, r)
    print("OK", name, json.dumps({k: r[k] for k in ("audio_s", "attempts", "text_sent", "text_normalization")}, ensure_ascii=False))
EOF'
