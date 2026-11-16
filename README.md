# zonos3-worker

ZONOS3-Mini TTS worker for Vast.ai Serverless (production) and RunPod Serverless, plus its multilingual voice bank.

## Image (`slim/`)

One image, `ghcr.io/olegphenomenon/zonos3-worker:runpod-slim-<sha>` (public), built by `.github/workflows/slim.yml`
without compiling: CUDA 12.8 runtime + prebuilt `zonos3-server` (release `ggml-cuda-sm86-sm89-96d9e0e`, Ampere/Ada)
+ `backbone-ptq.gguf`, `zac-keep2.gguf`, `spk-encoder.gguf` (~1.2 GB) + the voice bank clips (Opus, ~33 MB).
Argument `vast` starts the Vast PyWorker (port 3000, route `/tts`); no argument starts the RunPod handler.

Request payload (Vast: `{"auth_data": <route answer>, "payload": {...}}`; RunPod: `input`):

| field | |
|---|---|
| `text` | text to speak; write numbers and abbreviations as words (Zonos normalizes only en/zh/ja/ko/de/es/fr/it/pt) |
| `language` | Zonos code: `ru`, `en_us`, `de`, `es`, `fr_fr`, `pt_br`, `cmn`, … (`slim/voices/<code>/`) |
| `voice` | bank voice, see `slim/voices/voices.json` (5 F, 5 M; same names in every language); cloned one-shot from `slim/voices/<code>/<voice>.ogg`, encoded once per worker |
| `reference_wav_base64` | instead of `voice`: clone a clip. Use a clip in the target language: cross-language clones get an accent and stop early |
| `seed` | optional |

Answer: `{wav_base64, audio_s, attempts, generation_s, model_load_s, gpu}`. A result shorter than the text allows
(chars / 25 s) is regenerated with the next seed, up to 3 attempts.

## Voice bank (`bank/`)

`GCP_CREDENTIALS=sa.json python3 bank/build_bank.py [codes…]` — Gemini TTS (Vertex AI) reads a neutral ~15 s text in
every Zonos language with each bank voice (`bank/refs/`, not committed). Convert the clips to Opus into
`slim/voices/<code>/` (`ffmpeg -i X.wav -ac 1 -c:a libopus -b:a 32k X.ogg`), commit and rebuild the image.

## Vast.ai

`VAST_API_KEY=… scripts/vast_setup.sh <image tag>` creates or updates template `zonos3-tts`, endpoint `zonos3-tts`
and its workergroup: cheap 8–24 GB Ampere/Ada cards under `VAST_MAX_DPH` (default $0.15/h); one stopped worker with
the image already pulled is kept as the cache (`cold_workers=1`, storage-only billing); active workers stop after
120 s without traffic.
