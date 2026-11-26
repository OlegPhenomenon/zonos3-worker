# zonos3-worker

ZONOS3-Mini TTS worker for Vast.ai Serverless (production) and RunPod Serverless, plus its multilingual voice bank.

## Image (`slim/`)

One image, `ghcr.io/olegphenomenon/zonos3-worker:runpod-slim-<sha>` (public), built by `.github/workflows/slim.yml`
without compiling: Ubuntu + the CUDA libraries the server links (cudart, cublas, cublasLt; GPU code for sm_80/86/89 only, NCCL replaced by a stub
because every worker has one GPU) + a 2 MB ffmpeg with just the audio decoders (`zonos3-server` shells out to it for voice clips; the build checks it
bit for bit against Ubuntu's ffmpeg on every clip) + prebuilt `zonos3-server` (release `ggml-cuda-sm86-sm89-96d9e0e`, Ampere/Ada)
+ `backbone-ptq.gguf`, `zac-keep2.gguf`, `spk-encoder.gguf`, `steer.gguf` (~1.2 GB) + the voice bank clips (Opus, ~33 MB). Python carries only what the
PyWorker imports (no pip, no `vastai` CLI/SDK tree, see the `pydeps` stage).
Argument `vast` starts the Vast PyWorker (port 3000, route `/tts`); no argument starts the RunPod handler, which needs
`docker build --build-arg RUNPOD=1` (the `runpod` package is not in the default image).

Request payload (Vast: `{"auth_data": <route answer>, "payload": {...}}`; RunPod: `input`):

| field | |
|---|---|
| `text` | text to speak; write numbers and abbreviations as words (Zonos normalizes only en/zh/ja/ko/de/es/fr/it/pt) |
| `language` | Zonos code: `ru`, `en_us`, `de`, `es`, `fr_fr`, `pt_br`, `cmn`, … (`slim/voices/<code>/`) |
| `voice` | bank voice, see `slim/voices/voices.json` (5 F, 5 M; same names in every language); cloned one-shot from `slim/voices/<code>/<voice>.ogg`, encoded once per worker |
| `reference_wav_base64` | instead of `voice`: clone a clip. Use a clip in the target language: cross-language clones get an accent and stop early |
| `seed` | optional |
| `emotion` | optional: a name (`"happy"`, default dose) or `{name: dose}`. angry 0..6 (default 4), happy 0..6 (4), sad 0..8 (6), calm 0..8 (4), scared 0..14 (12), confused 0..4 (3); `valence`, `arousal` -6..6 (dose required). Doses are raw, clamped to the range; Zyphra's tested ranges are about two thirds of the maximum. Unknown name: error |
| `controls` | optional: `{"<set>.<direction>": dose}`, clamped: `pitch.pitch`, `pitch.pitch_range` ±3; `delivery.tempo` ±3 (negative = slower), `delivery.whisper` 0..3, `delivery.loud` 0..4, `delivery.conversational` ±5; `expressive.expressive` ±16; `disfluency.disfluent` -8..20; `identity.gender`, `identity.age` ±3; `texture.deep`/`shrill`/`husky`/`raspy`/`soft`/`authoritative` 0..6; `room.clean` -4..6; the emotion ids `emotions.angry`/`happy`/`sad`/`serene`/`scared`/`confused`, `mood.valence`/`arousal`. `emotion` is sugar for these; `controls` wins on a clash. Unknown id: error |
| `speaking_rate` | optional: UTF-8 bytes of text per second of speech, clamped 8..30 (default 22, the worker's fixed pace) |
| `steer_hold` | optional, `1` only (a no-op). The server's fade-after-4-s needs a `steer_release` the published `steer.gguf` lacks, so controls are a constant push for the whole line; anything below 1 is an error. `controls_cfg_scale` is not offered (the server refuses it together with the speaker CFG the worker always uses) |

No prosody field = the request the worker has always sent = the same audio. The model has no `pitch_std`/`emotion` parameter of its own;
emotions are steering directions (`steer.gguf`, from Zyphra's steering bundle) that move the model's hidden state, so the effect
differs per voice, text and seed. Other keys are ignored and named in the answer's `ignored_fields`.

Answer: `{wav_base64, audio_s, attempts, generation_s, model_load_s, gpu}`. A result shorter than the text allows
(chars / 25 s) is regenerated with the next seed, up to 3 attempts.

## Voice bank (`bank/`)

`GCP_CREDENTIALS=sa.json python3 bank/build_bank.py [codes…]` — Gemini TTS (Vertex AI) reads a neutral ~15 s text in
every Zonos language with each bank voice (`bank/refs/`, not committed). Convert the clips to Opus into
`slim/voices/<code>/` (`ffmpeg -i X.wav -ac 1 -c:a libopus -b:a 32k X.ogg`), commit and rebuild the image.

## Vast.ai

`VAST_API_KEY=… scripts/vast_setup.sh <image tag>` creates or updates template `zonos3-tts`, endpoint `zonos3-tts`
and its workergroup: cheap 8–24 GB Ampere/Ada cards under `VAST_MAX_DPH` (default $0.15/h); three stopped workers on
different hosts with the image already pulled are kept as the cache (`cold_workers=3`, storage-only billing); the next
worker wakes at 70% load (`target_util`); active workers stop after 120 s without traffic. Capacity and latency:
`research/loadtest.py` (one RTX 4070: ~36 s of audio per second, ~20 dubbing viewers).

Switching the image without template rights works only for a workergroup that has no template: `PUT /api/v0/workergroups/<id>/` with
`{"client_id":"me","autojob_id":<id>,"endpoint_id":<eid>,"endpoint_name":"…","launch_args":"--image ghcr.io/olegphenomenon/zonos3-worker:<tag> --env '-p 3000:3000' --disk 16 --onstart-cmd 'cd /app && exec python3 -u vast_worker.py'"}`
answers `{"success": true}` and Vast recycles the workers onto that image. On a workergroup that has a template the template's image wins
and `launch_args` does not change it (tested 2026-11-25); with `--onstart-cmd` the template's `args` runtype breaks container creation.

Prod since 2026-11-26: endpoint `zonos3-tts` (44925) runs workergroup 56558 without a template on `runpod-slim-a88e1e7971d6b9555f196ddbd43e0c3e07078c46`
(1.3 GB compressed against 2.4 GB for `runpod-slim-72cb61ef99fc75c7801fc7d7e74b535d3d6d4adf`; byte-identical audio to the old image on the same
GPU model, 19 requests each on RTX 4060 Ti, RTX 3060, RTX A4000), cards RTX 3060 / 3060 Ti / 3070 / A4000 / 4060 Ti up to $0.12/h,
`cold_workers=3`, `max_workers=3`. One pool serves the app's old path, the extensions' gateway and the studio's emotional dubs: the separate
`zonos3-tts-emo` endpoint (46194) is gone, its workergroup was moved onto 44925 (`PUT` with the new `endpoint_id`), and the templated
workergroup 55211 was deleted. `scripts/vast_setup.sh` still describes the templated setup.
