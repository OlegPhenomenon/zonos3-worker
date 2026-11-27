# studio-prep worker

Vast PyWorker for the realtime studio dub (`docs/architecture/studio-dub-realtime.md` in the app repo): the GPU and CPU heavy
steps next to the stems. Image `ghcr.io/olegphenomenon/zonos3-worker:studio-<sha>-lab-<lab sha>` (workflow
`.github/workflows/studio.yml`, `workflow_dispatch`, input `lab_ref`: the branch / commit of mobile-universal-dubber whose
`tool/studio_dub_lab` CI copies into `lab/` with the read-only deploy key `LAB_REPO_KEY`), endpoint `studio-prep`
(`scripts/studio_setup.sh <tag>`: cold 2, max 4, launch command in the workergroup's `launch_args`).

- `vast_worker.py` – routes `/prepare` (separation in the background, answers pending until done), `/ser` (emotion2vec+ large,
  ONNX on CUDA, through the lab's `emotion_ser.py`) and `/mix_piece` (the lab's `mix_clip.py` netflix on a piece cut from the
  cached stems and original, with the runner's dub of the piece), remote dispatch like `slim/`, disk cache `STUDIO_CACHE_DIR/<audioHash>/`
  with LRU cap `STUDIO_CACHE_GB` (40). Failures are answers `{"error": code, "detail"}` (HTTP 200), codes in the module docstring.
  The worker refuses to start when either model returns non-finite output or runs without CUDA.
- `runpod_lb.py` – the same routes as a plain HTTP server for a RunPod load-balancing endpoint (argument `runpod-lb`,
  `GET /ping` 204 while loading, 200 when ready): the fallback endpoint `studio-prep-fallback` (max 1 worker, so every
  request of a job reaches the worker holding its stems) that the runner uses when Vast has no worker.
- `sep.py` – dialogue/rest separation: Bandit v2 (multi) on ONNX Runtime GPU, no PyTorch. Speech branch only; `bg = mix - speech`.
  Chunks of 8 s, hop `SEP_HOP_S` (4), batch `SEP_BATCH` (2: ~2.9 GB GPU), fp16 weights with fp32 I/O.
- `tool/export_bandit.py` – how `bandit_speech.onnx` is made (needs a clone of kwatcharasupat/bandit-v2 and the Zenodo checkpoint,
  `python export_bandit.py checkpoint-multi.ckpt out.onnx speech`, then `onnxconverter_common.float16.convert_float_to_float16(keep_io_types=True)`).
  The file is the release asset `bandit-v2-speech-onnx` that CI bakes into the image; `emotion2vec-plus-large-onnx` likewise
  (made by the app repo's `tool/studio_dub_lab/emotion_onnx_export.py`).
- `assets/bench.flac` – the startup benchmark clip, `prune_fatbin.py` – copy of `slim/prune_fatbin.py`.

Model license: Bandit v2 code Apache-2.0, weights (Zenodo 12701995) CC BY-SA 4.0: commercial use allowed with attribution, the ONNX
conversion is an adaptation and stays CC BY-SA 4.0 (`NOTICE-BANDIT.txt` is in the image). Bandit v1 weights are CC BY-NC and are not used.

A new image reaches the pool only through a new workergroup (changing `launch_args` of a running group recycles its instances
in place with the old image): delete workergroup and endpoint, then run `scripts/studio_setup.sh <tag>`.
