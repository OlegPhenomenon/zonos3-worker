# studio-prep worker

Vast PyWorker for the realtime studio dub (`docs/architecture/studio-dub-realtime.md` in the app repo). Image
`ghcr.io/olegphenomenon/zonos3-worker:studio-<sha>` (workflow `.github/workflows/studio.yml`, `workflow_dispatch`),
endpoint `studio-prep` (`scripts/studio_setup.sh <tag>`: cold 2, max 4, launch command in the workergroup's `launch_args`).

- `vast_worker.py` – routes `/prepare`, `/cues`, `/mix` (remote dispatch like `slim/`), disk cache `STUDIO_CACHE_DIR/<audioHash>/`
  with LRU cap `STUDIO_CACHE_GB` (40). Failures are answers `{"error": code, "detail"}` (HTTP 200), codes in the module docstring.
- `sep.py` – dialogue/rest separation: Bandit v2 (multi) on ONNX Runtime GPU, no PyTorch. Speech branch only; `bg = mix - speech`.
  Chunks of 8 s, hop `SEP_HOP_S` (4), batch `SEP_BATCH` (2: ~2.9 GB GPU), fp16 weights with fp32 I/O.
- `tool/export_bandit.py` – how `bandit_speech.onnx` is made (needs a clone of kwatcharasupat/bandit-v2 and the Zenodo checkpoint,
  `python export_bandit.py checkpoint-multi.ckpt out.onnx speech`, then `onnxconverter_common.float16.convert_float_to_float16(keep_io_types=True)`).
  The file is the release asset `bandit-v2-speech-onnx` that CI bakes into the image.
- `mix/` – the mix library (own owner), `prune_fatbin.py` – copy of `slim/prune_fatbin.py`.

Model license: Bandit v2 code Apache-2.0, weights (Zenodo 12701995) CC BY-SA 4.0: commercial use allowed with attribution, the ONNX
conversion is an adaptation and stays CC BY-SA 4.0 (`NOTICE-BANDIT.txt` is in the image). Bandit v1 weights are CC BY-NC and are not used.

A new image reaches the pool only through a new workergroup (changing `launch_args` of a running group recycles its instances
in place with the old image): delete workergroup and endpoint, then run `scripts/studio_setup.sh <tag>`.
