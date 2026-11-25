"""Dialogue-vs-rest separation with Bandit v2 (multi) on ONNX Runtime, no PyTorch.

The ONNX graph (tool/export_bandit.py) takes one mono 8 s chunk as a normalized-STFT-domain spectrogram and returns the
speech spectrogram already multiplied by its complex mask. STFT, inverse STFT and the chunk overlap-add run in numpy,
exactly as upstream's StandardTensorChunkedInferenceHandler does (periodic Hann chunk window, reflect padding, the two
channels as independent mono items), except for the hop (SEP_HOP_S) and the padding, which is only as long as needed.
"""
import os
import queue
import threading
from pathlib import Path

import numpy as np
import scipy.fft

FS = 48000
N_FFT, HOP = 2048, 512
CHUNK = 8 * FS
NF = N_FFT // 2 + 1
T = CHUNK // HOP + 1  # 751
WIN = np.hanning(N_FFT + 1)[:-1].astype(np.float32)  # periodic Hann
NORM = float(np.sqrt((WIN.astype(np.float64) ** 2).sum()))  # torchaudio normalized=True
# Overlap-add envelope of the squared window for T frames (torch.istft divides by it); trimmed to the center region.
_env = np.zeros(CHUNK + N_FFT, np.float32)
for _k in range(T):
    _env[_k * HOP:_k * HOP + N_FFT] += WIN ** 2
ENV = _env[N_FFT // 2:N_FFT // 2 + CHUNK]


def stft(x):
    """x [B, CHUNK] -> [B, NF, T] complex64, divided by NORM."""
    xp = np.pad(x, ((0, 0), (N_FFT // 2, N_FFT // 2)), mode='reflect')
    frames = np.lib.stride_tricks.as_strided(
        xp, (xp.shape[0], T, N_FFT), (xp.strides[0], xp.strides[1] * HOP, xp.strides[1]))
    s = scipy.fft.rfft(frames * WIN, axis=-1, workers=-1)
    return (s.transpose(0, 2, 1) / NORM).astype(np.complex64)


def istft(s_re, s_im):
    """Masked normalized spectrogram [B, NF, T] (re, im) -> [B, CHUNK] float32."""
    s = (s_re + 1j * s_im).transpose(0, 2, 1) * NORM
    fr = scipy.fft.irfft(s, n=N_FFT, axis=-1, workers=-1).astype(np.float32) * WIN  # B,T,N_FFT
    B = fr.shape[0]
    out = np.zeros((B, CHUNK + N_FFT), np.float32)
    g = fr.reshape(B, T, N_FFT // HOP, HOP)  # frame k covers blocks k..k+3
    for j in range(N_FFT // HOP):
        out[:, j * HOP:j * HOP + T * HOP] += g[:, :, j, :].reshape(B, T * HOP)
    return out[:, N_FFT // 2:N_FFT // 2 + CHUNK] / ENV


class Separator:
    def __init__(self, model_path, providers=None, hop_s=None, batch=None):
        import onnxruntime as ort
        so = ort.SessionOptions()
        so.log_severity_level = 3
        if providers is None:
            # Allocate exactly what is asked: the default arena doubles and keeps the whole card.
            providers = [('CUDAExecutionProvider', {'arena_extend_strategy': 'kSameAsRequested'}), 'CPUExecutionProvider']
        self.sess = ort.InferenceSession(str(model_path), so, providers=providers)
        self.hop = int(float(hop_s or os.environ.get('SEP_HOP_S', 4)) * FS)
        self.batch = int(batch or os.environ.get('SEP_BATCH', 8))
        self.window = (np.hanning(CHUNK + 1)[:-1] / (CHUNK / (2 * self.hop))).astype(np.float32)
        self.inputs = [i.name for i in self.sess.get_inputs()]
        self.dtype = np.float16 if 'float16' in self.sess.get_inputs()[0].type else np.float32

    def _chunks(self, n):
        pad = CHUNK - self.hop  # overlap: the first and last `pad` samples only get partial window coverage
        n_chunks = -(-(n + 2 * pad - CHUNK) // self.hop) + 1
        end = (n_chunks - 1) * self.hop + CHUNK - n - pad
        return pad, end, n_chunks

    def speech(self, x, progress=None):
        """x [2, n] float32 at 48 kHz -> speech estimate [2, n] float32."""
        ch, n = x.shape
        pad, end, n_chunks = self._chunks(n)
        xp = np.pad(x, ((0, 0), (pad, end + pad)), mode='reflect') if n > 1 else np.pad(x, ((0, 0), (pad, end + pad)))
        # reflect-pad may be shorter than the request for tiny inputs: np.pad repeats the reflection, so lengths match.
        out = np.zeros((ch, xp.shape[1]), np.float32)
        items = [(c, k) for k in range(n_chunks) for c in range(ch)]
        q = queue.Queue(maxsize=3)

        def producer():
            for i in range(0, len(items), self.batch):
                part = items[i:i + self.batch]
                seg = np.stack([xp[c, k * self.hop:k * self.hop + CHUNK] for c, k in part])
                s = stft(seg)
                q.put((part, np.ascontiguousarray(s.real).astype(self.dtype), np.ascontiguousarray(s.imag).astype(self.dtype)))
            q.put(None)

        th = threading.Thread(target=producer, daemon=True)
        th.start()
        done = 0
        while (job := q.get()) is not None:
            part, re, im = job
            o_re, o_im = self.sess.run(None, {self.inputs[0]: re, self.inputs[1]: im})
            y = istft(o_re.astype(np.float32), o_im.astype(np.float32)) * self.window
            for (c, k), yy in zip(part, y):
                out[c, k * self.hop:k * self.hop + CHUNK] += yy
            done += len(part)
            if progress:
                progress(done / len(items))
        th.join()
        return out[:, pad:pad + n]


def separate_file(mix_flac, out_dir, sep, subtype='PCM_24'):
    """mix.flac (stereo 48 kHz) -> speech.flac, bg.flac (= mix - speech) of the same length. Returns seconds spent."""
    import soundfile as sf
    x, fs = sf.read(str(mix_flac), dtype='float32', always_2d=True)
    assert fs == FS and x.shape[1] == 2, f'mix must be stereo 48 kHz, got {fs} Hz x {x.shape[1]}'
    x = np.ascontiguousarray(x.T)
    sp = sep.speech(x)
    out_dir = Path(out_dir)
    sf.write(str(out_dir / 'speech.flac'), np.clip(sp.T, -1, 1), FS, subtype=subtype)
    sf.write(str(out_dir / 'bg.flac'), np.clip((x - sp).T, -1, 1), FS, subtype=subtype)
