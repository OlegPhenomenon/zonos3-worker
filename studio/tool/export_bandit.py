"""Export Bandit v2 (multi) to ONNX: masked complex spectrograms for the chosen stems.
python export_bandit.py checkpoint-multi.ckpt out.onnx speech[,music,sfx]
Input spec_r/spec_i [B,1025,T] (un-normalized STFT, n_fft 2048, hop 512, periodic hann, reflect center), output per stem
<stem>_r/<stem>_i masked spectrogram [B,1025,T] in the same (un-normalized) units.
"""
import sys
import torch
import torch.nn.functional as F
from torch import nn

sys.path.insert(0, 'b2')
from src.models.bandit.bandit import Bandit  # noqa: E402

KW = dict(in_channels=1, stems=['speech', 'music', 'sfx'], band_type='musical', n_bands=64,
          normalize_channel_independently=False, treat_channel_as_feature=True, n_sqm_modules=8, emb_dim=128,
          rnn_dim=256, bidirectional=True, rnn_type='GRU', mlp_dim=512, hidden_activation='Tanh',
          hidden_activation_kwargs=None, complex_mask=True, use_freq_weights=True, n_fft=2048, win_length=2048,
          hop_length=512, window_fn='hann_window', wkwargs=None, power=None, center=True, normalized=True,
          pad_mode='reflect', onesided=True, fs=48000)


def load(ckpt):
    m = Bandit(**KW)
    sd = torch.load(ckpt, map_location='cpu', weights_only=False)['state_dict']
    sd = {k[len('model.'):]: v for k, v in sd.items() if k.startswith('model.')}
    print(m.load_state_dict(sd, strict=False))
    return m.eval()


def unwrap(mod):
    return getattr(mod, '_orig_mod', mod)


class Net(nn.Module):
    def __init__(self, m, stems):
        super().__init__()
        self.m, self.stems = m, stems
        self.specs = m.band_specs.get_band_specs()
        self.nf = KW['n_fft'] // 2 + 1

    def forward(self, r, i):
        m = self.m
        x = torch.stack([r, i], -1).permute(0, 2, 1, 3)  # B,T,F,2
        B, T = x.shape[0], x.shape[1]
        zs = []
        for (a, b), nfm in zip(self.specs, m.band_split.norm_fc_modules):
            xb = x[:, :, a:b, :].reshape(B, T, -1)
            zs.append(unwrap(nfm).combined(xb))
        z = torch.stack(zs, 1)  # B,64,T,128
        for layer in unwrap(m.tf_model).seqband:
            z = layer(z)
        out = []
        for stem in self.stems:
            me = m.mask_estim[stem]
            mr = mi = 0
            for bi, (a, b) in enumerate(self.specs):
                qb = z[:, bi]
                mb = unwrap(me.norm_mlp[bi]).combined(qb)  # B,T,bw*2
                mb = mb.reshape(B, T, b - a, 2)
                fw = me.get_buffer(f'freq_weights/{bi}')[None, None, :, None]
                mb = (mb * fw).permute(0, 2, 1, 3)  # B,bw,T,2
                mr = mr + F.pad(mb[..., 0], (0, 0, a, self.nf - b))
                mi = mi + F.pad(mb[..., 1], (0, 0, a, self.nf - b))
            out += [r * mr - i * mi, r * mi + i * mr]
        return tuple(out)


if __name__ == '__main__':
    ckpt, out, stems = sys.argv[1], sys.argv[2], sys.argv[3].split(',')
    net = Net(load(ckpt), stems).eval()
    r = torch.randn(1, 1025, 751)
    i = torch.randn(1, 1025, 751)
    names = [f'{s}_{c}' for s in stems for c in 'ri']
    torch.onnx.export(net, (r, i), out, input_names=['spec_r', 'spec_i'], output_names=names, opset_version=17,
                      dynamic_axes={n: {0: 'b', 2: 't'} for n in ['spec_r', 'spec_i'] + names}, dynamo=False)
    print('exported', out)
