"""Phase 6 (Colab): pretrained efficient CNN students.

Instead of training a tiny CNN from scratch, start from EfficientAT MobileNetV3 models
(Schmid et al., ICASSP 2023; MIT licence, github.com/fschmid56/EfficientAT) that were
pre-trained on all of AudioSet by transformer-to-CNN distillation, and fine-tune them on the six
classes with the same simulated data and the same CNN14 teacher as phase 4.

    python -m safesound.phase6 --root /content/drive/MyDrive/safesound-tiny [--sizes mn04 mn02 mn01]

Front end, equivalent to EfficientAT's (32 kHz, 1024-FFT, 25 ms window, 10 ms hop, 128 mel bands
0-15 kHz, pre-emphasis 0.97) but computed natively on 16 kHz audio as the device would: 512-FFT
(same 31.25 Hz bins), 400-sample window, 160 hop, pre-emphasis applied as the 32 kHz filter's
response in the frequency domain, power x4 (window twice as long at 32 kHz); the mel bands above
8 kHz get no energy (constant floor).

Int8 is simulated: batch norm folded, weights per-channel symmetric int8, activations of every
convolution, linear layer, non-linearity, squeeze-excitation and residual block fake-quantised
(asymmetric, min/max calibrated on simulated training data).
"""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from safesound import bench
from safesound.classes import AUDIOSET_MIDS, CLASSES
from safesound.quant import ActQuant, q_weight
from safesound.sim import CONFIGS, LABEL
from safesound.train import HOP, WIN, loader

EAT_DIR = "/content/EfficientAT"
SIZES = {"mn01": (0.1, "mn01_as"), "mn02": (0.2, "mn02_as"), "mn04": (0.4, "mn04_as"), "mn10": (1.0, "mn10_as")}
N_FFT, WIN_LEN, HOPSIZE, N_MELS = 512, 400, 160, 128


# ------------------------------------------------------------------ front end

def kaldi_mel_banks(n_mels=128, n_fft=1024, sr=32000, fmin=0.0, fmax=15000.0):
    """Triangular mel filters as torchaudio.compliance.kaldi.get_mel_banks (no VTLN), plus a
    zero column for the Nyquist bin: (n_mels, n_fft // 2 + 1)."""
    mel = lambda f: 1127.0 * np.log(1.0 + np.asarray(f, float) / 700.0)
    bins = mel(sr / n_fft * np.arange(n_fft // 2))
    lo, hi = mel(fmin), mel(fmax)
    d = (hi - lo) / (n_mels + 1)
    fb = np.zeros((n_mels, n_fft // 2 + 1), np.float32)
    for i in range(n_mels):
        left, centre, right = lo + i * d, lo + (i + 1) * d, lo + (i + 2) * d
        fb[i, :-1] = np.maximum(0, np.minimum((bins - left) / (centre - left), (right - bins) / (right - centre)))
    return fb


class Mel16k(nn.Module):
    """EfficientAT-compatible log-mel computed from 16 kHz audio. (B, samples) -> (B, 1, 128, frames)."""

    def __init__(self):
        super().__init__()
        fb = kaldi_mel_banks()[:, : N_FFT // 2 + 1]                 # bins 0..8 kHz of the 32 kHz bank
        f = np.arange(N_FFT // 2 + 1) * 16000 / N_FFT
        pre = np.abs(1 - 0.97 * np.exp(-2j * np.pi * f / 32000)) ** 2   # 32 kHz pre-emphasis response
        self.register_buffer("fb", torch.tensor(fb * pre[None, :] * 4.0))
        self.register_buffer("window", torch.hann_window(WIN_LEN, periodic=False))

    def forward(self, x, specaug=False):
        with torch.autocast(x.device.type, enabled=False):
            s = torch.stft(x.float(), N_FFT, hop_length=HOPSIZE, win_length=WIN_LEN, window=self.window,
                           center=True, return_complex=True).abs() ** 2          # (B, 257, T)
            m = (torch.matmul(self.fb, s) + 1e-5).log()
            m = (m + 4.5) / 5.0
        if specaug:
            B, M, T = m.shape
            for b in range(B):                                                    # light SpecAugment
                f0 = np.random.randint(0, 96); m[b, f0:f0 + np.random.randint(0, 16)] = m[b].mean()
                t0 = np.random.randint(0, T - 20); m[b, :, t0:t0 + np.random.randint(0, 20)] = m[b].mean()
        return m.unsqueeze(1)


# ------------------------------------------------------------------ model

class Student(nn.Module):
    def __init__(self, size, mids):
        super().__init__()
        if not Path(EAT_DIR).exists():
            subprocess.run(f"git clone -q https://github.com/fschmid56/EfficientAT {EAT_DIR}", shell=True, check=True)
        if EAT_DIR not in sys.path:
            sys.path.insert(0, EAT_DIR)
        import contextlib
        import io
        import os
        width, name = SIZES[size]
        cwd = os.getcwd()
        os.chdir(EAT_DIR)                       # the repository reads files relative to its root
        try:
            from models.mn.model import get_model
            with contextlib.redirect_stdout(io.StringIO()):
                self.net = get_model(width_mult=width, pretrained_name=name)
        finally:
            os.chdir(cwd)
        old = self.net.classifier[5]
        new = nn.Linear(old.in_features, len(CLASSES))
        pos = {m: i for i, m in enumerate(mids)}
        with torch.no_grad():                                    # start from the AudioSet rows of each class
            for k, c in enumerate(CLASSES):
                rows = [pos[m] for m in AUDIOSET_MIDS[c] if m in pos]
                new.weight[k] = old.weight[rows].mean(0)
                new.bias[k] = old.bias[rows].mean(0)
        self.net.classifier[5] = new
        self.mel = Mel16k()
        self.size = size

    def logits_from_mel(self, m):
        out = self.net(m)[0]
        return out.view(m.shape[0], -1)

    def forward(self, wav, specaug=False):
        return self.logits_from_mel(self.mel(wav, specaug))


def count(model):
    """Parameters and multiply-accumulates of the CNN for one 2 s window."""
    macs = []

    def hook(mod, inp, out):
        if isinstance(mod, nn.Conv2d):
            macs.append(out.numel() / out.shape[0] * mod.weight[0].numel())
        else:
            macs.append(mod.in_features * mod.out_features)
    hs = [m.register_forward_hook(hook) for m in model.net.modules() if isinstance(m, (nn.Conv2d, nn.Linear))]
    with torch.no_grad():
        model(torch.zeros(2, WIN, device=next(model.parameters()).device))
    for h in hs:
        h.remove()
    return sum(p.numel() for p in model.net.parameters()), int(sum(macs))


def detector(model, logits_fn=None):
    fn = logits_fn or (lambda w: model(w))

    @torch.no_grad()
    def det(batch):
        wins = [bench.windows(x, WIN, HOP) for x in batch]
        flat = torch.from_numpy(np.concatenate(wins)).cuda()
        out = []
        for i in range(0, len(flat), 1024):
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=logits_fn is None):
                out.append(torch.sigmoid(fn(flat[i:i + 1024]).float()))
        p = torch.cat(out).cpu().numpy()
        return np.split(p, np.cumsum([len(w) for w in wins])[:-1])
    return det


def train(data, cfg, size, mids, teacher, steps, batch=128, lr=5e-4, kd=0.5):
    model = Student(size, mids).cuda().train()
    it = loader(data, cfg, batch)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps, pct_start=0.1)
    t, run = time.time(), 0.0
    for step in range(1, steps + 1):
        x, y = next(it)
        x, y = x.cuda(non_blocking=True)[:, -LABEL:], y.cuda(non_blocking=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(x, specaug=True).float()
            loss = F.binary_cross_entropy_with_logits(logits, y)
            if teacher is not None:
                with torch.no_grad():
                    tp = torch.sigmoid(teacher(x).float())
                loss = (1 - kd) * loss + kd * F.binary_cross_entropy_with_logits(logits, tp)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()
        run = 0.98 * run + 0.02 * loss.item()
        if step % 500 == 0:
            print(f"  step {step}/{steps} loss {run:.4f} {batch * 500 / (time.time() - t):.0f} ex/s", flush=True)
            t = time.time()
    return model.eval()


# ------------------------------------------------------------------ int8 simulation

def fold_all_bn(net):
    """Fold every BatchNorm2d that directly follows a Conv2d inside a Sequential."""
    from torch.nn.utils.fusion import fuse_conv_bn_eval
    for mod in net.modules():
        if isinstance(mod, nn.Sequential):
            keys = list(mod._modules.keys())
            for a, b in zip(keys, keys[1:]):
                if isinstance(mod._modules[a], nn.Conv2d) and isinstance(mod._modules[b], nn.BatchNorm2d):
                    mod._modules[a] = fuse_conv_bn_eval(mod._modules[a], mod._modules[b])
                    mod._modules[b] = nn.Identity()
    return net


def quantize(model, calib_wave):
    """Returns (logits_fn, n_bn_left). calib_wave: (N, samples) tensor on cpu."""
    import copy
    q = copy.deepcopy(model).eval()
    fold_all_bn(q.net)
    left = sum(isinstance(m, nn.BatchNorm2d) for m in q.net.modules())
    quants = []

    def add(mod):
        aq = ActQuant()
        quants.append(aq)
        mod.register_forward_hook(lambda m, i, o: aq(o))
    for name, mod in q.net.named_modules():
        if isinstance(mod, (nn.Conv2d, nn.Linear)):
            mod.weight.data = q_weight(mod.weight.data)
            add(mod)
        elif isinstance(mod, (nn.ReLU, nn.Hardswish, nn.Hardsigmoid, nn.Sigmoid)):
            mod.inplace = False
            add(mod)
        elif type(mod).__name__ in ("InvertedResidual", "SqueezeExcitation", "ConcurrentSEBlock"):
            add(mod)
    in_q = ActQuant()
    quants.append(in_q)

    def fn(w):
        return q.logits_from_mel(in_q(q.mel(w)))
    with torch.no_grad():
        for i in range(0, len(calib_wave), 256):
            fn(calib_wave[i:i + 256].to(next(q.parameters()).device))
    for a in quants:
        a.calibrating = False
    return fn, left


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--sizes", nargs="*", default=["mn04", "mn02", "mn01"])
    ap.add_argument("--steps", type=int, default=10000)
    ap.add_argument("--no-kd", action="store_true")
    a = ap.parse_args()
    from safesound.cnn14 import fetch
    from safesound.phases import load_teacher, macro
    data = bench.local_copy(a.root)
    _, mids = fetch()
    cfg = CONFIGS[json.load(open(a.root / "results" / "phase2_choice.json"))["config"]]
    teacher = None if a.no_kd else load_teacher(a.root, "p3_teacher_raw")
    summary_f = a.root / "results" / "phase6_summary.json"
    summary = json.load(open(summary_f)) if summary_f.exists() else {}
    for size in a.sizes:
        name = f"p6_{size}" + ("_hard" if a.no_kd else "_kd")
        ck = a.root / "checkpoints" / f"{name}.pt"
        if ck.exists():
            model = Student(size, mids).cuda()
            model.load_state_dict(torch.load(ck, map_location="cuda"))
            model.eval()
        else:
            print(f"\n### training {name}", flush=True)
            model = train(data, cfg, size, mids, teacher, a.steps)
            torch.save(model.state_dict(), ck)
        params, macs = count(model)
        out = a.root / "results" / f"{name}.json"
        r = json.load(open(out)) if out.exists() else bench.run(
            data, a.root, name, detector(model), hop=0.5, t0=0.5, out_json=out)
        qout = a.root / "results" / f"{name}_int8.json"
        if qout.exists():
            rq = json.load(open(qout))
        else:
            it = loader(data, cfg, 256, seed=99)
            calib = torch.cat([next(it)[0][:, -LABEL:] for _ in range(8)])
            fn, left = quantize(model, calib)
            print(f"  int8 simulation: {left} batch norms not folded", flush=True)
            rq = bench.run(data, a.root, name + "_int8", detector(model, fn), hop=0.5, t0=0.5, out_json=qout)
        summary[name] = {"params": params, "macs_per_inference": macs, "float": macro(r), "int8": macro(rq),
                         "tune_float": macro(r, "tune_recall")}
        json.dump(summary, open(summary_f, "w"), indent=1)
        print(f"{name}: params {params} MACs {macs / 1e6:.1f}M  float {summary[name]['float']}  int8 {summary[name]['int8']}",
              flush=True)


if __name__ == "__main__":
    main()
