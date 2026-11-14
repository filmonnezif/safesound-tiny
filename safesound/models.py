"""Student networks (depthwise-separable CNNs, MCU-friendly ops only) and the CNN14 teacher."""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from safesound.classes import AUDIOSET_MIDS, CLASSES
from safesound.frontend import FRAMES_2S, N_MELS


class ConvBN(nn.Module):
    def __init__(self, cin, cout, k, stride=1, groups=1, relu=True):
        super().__init__()
        self.conv = nn.Conv2d(cin, cout, k, stride, k // 2, groups=groups, bias=False)
        self.bn = nn.BatchNorm2d(cout)
        self.relu = relu

    def forward(self, x):
        x = self.bn(self.conv(x))
        return F.relu(x) if self.relu else x


class DSCNN(nn.Module):
    """DS-CNN (Zhang et al. 2017, keyword spotting) for 2 s of 40-band features.

    conv 3x3 /2  ->  N x [depthwise 3x3 (block 2 strided /2) + pointwise 1x1]
    -> mean over frequency -> max + mean over time -> linear (6 logits).
    Input features are normalised by fixed per-band mean/std (folded into the front end
    on the device)."""

    def __init__(self, width=64, blocks=4, n_mels=N_MELS, n_classes=len(CLASSES)):
        super().__init__()
        self.cfg = dict(width=width, blocks=blocks, n_mels=n_mels, n_classes=n_classes)
        self.register_buffer("feat_mean", torch.zeros(n_mels, 1))
        self.register_buffer("feat_std", torch.ones(n_mels, 1))
        layers = [ConvBN(1, width, 3, 2)]
        for b in range(blocks):
            layers += [ConvBN(width, width, 3, 2 if b == 1 else 1, groups=width), ConvBN(width, width, 1)]
        self.body = nn.Sequential(*layers)
        self.fc = nn.Linear(width, n_classes)

    def forward(self, feats):                  # (B, n_mels, frames)
        x = ((feats - self.feat_mean) / self.feat_std).unsqueeze(1)
        x = self.body(x).mean(2)               # (B, C, T)
        return self.fc(x.amax(2) + x.mean(2))


def count_params(m):
    return sum(p.numel() for p in m.parameters())


def count_macs(model, n_mels=N_MELS, frames=FRAMES_2S):
    macs = []

    def hook(mod, inp, out):
        if isinstance(mod, nn.Conv2d):
            k = mod.weight[0].numel()
            macs.append(out.numel() * k)
        elif isinstance(mod, nn.Linear):
            macs.append(mod.in_features * mod.out_features)
    hs = [m.register_forward_hook(hook) for m in model.modules() if isinstance(m, (nn.Conv2d, nn.Linear))]
    was = model.training
    model.eval()
    with torch.no_grad():
        model(torch.zeros(1, n_mels, frames, device=next(model.parameters()).device))
    model.train(was)
    for h in hs:
        h.remove()
    return int(sum(macs))


STUDENT_SIZES = {"S": (32, 4), "M": (64, 4), "L": (128, 5), "XL": (192, 6)}


class Teacher(nn.Module):
    """CNN14 (AudioSet-pretrained) with a new 6-class head, initialised from the AudioSet
    head rows of each class's labels."""

    def __init__(self, cnn14, audioset_mids):
        super().__init__()
        self.base = cnn14
        self.head = nn.Linear(2048, len(CLASSES))
        pos = {m: i for i, m in enumerate(audioset_mids)}
        with torch.no_grad():
            for k, c in enumerate(CLASSES):
                rows = [pos[m] for m in AUDIOSET_MIDS[c] if m in pos]
                self.head.weight[k] = cnn14.fc_audioset.weight[rows].mean(0)
                self.head.bias[k] = cnn14.fc_audioset.bias[rows].mean(0)

    def forward(self, wav, specaug=None):
        return self.head(self.base.embed(wav, specaug))


def feature_stats(feats: np.ndarray):
    """Per-band mean / std over a batch of features (B, n_mels, frames)."""
    m = feats.mean(axis=(0, 2))[:, None]
    s = feats.std(axis=(0, 2))[:, None] + 1e-3
    return torch.tensor(m, dtype=torch.float32), torch.tensor(s, dtype=torch.float32)
