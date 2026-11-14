"""int8 simulation of a trained DSCNN, following the TFLite int8 scheme:
batch norm folded into conv weights; weights int8 symmetric per output channel;
activations int8 asymmetric per tensor, ranges calibrated on simulated training data.
(Phase 6 converts to real TFLite Micro and checks the outputs match.)"""
import copy

import torch
import torch.nn as nn

from safesound.models import ConvBN, DSCNN


def fold_bn(model: DSCNN) -> DSCNN:
    m = copy.deepcopy(model).cpu().eval()
    for mod in m.modules():
        if isinstance(mod, ConvBN) and isinstance(mod.bn, nn.BatchNorm2d):
            bn, conv = mod.bn, mod.conv
            s = bn.weight / torch.sqrt(bn.running_var + bn.eps)
            w = conv.weight * s[:, None, None, None]
            b = bn.bias - bn.running_mean * s
            new = nn.Conv2d(conv.in_channels, conv.out_channels, conv.kernel_size, conv.stride, conv.padding,
                            groups=conv.groups, bias=True)
            new.weight.data, new.bias.data = w.detach(), b.detach()
            mod.conv, mod.bn = new, nn.Identity()
    return m


def q_weight(w):
    scale = w.abs().flatten(1).amax(1).clamp(min=1e-8) / 127
    shape = (-1,) + (1,) * (w.dim() - 1)
    return torch.round(w / scale.view(shape)).clamp(-127, 127) * scale.view(shape)


class ActQuant(nn.Module):
    """Records min/max while calibrating, then fake-quantises to int8 (asymmetric)."""

    def __init__(self):
        super().__init__()
        self.lo, self.hi, self.calibrating = None, None, True

    def forward(self, x):
        if self.calibrating:
            lo, hi = x.min().item(), x.max().item()
            self.lo = lo if self.lo is None else min(self.lo, lo)
            self.hi = hi if self.hi is None else max(self.hi, hi)
            return x
        lo, hi = min(self.lo, 0.0), max(self.hi, 0.0)
        scale = max(hi - lo, 1e-8) / 255
        zp = round(-lo / scale) - 128
        q = torch.clamp(torch.round(x / scale) + zp, -128, 127)
        return (q - zp) * scale


class QDSCNN(nn.Module):
    def __init__(self, folded: DSCNN):
        super().__init__()
        self.m = folded
        for mod in self.m.modules():
            if isinstance(mod, (nn.Conv2d, nn.Linear)):
                mod.weight.data = q_weight(mod.weight.data)
        self.q_in = ActQuant()
        self.q_layers = nn.ModuleList([ActQuant() for _ in self.m.body])
        self.q_pool, self.q_out = ActQuant(), ActQuant()

    def forward(self, feats):
        m = self.m
        x = self.q_in(((feats - m.feat_mean) / m.feat_std).unsqueeze(1))
        for layer, q in zip(m.body, self.q_layers):
            x = q(layer(x))
        x = x.mean(2)
        x = self.q_pool(x.amax(2) + x.mean(2))
        return self.q_out(m.fc(x))

    def set_calibrating(self, on):
        for mod in self.modules():
            if isinstance(mod, ActQuant):
                mod.calibrating = on


def quantize(model: DSCNN, calib_feats: torch.Tensor) -> QDSCNN:
    q = QDSCNN(fold_bn(model)).eval()
    with torch.no_grad():
        for i in range(0, len(calib_feats), 256):
            q(calib_feats[i:i + 256])
    q.set_calibrating(False)
    return q


def model_bytes_int8(model: DSCNN) -> int:
    """Weights int8 + biases int32 after folding (what goes to flash, excluding the graph)."""
    f = fold_bn(model)
    n = 0
    for mod in f.modules():
        if isinstance(mod, (nn.Conv2d, nn.Linear)):
            n += mod.weight.numel() + 4 * (mod.bias.numel() if mod.bias is not None else 0)
    return n
