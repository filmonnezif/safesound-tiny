"""Export a DS-CNN student for the WristEcho Lite Android app.

    python firmware/tools/export_android.py L ~/sound-aware/lite

Writes <app>/src/main/assets/dscnn.bin (float32 little-endian, BN folded, same order as
firmware/detector/ss_core.h), dscnn.json (shape, classes, thresholds at 1 FA/h) and a unit-test
fixture <app>/src/test/resources/fixture.json (int16 audio of a real siren clip + PyTorch scores).
"""
import json
import sys
import wave
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from export_model import load, thresholds  # noqa: E402
from safesound.classes import CLASSES  # noqa: E402
from safesound.frontend import mel_power  # noqa: E402

size, app = sys.argv[1], Path(sys.argv[2]).expanduser()
m = load(size)
W, B = m.cfg["width"], m.cfg["blocks"]
parts = [m.feat_mean, 1.0 / m.feat_std]
body = list(m.body)
parts += [body[0].conv.weight, body[0].conv.bias]
for b in range(B):
    dw, pw = body[1 + 2 * b].conv, body[2 + 2 * b].conv
    parts += [dw.weight, dw.bias, pw.weight, pw.bias]
parts += [m.fc.weight, m.fc.bias]
flat = np.concatenate([np.asarray(p.detach() if hasattr(p, "detach") else p, np.float32).ravel() for p in parts])
(app / "src/main/assets/dscnn.bin").write_bytes(flat.astype("<f4").tobytes())
meta = {"model": f"p4_{size}_logmel_kd", "width": W, "blocks": B, "classes": CLASSES,
        "thresholds_1fa_per_h": thresholds(size), "floats": int(flat.size)}
(app / "src/main/assets/dscnn.json").write_text(json.dumps(meta, indent=1))
print("weights", flat.size, "floats", meta)

# fixture: first 4 s of a real FSD50K siren clip, int16; PyTorch scores of the windows
w = wave.open(str(Path(__file__).resolve().parents[1] / "assets/clips/siren_58368.wav"))
x16 = np.frombuffer(w.readframes(4 * 16000), "<i2")
x = x16.astype(np.float32) / 32768
pad = 2 * 16000
F = np.log(mel_power(np.concatenate([np.zeros(pad, np.float32), x])) + 1.0)
ends = [(pad + (k + 1) * 8000 - 512) // 320 for k in range(len(x) // 8000)]
win = np.stack([F[:, e - 99:e + 1] for e in ends])
with torch.no_grad():
    p = torch.sigmoid(m(torch.from_numpy(win))).numpy()
fx = {"audio_int16": x16.tolist(), "pad": pad, "ends": ends,
      "mel_frame_first_window_last": F[:, ends[0]].tolist(), "probs": p.tolist()}
(app / "src/test/resources/fixture.json").write_text(json.dumps(fx))
print("fixture windows", len(ends), "max siren", p[:, 1].max())
