"""Phase 1 (Colab): lock the test set; zero-shot baselines YAMNet and pretrained CNN14.

    python -m safesound.phase1 --root /content/drive/MyDrive/safesound-tiny
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from safesound import bench
from safesound.classes import AUDIOSET_MIDS, CLASSES

WIN, HOP = 32000, 8000   # 2 s windows every 0.5 s (shared by every windowed model)


def mid_columns(mids_in_order):
    """For each class, the column indices of its AudioSet MIDs in a model's output."""
    pos = {m: i for i, m in enumerate(mids_in_order)}
    return [[pos[m] for m in AUDIOSET_MIDS[c] if m in pos] for c in CLASSES]


def yamnet_detector():
    import tensorflow as tf
    tf.config.set_visible_devices([], "GPU")      # leave the GPU to PyTorch
    import tensorflow_hub as hub
    model = hub.load("https://tfhub.dev/google/yamnet/1")
    cmap = pd.read_csv(model.class_map_path().numpy().decode())
    cols = mid_columns(list(cmap.mid))

    def det(batch):
        out = []
        for x in batch:
            s = model(x)[0].numpy()               # (frames, 521), 0.96 s windows every 0.48 s
            out.append(np.stack([s[:, c].max(1) for c in cols], 1))
        return out
    return det


def cnn14_detector(work="/content/work"):
    import torch
    from safesound.cnn14 import fetch, load_pretrained
    ck, mids = fetch(work)
    cols = mid_columns(mids)
    model = load_pretrained(ck).cuda().eval()

    @torch.no_grad()
    def det(batch):
        wins = [bench.windows(x, WIN, HOP) for x in batch]
        flat = torch.from_numpy(np.concatenate(wins)).cuda()
        p = torch.cat([model(flat[i:i + 1024]) for i in range(0, len(flat), 1024)]).cpu().numpy()
        p = np.stack([p[:, c].max(1) for c in cols], 1)
        return np.split(p, np.cumsum([len(w) for w in wins])[:-1])
    return det


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True)
    a = ap.parse_args()
    data = bench.local_copy(a.root)
    res_dir = a.root / "results"
    res_dir.mkdir(parents=True, exist_ok=True)

    # 1. lock the test set: counts + hash, written once
    lock = res_dir / "test_lock.json"
    h = bench.manifest_hash(data)
    if lock.exists():
        old = json.load(open(lock))
        assert old["sha256"] == h, "TEST SET CHANGED since it was locked"
    else:
        counts = {}
        for role in bench.TEST.values():
            c = bench.Clips(data, role)
            counts[role] = {"clips": len(c), "hours": round(c.meta.n.sum() / 16000 / 3600, 2),
                            **{cls: int(c.meta.classes.str.split(";").apply(lambda L: cls in L).sum())
                               for cls in CLASSES}}
        json.dump({"sha256": h, "counts": counts}, open(lock, "w"), indent=1)
    print("test set sha256:", h)
    print(json.dumps(json.load(open(lock))["counts"], indent=1))

    # 2. baselines
    bench.run(data, a.root, "yamnet_zeroshot", yamnet_detector(), hop=0.48, t0=0.96,
              out_json=res_dir / "phase1_yamnet.json")
    bench.run(data, a.root, "cnn14_zeroshot", cnn14_detector(), hop=HOP / 16000, t0=HOP / 16000,
              out_json=res_dir / "phase1_cnn14.json")


if __name__ == "__main__":
    main()
