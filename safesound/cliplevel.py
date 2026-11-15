"""Clip-level metrics (the usual literature view), from the cached window scores.

A clip's score for a class = max over its windows. Per class:
  positives = clips with that class (test_pos), negatives = every other test clip
  (other classes' event clips, AudioSet negatives, SINS home segments).
Reports AUC and average precision (threshold-free), and precision / recall / F1 / accuracy
at the threshold that maximises F1 on the TUNING clips (applied unchanged to test).

    python -m safesound.cliplevel --root /content/drive/MyDrive/safesound-tiny --data /content/cache
"""
import argparse
import json
import pickle
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score

from safesound.bench import TEST, TUNE, Clips
from safesound.classes import CLASSES


def clip_matrix(data, root, name, roles):
    """(scores [n_clips, 6], labels [n_clips, 6]) for the given roles."""
    S, Y = [], []
    for role in roles.values():
        c = Clips(data, role)
        sc = pickle.load(open(Path(root) / "scores" / name / f"{role}.pkl", "rb"))
        assert len(sc) == len(c), (name, role)
        S.append(np.stack([s.max(0) if len(s) else np.zeros(len(CLASSES)) for s in sc]))
        Y.append(np.array([[float(k in c.classes[i]) for k in CLASSES] for i in range(len(c))]))
    return np.concatenate(S), np.concatenate(Y)


def best_f1_threshold(y, s):
    p, r, t = precision_recall_curve(y, s)
    f1 = 2 * p * r / np.maximum(p + r, 1e-9)
    return float(t[np.argmax(f1[:-1])])


def evaluate(data, root, name):
    St, Yt = clip_matrix(data, root, name, TUNE)
    Ss, Ys = clip_matrix(data, root, name, TEST)
    out = {}
    for k, cls in enumerate(CLASSES):
        th = best_f1_threshold(Yt[:, k], St[:, k])
        y, s = Ys[:, k], Ss[:, k]
        pred = s >= th
        tp, fp = int((pred & (y == 1)).sum()), int((pred & (y == 0)).sum())
        fn, tn = int((~pred & (y == 1)).sum()), int((~pred & (y == 0)).sum())
        prec = tp / max(tp + fp, 1)
        rec = tp / max(tp + fn, 1)
        out[cls] = {"auc": roc_auc_score(y, s), "ap": average_precision_score(y, s), "precision": prec,
                    "recall": rec, "f1": 2 * prec * rec / max(prec + rec, 1e-9),
                    "accuracy": (tp + tn) / len(y), "balanced_acc": 0.5 * (rec + tn / max(tn + fp, 1))}
    macro = {m: float(np.mean([out[c][m] for c in CLASSES])) for m in out[CLASSES[0]]}
    return macro, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--data", type=Path, default=Path("/content/cache"))
    a = ap.parse_args()
    names = sorted(p.name for p in (a.root / "scores").iterdir()
                   if all((p / f"{r}.pkl").exists() for r in list(TEST.values()) + list(TUNE.values())))
    rows = {}
    print(f"{'model':24s} {'AUC':>6s} {'AP':>6s} {'prec':>6s} {'recall':>6s} {'F1':>6s} {'acc':>7s} {'bal.acc':>7s}")
    for n in names:
        macro, per = evaluate(a.data, a.root, n)
        rows[n] = {"macro": macro, "classes": per}
        print(f"{n:24s} {macro['auc']:6.3f} {macro['ap']:6.3f} {macro['precision']:6.1%} {macro['recall']:6.1%}"
              f" {macro['f1']:6.1%} {macro['accuracy']:7.2%} {macro['balanced_acc']:7.1%}")
    json.dump(rows, open(a.root / "results" / "cliplevel.json", "w"), indent=1)


if __name__ == "__main__":
    main()
