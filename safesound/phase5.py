"""Phase 5 (Colab): recall-first operating point, DSP detectors fused with the networks,
and the loud-sound safety net.

    python -m safesound.phase5 --root /content/drive/MyDrive/safesound-tiny --data /content/cache

Protocol (recall first): per class a target recall (95% alarm/siren, 90% others). On the TUNING
set we pick, per class, the method and the highest threshold that reach the target; the test
set then reports the recall actually reached and the false alarms per hour it costs.
Methods per class: nn | nn_smooth (mean of the last two windows) | or (max of nn and the
calibrated DSP score) | lr (logistic fusion of nn and DSP) - the one with the fewest tuning
false alarms at the target wins. The safety net adds a class-agnostic "loud sound" alert.
"""
import argparse
import json
import multiprocessing as mp
import pickle
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression

from safesound import bench, dsp
from safesound.classes import CLASSES
from functools import partial

from safesound.evaluate import Recording
from safesound.evaluate import evaluate as _evaluate

evaluate = partial(_evaluate, decision_hits=True)   # recall-first protocol

TARGET = {"alarm": 0.95, "siren": 0.95, "doorbell": 0.90, "knock": 0.90, "baby_cry": 0.90, "car_horn": 0.90}
MODELS = ["p3_teacher_raw", "p4_XL_logmel_kd_int8", "p4_L_logmel_kd_int8", "p4_M_logmel_kd_int8"]
HOP = 0.5
SPLITS = {"tune": bench.TUNE, "test": bench.TEST}


def load_scores(root, name, role):
    return pickle.load(open(Path(root) / "scores" / name / f"{role}.pkl", "rb"))


def dsp_scores(data, root):
    pool = mp.get_context("fork").Pool(max(1, mp.cpu_count() - 1))
    out = {}
    for split, roles in SPLITS.items():
        for k, role in roles.items():
            out[(split, k)] = bench.cached_scores(data, root, "dsp_v1", role, lambda b: dsp.detector(b, pool),
                                                  batch=256)
    pool.close()
    return out


def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def window_labels(clips, i, cls, n):
    """1 for windows overlapping an event of cls, 0 for windows of clips without it, -1 to skip
    (windows of a cls clip outside its events: label noise)."""
    ev = clips.events(i, cls)
    if not ev:
        return np.zeros(n)
    y = -np.ones(n)
    for k in range(n):
        t1 = (k + 1) * HOP
        if any(min(off, t1) - max(on, t1 - 2.0) > 0 for on, off in ev):
            y[k] = 1
    return y


class Fusion:
    """Per-class fused scores for one model."""

    def __init__(self, ci, nn, dspd, clips):
        self.ci, self.nn, self.dsp, self.clips = ci, nn, dspd, clips
        X, Y = [], []
        for k in ("pos", "neg_as", "neg_home"):
            c = clips[("tune", k)]
            for i in range(len(c)):
                a, d = nn[("tune", k)][i][:, ci], dspd[("tune", k)][i][:, ci]
                n = min(len(a), len(d))
                y = window_labels(c, i, CLASSES[ci], n)
                keep = y >= 0
                X.append(np.stack([logit(a[:n]), d[:n], logit(a[:n]) * d[:n]], 1)[keep])
                Y.append(y[keep])
        X, Y = np.concatenate(X), np.concatenate(Y)
        self.lr = LogisticRegression(class_weight="balanced", max_iter=500).fit(X, Y)
        self.cal = LogisticRegression(class_weight="balanced", max_iter=500).fit(X[:, 1:2], Y)

    def scores(self, method, a, d):
        n = min(len(a), len(d))
        a, d = a[:n], d[:n]
        if method == "nn":
            return a
        if method == "nn_smooth":
            return 0.5 * (a + np.concatenate([a[:1], a[:-1]]))
        if method == "dsp":
            return self.cal.predict_proba(d[:, None])[:, 1]
        if method == "or":
            return np.maximum(a, self.cal.predict_proba(d[:, None])[:, 1])
        if method == "lr":
            return self.lr.predict_proba(np.stack([logit(a), d, logit(a) * d], 1))[:, 1]
        raise ValueError(method)


def recordings(fusion, method, split, ci, extra=None):
    """Recordings for class ci under a method; `extra` = (loud threshold, ratio) adds the safety
    net: alert when max(score / th, loud / th_loud) >= 1, so we return that ratio with th = 1."""
    cls = CLASSES[ci]
    pos, neg = [], []
    for k in ("pos", "neg_as", "neg_home"):
        c = fusion.clips[(split, k)]
        for i in range(len(c)):
            s = fusion.scores(method, fusion.nn[(split, k)][i][:, ci], fusion.dsp[(split, k)][i][:, ci])
            if extra is not None:
                th, th_loud = extra
                loud = fusion.dsp[(split, k)][i][:len(s), 6]
                s = np.maximum(s / th, loud / th_loud)
            if k == "pos":
                ev = c.events(i, cls)
                if ev:
                    pos.append(Recording(s, HOP, ev, HOP))
            else:
                neg.append(Recording(s, HOP, [], HOP))
    return pos, neg


def threshold_for_recall(pos, target):
    """Highest threshold whose event recall on `pos` reaches the target (binary search)."""
    grid = np.unique(np.quantile(np.concatenate([r.scores for r in pos]), np.linspace(0, 1, 2001)))
    lo, hi = 0, len(grid) - 1
    while lo < hi:
        mid = (lo + hi + 1) // 2
        det, n, _, _, _ = evaluate(pos, grid[mid])
        if det / max(n, 1) >= target:
            lo = mid
        else:
            hi = mid - 1
    return float(grid[lo])


def fa_per_h(neg, th):
    _, _, nfa, h, _ = evaluate(neg, th)
    return nfa / h if h else 0.0


def split_neg(neg, fusion, split):
    n_as = len(fusion.clips[(split, "neg_as")])
    return neg[:n_as], neg[n_as:]


def run_model(name, data, root, clips, dspd, out):
    nn = {(s, k): load_scores(root, name, role) for s, roles in SPLITS.items() for k, role in roles.items()}
    res = {}
    for ci, cls in enumerate(CLASSES):
        if not any(clips[("tune", "pos")].events(i, cls) for i in range(len(clips[("tune", "pos")]))):
            continue
        fu = Fusion(ci, nn, dspd, clips)
        best = None
        rows = {}
        for method in ("dsp", "nn", "nn_smooth", "or", "lr"):
            pos_t, neg_t = recordings(fu, method, "tune", ci)
            th = threshold_for_recall(pos_t, TARGET[cls])
            fa_tune = fa_per_h(neg_t, th)
            pos, neg = recordings(fu, method, "test", ci)
            det, n, _, _, hits = evaluate(pos, th)
            nas, nhome = split_neg(neg, fu, "test")
            rows[method] = {"threshold": th, "tune_fa_per_h": round(fa_tune, 3), "test_recall": round(det / n, 4),
                            "test_fa_as": round(fa_per_h(nas, th), 3), "test_fa_home": round(fa_per_h(nhome, th), 3),
                            "occurrences": n}
            if method != "dsp" and (best is None or fa_tune < rows[best]["tune_fa_per_h"]):
                best = method
        res[cls] = {"chosen": best, "methods": rows}
        print(f"  {name} {cls:9s} chosen {best:9s} " + "  ".join(
            f"{m}: rec {r['test_recall']:.1%} FA {r['test_fa_as']:.1f}/{r['test_fa_home']:.2f}" for m, r in rows.items()),
            flush=True)

        # safety net on top of the chosen method: loud threshold set for 95% tuning recall of
        # this class's events by the loud detector alone (a generous net), then combined
        res[cls]["net"] = {}
        for th_loud in (0.7, 0.85, 1.0):
            pos, neg = recordings(fu, best, "test", ci, extra=(rows[best]["threshold"], th_loud))
            det, n, _, _, _ = evaluate(pos, 1.0)
            nas, nhome = split_neg(neg, fu, "test")
            res[cls]["net"][str(th_loud)] = {"test_recall": round(det / n, 4), "test_fa_as": round(fa_per_h(nas, 1.0), 3),
                                             "test_fa_home": round(fa_per_h(nhome, 1.0), 3)}
    out[name] = res


def summarise(out):
    print("\n=== Phase 5: recall-first (targets 95% alarm/siren, 90% others; thresholds from tuning)")
    print(f"{'model':24s} {'method':12s} {'macro test recall':>18s} {'FA/h AudioSet':>14s} {'FA/h home':>10s}")
    for name, res in out.items():
        for m in ("dsp", "nn", "nn_smooth", "or", "lr", "chosen"):
            pick = (lambda c: res[c]["methods"][res[c]["chosen"]]) if m == "chosen" else (lambda c: res[c]["methods"][m])
            cl = [c for c in CLASSES if c in res]
            rec = np.mean([pick(c)["test_recall"] for c in cl])
            fa = np.mean([pick(c)["test_fa_as"] for c in cl])
            fh = np.mean([pick(c)["test_fa_home"] for c in cl])
            print(f"{name:24s} {m:12s} {rec:18.1%} {fa:14.2f} {fh:10.2f}")
        for th in ("0.7", "0.85", "1.0"):
            cl = [c for c in CLASSES if c in res]
            rec = np.mean([res[c]["net"][th]["test_recall"] for c in cl])
            fa = np.mean([res[c]["net"][th]["test_fa_as"] for c in cl])
            fh = np.mean([res[c]["net"][th]["test_fa_home"] for c in cl])
            print(f"{name:24s} {'+net ' + th:12s} {rec:18.1%} {fa:14.2f} {fh:10.2f}")
        print("  per class (chosen): " + "  ".join(
            f"{c} {res[c]['chosen']} {res[c]['methods'][res[c]['chosen']]['test_recall']:.0%}" for c in CLASSES if c in res))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--data", type=Path, default=Path("/content/cache"))
    a = ap.parse_args()
    clips = {(s, k): bench.Clips(a.data, role) for s, roles in SPLITS.items() for k, role in roles.items()}
    print("DSP detector scores ...", flush=True)
    dspd = dsp_scores(a.data, a.root)
    out = {}
    for name in MODELS:
        print(f"\n### {name}", flush=True)
        run_model(name, a.data, a.root, clips, dspd, out)
    (a.root / "results").mkdir(parents=True, exist_ok=True)
    json.dump(out, open(a.root / "results" / "phase5.json", "w"), indent=1)
    summarise(out)


if __name__ == "__main__":
    main()
