"""Phase 5c (Colab, CPU, cached scores): a deployable alert policy that keeps as much recall as
possible while cutting false alarms to a livable rate.

    python -m safesound.phase5c --root /content/drive/MyDrive/safesound-tiny --data /content/cache

Per class, on top of the phase-5 method (network / smoothed / DSP fusion) and the DSP wake gate:
  persistence k-of-n: alert when the score is above threshold in k of the last n windows
                      (classic decision logic; long sounds such as alarms and sirens pass it,
                      one-off transients do not). Detection tolerance grows by the added delay.
  tiers: life-safety (alarm, siren) keep a recall target (95%, on tuning); the others get a
         false-alarm budget per class (on tuning negatives), car horn the strictest.
Each class picks the persistence that is best on TUNING data (fewest false alarms at the recall
target, or most recall within the budget). Totals = SUM over classes (what the user lives with).
"""
import argparse
import json
from functools import partial
from pathlib import Path

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from safesound import bench
from safesound.classes import CLASSES
from safesound.evaluate import Recording, threshold_for_budget
from safesound.evaluate import evaluate as _evaluate
from safesound.phase5 import HOP, SPLITS, Fusion, load_scores, threshold_for_recall
from safesound.phase5b import Gated, gate_open

GATE = ("G2_dsp", 0.3, 4)
PERSIST = [(1, 1), (2, 3), (2, 4), (3, 4), (3, 6), (4, 8)]
POLICIES = {
    # life-safety recall targets; other classes a budget on all tuning negatives
    "tiered": {"alarm": ("recall", 0.95), "siren": ("recall", 0.95), "doorbell": ("budget", 0.25),
               "knock": ("budget", 0.25), "baby_cry": ("budget", 0.25), "car_horn": ("budget", 0.05)},
    # deployment in a home: budgets on HOME tuning negatives only, summing to 1 alert per hour
    "home1": {"alarm": ("home_budget", 0.3), "siren": ("home_budget", 0.3), "doorbell": ("home_budget", 0.1),
              "knock": ("home_budget", 0.1), "baby_cry": ("home_budget", 0.1), "car_horn": ("home_budget", 0.1)},
    # same, 3 alerts per hour (alarm/siren get most of it)
    "home3": {"alarm": ("home_budget", 1.0), "siren": ("home_budget", 1.0), "doorbell": ("home_budget", 0.25),
              "knock": ("home_budget", 0.25), "baby_cry": ("home_budget", 0.25), "car_horn": ("home_budget", 0.25)},
}
POLICY = POLICIES["tiered"]
COLS = {"G2_dsp": [0, 1, 2, 3, 4, 5, 6], "G1_energy": [6]}


def kth_of_last_n(s, k, n):
    """Score whose crossing means 'k of the last n windows above threshold'."""
    if n == 1:
        return s
    pad = np.concatenate([np.zeros(n - 1, s.dtype), s])
    w = sliding_window_view(pad, n)
    return np.sort(w, axis=1)[:, n - k]


def build(items, k, n, gate):
    pos, neg_as, neg_home = [], [], []
    for kind, s, d, ev in items:
        if gate is not None:
            s = np.where(gate_open(d, COLS[gate[0]], gate[1], gate[2]), s, 0.0)
        r = Recording(kth_of_last_n(s, k, n), HOP, ev, HOP)
        (pos if kind == "pos" else neg_as if kind == "neg_as" else neg_home).append(r)
    return pos, neg_as, neg_home


def fa_rate(neg, th):
    _, _, nfa, h, _ = _evaluate(neg, th)
    return nfa / h if h else 0.0


def run(name, root, clips, dspd, p5, gate=GATE, policy=None):
    policy = policy or POLICY
    nn = {(s, k): load_scores(root, name, role) for s, roles in SPLITS.items() for k, role in roles.items()}
    out = {}
    for ci, cls in enumerate(CLASSES):
        fu = Fusion(ci, nn, dspd, clips)
        method = p5[name][cls]["chosen"]
        tune, test = Gated(fu, method, ci, "tune").items, Gated(fu, method, ci, "test").items
        kind, val = policy[cls]
        best = None
        for k, n in PERSIST:
            tol = 2.0 + (n - 1) * HOP
            ev = partial(_evaluate, decision_hits=True, tolerance_s=tol)
            tp, tas, thome = build(tune, k, n, gate)
            if kind == "recall":
                th = threshold_for_recall(tp, val) if n == 1 else _thr_recall(tp, val, ev)
                cost = fa_rate(tas + thome, th)
                key = cost
            elif kind == "home_budget":
                th = threshold_for_budget(thome, val)
                det, m, _, _, _ = ev(tp, th)
                key = -det / max(m, 1)
            else:
                th = threshold_for_budget(tas + thome, val)
                det, m, _, _, _ = ev(tp, th)
                key = -det / max(m, 1)
            if best is None or key < best[0]:
                best = (key, k, n, th, tol)
        _, k, n, th, tol = best
        ev = partial(_evaluate, decision_hits=True, tolerance_s=tol)
        sp, sas, shome = build(test, k, n, gate)
        det, m, _, _, hits = ev(sp, th)
        out[cls] = {"method": method, "policy": [kind, val], "k": k, "n": n, "threshold": th,
                    "test_recall": det / m, "occurrences": m, "fa_as": fa_rate(sas, th),
                    "fa_home": fa_rate(shome, th), "hits": hits.tolist()}
        print(f"  {name} {cls:9s} {kind:6s} {val:<5} k{k}/n{n}  recall {det / m:.1%}  FA/h AS {out[cls]['fa_as']:.2f} "
              f"home {out[cls]['fa_home']:.3f}", flush=True)
    tot = {"recall_macro": float(np.mean([out[c]["test_recall"] for c in CLASSES])),
           "recall_life_safety": float(np.mean([out[c]["test_recall"] for c in ("alarm", "siren") if c in out])),
           "total_fa_home_per_h": float(sum(out[c]["fa_home"] for c in CLASSES)),
           "total_fa_as_per_h": float(sum(out[c]["fa_as"] for c in CLASSES))}
    rng = np.random.default_rng(0)
    H = [np.array(out[c]["hits"], float) for c in CLASSES]
    boots = [np.mean([h[rng.integers(0, len(h), len(h))].mean() for h in H]) for _ in range(2000)]
    tot["recall_macro_ci"] = [float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975))]
    for c in CLASSES:
        out[c].pop("hits")
    out["total"] = tot
    print(f"  {name} TOTAL  macro recall {tot['recall_macro']:.1%} {np.round(tot['recall_macro_ci'], 3)}  "
          f"alarm+siren {tot['recall_life_safety']:.1%}  device FA/h home {tot['total_fa_home_per_h']:.2f}  "
          f"TV-like {tot['total_fa_as_per_h']:.1f}", flush=True)
    return out


def _thr_recall(pos, target, ev):
    """threshold_for_recall with a custom evaluate (tolerance for persistence delay)."""
    grid = np.unique(np.quantile(np.concatenate([r.scores for r in pos]), np.linspace(0, 1, 2001)))
    lo, hi = 0, len(grid) - 1
    while lo < hi:
        mid = (lo + hi + 1) // 2
        det, n, _, _, _ = ev(pos, grid[mid])
        if det / max(n, 1) >= target:
            lo = mid
        else:
            hi = mid - 1
    return float(grid[lo])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--data", type=Path, default=Path("/content/cache"))
    ap.add_argument("--policy", default="tiered", choices=list(POLICIES))
    ap.add_argument("--models", nargs="*", default=["p4_XL_logmel_kd_int8", "p3_teacher_raw"])
    a = ap.parse_args()
    clips = {(s, k): bench.Clips(a.data, role) for s, roles in SPLITS.items() for k, role in roles.items()}
    dspd = {(s, k): load_scores(a.root, "dsp_v1", role) for s, roles in SPLITS.items() for k, role in roles.items()}
    p5 = json.load(open(a.root / "results" / "phase5.json"))
    res = {}
    for name in a.models:
        print(f"\n### {name} policy {a.policy}", flush=True)
        res[name] = run(name, a.root, clips, dspd, p5, policy=POLICIES[a.policy])
        suffix = "" if a.policy == "tiered" else f"_{a.policy}"
        json.dump(res, open(a.root / "results" / f"phase5c{suffix}.json", "w"), indent=1)


if __name__ == "__main__":
    main()
