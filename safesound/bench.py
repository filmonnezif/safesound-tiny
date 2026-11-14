"""Shared benchmark: score a detector on tuning + test roles, set thresholds on tuning,
report test recall (with 95% CI) and false alarms per hour (docs/protocol.md).

A detector is a function   audio(float32 16 kHz) list -> list of (frames, 6) score arrays,
with a fixed hop and t0 (time of the first score). Scores are cached on Drive per role.
"""
import hashlib
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from safesound.audio import to_float
from safesound.classes import CLASSES
from safesound.evaluate import Recording, bootstrap_recall, evaluate, threshold_for_budget
from safesound.store import Store

TEST = {"pos": "test_pos", "neg_as": "test_neg_as", "neg_home": "test_neg_home"}
TUNE = {"pos": "tune_pos", "neg_as": "tune_neg_as", "neg_home": "tune_neg_home"}
BUDGETS = [0.1, 1.0]


class Clips:
    """All sources of one role, concatenated."""

    def __init__(self, root, role):
        self.stores = [Store(p.parent) for p in sorted((Path(root) / "stores" / role).glob("*/meta.csv"))]
        metas = []
        for k, s in enumerate(self.stores):
            m = s.meta.copy()
            m["_store"], m["_row"] = k, np.arange(len(m))
            m["src"] = s.path.name
            metas.append(m)
        self.meta = pd.concat(metas, ignore_index=True)

    def __len__(self):
        return len(self.meta)

    def audio(self, i):
        r = self.meta.iloc[i]
        return to_float(self.stores[r._store].audio(int(r._row)))

    def events(self, i, cls):
        return self.stores[self.meta._store.iloc[i]].events(int(self.meta._row.iloc[i]), cls)


def manifest_hash(root):
    """Hash of the locked test set (sorted uids per role)."""
    h = hashlib.sha256()
    for role in TEST.values():
        for uid in sorted(Clips(root, role).meta.uid):
            h.update(f"{role}:{uid}\n".encode())
    return h.hexdigest()


MIN_LEN = int(2.5 * 16000)   # very short clips (e.g. a single knock) get trailing silence


def cached_scores(data_root, out_root, name, role, detector, clips=None, batch=64):
    """Run `detector` over every clip of `role` (or load the cache). Returns list of arrays."""
    f = Path(out_root) / "scores" / name / f"{role}.pkl"
    if f.exists():
        return pickle.load(open(f, "rb"))
    clips = clips or Clips(data_root, role)

    def get(i):
        x = clips.audio(i)
        return np.pad(x, (0, max(0, MIN_LEN - len(x))))
    out = []
    for b in range(0, len(clips), batch):
        out += detector([get(i) for i in range(b, min(b + batch, len(clips)))])
    f.parent.mkdir(parents=True, exist_ok=True)
    pickle.dump([o.astype(np.float16) for o in out], open(f, "wb"))
    return out


def recordings(clips, scores, ci, cls, hop, t0, positive):
    recs = []
    for i in range(len(clips)):
        s = scores[i][:, ci].astype(np.float32)
        if positive:
            ev = clips.events(i, cls)
            if ev:
                recs.append(Recording(s, hop, ev, t0))
        else:
            recs.append(Recording(s, hop, [], t0))
    return recs


def run(data_root, out_root, name, detector, hop, t0, out_json=None):
    """Full benchmark of one detector. Returns a results dict and prints a table.
    `data_root` holds the stores (a fast local copy), `out_root` the score cache (Drive)."""
    sets = {}
    for split, roles in [("tune", TUNE), ("test", TEST)]:
        for k, role in roles.items():
            c = Clips(data_root, role)
            sets[(split, k)] = (c, cached_scores(data_root, out_root, name, role, detector, c))
    res = {"detector": name, "hop": hop, "t0": t0, "classes": {}}
    for ci, cls in enumerate(CLASSES):
        R = {s: {k: recordings(c, sc, ci, cls, hop, t0, k == "pos") for k, (c, sc) in
                 ((k, sets[(s, k)]) for k in ("pos", "neg_as", "neg_home"))} for s in ("tune", "test")}
        rc = {"tune_clips": len(R["tune"]["pos"]), "test_clips": len(R["test"]["pos"])}
        for b in BUDGETS:
            th = threshold_for_budget(R["tune"]["neg_as"] + R["tune"]["neg_home"], b)
            det, n, _, _, hits = evaluate(R["test"]["pos"], th)
            lo, hi = bootstrap_recall(hits)
            fa = {}
            for k in ("neg_as", "neg_home"):
                _, _, nfa, h, _ = evaluate(R["test"][k], th)
                fa[k] = round(nfa / h, 3) if h else None
            tdet, tn, _, _, _ = evaluate(R["tune"]["pos"], th)
            # recall per source (AudioSet vs FSD50K) on test
            src = {}
            c_pos = sets[("test", "pos")][0]
            idx = [i for i in range(len(c_pos)) if c_pos.events(i, cls)]
            for sname in sorted(c_pos.meta.src.iloc[idx].unique()) if idx else []:
                sub = [R["test"]["pos"][j] for j, i in enumerate(idx) if c_pos.meta.src.iloc[i] == sname]
                d, nn, _, _, _ = evaluate(sub, th)
                src[sname] = [int(d), int(nn)]
            rc[f"@{b}"] = {"threshold": th, "recall": round(det / max(n, 1), 4), "ci95": [round(lo, 3), round(hi, 3)],
                           "detected": int(det), "occurrences": int(n), "fa_per_h_test": fa,
                           "tune_recall": round(tdet / max(tn, 1), 4), "by_source": src}
        res["classes"][cls] = rc
    res["hours"] = {f"{s}_{k}": round(sum(r.hours for r in recordings(c, sc, 0, "", hop, t0, False)), 2)
                    for (s, k), (c, sc) in sets.items() if k != "pos"}
    print_table(res)
    if out_json:
        Path(out_json).parent.mkdir(parents=True, exist_ok=True)
        json.dump(res, open(out_json, "w"), indent=1)
    return res


def print_table(res):
    print(f"\n=== {res['detector']}  (negative hours: {res['hours']})")
    print(f"{'class':10s} " + " ".join(f"{'recall@' + str(b):>22s} {'FA/h as,home':>14s}" for b in BUDGETS))
    for cls, rc in res["classes"].items():
        cells = []
        for b in BUDGETS:
            r = rc[f"@{b}"]
            cells.append(f"{r['recall']:6.1%} [{r['ci95'][0]:.2f},{r['ci95'][1]:.2f}] n={r['occurrences']:<4d}"
                         f" {r['fa_per_h_test']['neg_as']:6.2f},{r['fa_per_h_test']['neg_home']:6.2f}")
        print(f"{cls:10s} " + " ".join(cells))
    rec = {b: np.mean([rc[f"@{b}"]["recall"] for rc in res["classes"].values()]) for b in BUDGETS}
    print("macro recall: " + ", ".join(f"@{b} FA/h: {v:.1%}" for b, v in rec.items()))


def local_copy(root, dest="/content/cache"):
    """Copy the stores from Drive to local disk once (random reads from Drive are slow)."""
    import subprocess
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    subprocess.run(f"rsync -a {Path(root) / 'stores'} {dest}/", shell=True, check=True)
    return dest


def windows(x, win, hop):
    """Causal sliding windows: window k ends at (k+1)*hop seconds-worth of samples; zero-padded start."""
    n = len(x) // hop
    xp = np.concatenate([np.zeros(win, np.float32), x])
    idx = np.arange(win)[None, :] + (np.arange(n)[:, None] + 1) * hop
    return xp[idx]
