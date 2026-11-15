"""Phase 5b (Colab, CPU, cached scores only): energy-saving DSP wake gate, recall-target margins,
and bootstrap confidence intervals.

    python -m safesound.phase5b --root /content/drive/MyDrive/safesound-tiny --data /content/cache

Wake gate: an always-on DSP detector decides, every 0.5 s, whether "something is happening";
only then does the classifier run (on the last 2 s, kept in a ring buffer, so no latency is
lost). After a trigger the gate stays open for H windows (hangover). Two gates:
  G1 energy: broadband level jump over a tracked noise floor (time-domain, ~2 ops/sample)
  G2 dsp:    max of all DSP evidence (tone, sweep, chime, onsets, voicing, horn, level jump)
Gate threshold and hangover are chosen on the TUNING set: the most closed gate whose macro
recall loss (vs. the ungated recall-first system of phase 5) is at most 1 point.

Margins: thresholds set on tuning for target + margin, to absorb the tuning->test shift.
CIs: 95% bootstrap over occurrences (class-stratified) for macro recall.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from safesound import bench
from safesound.classes import CLASSES
from safesound.evaluate import Recording, threshold_for_budget
from safesound.evaluate import evaluate as evaluate_fa
from safesound.phase5 import (HOP, SPLITS, TARGET, Fusion, evaluate, load_scores, recordings,
                              threshold_for_recall)

MODELS = ["p4_XL_logmel_kd_int8", "p4_L_logmel_kd_int8", "p3_teacher_raw"]
# compute per second of audio (multiply-accumulates or equivalent operations)
OPS_CLASSIFIER = {"p4_XL_logmel_kd_int8": 2 * 88.56e6, "p4_L_logmel_kd_int8": 2 * 36.22e6,
                  "p4_M_logmel_kd_int8": 2 * 8.75e6, "p3_teacher_raw": 2 * 2.0e9}
OPS_LOGMEL = 50 * (5 * 512 * 9 + 40 * 257)                     # 50 FFTs/s + mel matrix
OPS_GATE = {"G1_energy": 2 * 16000, "G2_dsp": 100 * (5 * 1024 * 10 + 6 * 513)}   # FFT 1024 / 10 ms + features
GATES = {"G1_energy": [6], "G2_dsp": [0, 1, 2, 3, 4, 5, 6]}


def gate_open(d, cols, tau, H):
    """Boolean per window: gate score >= tau now or within the last H windows."""
    trig = d[:, cols].max(1) >= tau
    if H <= 0 or not trig.any():
        return trig
    idx = np.flatnonzero(trig)
    open_ = np.zeros(len(trig), bool)
    for i in idx:
        open_[i:i + H + 1] = True
    return open_


class Gated:
    """Recordings of one class with the gate applied (classifier output = 0 when closed)."""

    def __init__(self, fu, method, ci, split):
        self.items = []
        for k in ("pos", "neg_as", "neg_home"):
            c = fu.clips[(split, k)]
            for i in range(len(c)):
                d = fu.dsp[(split, k)][i]
                s = fu.scores(method, fu.nn[(split, k)][i][:, ci], d[:, ci])
                n = min(len(s), len(d))
                ev = c.events(i, CLASSES[ci]) if k == "pos" else []
                if k == "pos" and not ev:
                    continue
                self.items.append((k, s[:n], d[:n], ev))

    def evaluate(self, th, gate=None, tau=0.0, H=0):
        pos, neg_as, neg_home, open_n, tot_n = [], [], [], {"pos": 0, "neg": 0}, {"pos": 0, "neg": 0}
        for k, s, d, ev in self.items:
            if gate is not None:
                g = gate_open(d, GATES[gate], tau, H)
                s = np.where(g, s, 0.0)
                key = "pos" if k == "pos" else "neg"
                open_n[key] += g.sum()
                tot_n[key] += len(g)
            r = Recording(s, HOP, ev, HOP)
            (pos if k == "pos" else neg_as if k == "neg_as" else neg_home).append(r)
        det, n, _, _, hits = evaluate(pos, th)
        fa = {}
        for name, neg in (("as", neg_as), ("home", neg_home)):
            _, _, nfa, h, _ = evaluate(neg, th)
            fa[name] = nfa / h if h else 0.0
        duty = {k: (open_n[k] / tot_n[k] if tot_n[k] else 1.0) for k in open_n} if gate else {"pos": 1.0, "neg": 1.0}
        return det / max(n, 1), hits, fa, duty


def boot_macro(hits_by_class, n_boot=2000, seed=0):
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_boot):
        vals.append(np.mean([h[rng.integers(0, len(h), len(h))].mean() for h in hits_by_class if len(h)]))
    return [float(np.quantile(vals, 0.025)), float(np.quantile(vals, 0.975))]


def run_model(name, root, clips, dspd, p5):
    nn = {(s, k): load_scores(root, name, role) for s, roles in SPLITS.items() for k, role in roles.items()}
    res = {"classes": {}}
    G = {}
    present = [c for c in CLASSES if c in p5[name]]
    for ci, cls in enumerate(CLASSES):
        if cls not in present:
            continue
        fu = Fusion(ci, nn, dspd, clips)
        method = p5[name][cls]["chosen"]
        th = p5[name][cls]["methods"][method]["threshold"]
        tune, test = Gated(fu, method, ci, "tune"), Gated(fu, method, ci, "test")
        rec, hits, fa, _ = test.evaluate(th)
        row = {"method": method, "threshold": th, "base": {"recall": rec, "fa_as": fa["as"], "fa_home": fa["home"]},
               "hits": hits}
        # margins: thresholds for a higher tuning target
        tpos = [Recording(s, HOP, ev, HOP) for k, s, d, ev in tune.items if k == "pos"]
        row["margin"] = {}
        for m in (0.02, 0.04):
            thm = threshold_for_recall(tpos, min(0.995, TARGET[cls] + m))
            r2, h2, fa2, _ = test.evaluate(thm)
            row["margin"][str(m)] = {"threshold": thm, "recall": r2, "fa_as": fa2["as"], "fa_home": fa2["home"], "hits": h2}
        res["classes"][cls] = row
        G[cls] = (tune, test, th, rec)
        print(f"  {name} {cls:9s} base rec {rec:.1%} FA {fa['as']:.1f}/{fa['home']:.2f}  margin+4 rec "
              f"{row['margin']['0.04']['recall']:.1%} FA {row['margin']['0.04']['fa_as']:.1f}/{row['margin']['0.04']['fa_home']:.2f}",
              flush=True)

    # gates: one setting for all classes, chosen on tuning (macro recall loss <= 1 point)
    base_tune = np.mean([G[c][0].evaluate(G[c][2])[0] for c in G])
    res["gates"] = {}
    for gate in GATES:
        best = None
        for H in (4, 10, 20):
            for tau in (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8):
                outs = [G[c][0].evaluate(G[c][2], gate, tau, H) for c in G]
                rec = np.mean([o[0] for o in outs])
                duty = np.mean([o[3]["neg"] for o in outs])
                if base_tune - rec <= 0.01 and (best is None or duty < best[2]):
                    best = (tau, H, duty, rec)
        if best is None:
            res["gates"][gate] = None
            continue
        tau, H = best[0], best[1]
        outs = {c: G[c][1].evaluate(G[c][2], gate, tau, H) for c in G}
        duty_neg = float(np.mean([o[3]["neg"] for o in outs.values()]))
        cls_ops = OPS_CLASSIFIER[name] + OPS_LOGMEL
        res["gates"][gate] = {
            "tau": tau, "hangover_windows": H, "tune_duty_neg": best[2], "tune_recall": best[3],
            "test_recall": float(np.mean([o[0] for o in outs.values()])),
            "test_recall_ci": boot_macro([o[1] for o in outs.values()]),
            "test_fa_as": float(np.mean([o[2]["as"] for o in outs.values()])),
            "test_fa_home": float(np.mean([o[2]["home"] for o in outs.values()])),
            "duty_negatives": duty_neg,
            "duty_events": float(np.mean([o[3]["pos"] for o in outs.values()])),
            "ops_per_s_always_on": cls_ops,
            "ops_per_s_gated": OPS_GATE[gate] + duty_neg * cls_ops,
            "compute_saving": 1 - (OPS_GATE[gate] + duty_neg * cls_ops) / cls_ops,
            "per_class_recall": {c: o[0] for c, o in outs.items()},
        }
        g = res["gates"][gate]
        print(f"  {name} gate {gate}: tau {tau} H {H}  duty(neg) {duty_neg:.1%}  recall {g['test_recall']:.1%} "
              f"{g['test_recall_ci']}  FA {g['test_fa_as']:.1f}/{g['test_fa_home']:.2f}  compute saving "
              f"{g['compute_saving']:.1%}", flush=True)

    P = present
    res["base_macro"] = {"recall": float(np.mean([res["classes"][c]["base"]["recall"] for c in P])),
                         "ci": boot_macro([res["classes"][c]["hits"] for c in P]),
                         "fa_as": float(np.mean([res["classes"][c]["base"]["fa_as"] for c in P])),
                         "fa_home": float(np.mean([res["classes"][c]["base"]["fa_home"] for c in P])),
                         "meets_target": {c: res["classes"][c]["base"]["recall"] >= TARGET[c] for c in P}}
    for m in ("0.02", "0.04"):
        res[f"margin_{m}_macro"] = {
            "recall": float(np.mean([res["classes"][c]["margin"][m]["recall"] for c in P])),
            "ci": boot_macro([res["classes"][c]["margin"][m]["hits"] for c in P]),
            "fa_as": float(np.mean([res["classes"][c]["margin"][m]["fa_as"] for c in P])),
            "fa_home": float(np.mean([res["classes"][c]["margin"][m]["fa_home"] for c in P])),
            "meets_target": {c: res["classes"][c]["margin"][m]["recall"] >= TARGET[c] for c in P}}
    for c in P:  # drop hit arrays before saving
        res["classes"][c].pop("hits")
        for m in res["classes"][c]["margin"].values():
            m.pop("hits")
    return res


def paired_claims(root, clips, a="p4_XL_logmel_kd_int8", b="yamnet_zeroshot"):
    """Paired bootstrap over test occurrences: model a vs model b at 1 FA/h (thresholds from
    tuning, phase 1-4 protocol), from the cached benchmark scores."""
    setup = {"yamnet_zeroshot": (0.48, 0.96)}
    hits = {}
    for name in (a, b):
        hop, t0 = setup.get(name, (0.5, 0.5))
        sc = {key: load_scores(root, name, SPLITS[key[0]][key[1]]) for key in clips}
        hits[name] = []
        for ci, cls in enumerate(CLASSES):
            R = {key: bench.recordings(clips[key], sc[key], ci, cls, hop, t0, key[1] == "pos") for key in clips}
            th = threshold_for_budget(R[("tune", "neg_as")] + R[("tune", "neg_home")], 1.0)
            hits[name].append(evaluate_fa(R[("test", "pos")], th)[4])
    rng = np.random.default_rng(1)
    diffs = []
    for _ in range(2000):
        d = []
        for x, y in zip(hits[a], hits[b]):
            i = rng.integers(0, len(x), len(x))
            d.append(x[i].mean() - y[i].mean())
        diffs.append(np.mean(d))
    out = {f"{a}_minus_{b}_recall_at_1fa": {
        "mean": float(np.mean([x.mean() - y.mean() for x, y in zip(hits[a], hits[b])])),
        "ci95": [float(np.quantile(diffs, 0.025)), float(np.quantile(diffs, 0.975))],
        "p_le_0": float(np.mean(np.array(diffs) <= 0))}}
    print("paired:", out, flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--data", type=Path, default=Path("/content/cache"))
    ap.add_argument("--curve", action="store_true", help="only the gate trade-off curve")
    ap.add_argument("--models", nargs="*", default=MODELS)
    a = ap.parse_args()
    f6 = a.root / "results" / "phase6_summary.json"
    if f6.exists():                                   # compute of the pretrained students
        for k, v in json.load(open(f6)).items():
            OPS_CLASSIFIER[k] = OPS_CLASSIFIER[k + "_int8"] = 2 * v["macs_per_inference"]
    if a.curve:
        for name in a.models:
            gate_curve(a.root, a.data, name)
        return
    clips = {(s, k): bench.Clips(a.data, role) for s, roles in SPLITS.items() for k, role in roles.items()}
    dspd = {(s, k): load_scores(a.root, "dsp_v1", role) for s, roles in SPLITS.items() for k, role in roles.items()}
    p5 = json.load(open(a.root / "results" / "phase5.json"))
    fo = a.root / "results" / "phase5b.json"
    out = json.load(open(fo)) if fo.exists() else {}
    if "paired" not in out:
        out["paired"] = paired_claims(a.root, clips) if (a.root / "scores" / "yamnet_zeroshot").exists() else {}
    for name in a.models:
        print(f"\n### {name}", flush=True)
        out[name] = run_model(name, a.root, clips, dspd, p5)
        json.dump(out, open(fo, "w"), indent=1)
    print("\n=== summary")
    for name in a.models:
        r = out[name]
        b = r["base_macro"]
        print(f"{name:22s} base   rec {b['recall']:.1%} {np.round(b['ci'], 3)} FA {b['fa_as']:.1f}/{b['fa_home']:.2f}")
        for m in ("0.02", "0.04"):
            x = r[f"margin_{m}_macro"]
            print(f"{name:22s} +{m}  rec {x['recall']:.1%} {np.round(x['ci'], 3)} FA {x['fa_as']:.1f}/{x['fa_home']:.2f}"
                  f" meets {sum(x["meets_target"].values())}/{len(x["meets_target"])}")
        for gname, g in r["gates"].items():
            if g:
                print(f"{name:22s} {gname:9s} rec {g['test_recall']:.1%} {np.round(g['test_recall_ci'], 3)} FA "
                      f"{g['test_fa_as']:.1f}/{g['test_fa_home']:.2f} duty {g['duty_negatives']:.1%} saving {g['compute_saving']:.1%}")



# ------------------------------------------------------------------ gate trade-off curve

def gate_curve(root, data, name="p4_XL_logmel_kd_int8"):
    """For every gate setting: test recall (all events, and events whose onset lies >= 1 s
    inside the clip, i.e. the gate could hear them start), classifier duty on home and on
    AudioSet negatives, compute saving for a home deployment, and false alarms."""
    clips = {(s, k): bench.Clips(data, role) for s, roles in SPLITS.items() for k, role in roles.items()}
    dspd = {(s, k): load_scores(root, "dsp_v1", role) for s, roles in SPLITS.items() for k, role in roles.items()}
    nn = {(s, k): load_scores(root, name, role) for s, roles in SPLITS.items() for k, role in roles.items()}
    p5 = json.load(open(root / "results" / "phase5.json"))[name]
    per_class = {}
    for ci, cls in enumerate(CLASSES):
        fu = Fusion(ci, nn, dspd, clips)
        method = p5[cls]["chosen"]
        per_class[cls] = (Gated(fu, method, ci, "test"), p5[cls]["methods"][method]["threshold"])
    cls_ops = OPS_CLASSIFIER[name] + OPS_LOGMEL
    rows = []
    for gate in GATES:
        for H in (4, 10, 20):
            for tau in (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9):
                rec, rec_on, fa_as, fa_home, open_h, n_h, open_a, n_a = [], [], [], [], 0, 0, 0, 0
                for cls, (G, th) in per_class.items():
                    pos, pos_on, nas, nhome = [], [], [], []
                    for k, s, d, ev in G.items:
                        g = gate_open(d, GATES[gate], tau, H) if tau > 0 else np.ones(len(s), bool)
                        r = Recording(np.where(g, s, 0.0), HOP, ev, HOP)
                        if k == "pos":
                            pos.append(r)
                            ev_on = [(on, off) for on, off in ev if on >= 1.0]
                            if ev_on:
                                pos_on.append(Recording(r.scores, HOP, ev_on, HOP))
                        elif k == "neg_as":
                            nas.append(r); open_a += g.sum(); n_a += len(g)
                        else:
                            nhome.append(r); open_h += g.sum(); n_h += len(g)
                    det, n, _, _, _ = evaluate(pos, th)
                    det2, n2, _, _, _ = evaluate(pos_on, th)
                    rec.append(det / n); rec_on.append(det2 / max(n2, 1))
                    for lst, neg in ((fa_as, nas), (fa_home, nhome)):
                        _, _, nfa, h, _ = evaluate(neg, th)
                        lst.append(nfa / h)
                duty_h, duty_a = open_h / n_h, open_a / n_a
                rows.append({"gate": gate, "H": H, "tau": tau, "recall": float(np.mean(rec)),
                             "recall_onset_seen": float(np.mean(rec_on)), "duty_home": float(duty_h),
                             "duty_audioset": float(duty_a), "fa_as": float(np.mean(fa_as)),
                             "fa_home": float(np.mean(fa_home)),
                             "saving_home": float(1 - (OPS_GATE[gate] + duty_h * cls_ops) / cls_ops)})
                r = rows[-1]
                print(f"{gate:9s} H{H:2d} tau {tau:.1f}  rec {r['recall']:.1%} (onset seen {r['recall_onset_seen']:.1%})"
                      f"  duty home {duty_h:.1%} AS {duty_a:.1%}  saving(home) {r['saving_home']:.1%}"
                      f"  FA {r['fa_as']:.1f}/{r['fa_home']:.2f}", flush=True)
    json.dump(rows, open(root / "results" / f"gate_curve_{name}.json", "w"), indent=1)
    return rows


if __name__ == "__main__":
    main()
