"""Phases 2-4 (Colab). Each phase skips work whose results already exist on Drive.

    python -m safesound.phases 2 --root /content/drive/MyDrive/safesound-tiny
    python -m safesound.phases 3 --root ...
    python -m safesound.phases 4 --root ...

Phase 2: DSP simulation ablation. A medium student (DS-CNN, width 64) trained with hard
         labels under each simulation setting A-E; the setting with the best TUNING macro
         recall at 1 FA/h is chosen for phases 3-4.
Phase 3: CNN14 teacher fine-tuned on raw data (A) and on the chosen simulation.
Phase 4: students S/M/L/XL distilled from the simulated-data teacher; log-mel vs PCEN;
         int8 simulation of each; size / compute / memory table.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from safesound import bench
from safesound.classes import CLASSES
from safesound.models import DSCNN, STUDENT_SIZES, count_macs, count_params
from safesound.quant import model_bytes_int8, quantize
from safesound.sim import CONFIGS
from safesound.train import (loader, student_detector, teacher_detector, train_student,
                             train_teacher)


def macro(res, key="recall"):
    return {b: round(float(np.mean([res["classes"][c][f"@{b}"][key] for c in CLASSES])), 4) for b in bench.BUDGETS}


def save_student(model, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"cfg": model.cfg, "state": model.state_dict()}, path)


def load_student(path):
    d = torch.load(path, map_location="cuda")
    m = DSCNN(d["cfg"]["width"], d["cfg"]["blocks"]).cuda()
    m.load_state_dict(d["state"])
    return m.eval()


def student_run(data, root, name, cfg, kind, size, steps, teacher=None):
    """Train (or load) one student and benchmark it. Returns (model, results)."""
    ck = root / "checkpoints" / f"{name}.pt"
    out = root / "results" / f"{name}.json"
    if ck.exists():
        model = load_student(ck)
    else:
        print(f"\n### training {name}", flush=True)
        w, b = STUDENT_SIZES[size]
        model = train_student(data, cfg, kind, w, b, steps=steps, teacher=teacher)
        save_student(model, ck)
    res = json.load(open(out)) if out.exists() else bench.run(
        data, root, name, student_detector(model, kind), hop=0.5, t0=0.5, out_json=out)
    return model, res


SEEDS = 3


def run_name(cname, seed):
    return f"p2_{cname}" if seed == 0 else f"p2_{cname}_s{seed}"


def phase2(data, root, steps):
    """Each config trained SEEDS times (independent runs); choice = best mean TUNING macro
    recall at 1 FA/h (ties: at 0.1 FA/h). Seeds run outermost so a restart keeps whole sets."""
    runs = {c: [] for c in CONFIGS}
    for seed in range(SEEDS):
        for cname, cfg in CONFIGS.items():
            _, r = student_run(data, root, run_name(cname, seed), cfg, "logmel", "M", steps)
            runs[cname].append({"tune": macro(r, "tune_recall"), "test": macro(r)})
    rows = {}
    for cname, rs in runs.items():
        rows[cname] = {split: {b: [round(float(np.mean([r[split][b] for r in rs])), 4),
                                   round(float(np.std([r[split][b] for r in rs])), 4)] for b in bench.BUDGETS}
                       for split in ("tune", "test")}
        rows[cname]["runs"] = len(rs)
    best = max(rows, key=lambda k: (rows[k]["tune"][1.0][0], rows[k]["tune"][0.1][0]))
    json.dump({"config": best, "rows": rows}, open(root / "results" / "phase2_choice.json", "w"), indent=1)
    print(f"\n=== Phase 2: macro recall, mean +- sd over {SEEDS} runs (tuning | test) at 0.1 and 1 FA/h")
    for k, v in rows.items():
        t, e = v["tune"], v["test"]
        print(f"{k:8s} tune {t[0.1][0]:.1%}+-{t[0.1][1]:.1%} {t[1.0][0]:.1%}+-{t[1.0][1]:.1%} | "
              f"test {e[0.1][0]:.1%}+-{e[0.1][1]:.1%} {e[1.0][0]:.1%}+-{e[1.0][1]:.1%}"
              + ("   <- chosen on tuning" if k == best else ""))


def load_teacher(root, name):
    from safesound.cnn14 import Cnn14, fetch
    from safesound.models import Teacher
    _, mids = fetch()
    t = Teacher(Cnn14(), mids)
    t.load_state_dict(torch.load(root / "checkpoints" / f"{name}.pt", map_location="cpu"))
    return t.cuda().eval()


def phase3(data, root, steps):
    from safesound.cnn14 import fetch, load_pretrained
    choice = json.load(open(root / "results" / "phase2_choice.json"))["config"]
    for name, cname in [("p3_teacher_raw", "A_raw"), ("p3_teacher_sim", choice)]:
        ck = root / "checkpoints" / f"{name}.pt"
        if ck.exists():
            model = load_teacher(root, name)
        else:
            print(f"\n### training {name} on {cname}", flush=True)
            path, mids = fetch()
            model = train_teacher(data, CONFIGS[cname], load_pretrained(path), mids, steps=steps)
            ck.parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), ck)
        out = root / "results" / f"{name}.json"
        if not out.exists():
            bench.run(data, root, name, teacher_detector(model), hop=0.5, t0=0.5, out_json=out)
        del model
        torch.cuda.empty_cache()


def calib_feats(data, cfg, kind, n_batches=16):
    it = loader(data, cfg, kind, 256, workers=16, seed=99)
    return torch.cat([next(it)[0] for _ in range(n_batches)])


def phase4(data, root, steps):
    choice = json.load(open(root / "results" / "phase2_choice.json"))["config"]
    cfg = CONFIGS[choice]
    teacher = load_teacher(root, "p3_teacher_sim")
    runs = [(f"p4_{s}_logmel_kd", s, "logmel") for s in STUDENT_SIZES] + [("p4_M_pcen_kd", "M", "pcen")]
    table = []
    for name, size, kind in runs:
        model, r = student_run(data, root, name, cfg, kind, size, steps, teacher=teacher)
        qname = name + "_int8"
        qout = root / "results" / f"{qname}.json"
        if qout.exists():
            rq = json.load(open(qout))
        else:
            q = quantize(model, calib_feats(data, cfg, kind)).cuda()
            rq = bench.run(data, root, qname, student_detector(q, kind), hop=0.5, t0=0.5, out_json=qout)
        table.append({"name": name, "size": size, "frontend": kind, "params": count_params(model),
                      "macs_per_inference": count_macs(model), "weights_bytes_int8": model_bytes_int8(model),
                      "float": macro(r), "int8": macro(rq), "tune_float": macro(r, "tune_recall")})
    # references: the hard-label student of phase 2 (same config) and the teacher
    ref = {"student_M_hard": macro(json.load(open(root / "results" / f"p2_{choice}.json"))),
           "teacher_sim": macro(json.load(open(root / "results" / "p3_teacher_sim.json")))}
    json.dump({"config": choice, "students": table, "references": ref},
              open(root / "results" / "phase4_summary.json", "w"), indent=1)
    print(f"\n=== Phase 4 (simulation {choice}): test macro recall at 0.1 / 1 FA/h")
    for k, v in ref.items():
        print(f"{k:22s} {v[0.1]:.1%} / {v[1.0]:.1%}")
    for t in table:
        print(f"{t['name']:22s} params {t['params']:>7d}  MACs {t['macs_per_inference'] / 1e6:6.2f}M  "
              f"int8 {t['weights_bytes_int8'] / 1024:6.1f} KB   float {t['float'][0.1]:.1%} / {t['float'][1.0]:.1%}"
              f"   int8 {t['int8'][0.1]:.1%} / {t['int8'][1.0]:.1%}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("phase", type=int, choices=[2, 3, 4])
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--steps", type=int, default=None)
    a = ap.parse_args()
    (a.root / "results").mkdir(parents=True, exist_ok=True)
    data = bench.local_copy(a.root)
    {2: lambda: phase2(data, a.root, a.steps or 12000),
     3: lambda: phase3(data, a.root, a.steps or 15000),
     4: lambda: phase4(data, a.root, a.steps or 12000)}[a.phase]()


if __name__ == "__main__":
    main()
