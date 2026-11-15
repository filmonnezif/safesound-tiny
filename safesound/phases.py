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


def claim(root, name, stale_h=3.0):
    """Claim a training run so several Colab runtimes can share a phase. A claim older than
    `stale_h` hours is treated as abandoned (its runtime died)."""
    import os
    import time
    lock = root / "checkpoints" / f"{name}.lock"
    lock.parent.mkdir(parents=True, exist_ok=True)
    try:
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.write(fd, str(time.time()).encode())
        os.close(fd)
        return True
    except FileExistsError:
        try:
            t = float(lock.read_text() or 0)
        except (OSError, ValueError):
            t = 0.0
        if time.time() - t > stale_h * 3600:
            lock.write_text(str(time.time()))
            return True
        return False


def wait_for(paths, what, wait_s=5 * 3600):
    import time
    t0 = time.time()
    while not all(Path(p).exists() for p in paths):
        if time.time() - t0 > wait_s:
            raise TimeoutError(f"{what} missing: " + str([Path(p).name for p in paths if not Path(p).exists()]))
        print(f"  waiting for {what} on another runtime ...", flush=True)
        time.sleep(120)


def student_run(data, root, name, cfg, kind, size, steps, teacher=None):
    """Train (or load) one student and benchmark it. Returns (model, results), or
    (None, None) if another runtime is training it."""
    ck = root / "checkpoints" / f"{name}.pt"
    out = root / "results" / f"{name}.json"
    if out.exists() and ck.exists():
        return load_student(ck), json.load(open(out))
    if ck.exists():
        model = load_student(ck)
    elif claim(root, name):
        print(f"\n### training {name}", flush=True)
        w, b = STUDENT_SIZES[size]
        model = train_student(data, cfg, kind, w, b, steps=steps, teacher=teacher)
        save_student(model, ck)
    else:
        print(f"  {name}: another runtime is training it", flush=True)
        return None, None
    res = bench.run(data, root, name, student_detector(model, kind), hop=0.5, t0=0.5, out_json=out)
    return model, res


SEEDS = 3


def run_name(cname, seed):
    return f"p2_{cname}" if seed == 0 else f"p2_{cname}_s{seed}"


def phase2(data, root, steps, seeds=range(SEEDS)):
    """Each config trained SEEDS times (independent runs; several runtimes can share the work
    with --seeds). Seeds run outermost so a restart keeps whole sets."""
    for seed in seeds:
        for cname, cfg in CONFIGS.items():
            student_run(data, root, run_name(cname, seed), cfg, "logmel", "M", steps)


def phase2_choose(root, wait_s=4 * 3600):
    """Choice = best mean TUNING macro recall at 1 FA/h (ties: 0.1 FA/h) over all SEEDS runs.
    Waits for runs still going on another runtime."""
    files = {c: [root / "results" / f"{run_name(c, s)}.json" for s in range(SEEDS)] for c in CONFIGS}
    wait_for([f for fs in files.values() for f in fs], "phase 2 runs", wait_s)
    rows = {}
    for cname, fs in files.items():
        rs = [json.load(open(f)) for f in fs]
        rs = [{"tune": macro(r, "tune_recall"), "test": macro(r)} for r in rs]
        rows[cname] = {split: {b: [round(float(np.mean([r[split][b] for r in rs])), 4),
                                   round(float(np.std([r[split][b] for r in rs])), 4)] for b in bench.BUDGETS}
                       for split in ("tune", "test")}
    best = max(rows, key=lambda k: (rows[k]["tune"][1.0][0], rows[k]["tune"][0.1][0]))
    json.dump({"config": best, "rows": rows, "runs": SEEDS},
              open(root / "results" / "phase2_choice.json", "w"), indent=1)
    print(f"\n=== Phase 2: macro recall, mean +- sd over {SEEDS} runs (tuning | test) at 0.1 and 1 FA/h")
    for k, v in rows.items():
        t, e = v["tune"], v["test"]
        print(f"{k:8s} tune {t[0.1][0]:.1%}+-{t[0.1][1]:.1%} {t[1.0][0]:.1%}+-{t[1.0][1]:.1%} | "
              f"test {e[0.1][0]:.1%}+-{e[0.1][1]:.1%} {e[1.0][0]:.1%}+-{e[1.0][1]:.1%}"
              + ("   <- chosen on tuning" if k == best else ""))
    return best


def load_teacher(root, name):
    from safesound.cnn14 import Cnn14, fetch
    from safesound.models import Teacher
    _, mids = fetch()
    t = Teacher(Cnn14(), mids)
    t.load_state_dict(torch.load(root / "checkpoints" / f"{name}.pt", map_location="cpu"))
    return t.cuda().eval()


def phase3(data, root, steps):
    from safesound.cnn14 import fetch, load_pretrained
    # the raw-data teacher does not depend on the phase 2 choice: train it first
    for name in ["p3_teacher_raw", "p3_teacher_sim"]:
        cname = "A_raw" if name.endswith("raw") else phase2_choose(root)
        ck = root / "checkpoints" / f"{name}.pt"
        if ck.exists():
            model = load_teacher(root, name)
        elif claim(root, name):
            print(f"\n### training {name} on {cname}", flush=True)
            path, mids = fetch()
            model = train_teacher(data, CONFIGS[cname], load_pretrained(path), mids, steps=steps)
            ck.parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), ck)
        else:
            print(f"  {name}: another runtime is training it", flush=True)
            continue
        out = root / "results" / f"{name}.json"
        if not out.exists():
            bench.run(data, root, name, teacher_detector(model), hop=0.5, t0=0.5, out_json=out)
        del model
        torch.cuda.empty_cache()


def calib_feats(data, cfg, kind, n_batches=16):
    from safesound.train import gpu_features
    it = loader(data, cfg, 256, seed=99)
    return torch.cat([gpu_features(next(it)[0].cuda(), kind).cpu() for _ in range(n_batches)])


def phase4(data, root, steps):
    choice = json.load(open(root / "results" / "phase2_choice.json"))["config"]
    cfg = CONFIGS[choice]
    wait_for([root / "checkpoints" / "p3_teacher_sim.pt"], "the simulated-data teacher")
    teacher = load_teacher(root, "p3_teacher_sim")
    runs = [(f"p4_{s}_logmel_kd", s, "logmel") for s in STUDENT_SIZES] + [("p4_M_pcen_kd", "M", "pcen")]
    for name, size, kind in runs:
        model, r = student_run(data, root, name, cfg, kind, size, steps, teacher=teacher)
        qout = root / "results" / f"{name}_int8.json"
        if model is not None and not qout.exists():
            q = quantize(model, calib_feats(data, cfg, kind)).cuda()
            bench.run(data, root, name + "_int8", student_detector(q, kind), hop=0.5, t0=0.5, out_json=qout)
    wait_for([root / "results" / f"{n}{x}.json" for n, _, _ in runs for x in ("", "_int8")], "phase 4 runs")
    table = []
    for name, size, kind in runs:
        model = load_student(root / "checkpoints" / f"{name}.pt")
        r = json.load(open(root / "results" / f"{name}.json"))
        rq = json.load(open(root / "results" / f"{name}_int8.json"))
        table.append({"name": name, "size": size, "frontend": kind, "params": count_params(model),
                      "macs_per_inference": count_macs(model), "weights_bytes_int8": model_bytes_int8(model),
                      "float": macro(r), "int8": macro(rq), "tune_float": macro(r, "tune_recall")})
    # references: the hard-label students of phase 2 (same config, all runs) and the teachers
    p2 = [macro(json.load(open(root / "results" / f"{run_name(choice, s)}.json"))) for s in range(SEEDS)]
    ref = {"student_M_hard (mean of runs)": {b: round(float(np.mean([m[b] for m in p2])), 4) for b in bench.BUDGETS},
           "teacher_raw": macro(json.load(open(root / "results" / "p3_teacher_raw.json"))),
           "teacher_sim": macro(json.load(open(root / "results" / "p3_teacher_sim.json")))}
    json.dump({"config": choice, "students": table, "references": ref},
              open(root / "results" / "phase4_summary.json", "w"), indent=1)
    print(f"\n=== Phase 4 (simulation {choice}): test macro recall at 0.1 / 1 FA/h")
    for k, v in ref.items():
        print(f"{k:30s} {v[0.1]:.1%} / {v[1.0]:.1%}")
    for t in table:
        print(f"{t['name']:22s} params {t['params']:>7d}  MACs {t['macs_per_inference'] / 1e6:6.2f}M  "
              f"int8 {t['weights_bytes_int8'] / 1024:6.1f} KB   float {t['float'][0.1]:.1%} / {t['float'][1.0]:.1%}"
              f"   int8 {t['int8'][0.1]:.1%} / {t['int8'][1.0]:.1%}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("phase", type=int, choices=[2, 3, 4])
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--seeds", type=int, nargs="*", default=None, help="phase 2: which repeat runs")
    a = ap.parse_args()
    (a.root / "results").mkdir(parents=True, exist_ok=True)
    data = bench.local_copy(a.root)
    {2: lambda: phase2(data, a.root, a.steps or 12000, a.seeds if a.seeds is not None else range(SEEDS)),
     3: lambda: phase3(data, a.root, a.steps or 15000),
     4: lambda: phase4(data, a.root, a.steps or 12000)}[a.phase]()


if __name__ == "__main__":
    main()
