"""Print every benchmark result on Drive as one compact table (macro and per-class recall).

    python -m safesound.report --root /content/drive/MyDrive/safesound-tiny [--classes]
"""
import argparse
import json
from pathlib import Path

from safesound.bench import BUDGETS
from safesound.classes import CLASSES


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--classes", action="store_true", help="also per-class test recall")
    ap.add_argument("--json", action="store_true", help="dump compact JSON of everything")
    a = ap.parse_args()
    rows = {}
    for f in sorted((a.root / "results").glob("*.json")):
        r = json.load(open(f))
        if "classes" not in r:
            continue
        row = {}
        for b in BUDGETS:
            row[f"test@{b}"] = round(sum(r["classes"][c][f"@{b}"]["recall"] for c in CLASSES) / len(CLASSES), 4)
            row[f"tune@{b}"] = round(sum(r["classes"][c][f"@{b}"]["tune_recall"] for c in CLASSES) / len(CLASSES), 4)
            row[f"fa_as@{b}"] = round(sum(r["classes"][c][f"@{b}"]["fa_per_h_test"]["neg_as"] or 0 for c in CLASSES) / len(CLASSES), 3)
            row[f"fa_home@{b}"] = round(sum(r["classes"][c][f"@{b}"]["fa_per_h_test"]["neg_home"] or 0 for c in CLASSES) / len(CLASSES), 3)
            row[f"cls@{b}"] = {c: r["classes"][c][f"@{b}"]["recall"] for c in CLASSES}
        rows[f.stem] = row
    if a.json:
        print(json.dumps(rows, separators=(",", ":")))
        return
    print(f"{'run':26s} {'test@0.1':>8s} {'test@1':>7s} {'tune@0.1':>8s} {'tune@1':>7s} {'FA/h as,home @1':>16s}")
    for k, v in rows.items():
        print(f"{k:26s} {v['test@0.1']:8.1%} {v['test@1.0']:7.1%} {v['tune@0.1']:8.1%} {v['tune@1.0']:7.1%}"
              f" {v['fa_as@1.0']:7.2f},{v['fa_home@1.0']:6.2f}")
        if a.classes:
            print(" " * 6 + "  ".join(f"{c} {v['cls@0.1'][c]:.0%}/{v['cls@1.0'][c]:.0%}" for c in CLASSES))


if __name__ == "__main__":
    main()
