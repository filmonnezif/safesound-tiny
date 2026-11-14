"""Phase 0: count real labelled events per target class in the public label files.

Usage: python -m safesound.inventory   (expects files under data/labels/, see docs/protocol.md)
"""
from pathlib import Path

import pandas as pd

from safesound.classes import CLASSES, MID_TO_CLASS

LABELS = Path(__file__).resolve().parent.parent / "data" / "labels"


def audioset_strong(split: str) -> pd.DataFrame:
    d = pd.read_csv(LABELS / f"audioset_{split}_strong.tsv", sep="\t")
    d["cls"] = d.label.map(MID_TO_CLASS)
    d["video"] = d.segment_id.str.rsplit("_", n=1).str[0]
    d["dur"] = d.end_time_seconds - d.start_time_seconds
    return d


def summary(d: pd.DataFrame) -> pd.DataFrame:
    t = d.dropna(subset=["cls"])
    return t.groupby("cls").agg(
        events=("dur", "size"), clips=("segment_id", "nunique"),
        videos=("video", "nunique"), median_s=("dur", "median"),
    ).reindex(CLASSES)


if __name__ == "__main__":
    for split in ["eval", "train"]:
        d = audioset_strong(split)
        n_neg = d.segment_id.nunique() - d.dropna(subset=["cls"]).segment_id.nunique()
        print(f"AudioSet strong {split}: {d.segment_id.nunique()} clips, {n_neg} with none of the six")
        print(summary(d).round(1), "\n")
