"""Phase 0b (runs on Colab): download the public datasets and build the clip stores.

    python -m safesound.data --root /content/drive/MyDrive/safesound-tiny --work /content/work

Stores are written to <root>/stores/<role>/<source>/ (see store.py). Roles:

  test_pos        real target events      AudioSet strong eval (all six) + FSD50K eval (4 classes)
  test_neg_as     real negatives, varied  AudioSet strong eval clips with none of the six (50%)
  test_neg_home   real negatives, home    SINS sessions held out for test
  tune_pos        tuning events           AudioSet strong train, 20% of videos + FSD50K dev 'val'
                                          + AudioSet balanced train (same 20% video split)
  tune_neg_as     tuning negatives        the same held-out AudioSet videos, none of the six
  tune_neg_home   tuning negatives, home  SINS tuning sessions
  train_pos       training events         the other 80% of strong train + FSD50K dev 'train' + bal train
  train_neg       training negatives      80% strong train: all hard negatives + 20% of the rest
  train_bg_home   backgrounds to mix into SINS training sessions

A source's store is skipped if its meta.csv exists, so the script can be re-run after a crash.
"""
import argparse
import hashlib
import json
import os
import shutil
import subprocess
import zipfile
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

from safesound.audio import decode, to_int16
from safesound.classes import AMBIGUOUS_MIDS, HARD_NEGATIVE_MIDS, MID_TO_CLASS
from safesound.store import StoreWriter, merge_parts

STRONG_URL = "http://storage.googleapis.com/us_audioset/youtube_corpus/strong/"
ZENODO = "https://zenodo.org/api/records/{rec}/files/{name}/content"
FSD_REC, SINS_REC = "4060432", "1247102"
FSD_CLASSES = {"Doorbell": "doorbell", "Knock": "knock", "Siren": "siren",
               "Vehicle_horn_and_car_horn_and_honking": "car_horn"}

TUNE_FRAC = 0.2


def unit(key: str, salt: str) -> float:
    """Deterministic uniform number in [0, 1) for a key (used for every split)."""
    return int(hashlib.md5(f"{salt}:{key}".encode()).hexdigest()[:8], 16) / 2**32


def sh(cmd):
    print("+", cmd, flush=True)
    subprocess.run(cmd, shell=True, check=True)


def aria(urls_names, out_dir, jobs=4):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    lst = out_dir / "aria.txt"
    lst.write_text("".join(f"{u}\n  out={n}\n" for u, n in urls_names if not (out_dir / n).exists()))
    if lst.read_text():
        sh(f"aria2c -q -x 8 -s 8 -j {jobs} --file-allocation=none --auto-file-renaming=false "
           f"-d {out_dir} -i {lst}")


# ---------------------------------------------------------------- AudioSet strong (CLAPv2 mirror)

def strong_index(work: Path, split: str):
    """video id -> (set of MIDs, events) from the official strong-label TSV."""
    f = work / "labels" / f"audioset_{split}_strong.tsv"
    if not f.exists():
        f.parent.mkdir(parents=True, exist_ok=True)
        sh(f"curl -sSfL -o {f} {STRONG_URL}audioset_{split}_strong.tsv")
    d = pd.read_csv(f, sep="\t")
    d["vid"] = d.segment_id.str.rsplit("_", n=1).str[0]
    idx = {}
    for vid, g in d.groupby("vid"):
        g = g[g.segment_id == g.segment_id.iloc[0]]  # one segment per video
        ev = [[MID_TO_CLASS[m], round(a, 3), round(b, 3)]
              for m, a, b in zip(g.label, g.start_time_seconds, g.end_time_seconds) if m in MID_TO_CLASS]
        idx[vid] = (set(g.label), ev)
    return idx


def strong_role(vid, mids, ev, split):
    if split == "eval":
        if ev:
            return "test_pos"
        if mids & AMBIGUOUS_MIDS:
            return None
        return "test_neg_as" if unit(vid, "neg") < 0.5 else None
    held = unit(vid, "tune") < TUNE_FRAC
    if ev:
        return "tune_pos" if held else "train_pos"
    if mids & AMBIGUOUS_MIDS:
        return None
    if held:
        return "tune_neg_as" if unit(vid, "neg") < 0.4 else None
    return "train_neg" if (mids & HARD_NEGATIVE_MIDS or unit(vid, "neg") < 0.2) else None


_G = {}


def _clap_file(args):
    path, split, parts = args
    import pyarrow.parquet as pq
    writers, n_seen = {}, 0
    for row in pq.read_table(path, columns=["index", "audio"]).to_pylist():
        vid = row["index"][-11:]
        if vid not in _G["idx"]:
            continue
        n_seen += 1
        mids, ev = _G["idx"][vid]
        role = strong_role(vid, mids, ev, split)
        if role is None:
            continue
        x = decode(row["audio"]["bytes"])
        if role not in writers:
            writers[role] = StoreWriter(Path(parts) / role / Path(path).stem)
        writers[role].add(to_int16(x), uid=f"as:{vid}", source="audioset_strong",
                          classes=";".join(sorted({e[0] for e in ev})), events=ev, weak=False,
                          group=vid, info=";".join(sorted(mids)))
    for w in writers.values():
        w.close()
    os.remove(path)
    return n_seen


def build_strong(root, work, split, workers):
    src = f"audioset_strong_{split}"
    roles = ["test_pos", "test_neg_as"] if split == "eval" else ["train_pos", "tune_pos", "train_neg", "tune_neg_as"]
    if all((root / "stores" / r / src / "meta.csv").exists() for r in roles):
        print("skip", src)
        return
    from huggingface_hub import snapshot_download
    _G["idx"] = strong_index(work, split)
    hf_split = "test" if split == "eval" else "train"
    d = snapshot_download("CLAPv2/audioset_strong", repo_type="dataset", local_dir=work / "clap",
                          allow_patterns=[f"data/{hf_split}/*.parquet"], max_workers=16)
    files = sorted(Path(d, "data", hf_split).glob("*.parquet"))
    parts = work / "parts" / src
    with Pool(workers) as p:
        seen = sum(p.imap_unordered(_clap_file, [(str(f), split, str(parts)) for f in files]))
    print(f"{src}: {seen} of {len(_G['idx'])} labelled videos have audio")
    for r in roles:
        n = merge_parts(sorted((parts / r).glob("*")), root / "stores" / r / src)
        print(f"  {r}: {n} clips")
    shutil.rmtree(parts, ignore_errors=True)


# ---------------------------------------------------------------- AudioSet balanced train (weak labels)

def _bal_file(args):
    path, parts = args
    import pyarrow.parquet as pq
    writers = {}
    for row in pq.read_table(path).to_pylist():
        vid = row["video_id"]
        if vid in _G["skip"]:
            continue
        classes = sorted({MID_TO_CLASS[m] for m in row["labels"] if m in MID_TO_CLASS})
        if not classes:
            continue
        role = "tune_pos" if unit(vid, "tune") < TUNE_FRAC else "train_pos"
        x = decode(row["audio"]["bytes"])
        dur = round(len(x) / 16000, 3)
        if role not in writers:
            writers[role] = StoreWriter(Path(parts) / role / Path(path).stem)
        writers[role].add(to_int16(x), uid=f"as:{vid}", source="audioset_bal", classes=";".join(classes),
                          events=[[c, 0.0, dur] for c in classes], weak=True, group=vid,
                          info=";".join(row["labels"]))
    for w in writers.values():
        w.close()
    os.remove(path)


def build_bal(root, work, workers):
    src = "audioset_bal"
    roles = ["train_pos", "tune_pos"]
    if all((root / "stores" / r / src / "meta.csv").exists() for r in roles):
        print("skip", src)
        return
    from huggingface_hub import snapshot_download
    # never duplicate a video we already have with strong labels, nor any eval video
    skip = set()
    for split in ["eval", "train"]:
        skip |= set(strong_index(work, split))
    _G["skip"] = skip
    d = snapshot_download("agkphysics/AudioSet", repo_type="dataset", local_dir=work / "bal",
                          allow_patterns=["data/bal_train/*.parquet"], max_workers=16)
    files = sorted(Path(d, "data", "bal_train").glob("*.parquet"))
    parts = work / "parts" / src
    with Pool(workers) as p:
        list(p.imap_unordered(_bal_file, [(str(f), str(parts)) for f in files]))
    for r in roles:
        n = merge_parts(sorted((parts / r).glob("*")), root / "stores" / r / src)
        print(f"  {r}: {n} clips")
    shutil.rmtree(parts, ignore_errors=True)


# ---------------------------------------------------------------- FSD50K (weak labels)

def build_fsd(root, work, workers):
    targets = {"eval": ["test_pos"], "dev": ["train_pos", "tune_pos"]}
    if all((root / "stores" / r / "fsd50k" / "meta.csv").exists() for rs in targets.values() for r in rs):
        print("skip fsd50k")
        return
    z = work / "fsd"
    aria([(ZENODO.format(rec=FSD_REC, name="FSD50K.ground_truth.zip"), "gt.zip")], z)
    sh(f"cd {z} && unzip -oq gt.zip")
    writers = {}
    for part, n_extra in [("eval", 1), ("dev", 5)]:
        names = [f"FSD50K.{part}_audio.zip"] + [f"FSD50K.{part}_audio.z{i:02d}" for i in range(1, n_extra + 1)]
        aria([(ZENODO.format(rec=FSD_REC, name=n), n) for n in names], z, jobs=6)
        full = z / f"{part}_full.zip"
        if not full.exists():
            sh(f"cd {z} && zip -q -s 0 FSD50K.{part}_audio.zip --out {full.name}")
        for n in names:
            (z / n).unlink(missing_ok=True)
        gt = pd.read_csv(z / "FSD50K.ground_truth" / f"{part}.csv")
        with zipfile.ZipFile(full) as zf:
            members = {Path(m).stem: m for m in zf.namelist() if m.endswith(".wav")}
            for _, r in gt.iterrows():
                classes = sorted({FSD_CLASSES[l] for l in r.labels.split(",") if l in FSD_CLASSES})
                if not classes:
                    continue
                role = "test_pos" if part == "eval" else ("tune_pos" if r.split == "val" else "train_pos")
                x = decode(zf.read(members[str(r.fname)]))
                dur = round(len(x) / 16000, 3)
                if role not in writers:
                    writers[role] = StoreWriter(root / "stores" / role / "fsd50k")
                writers[role].add(to_int16(x), uid=f"fsd:{r.fname}", source="fsd50k",
                                  classes=";".join(classes), events=[[c, 0.0, dur] for c in classes],
                                  weak=True, group=f"fsd:{r.fname}", info=r.labels)
        full.unlink()
    for r, w in writers.items():
        print(f"  {r}: {w.close()} clips")


# ---------------------------------------------------------------- SINS / DCASE 2018 task 5 (home negatives)

def sins_plan(meta: pd.DataFrame):
    """Split sessions (time) into train/tune/test by duration; one mic node per session."""
    meta = meta.copy()
    meta["node"] = meta.f.str.extract(r"DevNode(\d)")[0].astype(int)
    sess = meta.groupby("sess").size().reset_index(name="n")
    sess["u"] = [unit(s, "sins") for s in sess.sess]
    sess = sess.sort_values("u")
    frac = sess.n.cumsum() / sess.n.sum()
    sess["role"] = np.where(frac <= 0.5, "train_bg_home", np.where(frac <= 0.7, "tune_neg_home", "test_neg_home"))
    sess["node"] = [1 + int(unit(s, "node") * 4) for s in sess.sess]
    m = meta.merge(sess[["sess", "role", "node"]], on="sess", suffixes=("", "_pick"))
    m = m[m.node == m.node_pick]
    # keep 60% of training backgrounds (enough variety, less storage)
    m = m[(m.role != "train_bg_home") | np.array([unit(f, "bg") < 0.6 for f in m.f])]
    return m.set_index(m.f.map(lambda p: Path(p).name))


def _sins_zip(args):
    path, parts = args
    writers = {}
    with zipfile.ZipFile(path) as zf:
        for mem in zf.namelist():
            name = Path(mem).name
            if name not in _G["plan"].index:
                continue
            r = _G["plan"].loc[name]
            x = decode(zf.read(mem), channel=0)
            if r.role not in writers:
                writers[r.role] = StoreWriter(Path(parts) / r.role / Path(path).stem)
            writers[r.role].add(to_int16(x), uid=f"sins:{name}", source="sins", classes="", events=[],
                                weak=False, group=f"sins:{r.sess}", info=r.act)
    for w in writers.values():
        w.close()
    os.remove(path)


def build_sins(root, work, workers):
    roles = ["train_bg_home", "tune_neg_home", "test_neg_home"]
    if all((root / "stores" / r / "sins" / "meta.csv").exists() for r in roles):
        print("skip sins")
        return
    z = work / "sins"
    aria([(ZENODO.format(rec=SINS_REC, name="DCASE2018-task5-dev.meta.zip"), "meta.zip")], z)
    sh(f"cd {z} && unzip -oq meta.zip")
    meta = pd.read_csv(z / "DCASE2018-task5-dev" / "meta.txt", sep="\t", header=None, names=["f", "act", "sess"])
    _G["plan"] = sins_plan(meta)
    print(_G["plan"].groupby("role").size().mul(10 / 3600).round(1).rename("hours"))
    names = [f"DCASE2018-task5-dev.audio.{i}.zip" for i in range(1, 24)]
    parts = work / "parts" / "sins"
    # download in batches of 6 zips (12 GB) to bound disk use
    for b in range(0, len(names), 6):
        batch = names[b: b + 6]
        aria([(ZENODO.format(rec=SINS_REC, name=n), n) for n in batch], z, jobs=6)
        with Pool(min(workers, len(batch))) as p:
            list(p.imap_unordered(_sins_zip, [(str(z / n), str(parts)) for n in batch]))
    for r in roles:
        n = merge_parts(sorted((parts / r).glob("*")), root / "stores" / r / "sins")
        print(f"  {r}: {n} segments")
    shutil.rmtree(parts, ignore_errors=True)


# ---------------------------------------------------------------- summary

def summary(root):
    rows = []
    for meta in sorted((root / "stores").glob("*/*/meta.csv")):
        m = pd.read_csv(meta, keep_default_na=False)
        role, src = meta.parent.parent.name, meta.parent.name
        row = {"role": role, "source": src, "clips": len(m), "hours": round(m.n.sum() / 16000 / 3600, 2)}
        for c in ["alarm", "siren", "doorbell", "knock", "baby_cry", "car_horn"]:
            row[c] = int(m.classes.str.split(";").apply(lambda L: c in L).sum())
        rows.append(row)
    s = pd.DataFrame(rows)
    s.to_csv(root / "stores" / "summary.csv", index=False)
    print(s.to_string(index=False))
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--work", type=Path, default=Path("/content/work"))
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    ap.add_argument("--only", nargs="*", default=["eval", "train", "bal", "fsd", "sins"])
    a = ap.parse_args()
    a.work.mkdir(parents=True, exist_ok=True)
    steps = {"eval": lambda: build_strong(a.root, a.work, "eval", a.workers),
             "train": lambda: build_strong(a.root, a.work, "train", a.workers),
             "bal": lambda: build_bal(a.root, a.work, a.workers),
             "fsd": lambda: build_fsd(a.root, a.work, a.workers),
             "sins": lambda: build_sins(a.root, a.work, a.workers)}
    for k in a.only:
        steps[k]()
    summary(a.root)


if __name__ == "__main__":
    main()
