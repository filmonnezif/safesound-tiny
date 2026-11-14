"""A simple clip store: int16 16 kHz audio in a few large .npy shards plus meta.csv.

meta.csv columns: uid, source, classes (';'-joined), events (JSON [[cls, on, off], ...]),
weak (True if labels are clip-level only), group (video / session, for leakage checks),
info (free text), shard, offset, n.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

SHARD_BYTES = 512 * 2**20


class StoreWriter:
    def __init__(self, path):
        self.path = Path(path)
        self.path.mkdir(parents=True, exist_ok=True)
        self.rows, self.buf, self.buf_n, self.shard = [], [], 0, 0

    def add(self, audio: np.ndarray, **row):
        assert audio.dtype == np.int16
        row.update(shard=self.shard, offset=self.buf_n, n=len(audio))
        if isinstance(row.get("events"), list):
            row["events"] = json.dumps(row["events"])
        self.rows.append(row)
        self.buf.append(audio)
        self.buf_n += len(audio)
        if self.buf_n * 2 >= SHARD_BYTES:
            self._flush()

    def _flush(self):
        if self.buf:
            np.save(self.path / f"shard_{self.shard:03d}.npy", np.concatenate(self.buf))
            self.shard += 1
            self.buf, self.buf_n = [], 0

    def close(self):
        self._flush()
        pd.DataFrame(self.rows).to_csv(self.path / "meta.csv", index=False)
        return len(self.rows)


class Store:
    def __init__(self, path):
        self.path = Path(path)
        self.meta = pd.read_csv(self.path / "meta.csv", keep_default_na=False)
        self._shards = {}

    def __len__(self):
        return len(self.meta)

    def audio(self, i) -> np.ndarray:
        r = self.meta.iloc[i]
        s = int(r.shard)
        if s not in self._shards:
            self._shards[s] = np.load(self.path / f"shard_{s:03d}.npy", mmap_mode="r")
        return np.asarray(self._shards[s][int(r.offset): int(r.offset) + int(r.n)])

    def events(self, i, cls=None):
        ev = json.loads(self.meta.iloc[i].events or "[]")
        return [(on, off) for c, on, off in ev if cls is None or c == cls]

    def hours(self):
        return self.meta.n.sum() / 16000 / 3600


def merge_parts(part_dirs, out):
    """Merge many small StoreWriter outputs into one store."""
    w = StoreWriter(out)
    for p in part_dirs:
        p = Path(p)
        if not (p / "meta.csv").exists():
            continue
        s = Store(p)
        for i in range(len(s)):
            row = s.meta.iloc[i].to_dict()
            for k in ("shard", "offset", "n"):
                row.pop(k)
            w.add(s.audio(i), **row)
    return w.close()
