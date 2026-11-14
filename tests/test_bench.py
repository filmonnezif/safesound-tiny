import numpy as np

from safesound import bench
from safesound.store import StoreWriter


def fake_root(tmp_path):
    rng = np.random.default_rng(0)
    for split in ["test", "tune"]:
        w = StoreWriter(tmp_path / "stores" / f"{split}_pos" / "src")
        for k in range(6):
            x = (rng.normal(size=160000) * 300).astype(np.int16)
            x[48000:64000] = 20000  # loud "event" at 3-4 s
            w.add(x, uid=f"{split}{k}", source="s", classes="alarm", events=[["alarm", 3.0, 4.0]],
                  weak=False, group=f"g{k}", info="")
        w.close()
        for neg in ["neg_as", "neg_home"]:
            w = StoreWriter(tmp_path / "stores" / f"{split}_{neg}" / "src")
            for k in range(20):
                w.add((rng.normal(size=160000) * 300).astype(np.int16), uid=f"{split}{neg}{k}", source="s",
                      classes="", events=[], weak=False, group=f"n{k}", info="")
            w.close()
    return tmp_path


def energy_detector(batch):
    out = []
    for x in batch:
        w = bench.windows(x, 32000, 8000)
        e = np.sqrt((w ** 2).mean(1))
        out.append(np.repeat(e[:, None], 6, 1))
    return out


def test_windows_are_causal():
    x = np.arange(16000 * 3, dtype=np.float32)
    w = bench.windows(x, 32000, 8000)
    assert w.shape == (6, 32000)
    assert w[0, -1] == 7999 and w[0, 0] == 0  # first window ends at 0.5 s, zero-padded start
    assert w[-1, -1] == len(x) - 1


def test_bench_end_to_end(tmp_path):
    root = fake_root(tmp_path)
    res = bench.run(root, root, "energy", energy_detector, hop=0.5, t0=0.5)
    r = res["classes"]["alarm"]["@1.0"]
    assert r["recall"] == 1.0 and r["occurrences"] == 6
    assert r["fa_per_h_test"]["neg_as"] <= 2.0
    assert len(bench.manifest_hash(root)) == 64
