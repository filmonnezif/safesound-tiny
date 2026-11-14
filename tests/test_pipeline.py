import numpy as np
import torch

from safesound import bench, frontend, sim
from safesound.models import DSCNN, count_macs, count_params
from safesound.quant import model_bytes_int8, quantize
from safesound.store import StoreWriter


def fake_train_root(tmp_path):
    rng = np.random.default_rng(1)
    w = StoreWriter(tmp_path / "stores" / "train_pos" / "a")
    for k, c in enumerate(["alarm", "siren", "doorbell", "knock", "baby_cry", "car_horn"]):
        x = (rng.normal(size=160000) * 500).astype(np.int16)
        w.add(x, uid=f"p{k}", source="s", classes=c, events=[[c, 2.0, 3.0]], weak=False, group="g", info="")
        w.add(x[:8000], uid=f"w{k}", source="s", classes=c, events=[[c, 0.0, 0.5]], weak=True, group="g", info="")
    w.close()
    for role in ["train_neg", "train_bg_home"]:
        w = StoreWriter(tmp_path / "stores" / role / "a")
        for k in range(4):
            w.add((rng.normal(size=160000) * 200).astype(np.int16), uid=f"{role}{k}", source="s", classes="",
                  events=[], weak=False, group="g", info="")
        w.close()
    return tmp_path


def test_room_ir_and_mic():
    rng = np.random.default_rng(0)
    near = sim.room_ir(rng, rt60=0.6, dist=0.3, volume=60)   # inside critical distance (0.57 m)
    far = sim.room_ir(rng, rt60=0.6, dist=3.0, volume=60)
    for h in (near, far):
        assert abs((h ** 2).sum() - 1) < 1e-4
    # direct sound dominates near the source, reverberation far away
    assert near[0] ** 2 > 0.5 and far[0] ** 2 < 0.1
    x = sim.mic_chain(rng.normal(size=48000).astype(np.float32) * 0.01, rng)
    assert x.dtype == np.float32 and np.abs(x).max() <= 1


def test_generator_all_configs(tmp_path):
    root = fake_train_root(tmp_path)
    pos, neg, bg = (bench.Clips(root, r) for r in ["train_pos", "train_neg", "train_bg_home"])
    for name, cfg in sim.CONFIGS.items():
        g = sim.Generator(pos, neg, bg, cfg, seed=3)
        ys = []
        for _ in range(30):
            x, y = g()
            assert x.shape == (sim.CLIP,) and np.isfinite(x).all()
            ys.append(y)
        assert np.array(ys).sum() > 0, name


def test_frontend_shapes():
    x = np.random.default_rng(0).normal(size=48000).astype(np.float32) * 0.05
    for kind in ["logmel", "pcen"]:
        f = frontend.features(x, kind)
        assert f.shape == (40, 1 + (48000 - 512) // 320) and np.isfinite(f).all()
    w = frontend.stream_windows(np.zeros(160000, np.float32), "logmel")
    assert w.shape == (20, 40, 100)


def test_student_and_int8():
    m = DSCNN(32, 4).eval()
    f = torch.randn(64, 40, 100)
    q = quantize(m, f)
    a, b = torch.sigmoid(m(f)), torch.sigmoid(q(f))
    assert (a - b).abs().max() < 0.1
    assert count_params(m) > 1000 and count_macs(m) > 1e5 and model_bytes_int8(m) > 1000
