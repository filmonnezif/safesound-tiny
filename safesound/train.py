"""Training on simulated examples generated on the fly (CPU workers), and detectors for bench."""
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, IterableDataset

from safesound import bench
from safesound.frontend import FRAMES_2S, features, stream_windows
from safesound.models import DSCNN, feature_stats
from safesound.sim import LABEL, Generator, SimConfig


class SimData(IterableDataset):
    def __init__(self, data_root, cfg: SimConfig, kind="logmel", feats=True, wave=False, seed=0):
        self.data_root, self.cfg, self.kind, self.feats, self.wave, self.seed = data_root, cfg, kind, feats, wave, seed

    def __iter__(self):
        wi = torch.utils.data.get_worker_info()
        seed = self.seed * 1000 + (wi.id if wi else 0) + int.from_bytes(os.urandom(2), "little") * 7919
        gen = Generator(bench.Clips(self.data_root, "train_pos"), bench.Clips(self.data_root, "train_neg"),
                        bench.Clips(self.data_root, "train_bg_home"), self.cfg, seed=seed)
        rng = np.random.default_rng(seed)
        while True:
            x, y = gen()
            f = features(x, self.kind, rng)[:, -FRAMES_2S:] if self.feats else np.zeros(1, np.float32)
            w = x[-LABEL:] if self.wave else np.zeros(1, np.float32)
            yield f, w, y


def loader(data_root, cfg, kind, batch, feats=True, wave=False, workers=None, seed=0):
    workers = workers or max(1, os.cpu_count() - 4)
    return iter(DataLoader(SimData(data_root, cfg, kind, feats, wave, seed), batch_size=batch,
                           num_workers=workers, pin_memory=True, prefetch_factor=4))


def train_student(data_root, cfg, kind="logmel", width=64, blocks=4, steps=12000, batch=256, lr=3e-3,
                  teacher=None, kd=0.5, log_every=1000):
    """Hard-label training, or knowledge distillation from `teacher` (BCE to its probabilities)."""
    model = DSCNN(width, blocks).cuda()
    it = loader(data_root, cfg, kind, batch, wave=teacher is not None)
    sample = np.concatenate([next(it)[0].numpy() for _ in range(8)])
    model.feat_mean.copy_(feature_stats(sample)[0]), model.feat_std.copy_(feature_stats(sample)[1])
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps, pct_start=0.1)
    t, run = time.time(), 0.0
    for step in range(1, steps + 1):
        f, w, y = next(it)
        f, y = f.cuda(non_blocking=True), y.cuda(non_blocking=True)
        logits = model(f)
        loss = F.binary_cross_entropy_with_logits(logits, y)
        if teacher is not None:
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                tp = torch.sigmoid(teacher(w.cuda(non_blocking=True)).float())
            loss = (1 - kd) * loss + kd * F.binary_cross_entropy_with_logits(logits, tp)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()
        run = 0.98 * run + 0.02 * loss.item()
        if step % log_every == 0:
            print(f"  step {step}/{steps} loss {run:.4f} {batch * log_every / (time.time() - t):.0f} ex/s", flush=True)
            t = time.time()
    return model.eval()


def train_teacher(data_root, cfg, cnn14, mids, steps=15000, batch=64, lr=1e-4, log_every=1000):
    from torchlibrosa.augmentation import SpecAugmentation
    from safesound.models import Teacher
    model = Teacher(cnn14, mids).cuda().train()
    specaug = SpecAugmentation(time_drop_width=16, time_stripes_num=2, freq_drop_width=8, freq_stripes_num=2)
    it = loader(data_root, cfg, "logmel", batch, feats=False, wave=True)
    opt = torch.optim.AdamW([{"params": model.base.parameters(), "lr": lr},
                             {"params": model.head.parameters(), "lr": lr * 10}], weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[lr, lr * 10], total_steps=steps, pct_start=0.05)
    t, run = time.time(), 0.0
    for step in range(1, steps + 1):
        _, w, y = next(it)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(w.cuda(non_blocking=True), specaug)
        loss = F.binary_cross_entropy_with_logits(logits.float(), y.cuda(non_blocking=True))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()
        run = 0.98 * run + 0.02 * loss.item()
        if step % log_every == 0:
            print(f"  step {step}/{steps} loss {run:.4f} {batch * log_every / (time.time() - t):.0f} ex/s", flush=True)
            t = time.time()
    return model.eval()


# ------------------------------------------------------------------ detectors for bench.run

WIN, HOP = 32000, 8000   # 2 s windows every 0.5 s


_POOL = None


def _sw(args):
    return stream_windows(*args)


def student_detector(model, kind):
    @torch.no_grad()
    def det(batch):
        global _POOL
        if _POOL is None:
            import multiprocessing as mp
            _POOL = mp.get_context("fork").Pool(min(32, os.cpu_count()))
        wins = _POOL.map(_sw, [(x, kind) for x in batch])
        flat = torch.from_numpy(np.concatenate(wins)).cuda()
        p = torch.cat([torch.sigmoid(model(flat[i:i + 4096])) for i in range(0, len(flat), 4096)]).cpu().numpy()
        return np.split(p, np.cumsum([len(w) for w in wins])[:-1])
    return det


def teacher_detector(model):
    @torch.no_grad()
    def det(batch):
        wins = [bench.windows(x, WIN, HOP) for x in batch]
        flat = torch.from_numpy(np.concatenate(wins)).cuda()
        out = []
        for i in range(0, len(flat), 1024):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out.append(torch.sigmoid(model(flat[i:i + 1024]).float()))
        p = torch.cat(out).cpu().numpy()
        return np.split(p, np.cumsum([len(w) for w in wins])[:-1])
    return det
