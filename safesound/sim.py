"""DSP simulation of real listening conditions, used to make training examples on the fly.

Each example is 3 s of audio (1 s of warm-up for the front end + the 2 s the label is about)
and a 6-class label for the last 2 s. Simulation steps, switched on cumulatively for the
ablation (configs A-E):

  mix   real home background (SINS) added at a random SNR; random overall level
  rir   room reverberation: statistical room impulse response (see `room_ir`)
  occl  sound heard through a door or wall: low-pass transmission (mass law, 6-12 dB/octave)
  mic   cheap MEMS microphone (INMP441-like): port high-pass, response tilt, self-noise floor,
        clipping, 16-bit quantisation

Positives and negatives always get the same treatment, so the model cannot use the
simulation itself as a cue.
"""
from dataclasses import dataclass

import numpy as np
from scipy.signal import butter, fftconvolve, lfilter, sosfilt

from safesound.classes import CLASSES

SR = 16000
CLIP, LABEL = 3 * SR, 2 * SR


@dataclass(frozen=True)
class SimConfig:
    mix: bool = False
    rir: bool = False
    occl: bool = False
    mic: bool = False


CONFIGS = {
    "A_raw": SimConfig(),
    "B_mix": SimConfig(mix=True),
    "C_rir": SimConfig(mix=True, rir=True),
    "D_occl": SimConfig(mix=True, rir=True, occl=True),
    "E_mic": SimConfig(mix=True, rir=True, occl=True, mic=True),
}


def db(x):
    return 10 ** (x / 20)


def rms(x):
    return float(np.sqrt(np.mean(x ** 2) + 1e-12))


def active_rms(x, frac=0.2, frame=800):
    """RMS of the loudest 20% of 50 ms frames (level of the sound itself, not its gaps)."""
    n = len(x) // frame
    if n == 0:
        return rms(x)
    e = (x[: n * frame].reshape(n, frame) ** 2).mean(1)
    k = max(1, int(n * frac))
    return float(np.sqrt(np.sort(e)[-k:].mean() + 1e-12))


# ------------------------------------------------------------------ room acoustics

_BANDS = [(None, 500), (500, 1500), (1500, 4000), (4000, None)]
_BAND_RT = [1.2, 1.0, 0.8, 0.55]     # high frequencies decay faster (air + wall absorption)


def room_ir(rng, rt60=None, dist=None, volume=None):
    """Statistical room impulse response (Polack model).

    - direct path at t=0;
    - 6-12 early reflections in the first 50 ms (image-source amplitude dist / path length);
    - diffuse tail: Gaussian noise with exponential decay exp(-6.9 t / RT60), in four bands
      with band-dependent RT60;
    - direct-to-reverberant ratio set by distance vs critical distance
      r_c = 0.057 * sqrt(V / RT60) (Sabine).
    """
    rt60 = rt60 or rng.uniform(0.2, 1.0)
    volume = volume or rng.uniform(25, 200)          # m^3: small room .. large living room
    dist = dist or rng.uniform(0.5, 8.0)              # m
    n = int(SR * min(1.5 * rt60, 1.2)) + 1
    t = np.arange(n) / SR
    h = np.zeros(n, np.float32)
    h[0] = 1.0
    for _ in range(rng.integers(6, 13)):
        extra = rng.uniform(0.7, 17.0)                 # extra path length of the reflection (m)
        k = int(extra / 343.0 * SR)                    # arrives 2-50 ms after the direct sound
        # image-source amplitude: 1/r spreading relative to the direct path, wall loss 0.5-0.9
        h[k] += rng.choice([-1, 1]) * rng.uniform(0.5, 0.9) * dist / (dist + extra)
    tail = np.zeros(n)
    noise = rng.normal(size=n)
    for (lo, hi), f in zip(_BANDS, _BAND_RT):
        if lo is None:
            sos = butter(2, hi, "lowpass", fs=SR, output="sos")
        elif hi is None:
            sos = butter(2, lo, "highpass", fs=SR, output="sos")
        else:
            sos = butter(2, [lo, hi], "bandpass", fs=SR, output="sos")
        tail += sosfilt(sos, noise) * np.exp(-6.9 * t / (rt60 * f))
    tail[: int(0.005 * SR)] = 0
    rc = 0.057 * np.sqrt(volume / rt60)
    drr = (rc / dist) ** 2                           # direct / reverberant energy ratio
    e_dir = float((h ** 2).sum())
    tail *= np.sqrt(e_dir / (drr * (tail ** 2).sum() + 1e-12))
    h = h + tail.astype(np.float32)
    return h / np.sqrt((h ** 2).sum())


def occlude(x, rng):
    """Transmission through a closed door / wall: 1st or 2nd order low-pass (mass law)."""
    order = rng.choice([1, 2])
    fc = rng.uniform(300, 1500)
    sos = butter(order, fc, "lowpass", fs=SR, output="sos")
    return sosfilt(sos, x).astype(np.float32)


# ------------------------------------------------------------------ microphone

def high_shelf(gain_db, f0=3000.0, q=0.707):
    """RBJ cookbook high-shelf biquad (b, a)."""
    A = 10 ** (gain_db / 40)
    w0 = 2 * np.pi * f0 / SR
    alpha = np.sin(w0) / (2 * q)
    cw = np.cos(w0)
    b = [A * ((A + 1) + (A - 1) * cw + 2 * np.sqrt(A) * alpha),
         -2 * A * ((A - 1) + (A + 1) * cw),
         A * ((A + 1) + (A - 1) * cw - 2 * np.sqrt(A) * alpha)]
    a = [(A + 1) - (A - 1) * cw + 2 * np.sqrt(A) * alpha,
         2 * ((A - 1) - (A + 1) * cw),
         (A + 1) - (A - 1) * cw - 2 * np.sqrt(A) * alpha]
    return np.array(b) / a[0], np.array(a) / a[0]


def mic_chain(x, rng):
    """INMP441-like capture: port/DC high-pass, response tilt, self-noise, clipping, 16-bit."""
    x = sosfilt(butter(2, rng.uniform(50, 200), "highpass", fs=SR, output="sos"), x)
    b, a = high_shelf(rng.uniform(-4, 4))
    x = lfilter(b, a, x)
    x = x + rng.normal(0, db(rng.uniform(-80, -65)), len(x))
    if rng.random() < 0.03:
        x = x * db(15)                                # loud event close to the mic: clipping
    x = np.clip(x, -1, 1)
    return (np.round(x * 32767) / 32767).astype(np.float32)


# ------------------------------------------------------------------ example generator

class Generator:
    """Makes (audio 3 s, label 6) training examples. `pos` = Clips of positives,
    `neg` = Clips of negatives, `bg` = Clips of home backgrounds (bench.Clips-like objects)."""

    def __init__(self, pos, neg, bg, cfg: SimConfig, p_pos=0.5, seed=0):
        self.pos, self.neg, self.bg, self.cfg, self.p_pos = pos, neg, bg, cfg, p_pos
        self.rng = np.random.default_rng(seed)
        self.by_class = {c: [i for i in range(len(pos)) if pos.events(i, c)] for c in CLASSES}
        self.weak = pos.meta.weak.astype(str).str.lower().eq("true").to_numpy()
        self.rirs = [room_ir(self.rng) for _ in range(200)] if cfg.rir else []

    def crop(self, x, end):
        start = end - CLIP
        if start < 0:
            x = np.concatenate([np.zeros(-start, np.float32), x])
            end, start = CLIP, 0
        if end > len(x):
            x = np.concatenate([x, np.zeros(end - len(x), np.float32)])
        return x[start:end]

    def positive(self):
        rng = self.rng
        c = CLASSES[rng.integers(len(CLASSES))]
        i = self.by_class[c][rng.integers(len(self.by_class[c]))]
        x = self.pos.audio(i)
        n = len(x)
        if self.weak[i]:
            # weak label: put the label window on a loud part of the clip
            hop = SR // 4
            ends = np.arange(min(LABEL, n), n + 1, hop) if n >= LABEL else np.array([n])
            e = np.array([rms(x[max(0, k - LABEL):k]) for k in ends])
            p = e ** 2 / (e ** 2).sum() if e.sum() > 0 else None
            end = int(rng.choice(ends, p=p))
        else:
            on, off = self.pos.events(i, c)[rng.integers(len(self.pos.events(i, c)))]
            lo = on + min(0.3, (off - on))
            hi = max(lo, off + 1.7)
            end = int(min(rng.uniform(lo, hi), n / SR) * SR)
            end = max(end, int(lo * SR))
        y = np.zeros(len(CLASSES), np.float32)
        t1, t0 = end / SR, end / SR - LABEL / SR
        for k, ck in enumerate(CLASSES):
            if self.weak[i]:
                y[k] = float(ck in str(self.pos.meta.classes.iloc[i]).split(";"))
            else:
                for on, off in self.pos.events(i, ck):
                    if min(off, t1) - max(on, t0) >= min(0.2, off - on):
                        y[k] = 1.0
        return self.crop(x, end), y

    def negative(self):
        rng = self.rng
        src = self.neg if rng.random() < 0.5 else self.bg
        x = src.audio(rng.integers(len(src)))
        end = rng.integers(min(CLIP, len(x)), len(x) + 1)
        return self.crop(x, end), np.zeros(len(CLASSES), np.float32)

    def background(self):
        x = self.bg.audio(self.rng.integers(len(self.bg)))
        if len(x) < CLIP:
            x = np.tile(x, CLIP // max(len(x), 1) + 1)
        s = self.rng.integers(0, len(x) - CLIP + 1)
        return x[s: s + CLIP]

    def __call__(self):
        rng, cfg = self.rng, self.cfg
        positive = rng.random() < self.p_pos
        x, y = self.positive() if positive else self.negative()
        if cfg.rir and rng.random() < 0.7:
            x = fftconvolve(x, self.rirs[rng.integers(len(self.rirs))])[:CLIP].astype(np.float32)
        if cfg.occl and rng.random() < 0.25:
            x = occlude(x, rng)
        if cfg.mix:
            if rng.random() < 0.9:
                b = self.background()
                snr = rng.uniform(-5, 25)
                x = x / active_rms(x) * rms(b) * db(snr) + b
            x = x / (rms(x) + 1e-9) * db(rng.uniform(-45, -12))
        if cfg.mic:
            x = mic_chain(x, rng)
        x = np.clip(x, -1, 1).astype(np.float32)
        return x, y
