"""Student front end: framing + FFT + mel filterbank, then log or PCEN compression.

Designed to run on the ESP32-S3 as is: non-centred frames (n_fft 512 = 32 ms, hop 320 =
20 ms), periodic Hann window, power spectrum, 40 triangular mel filters 50-8000 Hz.
A 2 s window = 100 frames.

PCEN (per-channel energy normalisation, Wang et al. 2017) replaces the log with an
adaptive gain control: a first-order IIR low-pass M tracks each band's energy E and
    PCEN = (E / (eps + M)^alpha + delta)^r - delta^r,
which removes slow loudness changes (distance, mic gain) and keeps onsets.
"""
import numpy as np
from scipy.signal import lfilter

SR = 16000
N_FFT, HOP, N_MELS, FMIN, FMAX = 512, 320, 40, 50.0, 8000.0
FRAMES_2S = 100
DITHER_DBFS = -90.0   # every signal gets the same tiny noise floor, like a real mic


def hz_to_mel(f):
    return 2595.0 * np.log10(1.0 + np.asarray(f) / 700.0)


def mel_to_hz(m):
    return 700.0 * (10 ** (np.asarray(m) / 2595.0) - 1.0)


def mel_filterbank(n_mels=N_MELS, n_fft=N_FFT, sr=SR, fmin=FMIN, fmax=FMAX):
    """(n_mels, n_fft//2+1) triangular filters, area-normalised (Slaney style)."""
    freqs = np.linspace(0, sr / 2, n_fft // 2 + 1)
    edges = mel_to_hz(np.linspace(hz_to_mel(fmin), hz_to_mel(fmax), n_mels + 2))
    fb = np.zeros((n_mels, len(freqs)), np.float32)
    for m in range(n_mels):
        lo, c, hi = edges[m], edges[m + 1], edges[m + 2]
        up = (freqs - lo) / (c - lo)
        down = (hi - freqs) / (hi - c)
        fb[m] = np.maximum(0, np.minimum(up, down)) * 2.0 / (hi - lo)
    return fb


_FB = mel_filterbank()
_WIN = (0.5 - 0.5 * np.cos(2 * np.pi * np.arange(N_FFT) / N_FFT)).astype(np.float32)


def mel_power(x: np.ndarray) -> np.ndarray:
    """(n_mels, frames) mel power of float audio in [-1, 1] (computed on the int16 scale)."""
    x = np.asarray(x, np.float32) * 32768.0
    n = 1 + (len(x) - N_FFT) // HOP
    idx = np.arange(N_FFT)[None, :] + HOP * np.arange(n)[:, None]
    spec = np.abs(np.fft.rfft(x[idx] * _WIN, axis=1)) ** 2
    return (_FB @ spec.T).astype(np.float32)


def pcen(E, T=0.4, alpha=0.98, delta=2.0, r=0.5, eps=1e-6):
    hop_s = HOP / SR
    t_frames = T / hop_s
    b = (np.sqrt(1 + 4 * t_frames ** 2) - 1) / (2 * t_frames ** 2)
    # M[t] = (1-b) M[t-1] + b E[t], starting from M[-1] = E[0]
    M, _ = lfilter([b], [1, b - 1], E, axis=1, zi=((1 - b) * E[:, :1]))
    return (E / (eps + M) ** alpha + delta) ** r - delta ** r


def features(x, kind="logmel", rng=None):
    rng = rng or np.random.default_rng(len(x))
    x = x + rng.normal(0, 10 ** (DITHER_DBFS / 20), len(x)).astype(np.float32)
    E = mel_power(x)
    if kind == "logmel":
        return np.log(E + 1.0).astype(np.float32)
    if kind == "pcen":
        return pcen(E).astype(np.float32)
    raise ValueError(kind)


def stream_windows(x, kind, hop_s=0.5, win_s=2.0):
    """Features for a whole clip as a stream (2 s of silence before it), cut into windows of
    100 frames ending every 0.5 s. Returns (n_windows, n_mels, 100); first window ends at 0.5 s."""
    pad = int(win_s * SR)
    F = features(np.concatenate([np.zeros(pad, np.float32), x]), kind)
    n = len(x) // int(hop_s * SR)
    ends = [(pad + (k + 1) * int(hop_s * SR) - N_FFT) // HOP for k in range(n)]
    return np.stack([F[:, e - FRAMES_2S + 1: e + 1] for e in ends])
