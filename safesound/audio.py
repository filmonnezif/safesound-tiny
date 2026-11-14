"""Audio decoding and resampling to the project rate (16 kHz mono)."""
import io
from fractions import Fraction

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

SR = 16000


def to_mono16k(x: np.ndarray, sr: int, channel: int | None = None) -> np.ndarray:
    """float32 mono at 16 kHz. `channel` picks one channel instead of averaging."""
    x = np.asarray(x, dtype=np.float32)
    if x.ndim == 2:
        x = x[:, channel] if channel is not None else x.mean(axis=1)
    if sr != SR:
        f = Fraction(SR, sr).limit_denominator(1000)
        x = resample_poly(x, f.numerator, f.denominator).astype(np.float32)
    return x


def decode(data: bytes, channel: int | None = None) -> np.ndarray:
    x, sr = sf.read(io.BytesIO(data), dtype="float32", always_2d=False)
    return to_mono16k(x, sr, channel)


def to_int16(x: np.ndarray) -> np.ndarray:
    return (np.clip(x, -1.0, 1.0) * 32767).astype(np.int16)


def to_float(x: np.ndarray) -> np.ndarray:
    return x.astype(np.float32) / 32767.0
