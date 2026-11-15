"""Classical DSP detectors, one per class, plus a class-agnostic loud-onset detector.

All run causally on a 16 kHz stream: STFT (64 ms Hann, 10 ms hop), a per-bin noise floor
tracked by a minimum follower, and hand-designed features from the physics of each sound.
Scores are produced for the same windows as the networks (2 s ending every 0.5 s), so they can
be fused with them and evaluated with the same protocol.

  alarm     narrowband tone in 2-5 kHz that dominates the low band, at a fixed frequency,
            switching on and off (ISO 8201 T3 / periodic beeps)
  siren     sustained tonal peak in 500-1800 Hz whose frequency sweeps smoothly (wail / yelp)
  doorbell  tonal onset in 500-3000 Hz that keeps ringing and decays (chime)
  knock     repeated broadband onsets (spectral flux) with regular spacing (80-700 ms)
  baby_cry  voiced frames with f0 300-650 Hz, harmonics 1-3 present, pitch that wobbles
  car_horn  loud tone in 250-1500 Hz, steady pitch across the window, sustained
  loud      class-agnostic safety net: broadband level jumps well above the noise floor

The noise floor starts from the 10th percentile of each bin over the clip, standing in for a
detector that has been running in the room; after that it is strictly causal.
"""
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from scipy.ndimage import maximum_filter1d, uniform_filter1d

SR = 16000
NFFT, HOP = 1024, 160
FPS = SR / HOP
WIN_S, STEP_S = 2.0, 0.5
FREQS = np.fft.rfftfreq(NFFT, 1 / SR)
_W = np.hanning(NFFT).astype(np.float32)
OUT = ["alarm", "siren", "doorbell", "knock", "baby_cry", "car_horn", "loud"]


def band(lo, hi):
    return np.flatnonzero((FREQS >= lo) & (FREQS < hi))


B_ALARM, B_LOW = band(2000, 5000), band(100, 1500)
B_SIREN, B_BELL, B_HORN, B_KNOCK, B_VOICE = (band(500, 1800), band(500, 3000), band(250, 1500),
                                             band(200, 4000), band(250, 4000))
F0_CRY = np.arange(300, 660, 10.0)


def ramp(x, lo, hi):
    return np.clip((np.asarray(x, float) - lo) / (hi - lo), 0.0, 1.0)


def stft_power(x):
    n = 1 + (len(x) - NFFT) // HOP
    if n < 1:
        return np.full((len(FREQS), 1), 1e-10, np.float32)
    idx = np.arange(NFFT)[None, :] + HOP * np.arange(n)[:, None]
    return (np.abs(np.fft.rfft(x[idx] * _W, axis=1)) ** 2).T.astype(np.float32) + 1e-10


def noise_floor(P, rise_s=2.0):
    """Minimum follower: drops at once to a quieter value, rises ~6 dB per rise_s."""
    up = 10 ** (6 / 10 / (rise_s * FPS))
    N = np.empty_like(P)
    n = np.percentile(P, 10, axis=1)
    for t in range(P.shape[1]):
        n = np.minimum(n * up, P[:, t])
        N[:, t] = n
    return N


def tonal_peak(PdB, NdB, b):
    sub = PdB[b]
    k = np.argmax(sub, axis=0)
    cols = np.arange(sub.shape[1])
    peak = sub[k, cols]
    return FREQS[b][k], peak, peak - np.median(sub, axis=0), peak - NdB[b][k, cols]


def prev_max(v, n, lag=0):
    """max of v[t-lag-n .. t-lag-1] for every t."""
    padded = np.concatenate([np.zeros(n + lag, v.dtype), v])
    return sliding_window_view(padded, n).max(1)[:len(v)]


def frame_features(x):
    P = stft_power(x)
    N = noise_floor(P)
    PdB, NdB = 10 * np.log10(P), 10 * np.log10(N)
    T = P.shape[1]
    F = {}

    f, peak, prom, snr = tonal_peak(PdB, NdB, B_ALARM)
    dom = ramp(peak - PdB[B_LOW].max(0), -6, 6)                       # tone louder than low band
    F["alarm"] = ramp(prom, 10, 25) * ramp(snr, 6, 20) * dom
    F["alarm_f"] = f

    f, _, prom, snr = tonal_peak(PdB, NdB, B_SIREN)
    F["siren"] = ramp(prom, 8, 22) * ramp(snr, 6, 20)
    F["siren_f"] = f

    _, _, prom, snr = tonal_peak(PdB, NdB, B_BELL)
    bell = (ramp(prom, 10, 25) * ramp(snr, 10, 25)).astype(np.float32)
    F["bell"] = bell
    F["bell_on"] = np.maximum(0, bell - prev_max(bell, 10, lag=5))

    L = PdB[B_KNOCK]
    flux = np.maximum(0, np.diff(L, axis=1, prepend=L[:, :1])).mean(0)
    F["knock"] = ramp(flux, 3, 12) * ramp((L - NdB[B_KNOCK]).mean(0), 6, 18)

    mag = np.sqrt(P)
    hmin = np.full((len(F0_CRY), T), np.inf, np.float32)
    for h in (1, 2, 3):
        bins = np.clip(np.round(F0_CRY * h / (SR / NFFT)).astype(int), 0, len(FREQS) - 1)
        hmin = np.minimum(hmin, mag[bins])                             # all of h1-h3 must be there
    best = hmin.argmax(0)
    cols = np.arange(T)
    harm = 20 * np.log10(hmin[best, cols] / (mag[B_VOICE].mean(0) + 1e-9) + 1e-9)
    f0 = F0_CRY[best]
    m0 = mag[np.round(f0 / (SR / NFFT)).astype(int), cols]
    sub = np.maximum(mag[np.round(f0 / 2 / (SR / NFFT)).astype(int), cols],
                     mag[np.round(f0 / 3 / (SR / NFFT)).astype(int), cols])
    true_f0 = ramp(20 * np.log10(m0 / (sub + 1e-9) + 1e-9), 3, 12)     # no subharmonic energy
    F["cry"] = ramp(harm, 4, 14) * ramp((PdB[B_VOICE] - NdB[B_VOICE]).mean(0), 3, 15) * true_f0
    F["cry_f0"] = F0_CRY[best]

    f, peak, prom, snr = tonal_peak(PdB, NdB, B_HORN)
    F["horn"] = ramp(prom, 10, 25) * ramp(snr, 12, 30)
    F["horn_f"], F["horn_db"] = f, peak

    lvl = P.sum(0)
    floor = noise_floor(lvl[None, :])[0]
    F["loud"] = ramp(10 * np.log10(lvl / floor), 10, 30)
    return F


def _freq_std(f, w):
    sel = f[w > 0.3]
    return np.std(sel) if len(sel) > 3 else np.inf


def window_scores(x):
    """(n_windows, 7) scores for 2 s windows ending every 0.5 s (the first windows reach back
    before the clip start, where there is no evidence)."""
    x = np.asarray(x, np.float32)
    F = frame_features(x)
    T = len(F["loud"])
    n = len(x) // int(STEP_S * SR)
    W = int(WIN_S * FPS)
    out = np.zeros((n, 7), np.float32)
    for k in range(n):
        end = min(T, int(((k + 1) * STEP_S * SR - NFFT) // HOP) + 1)
        if end <= 0:
            continue
        s = slice(max(0, end - W), end)

        a = F["alarm"][s]
        edges = np.count_nonzero(np.diff((a > 0.3).astype(int)) == 1)
        steady = ramp(-_freq_std(F["alarm_f"][s], a), -60, -15)
        out[k, 0] = np.percentile(a, 90) * steady * (0.6 + 0.4 * min(edges, 3) / 3)

        sv, sf = F["siren"][s], F["siren_f"][s]
        fs = sf[sv > 0.3]
        if len(fs) > 0.3 * W:
            smooth = ramp(-np.median(np.abs(np.diff(fs))), -40, -5)
            cont = ramp(len(fs) / max(1, len(sf)), 0.5, 0.9)
            out[k, 1] = sv.mean() * ramp(np.ptp(fs), 150, 500) * smooth * cont

        on, bt = F["bell_on"][s], F["bell"][s]
        if on.max() > 0:
            i = int(np.argmax(on))
            later = bt[min(i + 25, len(bt) - 1)]
            rings = ramp(later, 0.05, 0.3) * ramp(-later / (bt[i] + 1e-6), -0.95, -0.4)
            out[k, 2] = on.max() * (0.4 + 0.6 * rings)

        kn = F["knock"][s]
        peaks = np.flatnonzero((kn > 0.3) & (kn >= maximum_filter1d(kn, size=9)))
        if len(peaks) >= 2:
            ioi = np.diff(peaks) / FPS
            ok = ((ioi > 0.08) & (ioi < 0.7)).mean()
            reg = ramp(-np.std(ioi) / (np.mean(ioi) + 1e-6), -0.6, -0.1)
            out[k, 3] = min(len(peaks), 4) / 4 * ok * reg * kn[peaks].mean()

        v, f0 = F["cry"][s], F["cry_f0"][s]
        run = uniform_filter1d((v > 0.4).astype(float), size=40).max()       # >= 0.4 s voiced
        wobble = ramp(_freq_std(f0, v), 5, 25) if np.isfinite(_freq_std(f0, v)) else 0.0
        out[k, 4] = v.mean() * ramp(run, 0.4, 0.9) * wobble

        h, hf, hdb = F["horn"][s], F["horn_f"][s], F["horn_db"][s]
        act = np.flatnonzero(h > 0.3)
        dur = len(act) / len(h)
        if len(act) > 5:
            post = act[act >= act[np.argmax(hdb[act])]]                          # after the level peak
            slope = np.polyfit(post / FPS, hdb[post], 1)[0] if len(post) > 5 else -8.0   # dB per second
            flat = ramp(slope, -12, -4)                                          # a horn does not decay
            out[k, 5] = np.percentile(h, 80) * ramp(dur, 0.05, 0.25) * ramp(-_freq_std(hf, h), -30, -8) * flat

        out[k, 6] = F["loud"][s].max()
    return out


def detector(batch, pool=None):
    return pool.map(window_scores, batch) if pool is not None else [window_scores(x) for x in batch]
