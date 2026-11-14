# Methods (working notes for the paper)

All numbers here are design choices made before seeing test results.

## Data

| Role | Sources |
|---|---|
| Training events | AudioSet strong-label train (80% of videos), AudioSet balanced train (weak labels, same video split), FSD50K dev `train` |
| Training negatives | AudioSet strong train clips with none of the six (all hard negatives + 20% of the rest) |
| Training backgrounds | SINS home recordings, 50% of the time (sessions), one mic node per session, 60% of segments |
| Tuning (thresholds, model choice) | held-out 20% of AudioSet strong-train videos; FSD50K dev `val`; SINS 20% of sessions |
| Test (locked) | AudioSet strong eval; FSD50K eval; SINS 30% of sessions |

Audio mirrors: `CLAPv2/audioset_strong` (strong train + eval, 48 kHz WAV), `agkphysics/AudioSet`
(balanced train), Zenodo (FSD50K 4060432, DCASE 2018 task 5 / SINS 1247102). Everything is
resampled to 16 kHz mono (`resample_poly`).

Clips with labels that may hide a target ("Alarm", "Emergency vehicle", "Crying, sobbing",
"Whimper") but none of the six are excluded from every negative set.

## Detection protocol

Every detector outputs a score per class every 0.5 s for the 2 s of audio that just ended
(causal; YAMNet uses its own 0.96 s / 0.48 s framing). An alert fires when a score crosses
the threshold, then that class is silent for 10 s. An occurrence (events of a class merged
across gaps up to 5 s) is detected if an alert falls between its onset and
max(offset, onset + 2 s). Thresholds are set per class on the tuning negatives for 0.1 and
1 false alarm per hour, then frozen and applied to the test set.

## DSP simulation (training only)

Examples are 3 s (1 s warm-up for the front end + the 2 s the label is about), generated on
the fly; 50% positives with classes drawn uniformly. Strong labels place the label window on
a random part of an event; weak labels on a loud part of the clip (energy-weighted).
Steps (cumulative configs A-E):

- **B mix:** with p = 0.9 add a SINS background at SNR ~ U(-5, 25) dB (SNR uses the RMS of
  the loudest 20% of 50 ms frames of the foreground); random level U(-45, -12) dBFS.
- **C rir:** with p = 0.7 convolve the foreground with a statistical room impulse response:
  direct path; 6-12 early reflections with image-source amplitude d / (d + extra path),
  2-50 ms; diffuse tail of Gaussian noise decaying as exp(-6.9 t / RT60) in four bands
  (RT60 x 1.2, 1.0, 0.8, 0.55 from low to high); direct-to-reverberant ratio
  (r_c / d)^2 with critical distance r_c = 0.057 sqrt(V / RT60). RT60 ~ U(0.2, 1.0) s,
  V ~ U(25, 200) m^3, d ~ U(0.5, 8) m.
- **D occl:** with p = 0.25, first- or second-order Butterworth low-pass at U(300, 1500) Hz
  (mass-law transmission through a door or wall, 6 or 12 dB/octave).
- **E mic:** INMP441-like capture: 2nd-order high-pass at U(50, 200) Hz (port / DC block),
  RBJ high-shelf at 3 kHz of U(-4, 4) dB (unit-to-unit response spread), white self-noise at
  U(-80, -65) dBFS, 3% of examples boosted 15 dB into clipping, 16-bit quantisation.

Positives and negatives go through the same steps.

## Models

- **Baselines:** YAMNet and CNN14-16k (PANNs), zero-shot: class score = max over the
  class's AudioSet labels.
- **Teacher:** CNN14-16k fine-tuned end to end on 2 s windows with a new 6-way head
  initialised from the mean of the class's AudioSet head rows; SpecAugment; AdamW, one-cycle,
  lr 1e-4 (head 1e-3), batch 64, bf16.
- **Students:** DS-CNN on 40 mel bands x 100 frames (n_fft 512, hop 20 ms, 50-8000 Hz):
  conv 3x3/2, N depthwise-separable blocks (second one strided), frequency mean, time max +
  mean, linear. Sizes S/M/L/XL = width 32/64/128/192, 4/4/5/6 blocks. Hard-label BCE, or
  distillation: 0.5 BCE(labels) + 0.5 BCE(teacher probabilities), teacher run on the same
  augmented example.
- **Front ends:** log(mel + 1) vs PCEN (T = 0.4 s, alpha 0.98, delta 2, r 0.5).
  Fixed per-band normalisation (training-set mean / std), folded into the front end.
- **int8:** batch norm folded; weights symmetric per channel; activations asymmetric per
  tensor with min/max calibrated on 4096 simulated examples (TFLite int8 scheme, simulated).
