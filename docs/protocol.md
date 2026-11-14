# Evaluation protocol

Written before any training. Changes must be recorded here with a date and reason.

## Rules

1. **Public data only.** No self-recorded audio.
2. **The test set is locked before training** and never used to choose models,
   thresholds, augmentations or hyperparameters. All choices use the tuning set.
3. **Test audio is real.** Simulated audio is for training and tuning only.
4. **No leakage.** No test clip (or another clip from the same source recording /
   Freesound uploader / YouTube video) appears in training. Pretrained teachers must
   not have trained on the test clips (the AudioSet eval split is held out from
   standard AudioSet training).

## Metrics

- **Event-based recall** per class: an event counts as detected if an alert of that
  class fires during the event or within 2 s after its onset.
- **False alarms per hour** per class, measured on hours of real home audio that
  contain none of the target sounds.
- Headline: recall at **0.1 FA/h** and **1 FA/h** per class; thresholds set on the
  tuning set, then frozen.
- 95% bootstrap confidence intervals (resampling events / recordings), since some
  classes will have few real test events.
- Detection latency (onset to alert).
- Deployment: parameters, int8 model size, tensor arena (RAM), flash, compute per
  second of audio, and latency / power (measured on a board when available,
  otherwise marked as estimated).

## Test set (to be filled in Phase 0)

| Source | Use | Classes / content | Hours / events |
|---|---|---|---|
| AudioSet strong labels, eval split | real events | all six | TBD |
| FSD50K eval | real events | all six | TBD |
| SINS | real negatives | home, continuous | TBD |
| CHiME-Home | real negatives | home | TBD |
| DCASE home scenes | real negatives | home | TBD |
