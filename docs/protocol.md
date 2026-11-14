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

## Phase 0 inventory (2026-11-14)

Real labelled clips per class (`python -m safesound.inventory`). One clip = one
10 s AudioSet segment from a distinct YouTube video; "events" are individual labelled
sounds (an alarm's separate beeps count separately).

| Class | AudioSet strong eval clips (events) | FSD50K eval clips | AudioSet strong train clips |
|---|---|---|---|
| alarm (smoke / fire) | 63 (215) | none (only the mixed "Alarm" parent) | 419 |
| siren | 228 (325) | 55 | 1796 |
| doorbell | 61 (111) | 37 | 330 |
| knock | 33 (109) | 103 | 133 |
| baby_cry | 64 (201) | none (only "Crying and sobbing", any age) | 492 |
| car_horn | 216 (456) | 68 | 1550 |

AudioSet strong eval has 16,347 clips (~45 h) with none of the six; strong labels
cover 456 classes, so "absent" is reliable. Audio: Hugging Face `agkphysics/AudioSet`
eval split (23.6 GB parquet, CC-BY-4.0 copy downloaded March 2023, 17,141 of 20,371
eval clips still available); the exact coverage of our test clips is counted on Colab.

Real home audio for false alarms: SINS / DCASE 2018 Task 5 development set
(one home, 4 mics, ~200 h of 10 s segments, 45 GB, Zenodo 1247102).

## Test set (locked once the coverage counts are in)

| Source | Use | Content |
|---|---|---|
| AudioSet strong eval | real events, all six classes | clips above that have audio |
| FSD50K eval | real events: siren, doorbell, knock, car_horn | clips above |
| AudioSet strong eval, none of the six | real negatives, varied | ~45 h before coverage loss |
| SINS / DCASE18-T5, held-out days | real negatives, home | days chosen before training |

## Tuning set (never test)

| Source | Use |
|---|---|
| FSD50K dev, `val` split | real events (siren, doorbell, knock, car_horn) |
| AudioSet strong train, 20% of videos held out from all training | real events, all six (only source for alarm and baby_cry) |
| SINS / DCASE18-T5, other days | real home negatives |
| AudioSet strong train held-out clips with none of the six | varied negatives |

## Known caveats

- **Knock** has only 33 AudioSet test clips; FSD50K eval adds 103 (close-mic Freesound
  recordings). Expect wide confidence intervals for knock and doorbell.
- **One home.** SINS is a single home, so false-alarm rates are reported for it and,
  separately, for the varied AudioSet negatives.
- **Unlabelled targets in negatives.** SINS was labelled for activities, not for these
  sounds; a doorbell or alarm may hide in "other". The top-scoring false alarms are
  audited and any true target sound is reported and removed, with counts.
- **Tuning on AudioSet train.** AudioSet-pretrained models saw these clips' weak labels
  in training, so their scores there may be over-confident. Where possible thresholds
  are checked on FSD50K dev `val`, which no AudioSet model has seen.
