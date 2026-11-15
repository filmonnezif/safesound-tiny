# Results (live; numbers copied from `python -m safesound.report`)

Test set locked 2026-11-14, sha256 `752f6bb5…` (910 real event clips; 21.8 h AudioSet
negatives; 15.6 h SINS home negatives). Macro recall = mean over the six classes of
event recall at the threshold set on the tuning set for that false-alarm budget.

## Phase 1: zero-shot baselines and the raw-data teacher

| Model | Params | Test @0.1 FA/h | Test @1 FA/h |
|---|---|---|---|
| YAMNet, zero-shot | 3.7 M | 13.9% | 55.5% |
| CNN14-16k, zero-shot | 80 M | 16.6% | 66.1% |
| CNN14-16k fine-tuned on raw clips (teacher A) | 80 M | 35.8% | 66.8% |

Per class at 1 FA/h (zero-shot CNN14 -> fine-tuned): alarm 89 -> 83, siren 81 -> 76,
doorbell 48 -> 59, knock 66 -> 75, baby cry 59 -> 47, car horn 53 -> 61.
Fine-tuning mostly helps at the strict budget (0.1 FA/h: 16.6% -> 35.8%).

## Phase 2: DSP simulation ablation (DS-CNN M, 21k params, hard labels, 2 runs each)

| Config | Test @0.1 | Test @1 | Tuning @1 | runs (test @1) |
|---|---|---|---|---|
| A raw clips | 22.3% | 45.5% | 43.3% | 44.8, 46.2 |
| B + home background mixing, random level | 22.1% | 49.9% | 46.8% | 49.3, 50.5 |
| C + statistical room reverberation | 16.9% | 49.5% | 47.3% | 50.6, 48.3 |
| D + occlusion (door / wall low-pass) | 22.1% | 49.3% | **47.3%** | 50.0, 48.6 |
| E + INMP441 mic model | 23.7% | 49.1% | 47.1% | 48.1, 50.0 |

Chosen on tuning recall at 1 FA/h: **D** (C, D, E within 0.2 points of each other).

Findings:
- Mixing real home backgrounds is the main gain (+4.4 test, +3.5 tuning at 1 FA/h), in
  both runs.
- Room, occlusion and mic simulation add nothing measurable on top on this test set, and
  reverberation alone lowers strict-budget recall. The test events are mostly close-mic
  YouTube / Freesound recordings, so far-field realism has little to show here; a far-field
  test set would be needed to measure it (limitation).
- False alarms in the real home (SINS) stay near zero (<= 0.15/h) for every model; the
  budget is spent on the varied AudioSet negatives, whose test FA/h is about twice the
  tuning target (distribution shift between AudioSet train and eval negatives).
- At 0.1 FA/h the threshold rests on about 3 tuning false alarms, so single runs vary by
  several points; report means over runs.

## Phase 3: teachers

| Teacher | Test @0.1 | Test @1 | Tuning @0.1 | Tuning @1 |
|---|---|---|---|---|
| CNN14 fine-tuned, raw clips (A) | **35.8%** | **66.8%** | **35.4%** | **65.8%** |
| CNN14 fine-tuned, simulation D | 29.5% | 65.4% | 30.7% | 62.5% |

Simulation makes the large teacher worse on both tuning and test, consistent with the
ablation: the evaluation audio is mostly close-mic, so far-field simulation moves the model
away from it. The phase 4 teacher is chosen on tuning recall like everything else, so the
students are distilled from the raw-data teacher (they still train on simulation D inputs;
the teacher labels the same simulated example).

## Phase 4: distilled students (simulation D inputs, teacher = raw-data CNN14)

| Student | Params | MACs / inference | int8 weights | Test @0.1 | Test @1 | int8 @0.1 | int8 @1 |
|---|---|---|---|---|---|---|---|
| S (DS-CNN w32x4) | 6.3 k | 2.6 M | 6.7 KB | 10.3% | 45.1% | 11.1% | 42.7% |
| M (w64x4) | 20.8 k | 8.8 M | 21.5 KB | 28.1% | 52.5% | 22.7% | 53.0% |
| L (w128x5) | 92.4 k | 36.2 M | 93.0 KB | 28.2% | 57.8% | 27.8% | 57.1% |
| **XL (w192x6)** | 239 k | 88.6 M | 239 KB | **32.4%** | **59.3%** | 28.6% | **59.5%** |
| M, PCEN front end | 20.8 k | 8.8 M | 21.5 KB | 18.2% | 50.1% | 12.6% | 49.2% |
| reference: M hard labels (phase 2 D, mean) | 20.8 k | 8.8 M | | 22.1% | 49.3% | | |
| reference: teacher CNN14 (raw) | 80 M | ~ 2 G | | 35.8% | 66.8% | | |

One inference = 2 s of audio; the device runs two per second.

Findings:
- Distillation helps the same network clearly: M 22.1% -> 28.1% at 0.1 FA/h and
  49.3% -> 52.5% at 1 FA/h (tuning set agrees: 21.7-24.6% -> 28.6%, 46.7-47.9% -> 49.0%).
- Recall grows with size; XL (239 KB int8) keeps 90% of the teacher's strict-budget recall
  and 89% at 1 FA/h with 1/330 of its parameters, and beats YAMNet zero-shot at both budgets.
- int8 (simulated TFLite scheme) is lossless at 1 FA/h for L and XL; at 0.1 FA/h it costs
  0-6 points (M, XL). Quantisation-aware training is the next step.
- PCEN is worse than log-mel here (M: 18.2% vs 28.1% at 0.1 FA/h). The evaluation audio has
  little slow loudness variation for PCEN to remove; it also quantises worse.
- ESP32-S3 budget: L (36 M MACs, 93 KB) and XL (89 M MACs, 239 KB) both fit the planned
  memory (XL needs PSRAM or careful arena use); latency to be measured or estimated in phase 6.

## Clip-level view (the usual literature metrics)

`python -m safesound.cliplevel`: clip score = max over its windows; per class, positives =
test clips with that class, negatives = every other test clip (99% of clips). Threshold for
precision / recall / F1 / accuracy = best F1 on the tuning clips, applied to test.

| Model | Size | AUC | AP | Precision | Recall | F1 | Accuracy |
|---|---|---|---|---|---|---|---|
| YAMNet, zero-shot | 3.7 M params | 0.948 | 0.635 | 64.0% | 60.0% | 61.0% | 99.20% |
| CNN14, zero-shot | 80 M | 0.974 | 0.717 | 74.8% | 66.1% | 69.3% | 99.40% |
| CNN14 fine-tuned (teacher) | 80 M | 0.978 | 0.731 | 74.3% | 66.6% | 68.8% | 99.42% |
| Student S int8 | 6.7 KB | 0.950 | 0.490 | 51.7% | 52.4% | 51.5% | 98.93% |
| Student M int8 | 21.5 KB | 0.962 | 0.592 | 61.5% | 55.5% | 57.5% | 99.06% |
| Student L int8 | 93 KB | 0.965 | 0.650 | 68.9% | 58.3% | 62.4% | 99.20% |
| Student XL int8 | 239 KB | 0.965 | 0.661 | 68.7% | 60.8% | 63.8% | 99.27% |

- Accuracy is ~99% for every model because 99% of clips are negatives (a model that never
  fires scores ~98.5%); it says nothing about quality for rare sounds.
- The int8 XL student beats YAMNet (3.7 M params) on every clip-level metric.
- Fine-tuning barely moves clip-level AP (0.717 -> 0.731) but doubles event recall at
  0.1 FA/h (16.6% -> 35.8%): clip-level metrics hide the strict-false-alarm regime that matters
  for a safety device. This motivates the event-level FA/h protocol.

## Phase 5: recall first, DSP detectors and fusion (`python -m safesound.phase5`)

Protocol: target recall per class (95% alarm and siren, 90% others). On the tuning set, per class,
the method and the highest threshold reaching the target are chosen; the test set reports the
recall actually reached and the false alarms per hour it costs. In this protocol an event counts
as detected when the decision is positive during it, even if the buzz was suppressed by the 10 s
refractory period after an earlier alert of the same class. Methods: network; smoothed network
(mean of the last two windows); network OR calibrated DSP score; logistic fusion of network and
DSP (fitted on tuning windows). Safety net: a class-agnostic loud-onset alert on top.

Macro over classes (test):

| Model | Method | Recall | FA/h AudioSet | FA/h home |
|---|---|---|---|---|
| Teacher CNN14 | network | 91.7% | 48.6 | 9.1 |
| Teacher CNN14 | per-class choice | 91.5% | 45.6 | 8.2 |
| XL int8 (239 KB) | network | 91.3% | 105.9 | 18.0 |
| XL int8 | per-class choice | 91.2% | 95.7 | 13.7 |
| L int8 (93 KB) | network | 92.4% | 116.4 | 18.8 |
| L int8 | per-class choice | 91.5% | 96.3 | 12.0 |
| M int8 (21.5 KB) | network | 90.9% | 127.5 | 14.0 |
| M int8 | per-class choice | 90.2% | 113.5 | 10.5 |
| XL int8 | per-class choice + safety net | 95.5% | 205.3 | 16.3 |

XL int8 per class (test recall, FA/h AudioSet / home):

| Class | DSP alone | Network | Network + DSP (logistic) | + safety net |
|---|---|---|---|---|
| alarm | 100%, 365 / 360 | 95%, 84 / 16.8 | **97%, 77 / 15.2** (chosen) | 98%, 186 / 17.7 |
| siren | 95%, 332 / 114 | 90%, 98 / 4.9 | 90%, **75 / 2.2** (chosen) | 94%, 202 / 5.1 |
| doorbell | 89%, 365 / 360 | 82%, 31 / 1.0 | 82%, 28 / 0.9 (chosen) | 92%, 162 / 3.6 |
| knock | 100%, 365 / 360 | 91%, 62 / 36.2 | 90%, **34 / 14.0** (chosen) | 94%, 160 / 16.1 |
| baby cry | 100%, 365 / 360 | 92%, 91 / 9.6 (chosen) | 92%, 98 / 10.2 | 97%, 197 / 12.3 |
| car horn | 100%, 365 / 360 | 96%, 271 / 39.7 | 96%, 270 / 39.4 | 98%, 326 / 42.7 |

(nn_smooth was chosen for car horn; 365 FA/h = an alert every refractory period, i.e. always on.)

Findings:
- About 90-97% recall per class is reachable with every model; the price is false alarms,
  mostly on the varied YouTube negatives (TV, music, other alarms), much less in the real home.
- The hand-designed DSP detectors alone cannot reach high recall without firing constantly.
- Fused with the network (logistic fusion chosen on tuning), they cut false alarms at the same
  recall where the sound has a clear physical signature: siren home FA 4.9 -> 2.2/h, knock
  36 -> 14/h, alarm 84 -> 77/h; macro home FA for the deployable students drops 24-36% (XL 18.0 -> 13.7, L 18.8 -> 12.0, M 14.0 -> 10.5/h).
- The loud-sound safety net lifts every class to 92-98% recall (macro 95.5%) at about twice the
  false-alarm rate on varied audio but only +2.6/h in the home.
- Car horn is the costly class at high recall (it is mostly outdoor, and TV/traffic negatives
  are full of horns).

## Export (`python -m safesound.export`)

| Student | TFLite int8 file | TFLite vs float: mean prob. error | vs simulated int8 | decision agreement |
|---|---|---|---|---|
| XL | 317,568 B | 0.0040 | 0.0045 | 99.57% |
| L | 141,144 B | 0.0049 | 0.0049 | 99.61% |
| M | 46,392 B | 0.0070 | 0.0065 | 99.48% |

On 2,929 windows of real test clips. Ops: CONV_2D, DEPTHWISE_CONV_2D, PAD, MEAN, REDUCE_MAX,
ADD, SUB, MUL, FULLY_CONNECTED, all supported by TFLite Micro. The real int8 models agree with the
simulated int8 used in phases 4-5, so those numbers stand. Files (and C arrays for ESP-IDF) are
on Drive in `results/export/`. On-device latency and power: phase 6 (hardware).
