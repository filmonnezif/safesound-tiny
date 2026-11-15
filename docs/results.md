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

## Phase 2: DSP simulation ablation (DS-CNN M, 45k params, hard labels, 2 runs each)

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

## Phase 4: distilled int8 students

(running)
