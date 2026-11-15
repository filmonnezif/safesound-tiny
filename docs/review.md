# Independent assessment (2026-11-15)

Written as a reviewer would, after phases 0-5. Each weakness lists what was done about it.

## Verdict

A solid, honest workshop-level study with an unusually careful protocol. As a full conference
paper it needs: an energy story (now added), statistical support for the headline claims (now
added), stronger baselines, and a far-field test set.

## Strengths

1. **Protocol.** Test set hashed and locked before training; thresholds only from separate tuning
   data; false alarms per hour on 37 h of real audio (one real home + varied YouTube audio).
   Event-level and clip-level views both reported.
2. **Reproducible.** Public data and code, every Colab step resumable from Drive.
3. **Honest negative results.** Training-time simulation does not help on this test set and makes
   the large teacher worse; reported with an explanation and a limitation.
4. **Deployment checked, not assumed.** Real int8 TFLite models agree with the int8 simulation
   (99.5-99.6% decisions) and use only TFLite Micro ops.

## Major weaknesses

| # | Weakness | Status |
|---|---|---|
| 1 | No energy argument for an always-listening wearable: the classifier runs 2x per second forever (XL: ~177 M MAC/s). | **Addressed:** DSP wake gate (phase 5b): classifier asleep 63% of the time in the real home, ~60% compute saved, recall 91.2% -> 90.5% (within CI), home FA -25%. A pure energy gate loses 12 points of recall and is rejected. |
| 2 | Recall targets missed on test (doorbell 82% vs 90%, siren 90% vs 95%): tuning-to-test shift not compensated. | **Addressed:** margins of +4 points meet 5/6 targets (95.0% macro) but cost 101 FA/h in the home; the loud-sound safety net reaches 95.5% at 16.3 FA/h, so it is the recommended way to buy recall. |
| 3 | No confidence intervals for phase 5, no paired test for "beats YAMNet". | **Addressed:** bootstrap CIs (XL recall-first 91.2% [88.9, 93.2]); paired bootstrap XL int8 vs YAMNet at 1 FA/h: +4.0 points [0.4, 7.6], p = 0.014. |
| 4 | Baselines are 2019-2020 models (YAMNet, CNN14); EfficientAT / BEATs missing. | **Partly addressed (phase 6):** EfficientAT MobileNetV3 used as pretrained students; the 716 k-parameter one is the best small model (32.3 / 62.5 int8). Stronger teachers (BEATs, strong-label pretraining) remain open. |
| 5 | Simulation contribution is a null result on close-mic test audio. | **Open:** needs a far-field test set (e.g. re-recorded or public far-field data); stated as a limitation. |

## Minor weaknesses

- Two runs per ablation configuration (three planned); differences at 0.1 FA/h are within noise.
- Real negatives from one home (SINS); the home false-alarm rate may not generalise.
- AudioSet test clips often start mid-event, which is pessimistic for onset-based logic
  (affects the energy gate; see phase 5b).
- Car horn is costly at high recall (~270 FA/h on varied audio); a per-class policy (e.g. lower
  priority indoors) is a product decision, not tested here.
- Window-level classifier with fixed 2 s context; longer context could help sirens and cries.
- No on-device latency / power yet (hardware step, deliberately left for later).
