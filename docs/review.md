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
| 1 | No energy argument for an always-listening wearable: the classifier runs 2x per second forever (XL: ~177 M MAC/s). | **Addressed:** DSP wake gates (phase 5b) that run the classifier only when something happens. |
| 2 | Recall targets missed on test (doorbell 82% vs 90%, siren 90% vs 95%): tuning-to-test shift not compensated. | **Measured:** threshold margins (+2, +4 points on tuning) recover the targets but the last points of recall are very expensive (e.g. siren 90% -> 99% costs home FA 2 -> 99/h). Reported as a trade-off, not hidden. |
| 3 | No confidence intervals for phase 5, no paired test for "beats YAMNet". | **Addressed:** class-stratified bootstrap CIs for macro recall; paired bootstrap XL int8 vs YAMNet at 1 FA/h. |
| 4 | Baselines are 2019-2020 models (YAMNet, CNN14); EfficientAT / BEATs missing. | **Open** (GPU, ~2-3 h): run them zero-shot and fine-tuned under the same protocol. |
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
