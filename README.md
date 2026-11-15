# SafeSound Tiny (working name)

A small, DSP-grounded detector of the sounds that matter most to deaf and
hard-of-hearing people, built to run on a ~$5 microcontroller (ESP32-S3).

**Goal:** the highest recall of safety sounds at a strict false-alarm budget
(false alarms per hour in real homes), with the smallest model that gets there.

## Target sounds

| Class | Why it matters |
|---|---|
| Smoke / fire alarm | life safety |
| Siren | approaching emergency vehicle |
| Doorbell | visitor, delivery |
| Knock | visitor |
| Baby crying | care |
| Car horn | street safety |

## Approach

1. **DSP-based simulation of real conditions.** Clean events are mixed into real
   home backgrounds, convolved with measured room impulse responses, swept over
   distance and SNR, filtered as if heard through a door, and passed through a
   model of the INMP441 MEMS mic (frequency response, automatic gain, clipping).
2. **Teacher.** A large AudioSet-pretrained model fine-tuned on the simulated data.
3. **Student.** Tiny CNNs (30k–300k parameters) distilled from the teacher and
   trained with quantization in the loop (int8). Log-mel vs PCEN front end.
4. **Classical DSP paths.** A Goertzel + ISO 8201 (T3) pattern detector for smoke
   alarms, and an energy / spectral-flux gate that wakes the network only when needed.
5. **ESP32-S3.** Front end + model on TensorFlow Lite Micro with ESP-NN.

## Evaluation (see [docs/protocol.md](docs/protocol.md))

Event-based recall at 0.1 and 1 false alarm per hour per class, on a **real, public,
locked** test set: real events (AudioSet strong-label eval, FSD50K eval) and hours
of real home audio without the target sounds (SINS, CHiME-Home, DCASE home scenes).

## Plan

| Phase | Work | Output |
|---|---|---|
| 0 | Dataset inventory, licences, locked test set + tuning set | protocol, event counts |
| 1 | Baselines: YAMNet off the shelf; teacher on raw clips | the number to beat |
| 2 | DSP simulation pipeline | ablation: what each part adds |
| 3 | Teacher on simulated data | upper bound |
| 4 | Distilled int8 students, size sweep, log-mel vs PCEN | recall vs size curve |
| 5 | Alarm pattern detector, wake gate, fusion | recall / FA / duty-cycle gains |
| 6 | ESP32-S3 port (QEMU / host build first) | memory, flash, latency, power |
| 7 | Paper | draft |

## Status (2026-11-15)

Done: phases 0-5, int8 TFLite export, paper draft ([paper/main.tex](paper/main.tex)); results in
[docs/results.md](docs/results.md). Remaining: phase 6 on hardware (ESP32-S3 latency, memory, power).

- Teacher (CNN14 fine-tuned): 35.8% / 66.8% event recall at 0.1 / 1 false alarm per hour.
- Best tiny model: distilled DS-CNN XL, 239 KB int8 (318 KB TFLite): 28.6% / 59.5%; beats YAMNet
  on every clip-level metric.
- Recall first (95% alarm/siren, 90% others): 90-97% recall per class; DSP fusion cuts home false
  alarms 24-36% at equal recall; with the loud-sound safety net the XL model reaches 95.5% recall
  at 16 false alarms per hour of real home audio.
- Energy: a DSP wake gate runs the classifier only when physics-based evidence fires; in the real
  home the classifier sleeps 63% of the time (~60% compute saved) for 0.7 points of recall.
- Independent review and what was done about it: [docs/review.md](docs/review.md).

## Layout

```
safesound/   python package: data, simulation, front ends, models, evaluation
notebooks/   Colab notebooks (all heavy compute runs on Colab)
firmware/    ESP-IDF project for the ESP32-S3
docs/        protocol, notes
results/     small result tables / figures (no audio, no checkpoints)
```

Data and checkpoints are not in git; they live on Colab / Google Drive.
