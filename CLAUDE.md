# SafeSound Tiny

Research project (DSP course, Dr. Imdad; aimed at a paper, venue undecided): highest
recall of safety sounds for deaf and hard-of-hearing people at a strict false-alarm
budget, with a tiny int8 model on an ESP32-S3. A fresh start: do not reuse or cite the
earlier WristEcho / wristecho-dsp-frontend experiments.

Constraints from the user:
- Public datasets only; the user cannot record their own audio.
- Heavy compute on Google Colab only (the laptop is weak). Local code stays light.
- Classical DSP must play a visible, credible role.
- No board on hand yet: deployment numbers come from QEMU / host builds and are
  labelled "estimated" unless measured on hardware.
- Follow [docs/protocol.md](docs/protocol.md): the real test set is locked; tune on
  the tuning set only.
- Author block stays "Anonymous" until the user gives names.
