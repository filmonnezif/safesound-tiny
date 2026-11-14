"""Generate the Colab notebooks (python notebooks/make.py). Each notebook is thin: it
mounts Drive, gets the repo, and calls a module; all logic lives in safesound/."""
import json
from pathlib import Path

ROOT = "/content/drive/MyDrive/safesound-tiny"
SETUP = f"""from google.colab import drive
drive.mount('/content/drive')
%cd /content
!test -d safesound-tiny || git clone -q https://github.com/filmonnezif/safesound-tiny
%cd /content/safesound-tiny
!git pull -q
!pip -q install -e . {{EXTRA}}
!mkdir -p {ROOT}
import os; print(os.cpu_count(), 'CPUs'); !nvidia-smi -L; !df -h /content /content/drive | tail -2"""

NOTEBOOKS = {
    "00_data": ("Phase 0b: build the clip stores on Drive (~1-2 h, mostly downloads). "
                "CPU-heavy: use a runtime with many cores.", "", [
        "!apt-get -qq install -y aria2 zip unzip > /dev/null",
        f"!python -m safesound.data --root {ROOT} --work /content/work 2>&1 | grep -v Warning",
    ]),
    "01_baselines": ("Phase 1: lock the test set, run YAMNet and pretrained CNN14 zero-shot.",
                     "tensorflow-hub torchlibrosa", [
        f"!python -m safesound.phase1 --root {ROOT}",
    ]),
    "02_simulation": ("Phase 2: DSP simulation ablation (medium student trained per simulation setting).",
                      "pyroomacoustics", [
        f"!python -m safesound.phase2 --root {ROOT}",
    ]),
    "03_teacher": ("Phase 3: fine-tune the CNN14 teacher on simulated data.", "torchlibrosa pyroomacoustics", [
        f"!python -m safesound.phase3 --root {ROOT}",
    ]),
    "04_students": ("Phase 4: distilled int8 students, size sweep, log-mel vs PCEN.",
                    "pyroomacoustics", [
        f"!python -m safesound.phase4 --root {ROOT}",
    ]),
}


def cell(src, kind="code"):
    c = {"cell_type": kind, "metadata": {}, "source": src.splitlines(keepends=True)}
    if kind == "code":
        c.update(outputs=[], execution_count=None)
    return c


for name, (title, extra, cells) in NOTEBOOKS.items():
    nb = {"nbformat": 4, "nbformat_minor": 0,
          "metadata": {"colab": {"provenance": []}, "kernelspec": {"name": "python3", "display_name": "Python 3"},
                       "accelerator": "GPU"},
          "cells": [cell(f"# {name}\n{title}", "markdown"), cell(SETUP.replace("{EXTRA}", extra))]
          + [cell(c) for c in cells]}
    Path(__file__).with_name(f"{name}.ipynb").write_text(json.dumps(nb, indent=1))
    print("wrote", name)
