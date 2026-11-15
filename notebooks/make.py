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
import os; print(os.cpu_count(), 'CPUs')
!nvidia-smi -L
!df -h /content /content/drive | tail -2"""

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
    "02_04_train": ("Phases 2-4 as one background job (survives notebook disconnects); log on Drive. "
                    "Re-run the last cell to see progress.", "torchlibrosa", [
        f"!nohup sh -c 'python -u -m safesound.phases 2 --root {ROOT} && python -u -m safesound.phases 3 --root {ROOT}"
        f" && python -u -m safesound.phases 4 --root {ROOT}' > {ROOT}/run.log 2>&1 &",
        f"!sleep 5; grep -v -i warn {ROOT}/run.log | tail -n 40",
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
