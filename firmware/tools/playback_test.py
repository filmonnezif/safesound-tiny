"""Play test clips through the PC speakers and record what the ESP32 detector says.

    python firmware/tools/playback_test.py [clip.wav ...]     (default: firmware/assets/clips/*.wav)

For each clip: plays at most MAX_S seconds, keeps listening 2.5 s more, then reports the
highest probability per class and every alert the board printed. The expected class is the
file-name prefix (siren_..., knock_...; speech/dishes/etc. = no target).
"""
import glob
import re
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

import serial

CLASSES = ["alarm", "siren", "doorbell", "knock", "baby_cry", "car_horn"]
MAX_S = 10
ROOT = Path(__file__).resolve().parents[2]

clips = sys.argv[1:] or sorted(glob.glob(str(ROOT / "firmware/assets/clips/*.wav")))
port = glob.glob("/dev/ttyACM*")[0]
ser = serial.Serial(port, 921600, timeout=0.2)
lines, lock = [], threading.Lock()


def reader():
    buf = b""
    while True:
        buf += ser.read(4096)
        while b"\n" in buf:
            ln, buf = buf.split(b"\n", 1)
            with lock:
                lines.append((time.time(), ln.decode(errors="ignore").strip()))


threading.Thread(target=reader, daemon=True).start()
time.sleep(3)
rows = []
for c in clips:
    name = Path(c).stem
    expect = next((k for k in CLASSES if name.startswith(k)), None)
    with tempfile.NamedTemporaryFile(suffix=".wav") as tmp:
        subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", c, "-t", str(MAX_S), tmp.name], check=True)
        t0 = time.time()
        subprocess.run(["pw-play", tmp.name], check=True)
        time.sleep(2.5)
        t1 = time.time()
    with lock:
        seg = [ln for t, ln in lines if t0 + 0.5 <= t <= t1]
    best = {k: 0.0 for k in CLASSES}
    alerts = []
    for ln in seg:
        for k in CLASSES:
            m = re.search(rf"{k} ([0-9.]+)", ln)
            if m:
                best[k] = max(best[k], float(m.group(1)))
        m = re.search(r"ALERT (\S+)", ln)
        if m:
            alerts += m.group(1).split(",")
    fired = sorted(set(alerts))
    ok = (expect in fired) if expect else not fired
    rows.append((name, expect, fired, ok))
    top = max(best, key=best.get)
    print(f"{name:22s} expect {str(expect):9s} alerts {','.join(fired) or '-':18s} "
          f"top {top} {best[top]:.2f}  {'OK' if ok else 'MISS' if expect else 'FALSE ALARM'}", flush=True)
    time.sleep(1.5)
n_ok = sum(r[3] for r in rows)
print(f"\n{n_ok}/{len(rows)} correct")
