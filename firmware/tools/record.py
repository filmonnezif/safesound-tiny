"""Record 5 s from the ESP32 mic test firmware and save a WAV.

    python firmware/tools/record.py /dev/ttyUSB0 rec.wav      (Linux; Windows: COM5)

Then: play rec.wav, or in MATLAB: safesound_demo('rec.wav').
Needs: pip install pyserial numpy
"""
import sys
import time
import wave

import numpy as np
import serial

port = sys.argv[1] if len(sys.argv) > 1 else "/dev/ttyUSB0"
out = sys.argv[2] if len(sys.argv) > 2 else "rec.wav"

with serial.Serial(port, 921600, timeout=10) as s:
    time.sleep(0.5)
    s.reset_input_buffer()
    s.write(b"r")
    while True:                                   # skip level lines until the header
        line = s.readline().decode(errors="ignore").strip()
        if not line and not s.in_waiting:
            sys.exit("no answer from the board (wrong port, or firmware not running?)")
        if line.startswith("REC"):
            _, sr, n = line.split()
            sr, n = int(sr), int(n)
            break
    print(f"recording {n / sr:.1f} s ...")
    data = s.read(2 * n)
    if len(data) < 2 * n:
        sys.exit(f"only got {len(data) // 2} of {n} samples")

x = np.frombuffer(data, "<i2")
with wave.open(out, "wb") as w:
    w.setnchannels(1)
    w.setsampwidth(2)
    w.setframerate(sr)
    w.writeframes(x.tobytes())
rms = np.sqrt(np.mean((x / 32768.0) ** 2))
print(f"saved {out}: {len(x) / sr:.1f} s, level {20 * np.log10(rms + 1e-12):.1f} dBFS, "
      f"peak {np.abs(x).max() / 32768:.2f} of full scale")
if np.abs(x).max() == 0:
    print("WARNING: all zeros - the mic is not sending data (check wiring)")
