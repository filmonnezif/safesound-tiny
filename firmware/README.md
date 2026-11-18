# Firmware

Step 1 (now): `mic_test/` checks that the board and the INMP441 microphone work.
Next: the detector (mel features + DSP detectors + wake gate + int8 network).

## Wiring (INMP441 -> ESP32-S3 CAM board)

| INMP441 pin | ESP32-S3 | classic ESP32 DevKit |
|---|---|---|
| VDD | 3V3 (never 5V) | 3V3 |
| GND | GND | GND |
| L/R | GND (left channel) | GND |
| SCK | GPIO 41 | GPIO 14 |
| WS  | GPIO 42 | GPIO 15 |
| SD  | GPIO 2  | GPIO 32 |

These GPIOs are not used by the camera, PSRAM or SD-card slot on the common ESP32-S3 CAM
layout. If your board has no pin labelled 41, 42 or 2, pick other free pins and change
`PIN_SCK`, `PIN_WS`, `PIN_SD` at the top of `mic_test.ino`. Avoid 4-18 (camera),
35-37 (PSRAM), 38-40 (SD card), 0/3/45/46 (boot pins), 19/20 (USB).

## Run

```sh
firmware/flash.sh                 # ESP32-S3; or: firmware/flash.sh esp32
```
Use the USB-C port marked COM/UART if the board has two. If upload fails, hold BOOT,
tap RESET, release BOOT, and run it again.

Expected: lines like `-58.3  125  ###############...` ten times a second (quiet room about -55 to -62 dBFS, voices -40 to -50); clapping makes the
level jump to around -20 dBFS, whistling shows a steady peak frequency. A level stuck at
-180 or the "only zeros" warning means a wiring fault.

Record 5 s to a WAV (close the monitor first with Ctrl+C):
```sh
.venv/bin/python firmware/tools/record.py /dev/ttyUSB0 rec.wav
```
then listen to it, or open it in MATLAB: `safesound_demo('rec.wav')`.
