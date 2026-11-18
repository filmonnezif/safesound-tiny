#!/usr/bin/env bash
# Build and upload the mic test, then open the serial monitor.
#   firmware/flash.sh            ESP32-S3 (N16R8), first serial port found
#   firmware/flash.sh esp32      classic ESP32 DevKit
#   firmware/flash.sh s3 /dev/ttyACM0
set -e
cd "$(dirname "$0")"
board=${1:-s3}
port=${2:-$(ls /dev/ttyACM* /dev/ttyUSB* 2>/dev/null | head -1)}
[ -n "$port" ] || { echo "no board found: plug it in (use a data cable)"; exit 1; }
if [ "$board" = esp32 ]; then fqbn=esp32:esp32:esp32
else fqbn="esp32:esp32:esp32s3:PSRAM=opi,FlashSize=16M,PartitionScheme=app3M_fat9M_16MB"; fi
echo "board $fqbn on $port"
arduino-cli compile --fqbn "$fqbn" mic_test
arduino-cli upload --fqbn "$fqbn" -p "$port" mic_test
arduino-cli monitor -p "$port" -c baudrate=921600
