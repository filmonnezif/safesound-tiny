#!/usr/bin/env bash
# SafeSound hardware demo: plays each sound with a pause; watch the board's serial output or LED.
#   ./play_demo.sh          all 8 sounds        ./play_demo.sh 5     only sound 5
cd "$(dirname "$0")"
player=$(command -v pw-play || command -v paplay || command -v aplay || command -v afplay)
for f in $(ls [1-8]_*.wav | grep "^${1:-}"); do
  echo ">>> $f"
  $player "$f"
  sleep 4
done
