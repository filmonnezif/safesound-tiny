# SafeSound demo sounds

Play these near the board (laptop speakers at high volume, mic within ~30 cm). The board's
LED turns red and the serial monitor prints `>>> ALERT <class>`.

| File | Board should say | Reliability in our tests |
|---|---|---|
| 1_smoke_alarm.wav | alarm | 3/3 |
| 2_siren_wail.wav | siren | 3/3 |
| 3_siren_yelp.wav | siren | 3/3 |
| 4_car_horn.wav | car_horn | 3/3 |
| 5_baby_crying.wav | baby_cry | 3/3 |
| 6_door_knock.wav | knock | 4/5 (knocks lose their thump on small speakers; a real knock on a door works better) |
| 7_speech_no_alert.wav | nothing | 3/3 |
| 8_dishes_no_alert.wav | nothing | 3/3 |

`all_demo.wav` = all eight with 4 s pauses (~1 min 40 s). `./play_demo.sh` plays them one by one.
Live output: `arduino-cli monitor -p /dev/ttyACM0 -c baudrate=921600`

## Sources (Freesound; FSD50K evaluation clips where noted)
- 1: synthetic ISO 8201 T3 smoke-alarm pattern, 3.1 kHz (generated)
- 2: "Siren.wav" by daveincamas, freesound.org/s/43806, CC BY 4.0 (FSD50K eval)
- 3: "Ambulence - WARNING LOUD" by eaglechopper, freesound.org/s/238228, CC0 (FSD50K eval)
- 4: "Car Horn.mp3" by DigPro120, freesound.org/s/432641, CC0 (FSD50K eval)
- 5: "Aiden Cry.m4a" by royal.cline1, freesound.org/s/518184, CC0
- 6: "Knocking_on_Door_Rattle_knob.wav" by Mediaman57, freesound.org/s/347142, CC0
- 7: "capu kiosk money rambla.wav" by sgel05, freesound.org/s/101807, CC Sampling+ (FSD50K eval)
- 8: "Quarter Spinning on Glass by Fajack" by postworkflow, freesound.org/s/360681, CC BY-NC 4.0 (FSD50K eval)
All clips are 16 kHz mono, loudness-normalised (-16 LUFS); 2 and 4 are looped to ~8-9 s.
