# MATLAB demo

```matlab
cd matlab
safesound_demo            % built-in test: quiet room, smoke alarm (5-13 s), door slam (16 s)
safesound_demo('x.wav')   % any audio file
safesound_demo('mic')     % record 10 s from the microphone
```

Base MATLAB only (no toolboxes). It plots the waveform, the spectrogram, the per-frame DSP
evidence and the 2 s window scores, prints when the alarm was detected, and plays the sound.

Expected on the built-in test: the alarm detector alerts from about 6 s to 13 s; the plain
loudness (energy) detector ignores the quiet alarm but fires on the door slam (16.5-18 s).
The detector is the alarm detector of `safesound/dsp.py` (checked against it in Python).
