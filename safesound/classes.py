"""The six target classes and the AudioSet labels (MIDs) that map onto each."""

CLASSES = ["alarm", "siren", "doorbell", "knock", "baby_cry", "car_horn"]

# AudioSet ontology IDs per class (parent label plus its relevant children).
AUDIOSET_MIDS = {
    "alarm": ["/m/01y3hg", "/m/0c3f7m"],            # Smoke detector; Fire alarm
    "siren": ["/m/03kmc9", "/m/04qvtq", "/m/012n7d",  # Siren; Police car; Ambulance
              "/m/012ndj", "/m/0dgbq"],              # Fire engine; Civil defense
    "doorbell": ["/m/03wwcy", "/m/07r67yg"],         # Doorbell; Ding-dong
    "knock": ["/m/07r4wb8"],                         # Knock
    "baby_cry": ["/t/dd00002"],                      # Baby cry, infant cry
    "car_horn": ["/m/0912c9", "/m/05x_td",           # Vehicle horn; Air horn, truck horn
                 "/m/07qv_d5"],                      # Toot (weak labels only)
}

# Clips carrying these labels but none of the six are left out of every negative
# set: they may contain a target sound under a vaguer label.
AMBIGUOUS_MIDS = {
    "/m/07pp_mv",   # Alarm (generic)
    "/m/03j1ly",    # Emergency vehicle
    "/m/0463cq4",   # Crying, sobbing (age unknown)
    "/m/07qz6j3",   # Whimper
}

# Hard negatives: sounds easily confused with the targets. Every training clip with
# one of these is kept; other negatives are subsampled.
HARD_NEGATIVE_MIDS = {
    "/m/046dlr", "/m/07cx4", "/m/07pp8cl", "/m/01hnzm",       # alarm clock, telephone, ringtone
    "/m/02fs_r", "/m/030rvx", "/m/0395lw", "/m/03w41f",       # beep, buzzer, bell, church bell
    "/m/0gy1t2s", "/m/027m70_", "/m/0239kh", "/m/02mfyn",     # bicycle/jingle bell, cowbell, car alarm
    "/m/02rhddq", "/m/0fx9l", "/m/0l156k", "/g/11b630rrvh",   # reversing beeps, microwave, whistles
    "/m/07qcpgn", "/m/02dgv", "/m/07rjzl8", "/m/07phxs1",     # tap, door, slam, ding
    "/m/0f8s22", "/m/026fgl", "/t/dd00001", "/m/0284vy3",     # chime, wind chime, baby laughter, train horn
    "/m/04fq5q", "/m/07c52", "/m/06bz3",                      # foghorn, television, radio
}

MID_TO_CLASS = {m: c for c, ms in AUDIOSET_MIDS.items() for m in ms}
