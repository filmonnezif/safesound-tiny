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
    "car_horn": ["/m/0912c9", "/m/05x_td"],          # Vehicle horn; Air horn, truck horn
}

MID_TO_CLASS = {m: c for c, ms in AUDIOSET_MIDS.items() for m in ms}
