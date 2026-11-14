"""Event-based evaluation: recall at a false-alarm-per-hour budget (docs/protocol.md).

A detector produces, for one class, a score per frame (hop `hop_s` seconds).
Alerts: the score crossing `threshold` fires an alert, then the class is silent for
`refractory_s` (one alert per occurrence, like a real device).

- Occurrence: labelled events of the same class in the same recording, merged when
  the gap between them is <= `merge_gap_s` (an alarm's beeps form one occurrence).
- Detected: an alert at a time in [onset, max(offset, onset + tolerance_s)].
- False alarms: alerts in negative recordings (none of the six classes), per hour.
"""
from dataclasses import dataclass

import numpy as np

MERGE_GAP_S = 5.0
TOLERANCE_S = 2.0
REFRACTORY_S = 10.0


@dataclass
class Recording:
    scores: np.ndarray                  # (frames,) score for one class
    hop_s: float
    events: list[tuple[float, float]]   # (onset, offset) of this class; [] for negatives
    t0: float = 0.0                     # time (s) of the first score (end of its window)

    @property
    def hours(self) -> float:
        return len(self.scores) * self.hop_s / 3600


def merge_events(events, gap_s=MERGE_GAP_S):
    out = []
    for on, off in sorted(events):
        if out and on - out[-1][1] <= gap_s:
            out[-1][1] = max(out[-1][1], off)
        else:
            out.append([on, off])
    return [tuple(e) for e in out]


def alert_times(scores, hop_s, threshold, refractory_s=REFRACTORY_S, t0=0.0):
    """Times (s) at which alerts fire: rising crossing of threshold, then a quiet period."""
    times, quiet_until = [], -np.inf
    above = np.flatnonzero(scores >= threshold)
    for i in above:
        t = t0 + i * hop_s
        if t >= quiet_until:
            times.append(t)
            quiet_until = t + refractory_s
    return times


def evaluate(recordings, threshold, tolerance_s=TOLERANCE_S, refractory_s=REFRACTORY_S):
    """Returns (detected, occurrences, false_alarms, negative_hours, per_occurrence_hits)."""
    hits, n_fa, neg_h = [], 0, 0.0
    for r in recordings:
        alerts = alert_times(r.scores, r.hop_s, threshold, refractory_s, r.t0)
        if not r.events:
            n_fa += len(alerts)
            neg_h += r.hours
            continue
        for on, off in merge_events(r.events):
            end = max(off, on + tolerance_s)
            hits.append(any(on <= t <= end for t in alerts))
    return sum(hits), len(hits), n_fa, neg_h, np.array(hits, bool)


def threshold_for_budget(recordings, fa_per_hour, grid=None):
    """Lowest threshold whose false-alarm rate on the negatives is <= the budget.

    Binary search over a grid of score quantiles (false alarms fall as the threshold
    rises). Call this on the TUNING set only; then apply the threshold to the test set.
    """
    negs = [r for r in recordings if not r.events]
    if grid is None:
        neg = np.concatenate([r.scores for r in negs])
        grid = np.unique(np.quantile(neg, np.linspace(0.9, 1.0, 4001)))
        grid = np.append(grid, neg.max() + 1e-6)
    grid = np.sort(grid)

    def ok(th):
        _, _, n_fa, neg_h, _ = evaluate(negs, th)
        return neg_h > 0 and n_fa / neg_h <= fa_per_hour

    lo, hi = 0, len(grid) - 1
    while lo < hi:
        mid = (lo + hi) // 2
        if ok(grid[mid]):
            hi = mid
        else:
            lo = mid + 1
    return float(grid[lo])


def bootstrap_recall(hits, n_boot=2000, seed=0):
    """95% bootstrap interval of recall over occurrences."""
    if len(hits) == 0:
        return (np.nan, np.nan)
    rng = np.random.default_rng(seed)
    boots = rng.choice(hits, size=(n_boot, len(hits)), replace=True).mean(axis=1)
    return tuple(np.quantile(boots, [0.025, 0.975]))
