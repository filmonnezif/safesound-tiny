import numpy as np

from safesound.evaluate import (Recording, alert_times, bootstrap_recall, evaluate,
                                merge_events, threshold_for_budget)

HOP = 0.5


def rec(n, events=(), spikes=()):
    s = np.zeros(n)
    for i, v in spikes:
        s[i] = v
    return Recording(s, HOP, list(events))


def test_merge_joins_beeps_into_one_occurrence():
    assert merge_events([(0, 0.5), (1.5, 2), (20, 21)]) == [(0, 2), (20, 21)]


def test_refractory_gives_one_alert_per_burst():
    s = np.zeros(100)
    s[10:20] = 1.0           # 5 s above threshold -> one alert
    s[60] = 1.0              # 25 s later -> second alert
    assert alert_times(s, HOP, 0.5) == [5.0, 30.0]


def test_detection_window_and_false_alarms():
    pos_hit = rec(40, [(5, 5.2)], [(11, 0.9)])    # alert at 5.5 s, within onset + 2 s
    pos_miss = rec(40, [(5, 5.2)], [(30, 0.9)])   # alert at 15 s, too late
    neg = rec(7200, [], [(100, 0.9), (3000, 0.4)])  # one hour, one alert above 0.5
    det, n, fa, hours, hits = evaluate([pos_hit, pos_miss, neg], 0.5)
    assert (det, n, fa) == (1, 2, 1)
    assert abs(hours - 1.0) < 1e-9
    assert hits.tolist() == [True, False]


def test_threshold_for_budget_meets_budget():
    neg = rec(7200, [], [(100, 0.9), (3000, 0.4)])  # 1 h
    th = threshold_for_budget([neg], fa_per_hour=0.5)
    _, _, fa, hours, _ = evaluate([neg], th)
    assert fa / hours <= 0.5 and th > 0.4


def test_bootstrap_interval_contains_point_estimate():
    hits = np.array([1, 1, 0, 1, 0, 1, 1, 1], bool)
    lo, hi = bootstrap_recall(hits)
    assert lo <= hits.mean() <= hi
