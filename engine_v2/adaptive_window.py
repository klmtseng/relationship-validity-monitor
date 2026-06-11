"""ADWIN-adaptive-window t-stat — replaces v1's FIXED trailing window W.

v1 hard-codes W (e.g. 260wk). That is a guess: too long and it is slow to notice a
break; too short and it is noisy. ADWIN (Bifet & Gavalda 2007) keeps a VARIABLE-length
window: it grows the window while the stream's mean is stable and automatically SHRINKS
it the moment a change in mean is detected, dropping the now-stale pre-break data.

We feed ADWIN the edge series (sign * z(driver) * ret, whose mean IS the predictive
covariance we care about). At each step we read ADWIN's current window width and compute
the SAME Newey-West t-stat as v1 over exactly that adaptive window. Result: a t-stat path
directly comparable to v1's, but with a data-driven window instead of a fixed guess.
"""
from __future__ import annotations
import numpy as np
from river import drift

from engine_v2 import nw_lrv


def adaptive_tstat(v: np.ndarray, hac: int = 26, min_w: int = 40,
                   delta: float = 0.002, cap_w: int | None = None):
    """Return (tstat_path, width_path) using an ADWIN-chosen trailing window.

    delta = ADWIN confidence (smaller = more conservative, fewer window cuts).
    min_w = minimum window before a t-stat is reported (HAC needs enough points).
    cap_w = optional hard cap on window length (None = let ADWIN grow freely).
    """
    v = np.asarray(v, float)
    n = len(v)
    ts = np.full(n, np.nan)
    width = np.full(n, np.nan)
    ad = drift.ADWIN(delta=delta)
    for t in range(n):
        ad.update(float(v[t]))
        w = int(ad.width)
        if cap_w is not None:
            w = min(w, cap_w)
        w = min(w, t + 1)
        width[t] = w
        if w >= min_w:
            win = v[t - w + 1: t + 1]
            ts[t] = np.sqrt(w) * win.mean() / np.sqrt(nw_lrv(win, hac))
    return ts, width


def adwin_changepoints(v: np.ndarray, delta: float = 0.002):
    """Indices where ADWIN flags a change in the mean of the edge series."""
    v = np.asarray(v, float)
    ad = drift.ADWIN(delta=delta)
    cps = []
    for t in range(len(v)):
        ad.update(float(v[t]))
        if ad.drift_detected:
            cps.append(t)
    return cps
