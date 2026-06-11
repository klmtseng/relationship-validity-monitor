"""engine_v2 — the validity engine rebuilt with the mature off-the-shelf components.

v1 (the demos in scripts/) = fixed trailing-W Newey-West t-stat, fixed bar 1.65/3.0,
circular-block-bootstrap null panel. Sound but a plain member of a mature family.

v2 swaps each part for the literature's stronger tool:
  - fixed window W        -> ADWIN adaptive windowing (Bifet-Gavalda 2007, via river)
  - fixed bar applied repeatedly -> Chu-Stinchcombe-White (1996) horizon-controlled
    monitoring boundary (controls path-wise false-alarm; the fixed bar does NOT)
  - bootstrap null panel  -> Deflated Sharpe Ratio + PBO/CSCV (Bailey-Lopez de Prado)

This file holds the ONE shared Newey-West long-run variance so the v1 baseline and the
v2 ADWIN t-stat use IDENTICAL HAC — otherwise the head-to-head would be apples-to-oranges.
It is byte-identical in behaviour to scripts.engine_demo_factor_zoo.nw_lrv.
"""
from __future__ import annotations
import numpy as np


def nw_lrv(x: np.ndarray, lag: int) -> float:
    x = np.asarray(x, float)
    x = x - x.mean()
    n = len(x)
    if n < 5:
        return float(np.var(x)) + 1e-12
    g0 = (x @ x) / n
    lrv = g0
    for k in range(1, min(lag, n - 1) + 1):
        w = 1 - k / (lag + 1)
        lrv += 2 * w * (x[k:] @ x[:-k]) / n
    return max(lrv, 1e-12)


def rolling_tstat_fixed(v: np.ndarray, w: int, hac: int) -> np.ndarray:
    """v1 baseline: fixed trailing-w window Newey-West t-stat (the thing v2 replaces)."""
    v = np.asarray(v, float)
    n = len(v)
    ts = np.full(n, np.nan)
    for t in range(w - 1, n):
        win = v[t - w + 1: t + 1]
        ts[t] = np.sqrt(w) * win.mean() / np.sqrt(nw_lrv(win, hac))
    return ts
