"""Chu-Stinchcombe-White (1996) monitoring boundary — replaces v1's FIXED 1.65 bar.

THE PROBLEM v1 has: it applies the bar 1.65 at EVERY date. Each look is a 5% test, but
looking hundreds of times inflates the path-wise false alarm enormously — we measured it:
a pure-null relationship pokes above 1.65 at least once with prob ~0.60 over a full sample.
So "it was significant once" is nearly worthless under v1.

CSW (Econometrica 1996, "Monitoring Structural Change") fixes exactly this. After a stable
training period of size m it monitors a recursive detector against a boundary
    b_a(s) = sqrt( s * (a^2 + log s) ),   s = (k - m) / m
whose constant a is chosen so the probability of the detector EVER crossing the boundary
over the whole monitoring horizon equals alpha. One number, controlled once, for the whole
path — not 5% per look.

We pick a by MONTE CARLO on the data's own null (circular block-bootstrap of the demeaned
series, preserving autocorrelation) rather than trusting a textbook constant — same
calibrate-against-your-own-null discipline the v1 demos used, and it sidesteps the fragile
asymptotic critical values.

Detector here is ONE-SIDED for an emerging POSITIVE edge (the v1 "alive" question):
    Q(k) = sum_{j=m+1..k} x_j / (sigma_hat * sqrt(m))
A genuinely alive (positive-mean) series makes Q grow ~linearly and eventually clears the
sqrt-boundary (power); a null series clears it only alpha of the time (size).
"""
from __future__ import annotations
import numpy as np

from engine_v2 import nw_lrv


def _circular_block_bootstrap(x: np.ndarray, n: int, block: int, rng) -> np.ndarray:
    x = np.asarray(x, float)
    L = len(x)
    out = np.empty(n)
    filled = 0
    while filled < n:
        start = rng.integers(0, L)
        take = min(block, n - filled)
        idx = (start + np.arange(take)) % L
        out[filled:filled + take] = x[idx]
        filled += take
    return out


def _detector_path(x: np.ndarray, m: int, hac: int, sigma: float | None = None):
    """One-sided CSW detector Q(k) and relative time s for k=m+1..n."""
    x = np.asarray(x, float)
    n = len(x)
    if sigma is None:
        sigma = np.sqrt(nw_lrv(x[:m], hac))
    csum = np.cumsum(x[m:])                       # sum_{j=m+1..k}
    Q = csum / (sigma * np.sqrt(m))
    k = np.arange(m + 1, n + 1)
    s = (k - m) / m
    return Q, s


def _boundary(s: np.ndarray, a: float) -> np.ndarray:
    val = a * a + np.log(s)
    b = np.where(val > 0, np.sqrt(s * np.maximum(val, 0.0)), np.inf)
    return b


def calibrate_a(x: np.ndarray, m: int, alpha: float = 0.05, hac: int = 26,
                block: int = 26, n_mc: int = 2000, seed: int = 0) -> float:
    """Bisection on a so P(detector ever crosses boundary | null) = alpha."""
    rng = np.random.default_rng(seed)
    n = len(x)
    pool = np.asarray(x, float) - np.mean(x)      # demeaned -> true null edge = 0
    null_max_ratio = np.empty(n_mc)
    for i in range(n_mc):
        xb = _circular_block_bootstrap(pool, n, block, rng)
        Q, s = _detector_path(xb, m, hac)          # sigma re-estimated per path
        # ratio Q / sqrt(s) ; boundary is sqrt(s)*sqrt(a^2+log s) -> cross when
        # Q/sqrt(s) >= sqrt(a^2+log s) i.e. (Q^2/s - log s) >= a^2 . track max of LHS.
        with np.errstate(divide="ignore", invalid="ignore"):
            lhs = Q ** 2 / s - np.log(s)
        lhs = lhs[np.isfinite(lhs) & (Q > 0)]      # one-sided: only positive excursions
        null_max_ratio[i] = lhs.max() if len(lhs) else -np.inf
    # a^2 = the (1-alpha) quantile of the per-path max LHS  -> P(max LHS > a^2)=alpha
    a2 = np.quantile(null_max_ratio, 1 - alpha)
    return float(np.sqrt(max(a2, 0.01)))


def csw_monitor(x: np.ndarray, m: int, alpha: float = 0.05, hac: int = 26,
                block: int = 26, n_mc: int = 2000, seed: int = 0) -> dict:
    """Run the calibrated CSW monitor on series x. Returns detection + diagnostics."""
    x = np.asarray(x, float)
    n = len(x)
    a = calibrate_a(x, m, alpha, hac, block, n_mc, seed)
    Q, s = _detector_path(x, m, hac)
    b = _boundary(s, a)
    crossed = (Q >= b) & np.isfinite(b)
    first = int(np.argmax(crossed)) if crossed.any() else -1
    detect_k = (m + 1 + first) if first >= 0 else None     # 1-based position in x
    return dict(a=a, Q=Q, s=s, boundary=b, crossed=crossed.any(),
                detect_pos=detect_k, m=m, alpha=alpha)


def fixed_bar_ever(tstat_path: np.ndarray, bar: float) -> bool:
    """v1 behaviour: did the fixed-bar t-stat EVER exceed bar over the path?"""
    t = np.asarray(tstat_path, float)
    t = t[~np.isnan(t)]
    return bool((t > bar).any())
