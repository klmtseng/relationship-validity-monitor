"""Deflated Sharpe Ratio + Probability of Backtest Overfitting — replace v1's bootstrap null.

v1 controlled false-discovery with a circular-block-bootstrap NULL panel (re-run the engine
on demeaned data, count spurious 'alive' fractions). That is sound but home-grown. Bailey &
Lopez de Prado give the field-standard closed-form / cross-validated tools:

  DSR (Deflated Sharpe Ratio, 2014): the probability the best of N tried strategies has a
  TRUE Sharpe > 0, after correcting for (a) selection across N trials, (b) sample length,
  (c) skew & kurtosis of returns. It deflates the benchmark to the EXPECTED MAXIMUM Sharpe
  a bunch of zero-skill trials would have produced, then asks if the observed best clears it.

  PBO (Probability of Backtest Overfitting, 2017) via CSCV: take the panel of N strategies'
  returns, combinatorially split time into in-sample / out-of-sample halves, pick the IS-best
  each split, and measure how often it lands BELOW the OOS median. High PBO => 'best in
  sample' is noise. This needs no distributional assumption.
"""
from __future__ import annotations
import numpy as np
from itertools import combinations
from scipy.stats import norm

GAMMA = 0.5772156649015329          # Euler-Mascheroni
E = np.e


def sharpe(returns: np.ndarray) -> float:
    r = np.asarray(returns, float)
    sd = r.std(ddof=1)
    return float(r.mean() / sd) if sd > 0 else 0.0


def psr(sr: float, n: int, skew: float, kurt: float, sr_benchmark: float = 0.0) -> float:
    """Probabilistic Sharpe Ratio: P(true SR > benchmark). sr,benchmark per-period; kurt non-excess."""
    denom = np.sqrt(max(1.0 - skew * sr + (kurt - 1.0) / 4.0 * sr * sr, 1e-12))
    z = (sr - sr_benchmark) * np.sqrt(max(n - 1, 1)) / denom
    return float(norm.cdf(z))


def expected_max_sr(sr_std: float, n_trials: int) -> float:
    """E[max SR] of n_trials independent zero-skill strategies (per-period units)."""
    if n_trials < 2 or sr_std <= 0:
        return 0.0
    a = norm.ppf(1.0 - 1.0 / n_trials)
    b = norm.ppf(1.0 - 1.0 / (n_trials * E))
    return float(sr_std * ((1.0 - GAMMA) * a + GAMMA * b))


def deflated_sharpe_ratio(ret_matrix: np.ndarray, selected: int | None = None) -> dict:
    """ret_matrix: T x N (each column a strategy/factor's per-period returns).

    Computes the DSR of the selected strategy (default = highest in-sample Sharpe), deflating
    against the expected-max Sharpe of N zero-skill trials with the observed cross-trial SR spread.
    """
    R = np.asarray(ret_matrix, float)
    if R.ndim == 1:
        R = R[:, None]
    T, N = R.shape
    srs = np.array([sharpe(R[:, j]) for j in range(N)])
    j = int(np.argmax(srs)) if selected is None else selected
    sr = srs[j]
    sr_std = float(np.std(srs, ddof=1)) if N > 1 else 0.0
    sr0 = expected_max_sr(sr_std, N)
    r = R[:, j]
    rs = (r - r.mean()) / (r.std(ddof=1) + 1e-12)
    skew = float(np.mean(rs ** 3))
    kurt = float(np.mean(rs ** 4))                  # non-excess
    dsr = psr(sr, T, skew, kurt, sr_benchmark=sr0)
    psr0 = psr(sr, T, skew, kurt, sr_benchmark=0.0)
    return dict(selected=j, sr=sr, sr0_deflated=sr0, sr_std=sr_std, n_trials=N,
                T=T, skew=skew, kurt=kurt, PSR_vs0=psr0, DSR=dsr)


def pbo_cscv(ret_matrix: np.ndarray, S: int = 14) -> dict:
    """Probability of Backtest Overfitting via combinatorially-symmetric cross-validation.

    ret_matrix: T x N. Split rows into S equal blocks; for each way of choosing S/2 blocks as
    IS, pick the IS-best strategy (by Sharpe), look up its OOS Sharpe rank, and form the logit
    of its relative OOS rank. PBO = P(logit <= 0) = P(IS-best is below OOS median).
    """
    R = np.asarray(ret_matrix, float)
    T, N = R.shape
    if N < 2:
        return dict(PBO=np.nan, n_splits=0, logits=np.array([]))
    S = S if S % 2 == 0 else S - 1
    cut = (T // S) * S
    blocks = np.array_split(R[:cut], S, axis=0)
    idx = list(range(S))
    logits = []
    for is_sel in combinations(idx, S // 2):
        is_set = set(is_sel)
        oos_sel = [b for b in idx if b not in is_set]
        IS = np.vstack([blocks[b] for b in is_sel])
        OOS = np.vstack([blocks[b] for b in oos_sel])
        is_sr = np.array([sharpe(IS[:, j]) for j in range(N)])
        oos_sr = np.array([sharpe(OOS[:, j]) for j in range(N)])
        n_star = int(np.argmax(is_sr))
        # relative OOS rank of the IS-best (fraction of strategies it beats OOS)
        rank = (np.sum(oos_sr < oos_sr[n_star]) + 0.5) / N
        rank = min(max(rank, 1e-6), 1 - 1e-6)
        logits.append(np.log(rank / (1 - rank)))
    logits = np.array(logits)
    pbo = float(np.mean(logits <= 0))
    return dict(PBO=pbo, n_splits=len(logits), logits=logits,
                median_logit=float(np.median(logits)))
