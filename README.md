# Relationship-Validity Monitor

A small, honest toolkit for one question quant research keeps getting wrong:

> **Is this statistical relationship still alive — or am I looking at noise that crossed a threshold once?**

This is **not an alpha engine.** It generates no signals and makes no return forecasts.
It is a *falsifier*: a calibrated monitor that tells you when a documented edge has decayed,
and — just as importantly — refuses to certify edges that don't survive honest multiple-testing
control. Its value is in what it **kills**, not what it discovers. It is a research-hygiene /
no-trade-discipline gate.

The repo also does something most "I built a detector" projects don't: it **swaps each of its
own home-grown parts for the mature, field-standard component and reports — honestly — whether
the upgrade changed any conclusion.** The headline finding is that *newer is not always better*.

## What's inside

```
engine_v2/                 the monitor, built from field-standard components
  __init__.py              shared Newey-West (HAC) long-run variance + v1 fixed-window baseline
  adaptive_window.py       ADWIN adaptive windowing (Bifet-Gavaldà 2007, via `river`)
  csw_monitor.py           Chu-Stinchcombe-White (1996) path-controlled monitoring boundary
  dsr_pbo.py               Deflated Sharpe Ratio + Probability of Backtest Overfitting (Bailey-LdP)
demos/
  factor_zoo.py            death & revival detection on 8 Fama-French premia + false-discovery control
  v1_vs_v2_headtohead.py   v1 (fixed window / fixed bar / bootstrap null) vs v2 (ADWIN / CSW / DSR+PBO)
scripts/
  download_ff_data.py      fetch the public Ken French factor library (no data is committed)
results/
  FINDINGS_factor_zoo.md   what the monitor finds, and how the null controls keep it honest
  FINDINGS_v1_vs_v2.md     the three head-to-head verdicts (the interesting part)
  figures/                 generated PNGs
```

## The core idea

The monitor is the *online / point-in-time* form of a structural-stability test: a rolling,
HAC-corrected (Newey-West) **t-statistic of an "edge series"** (e.g. a factor's mean weekly
premium), declared *alive* while it stays above a bar. That much is standard. The discipline is
in the controls:

- **A null relationship crosses a 1.65 bar at least once with probability ≈ 0.60** over a long
  sample. So *counting single threshold crossings is noise.* The only trustworthy statistics are
  the **alive-fraction vs a bootstrap null (~5%)** and crossings of a high **discovery bar (3.0,
  per Harvey-Liu-Zhu 2016)**.
- Every claim is checked against **circular-block-bootstrap null factors** (zero true edge,
  realistic vol clustering and autocorrelation) so the false-discovery rate is measured, not assumed.

## v1 → v2: does the mature component change the conclusion?

`demos/v1_vs_v2_headtohead.py` replaces each home-grown part with the literature's stronger tool
and reports the result without spin:

| Part | v1 (home-grown) | v2 (field-standard) | Verdict |
|---|---|---|---|
| Window | fixed trailing W=260 | **ADWIN** adaptive window | **Worse — don't swap.** On low-SNR weekly edges ADWIN never cuts, collapses to a global window, and *misses recent decay.* The fixed window's deliberate forgetting is the correct design here. |
| Boundary | fixed bar 1.65 / 3.0, checked every step | **Chu-Stinchcombe-White** path-controlled boundary | **The one real upgrade.** Null-panel false-alarm drops from 0.97 (ever>1.65) / 0.42 (ever>3.0) to **0.05** by design. Cost: power — once correctly controlled, most weekly edges (even the equity premium) stop clearing the bar. |
| False discovery | bootstrap null panel | **Deflated Sharpe + PBO/CSCV** | **Tie, but standard.** Same conclusion as the bootstrap null, now in the language reviewers and allocators already speak. Caveat: it measures *gross* statistical edge, not *net* tradeability. |

**Takeaway:** swapping in fancier tools overturned *none* of the original conclusions — it
*reinforced* them, and in one case (ADWIN) the upgrade was a regression. The honest negative
result is the point.

## Quick start

```bash
pip install -r requirements.txt
python -m scripts.download_ff_data       # fetch public Ken French data (cached locally, not committed)
python -m demos.factor_zoo               # demo 1: factor death/revival + false-discovery control
python -m demos.v1_vs_v2_headtohead      # demo 2: v1 vs v2 head-to-head
```

Run from the repository root. Figures land in `results/figures/`.

## Scope & honesty notes

- **No market data is shipped.** The demos use the freely-redistributable Ken French factor
  library, downloaded on first run. No vendor data, no licensing entanglements.
- This tool finds *whether a relationship is statistically alive*, not *whether it is profitable
  after costs.* Short-term reversal is the cautionary example: statistically rock-solid (Deflated
  Sharpe = 1.0), but eaten alive by transaction costs in practice. **Statistical significance ≠
  tradeability.**
- It works on **high-frequency, continuous relationships** (factor premia, beta links), where
  there are enough observations to calibrate. It is *not* built for rare binary events.
- Hyperparameters (windows, bars, HAC lag) are fixed by theory and **never tuned to outcomes.**

## References

See [`REFERENCES.md`](REFERENCES.md).
