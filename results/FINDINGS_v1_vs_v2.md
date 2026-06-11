# Findings — v1 vs v2: does swapping in the mature component change any conclusion?

**Data:** 8 Fama-French factor premia, weekly, 1990–2026 (1892 weeks).
**Script:** `demos/v1_vs_v2_headtohead.py` · **Figure:** `results/figures/engine_v2_vs_v1.png`

## Motivation

The home-grown monitor (a rolling HAC t-stat with a fixed window, a fixed bar, and a
bootstrap null) has a more mature counterpart for *every* part in the change-detection,
sequential-monitoring, and backtest-overfitting literatures. So we swapped each part for the
field-standard tool and asked one honest question: **after the upgrade, does any earlier
conclusion change — and is each swap actually worth it?** Same data, v1 and v2 side by side.

| Part | v1 (home-grown) | v2 (field-standard) |
|---|---|---|
| Window | fixed W=260 trailing | **ADWIN** adaptive window (Bifet-Gavaldà 2007, `river`) |
| Boundary | fixed 1.65 / 3.0, checked every step | **Chu-Stinchcombe-White (1996)** path-controlled boundary |
| False discovery | circular-block-bootstrap null panel | **Deflated Sharpe + PBO/CSCV** (Bailey-López de Prado) |

## 1) Window: ADWIN — no improvement, in fact worse

| Factor | v1 fixed now_t | v2 ADWIN now_t | ADWIN window | ADWIN cuts |
|---|---|---|---|---|
| Mkt-RF | +1.32 | +3.52 | 1892 (full) | **0** |
| ST_Rev | +0.36 | **+6.94** | 1892 | **0** |
| Mom | +2.28 | +2.34 | 1892 | 0 |
| (all 8) | … | … | all 1892 | **all 0** |

**ADWIN never cut the window on real weekly factor data** — the window always collapsed to the
full sample. ADWIN's change test is tuned for distribution shifts that are large relative to
noise; a weekly factor's edge change is tiny relative to its volatility (weekly Sharpe ≈ 0.07),
far below ADWIN's detection threshold.

The consequence matters: ST_Rev's v2 (full-sample) now_t = **+6.94**, but v1's fixed-window
now_t is only **+0.36** — because ST_Rev's edge has **decayed / been eaten by costs in recent
years**, and the fixed 5-year window *deliberately forgets old data*, so it correctly reports
"weak now." ADWIN, never cutting, averages in the strong 1990s and falsely reports "still alive."

> **Verdict 1: for the question "is this relationship alive *now*," the fixed window's
> deliberate forgetting is a feature, not a bug. ADWIN only forgets on a violent distribution
> break, which low-SNR financial edges never produce → ADWIN degrades to a global window:
> slower than v1 and prone to miss recent decay. Don't swap this part.**

## 2) Boundary: CSW — the genuine upgrade, and it changes the conclusion

### False-discovery panel (60 zero-edge block-bootstrap factors)

| Criterion | P(path ever falsely alarms) |
|---|---|
| v1 "ever t > 1.65" | **0.967** |
| v1 "ever t > 3.0" (Harvey-Liu-Zhu patch) | **0.417** |
| **v2 CSW path-control** | **0.050** (target 5%, exact hit) |

This is the hardest number in the exercise. Looking at the 1.65 bar repeatedly over a long path,
**97% of pure-noise factors false-alarm at least once**; even the 3.0 patch still false-alarms
**42% of the time** over a long horizon. CSW is *designed* to control the family-wise false-alarm
rate over the whole monitoring horizon — 5% in one shot.

> CSW's boundary constant was Monte-Carlo-**calibrated on the data's own block-bootstrap null**,
> giving a ≈ 3.5–4.4, clearly above the textbook iid constant (~2.79). That is not a bug — financial
> series are autocorrelated and fat-tailed, so the textbook constant *understates* the boundary and
> admits too many false alarms. Self-calibration is the honest choice.

### The cost: power

| Series | CSW fires? | Reading |
|---|---|---|
| STABLE control (fixed true edge, weekly SR 0.075) | **yes** (2015) | real edge, correctly caught |
| LT_Rev | yes (2002) | a strong accumulation in the early 2000s |
| Mkt-RF / RMW / Mom / ST_Rev … | **no** | too weak to clear the strict sequential bar |
| NOISE control | no | correctly silent |

**CSW misses the equity premium (Mkt-RF)** — which is real (full-sample t ≈ 3.1) but **so weak
that once you correctly correct for "having looked thousands of times," it is indistinguishable
from noise.** That is not a flaw; it is the honest price of multiple-testing correction.

> **Verdict 2: CSW is the one part worth swapping, and it changes the conclusion — it turns the
> boundary problem from "hand-tuned 1.65/3.0" into "5% path-level control." But once correctly
> controlled, most weekly factors' "alive" verdict evaporates (even the equity premium can't
> survive). This *strengthens* the original positioning: these edges are extremely marginal, and
> v1's "was once significant" carries almost no evidentiary weight (97% false-alarm) — CSW
> quantifies that beyond dispute.**

## 3) False discovery: DSR + PBO — a tie, but more standard

8 real factors, T = 1892:

| Metric | Value | Reading |
|---|---|---|
| IS-best (highest Sharpe) | **ST_Rev** (weekly SR 0.170, ann ~1.23) | |
| Deflated benchmark SR0 (E[max] of 8 trials) | 0.074 | |
| PSR vs 0 (naive, ignores multiple testing) | 1.000 | |
| **DSR vs SR0** (deflated, P true SR>0) | **1.000** | ST_Rev's *gross* edge survives deflation |
| **PBO** (CSCV, 3432 splits) | **0.041** | low overfitting probability, stable OOS |
| widen to 58 trials (+50 decoys) | DSR=1.000, PBO=0.001 | a truly strong factor isn't drowned by decoys |

DSR/PBO reach the **same** conclusion as v1's bootstrap null panel (wider search reaps more noise,
but a genuinely strong factor holds up) — just expressed with **field-standard** closed-form +
cross-validation rather than a home-grown bootstrap.

> **Important caveat: DSR/PBO measure whether the *gross* statistical edge is real, not whether
> the *net* return is tradeable.** ST_Rev is the living example: DSR = 1.0 (gross edge rock-solid),
> yet in practice it is **eaten by transaction costs.** Statistical significance ≠ profitability.
>
> **Verdict 3: this swap is "a tie but more standard / more communicable" — the conclusion is
> unchanged, but stated in the language reviewers and allocators already accept.**

## Bottom line: after the upgrade, the core positioning is unchanged — and reinforced

| Part | After swapping | One line |
|---|---|---|
| Window → ADWIN | **worse, don't swap** | under low SNR ADWIN never cuts → degrades to a global window, misses recent decay |
| Boundary → CSW | **real upgrade, do swap** | path false-alarm 0.97 → 0.05; but once controlled, most edges evaporate |
| False discovery → DSR/PBO | **tie, worth swapping** | same conclusion, more standard language; measures gross, not net |

**The biggest lesson (honest version): "more advanced" ≠ "better."**
- ADWIN proves a fancier tool can *regress* — the fixed window's deliberate forgetting is the
  right design for "is it alive now."
- CSW is the genuine methodological upgrade: it turns the "a single crossing = noise" rule we'd
  always asserted verbally into a *guaranteed* 5% path-level control — and confirms, with a strict
  standard, that weekly factor edges are so weak even the equity premium can't survive sequential
  multiple testing.
- DSR/PBO put false-discovery control in the field's common language, while reminding us it
  measures gross, not net.

v2 overturns **none** of v1's conclusions (the engine is a calibrated relationship-decay monitor /
discipline gate, not an alpha machine) — it nails them down with stricter tools. The forward
recipe: **keep the fixed rolling window (not ADWIN), replace the fixed bar with CSW, replace the
home-grown null with DSR/PBO** — rather than re-proving the engine is "right" one more time.
