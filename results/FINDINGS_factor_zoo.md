# Findings — factor-zoo death & revival detection + false-discovery control

**Data:** 8 Fama-French premia, weekly, 1990–2026 (1892 weeks).
**Script:** `demos/factor_zoo.py`
**Figures:** `results/figures/engine_demo_factor_zoo.png` (8 t-stat paths),
`engine_demo_factor_zoo_fdr.png` (null false-discovery distribution)

## The question

Two things: (1) can the monitor find factors that came *back alive*, not just ones that *died*;
and (2) can we cast a wide net across many factors to backtest their past deaths/revivals? This
demo answers both — and uses random null factors to quantify exactly how much false signal a wide
net produces.

The panel runs 8 documented French-library factors (all with a positive documented sign):
Mkt-RF, SMB, HML, RMW (profitability), CMA (investment), Mom, ST_Rev (short-term reversal),
LT_Rev (long-term reversal).

**Asymmetric bars (set by theory a priori, never tuned):**
- **Death** bar = 1.65 — you *already had a prior* the factor worked, so losing significance is
  trustworthy.
- **Revival** bar = 3.0 — declaring a dead factor "alive again" is a *selection* across many
  candidates → multiple testing. Harvey-Liu-Zhu (2016) argue the t-bar for *discovered* factors
  should be ~3.0.

## Core result: false-discovery control (the whole point)

100 null factors = circular block-bootstrap of the **demeaned** real returns (preserves vol
clustering / autocorrelation, but true edge = 0).

| Metric | null mean | Reading |
|---|---|---|
| alive-fraction | **0.052** | exact hit on the 1.65 bar's 5% nominal size — the monitor stays calibrated even on *realistic-structure* nulls |
| deaths per factor @1.65 | **5.13** | a zero-edge factor fakes ~5 "deaths" on average |
| revivals per factor @1.65 | **5.09** (P≥1 = 0.85) | likewise ~5 fake "revivals" |
| revivals per factor @3.00 | **0.36** (P≥1 = **0.10**) | raise to 3.0 and 90% of null factors produce **zero** fake revivals |

**Against the real factors:** the 8 factors cross 1.65 a total of 101 times, of which the null
predicts ~**40.7** are *false* (~40%); at 3.0 they cross 36 times, of which ~**2.9** are expected
false (FDR ~8%).

→ **At the 1.65 bar, nearly half of all "revival/death events" are noise; only crossings above
3.0 clearly exceed the null.** (See the FDR figure: 90% of the right-hand null mass is at zero.)

> Note: of the 36 crossings at 3.0, **Mkt-RF accounts for 24** — that is a *strong* factor
> oscillating at a high level, not "death and resurrection." The one clean die-then-revive case
> (HML 2022) **never clears 3.0 at any window.**

## HML 2021–22 value revival: the sensitivity / false-discovery tradeoff

The literature clearly documents the 2021–22 value comeback. Can we catch it in time?

| Window W | first t>1.65 post-2020 | first t>3.0 | null fake-revivals/factor @3.0 |
|---|---|---|---|
| 104wk (2y) | **2022-04-22** (caught!) | never | **2.57** |
| 156wk | never | never | 1.17 |
| 208wk | never | never | 0.46 |
| 260wk (5y) | never | never | 0.31 |

**Punchline:** HML's real revival is only caught at **W=104 + the 1.65 bar** — exactly the setting
where null factors average ~5 fake revivals and even the 3.0 count inflates to 2.57. **The
sensitivity that catches the real revival is the same sensitivity at which real and fake revivals
look identical.** Lengthen the window / raise the bar enough to control false discovery (W=260,
bar 3.0) and the real HML revival washes out. No free lunch.

## Honest answers to the two questions

**Q1 "Can it find revived factors?"** Mechanically yes — HML's 2022 value revival is detectable.
But only at a *short window + low bar*, where true and false are indistinguishable. Raising to the
Harvey-Liu-Zhu 3.0 bar makes revival credible (FDR ~8%) at the cost of being too slow to catch a
moderate real revival.

**Q2 "Cast a wide net to mine many factors for deaths/revivals?"** That is exactly what this demo
does, and the conclusion is that **a wide net reaps noise**: zero-edge factors fake ~5 deaths and
~5 revivals each at 1.65. Therefore:
1. **The trustworthy statistic is alive-fraction / is-it-alive-now (calibrated at 5%), not the
   count of flips** — counting crossings is intrinsically noisy.
2. **Revival detection requires: a pre-specified, theory-motivated candidate set (not a scan of
   1000), the 3.0 bar, and an accompanying null false-discovery panel.** Drop any of the three and
   "finding a revived factor" is just data mining.

## Consistent with the broader positioning

This extends and *quantifies* the monitor's role:
- **Death detection is trustworthy because of the prior** (you only ask about factors you already
  believed in), not because a single crossing is clean — the null shows single crossings at 1.65
  are very noisy.
- **Revival / birth detection is heavily penalized by multiple testing**; the 3.0 bar controls it
  but is conservative.
- The tool remains a **calibrated monitor + discipline gate, not a signal generator.** Its
  strongest selling point — alive-fraction = 5.2% on realistic-structure nulls, i.e. it does not
  hallucinate marginal edges — holds again.
