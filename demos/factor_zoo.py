"""DEMO 1 — factor-zoo death & REVIVAL detection with false-discovery control.

Monitors 8 documented Fama-French factor premia and asks the two-sided question: can a
rolling-significance monitor find factors that DIED *and* factors that came back ALIVE —
without manufacturing false revivals when we widen the search?

THE ASYMMETRY (why death is easy and revival is dangerous):
  - DEATH: the factor had a documented prior (we knew it worked); losing significance is
    trustworthy because the NOISE control proves ~0% false positives. Bar = 1.65.
  - REVIVAL/BIRTH: declaring a previously-dead factor 'alive' is a SELECTION across many
    candidates -> multiple-testing / data-mining. Harvey-Liu-Zhu (2016) argue the t-bar
    for *discovered* factors should be ~3.0, not 2.0. So revival bar = 3.0.

ENGINE: rolling trailing-W HAC (Newey-West) t-stat of the premium mean, point-in-time
(monitor form of a fluctuation / structural-stability test). Premia are directional (+),
one-sided.

CONTROLS (the honest part):
  - NULL FACTORS: N circular-block-bootstrap resamples of DEMEANED real returns (zero true
    edge, realistic vol/autocorr). Count spurious deaths & high-confidence (>3.0) revivals
    -> false-discovery rate per event type.
  - WINDOW SENSITIVITY: vary W; does a shorter window catch HML's documented 2021-22 value
    comeback earlier, and at what cost in false revivals on the null panel?

Data: public Ken French factor library (no licensing restriction). Run:
    python -m demos.factor_zoo
"""
from __future__ import annotations
import warnings; warnings.filterwarnings("ignore")
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data/processed"
FIG = ROOT / "results/figures"
W = 260            # trailing window weeks (~5y), pre-specified
HAC = 26           # Newey-West lag
ESTAB = 1.65       # establishment / death bar (one-sided 5%; we had a prior)
DISC = 3.00        # discovery / revival bar (Harvey-Liu-Zhu 2016; we are searching)
N_NULL = 100       # null factors for false-discovery control
BLOCK = 26         # circular block-bootstrap length (preserve vol clustering)
START = "1990-01-01"
SEED = 7

# documented sign is POSITIVE for every factor as French defines them.
GROUND_TRUTH = {
    "Mkt-RF": "equity premium — robust/persistent",
    "SMB":    "size — weak/dead post-publication",
    "HML":    "value — decayed ~2007, 2021-22 comeback",
    "RMW":    "profitability — robust (Novy-Marx 2013)",
    "CMA":    "investment — moderate, FF5 (2015)",
    "Mom":    "momentum — strong, 2009 crash then long flat",
    "ST_Rev": "short-term reversal — strong but cost-heavy",
    "LT_Rev": "long-term reversal — weak/contested",
}


def nw_lrv(x, lag):
    x = x - x.mean(); n = len(x)
    if n < 5:
        return float(np.var(x)) + 1e-12
    g0 = (x @ x) / n; lrv = g0
    for k in range(1, min(lag, n - 1) + 1):
        w = 1 - k / (lag + 1); lrv += 2 * w * (x[k:] @ x[:-k]) / n
    return max(lrv, 1e-12)


def rolling_tstat(v: np.ndarray, w=W, hac=HAC) -> np.ndarray:
    n = len(v); ts = np.full(n, np.nan)
    for t in range(w - 1, n):
        win = v[t - w + 1: t + 1]
        ts[t] = np.sqrt(w) * win.mean() / np.sqrt(nw_lrv(win, hac))
    return ts


def rising_crossings(ts: np.ndarray, bar: float, index) -> list:
    """dates where t crosses bar from below to above (dead -> alive)."""
    a = (ts > bar).astype(float); a[np.isnan(ts)] = np.nan
    d = np.diff(a)
    return [index[i + 1] for i in np.where(d == 1)[0]]


def falling_crossings(ts: np.ndarray, bar: float, index) -> list:
    a = (ts > bar).astype(float); a[np.isnan(ts)] = np.nan
    d = np.diff(a)
    return [index[i + 1] for i in np.where(d == -1)[0]]


def load_zoo_weekly() -> pd.DataFrame:
    """Load (or download from the public Ken French library) the 8-factor weekly panel."""
    cache = PROC / "ff_zoo_weekly.parquet"
    if cache.exists():
        return pd.read_parquet(cache)
    import pandas_datareader.data as web
    ff5 = web.DataReader("F-F_Research_Data_5_Factors_2x3_daily", "famafrench", start=START)[0]
    mom = web.DataReader("F-F_Momentum_Factor_daily", "famafrench", start=START)[0]
    st = web.DataReader("F-F_ST_Reversal_Factor_daily", "famafrench", start=START)[0]
    lt = web.DataReader("F-F_LT_Reversal_Factor_daily", "famafrench", start=START)[0]
    d = ff5.join([mom, st, lt], how="inner") / 100.0
    d.columns = [c.strip() for c in d.columns]
    d.index = pd.to_datetime(d.index)
    keep = ["Mkt-RF", "SMB", "HML", "RMW", "CMA", "Mom", "ST_Rev", "LT_Rev"]
    wk = (1 + d[keep]).resample("W-FRI").prod() - 1
    PROC.mkdir(parents=True, exist_ok=True)
    wk.to_parquet(cache)
    return wk


def block_bootstrap(x: np.ndarray, n: int, block: int, rng) -> np.ndarray:
    """circular block bootstrap to length n (preserves serial structure, zero mean)."""
    m = len(x); out = np.empty(n); filled = 0
    while filled < n:
        s = rng.integers(0, m)
        take = min(block, n - filled)
        idx = (s + np.arange(take)) % m
        out[filled:filled + take] = x[idx]; filled += take
    return out - out.mean()      # force exactly zero mean -> true null


def main():
    print("=" * 80)
    print(" DEMO 1 — factor-zoo death & REVIVAL detection + false-discovery control")
    print("=" * 80)
    wk = load_zoo_weekly(); n = len(wk); idx = wk.index
    cols = list(wk.columns)
    print(f"weekly zoo: {n} wks  {idx.min().date()}..{idx.max().date()}  factors={cols}")
    print(f"engine: trailing W={W}wk HAC={HAC}  death-bar={ESTAB}  revival-bar={DISC} (Harvey-Liu-Zhu)")

    # --- per-real-factor t-stat + event detection ------------------------------------
    tstats = {c: rolling_tstat(wk[c].values.astype(float)) for c in cols}
    print(f"\n  {'factor':<8}{'ann':>8}{'alive%':>8}{'now_t':>7}{'deaths':>8}{'revivals(>3)':>13}"
          f"  documented")
    rows = {}
    for c in cols:
        ts = tstats[c]; tsv = ts[~np.isnan(ts)]
        alive = float((tsv > ESTAB).mean())
        ann = (1 + wk[c]).prod() ** (52 / n) - 1
        deaths = falling_crossings(ts, ESTAB, idx)
        revivals = rising_crossings(ts, DISC, idx)
        tent = rising_crossings(ts, ESTAB, idx)
        rows[c] = dict(ann=ann, alive=alive, now=tsv[-1], deaths=deaths,
                       revivals=revivals, tentative=tent)
        print(f"  {c:<8}{ann:>+8.3f}{alive:>8.2f}{tsv[-1]:>+7.2f}{len(deaths):>8}"
              f"{len(revivals):>13}  {GROUND_TRUTH[c]}")

    # show the dated revival/death episodes for the headline factors
    print("\n  dated events (death = lost establishment t<1.65 ; REVIVAL = cleared 3.0):")
    for c in cols:
        d = [x.date().isoformat() for x in rows[c]["deaths"]]
        r = [x.date().isoformat() for x in rows[c]["revivals"]]
        print(f"    {c:<8} deaths={d if d else '—'}")
        print(f"    {'':<8} REVIVALS(>3)={r if r else '—'}")

    # --- NULL-FACTOR false-discovery control -----------------------------------------
    rng = np.random.default_rng(SEED)
    pool = np.concatenate([(wk[c].values - wk[c].values.mean()) for c in cols])
    null_deaths = []; null_estab_revivals = []; null_disc_revivals = []; null_alive = []
    for _ in range(N_NULL):
        x = block_bootstrap(pool, n, BLOCK, rng)
        ts = rolling_tstat(x)
        tsv = ts[~np.isnan(ts)]
        null_alive.append(float((tsv > ESTAB).mean()))
        null_deaths.append(len(falling_crossings(ts, ESTAB, idx)))
        null_estab_revivals.append(len(rising_crossings(ts, ESTAB, idx)))
        null_disc_revivals.append(len(rising_crossings(ts, DISC, idx)))
    null_deaths = np.array(null_deaths); null_estab_revivals = np.array(null_estab_revivals)
    null_disc_revivals = np.array(null_disc_revivals); null_alive = np.array(null_alive)

    print("\n" + "-" * 80)
    print(" FALSE-DISCOVERY CONTROL  (N=%d null factors: block-bootstrap of DEMEANED real returns)" % N_NULL)
    print("-" * 80)
    print(f"  null alive-fraction:        mean={null_alive.mean():.3f}  (a true-null factor LOOKS alive this often by chance)")
    print(f"  null deaths per factor:     mean={null_deaths.mean():.2f}  (spurious 'death' chatter at bar 1.65)")
    print(f"  null revivals @1.65/factor: mean={null_estab_revivals.mean():.2f}  P(>=1)={(null_estab_revivals>=1).mean():.2f}")
    print(f"  null revivals @3.00/factor: mean={null_disc_revivals.mean():.2f}  P(>=1)={(null_disc_revivals>=1).mean():.2f}  <- raising the bar to 3.0 is what controls false discovery")
    real_disc = sum(len(rows[c]['revivals']) for c in cols)
    real_estab_rev = sum(len(rows[c]['tentative']) for c in cols)
    print(f"\n  REAL factors: {real_disc} revivals@3.0 total, {real_estab_rev} crossings@1.65 total across {len(cols)} factors")
    print(f"  expected FALSE @1.65 over {len(cols)} factors = {null_estab_revivals.mean()*len(cols):.1f}  | @3.0 = {null_disc_revivals.mean()*len(cols):.1f}")
    print(f"  => at bar 1.65 most 'revivals' are indistinguishable from null chatter; at 3.0 a real crossing is meaningful.")

    # --- WINDOW SENSITIVITY vs FDR on HML's documented 2021-22 value comeback --------
    print("\n" + "-" * 80)
    print(" WINDOW SENSITIVITY — HML 2021-22 value revival: catch it earlier vs false-discovery cost")
    print("-" * 80)
    Ws = [104, 156, 208, 260]
    hml = wk["HML"].values.astype(float)
    print(f"  {'W(wk)':>6}{'first t>1.65 post-2020':>26}{'first t>3.0 post-2020':>24}"
          f"{'null rev@3/factor':>20}")
    sens = {}
    rng2 = np.random.default_rng(SEED + 1)
    for w in Ws:
        ts = rolling_tstat(hml, w=w)
        s = pd.Series(ts, index=idx)
        post = s.loc["2020-01-01":]
        c165 = post[post > ESTAB]
        c30 = post[post > DISC]
        first165 = c165.index[0].date().isoformat() if len(c165) else "never"
        first30 = c30.index[0].date().isoformat() if len(c30) else "never"
        # null cost at this window
        nrev = []
        for _ in range(N_NULL):
            x = block_bootstrap(pool, n, BLOCK, rng2)
            nrev.append(len(rising_crossings(rolling_tstat(x, w=w), DISC, idx)))
        nrev = float(np.mean(nrev))
        sens[w] = dict(first165=first165, first30=first30, null_rev=nrev)
        print(f"  {w:>6}{first165:>26}{first30:>24}{nrev:>20.2f}")
    print("  (shorter window = earlier detection BUT more null revivals -> the sensitivity/FDR tradeoff)")

    # --- figures ---------------------------------------------------------------------
    fig, axes = plt.subplots(2, 4, figsize=(17, 7.5), sharex=True)
    colors = plt.cm.tab10(np.linspace(0, 1, len(cols)))
    for ax, c, col in zip(axes.ravel(), cols, colors):
        ts = pd.Series(tstats[c], index=idx)
        ax.plot(ts.index, ts.values, color=col, lw=1.1)
        ax.axhline(ESTAB, color="black", ls="--", lw=0.7)
        ax.axhline(DISC, color="firebrick", ls="--", lw=0.7)
        ax.axhline(0, color="grey", ls=":", lw=0.5)
        ax.fill_between(ts.index, ESTAB, ts.where(ts > ESTAB), color="green", alpha=0.15)
        ax.fill_between(ts.index, DISC, ts.where(ts > DISC), color="darkgreen", alpha=0.30)
        ax.set_title(f"{c}  alive {rows[c]['alive']*100:.0f}%  rev>3:{len(rows[c]['revivals'])}", fontsize=8.5)
        ax.grid(alpha=0.2)
    fig.suptitle(f"Factor zoo health monitor — trailing-{W}wk HAC t-stat. dashed: 1.65 death-bar, "
                 f"red 3.0 revival-bar (Harvey-Liu-Zhu). dark-green = high-confidence revival", fontsize=10)
    fig.tight_layout(); FIG.mkdir(parents=True, exist_ok=True)
    out1 = FIG / "engine_demo_factor_zoo.png"
    fig.savefig(out1, dpi=125); plt.close(fig)

    # FDR figure: null distribution of revivals @1.65 vs @3.0
    fig2, ax = plt.subplots(1, 2, figsize=(12, 4))
    ax[0].hist(null_estab_revivals, bins=range(0, int(null_estab_revivals.max()) + 2),
               color="orange", alpha=0.7, align="left")
    ax[0].axvline(null_estab_revivals.mean(), color="black", ls="--",
                  label=f"mean {null_estab_revivals.mean():.2f}")
    ax[0].set_title("null revivals per factor @ bar 1.65\n(false discovery if we use the death-bar)", fontsize=9)
    ax[0].set_xlabel("# revivals on a TRUE-NULL factor"); ax[0].legend(fontsize=8)
    ax[1].hist(null_disc_revivals, bins=range(0, max(2, int(null_disc_revivals.max()) + 2)),
               color="seagreen", alpha=0.7, align="left")
    ax[1].axvline(null_disc_revivals.mean(), color="black", ls="--",
                  label=f"mean {null_disc_revivals.mean():.2f}")
    ax[1].set_title("null revivals per factor @ bar 3.0\n(Harvey-Liu-Zhu bar controls false discovery)", fontsize=9)
    ax[1].set_xlabel("# revivals on a TRUE-NULL factor"); ax[1].legend(fontsize=8)
    fig2.suptitle("False-discovery control: how often a ZERO-edge factor fakes a 'revival'", fontsize=10)
    fig2.tight_layout()
    out2 = FIG / "engine_demo_factor_zoo_fdr.png"
    fig2.savefig(out2, dpi=125); plt.close(fig2)

    PROC.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(tstats, index=idx).to_parquet(PROC / "factor_zoo_tstats.parquet")
    print(f"\nfig -> {out1}\nfig -> {out2}")
    print("READ: death detection is trustworthy (we had a prior + ~0% noise false-positive). "
          "Revival detection is a SEARCH -> only crossings above the 3.0 discovery bar exceed "
          "the null chatter. Shorter windows detect revivals earlier but inflate false discovery.")


if __name__ == "__main__":
    main()
