"""HEAD-TO-HEAD: validity engine v1 (fixed window / fixed bar / bootstrap null) vs
v2 (ADWIN adaptive window / Chu-Stinchcombe-White boundary / Deflated-Sharpe + PBO).

The honest question is NOT "is v2 fancier" (it is) but "does swapping in the mature
components CHANGE ANY CONCLUSION we already reached, and where does each tool actually
earn its keep?"  Three comparisons, each on the same data:

  1. WINDOW   : v1 fixed W=260 trailing t-stat  vs  v2 ADWIN-adaptive-window t-stat.
                Does ADWIN's data-driven window catch decay/revival faster, or does the
                low signal-to-noise of weekly factor edges keep it pinned to a long window?
  2. BOUNDARY : v1 'ever exceeded bar 1.65' (we measured P(false)~0.60)  vs  v2 CSW
                path-controlled boundary (calibrated to 5% path-wise false alarm).
                Headline metric: false-alarm on a null-factor panel, and POWER on real factors.
  3. FDR      : v1 circular-block-bootstrap null alive-fraction (~0.052)  vs  v2 Deflated
                Sharpe Ratio + Probability of Backtest Overfitting on the factor selection.

Run: python -m demos.v1_vs_v2_headtohead
"""
from __future__ import annotations
import warnings; warnings.filterwarnings("ignore")
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib.pyplot as plt

from engine_v2 import nw_lrv, rolling_tstat_fixed
from engine_v2.adaptive_window import adaptive_tstat, adwin_changepoints
from engine_v2.csw_monitor import csw_monitor, fixed_bar_ever
from engine_v2.dsr_pbo import deflated_sharpe_ratio, pbo_cscv, sharpe
from demos.factor_zoo import load_zoo_weekly, block_bootstrap, GROUND_TRUTH

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "results/figures"
W = 260; HAC = 26; ESTAB = 1.65; DISC = 3.00
N_NULL = 60; SEED = 7
MC = 800            # MC paths for CSW boundary calibration


# =====================================================================================
def part1_window(wk: pd.DataFrame):
    print("\n" + "=" * 96)
    print(" 1) WINDOW — v1 fixed W=260 trailing t-stat  vs  v2 ADWIN adaptive-window t-stat")
    print("=" * 96)
    cols = list(wk.columns)
    print(f"  {'factor':<8}{'v1 now_t':>9}{'v1 maxt':>8}{'v2 now_t':>9}{'v2 win_now':>11}"
          f"{'ADWIN cuts':>11}   documented")
    out = {}
    for c in cols:
        v = wk[c].values.astype(float)
        t1 = rolling_tstat_fixed(v, W, HAC)
        t2, width = adaptive_tstat(v, hac=HAC, min_w=40)
        cps = adwin_changepoints(v)
        out[c] = dict(t1=t1, t2=t2, width=width, cps=cps)
        m1 = np.nanmax(t1); n1 = t1[~np.isnan(t1)][-1]
        n2 = t2[~np.isnan(t2)][-1]
        print(f"  {c:<8}{n1:>+9.2f}{m1:>+8.2f}{n2:>+9.2f}{int(width[-1]):>11}{len(cps):>11}"
              f"   {GROUND_TRUTH[c]}")
    print("\n  read: if 'ADWIN cuts'=0 and win_now is large, the adaptive window collapsed to a")
    print("        near-global window -> on low-SNR weekly factor edges ADWIN rarely fires, so")
    print("        it behaves like (and is slower than) a deliberately-short fixed window.")
    return out


# =====================================================================================
def _make_controls(n, idx, rng):
    """synthetic NOISE (no edge) and STABLE (constant modest edge) for calibration."""
    noise = pd.Series(rng.normal(0, 0.02, n), index=idx)
    stable = pd.Series(rng.normal(0.0015, 0.02, n), index=idx)   # ~0.075 weekly Sharpe
    return {"NOISE(ctrl)": noise, "STABLE(ctrl)": stable}


def part2_boundary(wk: pd.DataFrame):
    print("\n" + "=" * 96)
    print(" 2) BOUNDARY — v1 'ever t>1.65' (P_false~0.60)  vs  v2 CSW path-controlled (5%)")
    print("=" * 96)
    cols = list(wk.columns); n = len(wk); idx = wk.index
    rng = np.random.default_rng(SEED)
    m = n // 3   # CSW training (first third assumed informative baseline)

    series = {c: wk[c] for c in cols}
    series.update(_make_controls(n, idx, rng))

    print(f"  {'series':<13}{'v1 ever>1.65':>13}{'v1 ever>3.0':>12}{'v2 CSW fires':>13}"
          f"{'CSW a':>7}{'CSW detect':>13}")
    res = {}
    for name, s in series.items():
        v = s.values.astype(float)
        t1 = rolling_tstat_fixed(v, W, HAC)
        ev165 = fixed_bar_ever(t1, ESTAB)
        ev30 = fixed_bar_ever(t1, DISC)
        c = csw_monitor(v, m=m, alpha=0.05, hac=HAC, n_mc=MC, seed=SEED)
        det = (idx[c["detect_pos"] - 1].date().isoformat() if c["detect_pos"] else "—")
        res[name] = dict(ev165=ev165, ev30=ev30, csw=c["crossed"], a=c["a"], det=det)
        print(f"  {name:<13}{str(ev165):>13}{str(ev30):>12}{str(c['crossed']):>13}"
              f"{c['a']:>7.2f}{det:>13}")

    # NULL-FACTOR PANEL: the calibration test. demean real returns -> zero edge.
    print(f"\n  NULL-FACTOR PANEL ({N_NULL} circular-block-bootstrap zero-edge factors):")
    pool = wk["Mkt-RF"].values.astype(float)
    fa_v1_165 = fa_v1_30 = fa_csw = 0
    for i in range(N_NULL):
        z = block_bootstrap(pool, n, HAC, rng)
        t1 = rolling_tstat_fixed(z, W, HAC)
        fa_v1_165 += int(fixed_bar_ever(t1, ESTAB))
        fa_v1_30 += int(fixed_bar_ever(t1, DISC))
        c = csw_monitor(z, m=m, alpha=0.05, hac=HAC, n_mc=400, seed=1000 + i)
        fa_csw += int(c["crossed"])
    print(f"    P(false alarm)  v1 ever>1.65 = {fa_v1_165/N_NULL:.3f}   "
          f"v1 ever>3.0 = {fa_v1_30/N_NULL:.3f}   v2 CSW = {fa_csw/N_NULL:.3f}  (target 0.05)")
    print("    => CSW delivers the 5% path-wise control that v1's repeated 1.65 bar cannot;")
    print("       v1's 3.0 bar is a hand-tuned patch for the same disease CSW cures by design.")
    res["_null"] = dict(v1_165=fa_v1_165/N_NULL, v1_30=fa_v1_30/N_NULL, csw=fa_csw/N_NULL)
    return res


# =====================================================================================
def part3_fdr(wk: pd.DataFrame):
    print("\n" + "=" * 96)
    print(" 3) FDR — v1 bootstrap-null alive-fraction  vs  v2 Deflated Sharpe + PBO/CSCV")
    print("=" * 96)
    cols = list(wk.columns)
    R = wk[cols].values                                  # T x N weekly returns
    T, N = R.shape

    srs = np.array([sharpe(R[:, j]) for j in range(N)])
    best = int(np.argmax(srs))
    d = deflated_sharpe_ratio(R)
    pbo = pbo_cscv(R, S=14)
    print(f"  panel: {N} real FF factors, T={T} weeks")
    print(f"  per-factor weekly Sharpe: " + "  ".join(f"{c}={srs[i]:+.3f}" for i, c in enumerate(cols)))
    print(f"\n  selected (IS-best Sharpe) = {cols[best]}  (weekly SR {d['sr']:+.3f}, "
          f"ann ~{d['sr']*np.sqrt(52):+.2f})")
    print(f"  deflated benchmark SR0 (E[max] of {N} zero-skill trials) = {d['sr0_deflated']:.3f}")
    print(f"  PSR vs 0      = {d['PSR_vs0']:.3f}   (naive: ignores multiple testing)")
    print(f"  DSR vs SR0    = {d['DSR']:.3f}   (deflated: P best factor has TRUE SR>0 after {N} trials)")
    print(f"  PBO (CSCV)    = {pbo['PBO']:.3f}   over {pbo['n_splits']} splits "
          f"(P that IS-best is below OOS median)")

    # WIDEN THE SEARCH: add null factors as decoy 'strategies' -> selection bias bites
    rng = np.random.default_rng(SEED)
    pool = wk["Mkt-RF"].values.astype(float)
    decoys = np.column_stack([block_bootstrap(pool, T, HAC, rng) for _ in range(50)])
    Rw = np.column_stack([R, decoys])                    # 8 real + 50 null = 58 trials
    dw = deflated_sharpe_ratio(Rw)
    pbow = pbo_cscv(Rw, S=14)
    wins = "real" if dw["selected"] < N else "a NULL decoy"
    print(f"\n  widen search to {Rw.shape[1]} trials (8 real + 50 zero-edge decoys):")
    print(f"    IS-best is now {wins} (col {dw['selected']});  SR0 deflates to {dw['sr0_deflated']:.3f}")
    print(f"    DSR = {dw['DSR']:.3f}   PBO = {pbow['PBO']:.3f}")
    print("    => DSR/PBO quantify selection bias in one number as the trial count grows — the")
    print("       same lesson as v1's null panel (counting/searching reaps noise), now field-standard.")
    return dict(narrow=dict(d=d, pbo=pbo), wide=dict(d=dw, pbo=pbow), cols=cols, srs=srs)


# =====================================================================================
def make_figure(p1, p2, p3, wk):
    cols = list(wk.columns); idx = wk.index
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    # (a) window: HML v1 vs v2 t-stat
    c = "HML"
    ax = axes[0, 0]
    ax.plot(idx, p1[c]["t1"], label="v1 fixed W=260", lw=1.1)
    ax.plot(idx, p1[c]["t2"], label="v2 ADWIN window", lw=1.1)
    ax.axhline(ESTAB, color="k", ls="--", lw=0.6); ax.axhline(DISC, color="firebrick", ls="--", lw=0.6)
    ax.set_title(f"WINDOW: {c} t-stat — v1 fixed vs v2 ADWIN", fontsize=10); ax.legend(fontsize=8); ax.grid(alpha=0.2)
    # (b) window width path
    ax = axes[0, 1]
    for c in ["Mkt-RF", "HML", "Mom"]:
        ax.plot(idx, p1[c]["width"], label=c, lw=1.0)
    ax.axhline(W, color="grey", ls="--", lw=0.8, label="v1 fixed 260")
    ax.set_title("ADWIN chosen window width (weeks)", fontsize=10); ax.legend(fontsize=8); ax.grid(alpha=0.2)
    # (c) boundary false-alarm bar chart
    ax = axes[1, 0]
    nf = p2["_null"]
    bars = ["v1 ever>1.65", "v1 ever>3.0", "v2 CSW"]
    vals = [nf["v1_165"], nf["v1_30"], nf["csw"]]
    ax.bar(bars, vals, color=["#c44", "#e89", "#4a4"])
    ax.axhline(0.05, color="k", ls="--", lw=0.8, label="5% target")
    for i, v in enumerate(vals):
        ax.text(i, v + 0.01, f"{v:.2f}", ha="center", fontsize=9)
    ax.set_title("BOUNDARY: false-alarm on null-factor panel", fontsize=10); ax.legend(fontsize=8)
    ax.set_ylim(0, 0.75); ax.grid(alpha=0.2, axis="y")
    # (d) FDR: PSR vs DSR as search widens
    ax = axes[1, 1]
    dn, dw = p3["narrow"]["d"], p3["wide"]["d"]
    labels = ["PSR vs 0", f"DSR ({dn['n_trials']} trials)", f"DSR ({dw['n_trials']} trials)"]
    vals = [dn["PSR_vs0"], dn["DSR"], dw["DSR"]]
    ax.bar(labels, vals, color=["#48c", "#28a", "#048"])
    for i, v in enumerate(vals):
        ax.text(i, v + 0.01, f"{v:.2f}", ha="center", fontsize=9)
    ax.set_title("FDR: naive PSR deflates as trial count grows", fontsize=10)
    ax.set_ylim(0, 1.05); ax.grid(alpha=0.2, axis="y")
    fig.suptitle("Validity engine v1 vs v2 — ADWIN window / CSW boundary / DSR+PBO", fontsize=12)
    fig.tight_layout()
    FIG.mkdir(parents=True, exist_ok=True)
    out = FIG / "engine_v2_vs_v1.png"
    fig.savefig(out, dpi=125); plt.close(fig)
    print(f"\n  fig -> {out}")


def main():
    print("#" * 96)
    print(" ENGINE v1 vs v2 HEAD-TO-HEAD — do the mature components change any conclusion?")
    print("#" * 96)
    wk = load_zoo_weekly()
    print(f" weekly factor zoo: {len(wk)} wks  {wk.index.min().date()}..{wk.index.max().date()}  {list(wk.columns)}")
    p1 = part1_window(wk)
    p2 = part2_boundary(wk)
    p3 = part3_fdr(wk)
    make_figure(p1, p2, p3, wk)
    print("\n" + "#" * 96)
    print(" VERDICT: see results/FINDINGS_v1_vs_v2.md")
    print("#" * 96)


if __name__ == "__main__":
    main()
