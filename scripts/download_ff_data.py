"""Fetch the public Ken French factor library and cache it as a weekly parquet panel.

The repository ships NO market data. This script downloads the 8 Fama-French factor
premia (Mkt-RF, SMB, HML, RMW, CMA, Mom, ST_Rev, LT_Rev) from the public Ken French Data
Library via pandas-datareader and writes data/processed/ff_zoo_weekly.parquet, which the
demos read. The French library is freely redistributable for research; we still keep the
derived file out of git so the repo stays code-only.

Run once before the demos:
    python -m scripts.download_ff_data
"""
from __future__ import annotations
from demos.factor_zoo import load_zoo_weekly


def main():
    wk = load_zoo_weekly()
    print(f"cached weekly factor panel: {len(wk)} weeks  "
          f"{wk.index.min().date()}..{wk.index.max().date()}")
    print(f"factors: {list(wk.columns)}")
    print("-> data/processed/ff_zoo_weekly.parquet ready; run the demos next.")


if __name__ == "__main__":
    main()
