
from pathlib import Path
import numpy as np
import pandas as pd

from src.research_v2 import (
    build_point_in_time_panel,
    run_next_open_backtest,
    factor_diagnostics,
    performance_summary,
)

BASE = Path(__file__).resolve().parent
OUT = BASE/"outputs"/"demo_v2"
OUT.mkdir(parents=True, exist_ok=True)

def build_demo_inputs():
    prices = pd.read_csv(BASE/"data"/"demo_prices.csv", parse_dates=["date"])
    # Create plausible open from close without using future data; only for pipeline demonstration.
    prices = prices.sort_values(["ticker","date"])
    prev_close = prices.groupby("ticker")["close"].shift(1)
    rng = np.random.default_rng(7)
    gap = rng.normal(0, 0.002, len(prices))
    prices["open"] = prev_close.fillna(prices["close"]) * (1+gap)
    prices["high"] = prices[["open","close"]].max(axis=1) * 1.003
    prices["low"] = prices[["open","close"]].min(axis=1) * 0.997
    prices["amount"] = prices["close"] * prices["volume"]

    # Demo daily basic from monthly fundamentals, expanded only on month-end dates.
    mf = pd.read_csv(BASE/"data"/"demo_fundamentals.csv", parse_dates=["date"])
    db = mf[["date","ticker","pe_ttm","pb"]].copy()
    db["turnover_rate"] = 1.0
    db["total_mv"] = 1_000_000.0
    db["circ_mv"] = 700_000.0

    # Point-in-time accounting: announce roughly 45 days after report date.
    f = mf[["date","ticker","roe","debt_ratio","profit_growth"]].copy()
    f = f.rename(columns={"date":"report_date"})
    f["ann_date"] = f["report_date"] + pd.Timedelta(days=45)
    f["roe_dt"] = f["roe"]

    # Membership: same 30-stock demo universe each month.
    tickers = sorted(prices["ticker"].unique())
    month_ends = pd.date_range(prices["date"].min(), prices["date"].max(), freq="ME")
    mem = pd.DataFrame(
        [(d,t,100/len(tickers)) for d in month_ends for t in tickers],
        columns=["effective_date","ticker","weight"]
    )
    mem["index_code"] = "DEMO300"
    return prices, db, f, mem

def main():
    prices, db, f, mem = build_demo_inputs()
    panel = build_point_in_time_panel(
        prices, db, f, mem,
        start_date="2023-08-01",
        end_date="2025-12-31",
        momentum_lookback_days=126
    )
    panel.to_csv(OUT/"factor_panel.csv", index=False)

    ret, holdings, turnover = run_next_open_backtest(
        prices, panel, top_n=10, transaction_cost_bps=10
    )
    holdings.to_csv(OUT/"holdings.csv", index=False)
    turnover.to_csv(OUT/"turnover.csv", index=False)

    # Equal-weight demo benchmark
    p = prices.pivot(index="date", columns="ticker", values="close").sort_index()
    bench = p.pct_change().mean(axis=1).fillna(0).reindex(ret.index).fillna(0)

    active = ret.index[ret.ne(0)]
    if len(active):
        ret = ret.loc[active.min():]
        bench = bench.loc[active.min():]

    summary = pd.DataFrame([performance_summary(ret, bench)])
    summary.to_csv(OUT/"performance_summary.csv", index=False)

    fr, ic_ts, quint, ic_summary = factor_diagnostics(prices, panel)
    fr.to_csv(OUT/"forward_returns.csv", index=False)
    ic_ts.to_csv(OUT/"factor_ic_timeseries.csv", index=False)
    quint.to_csv(OUT/"quintile_returns.csv", index=False)
    ic_summary.to_csv(OUT/"ic_summary.csv", index=False)

    print("Demo-v2 pipeline completed.")
    print(summary.round(4).to_string(index=False))
    if not ic_summary.empty:
        print("\nIC Summary")
        print(ic_summary.round(4).to_string(index=False))

if __name__ == "__main__":
    main()
