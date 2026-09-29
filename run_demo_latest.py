
from pathlib import Path
import numpy as np
import pandas as pd

from src.latest_selection import build_latest_selection, build_freshness_summary

BASE = Path(__file__).resolve().parent
OUT = BASE/"outputs"/"demo_latest"
OUT.mkdir(parents=True, exist_ok=True)

def build_demo_inputs():
    prices = pd.read_csv(BASE/"data"/"demo_prices.csv", parse_dates=["date"]).sort_values(["ticker","date"])
    prev_close = prices.groupby("ticker")["close"].shift(1)
    rng = np.random.default_rng(11)
    prices["open"] = prev_close.fillna(prices["close"]) * (1+rng.normal(0,0.002,len(prices)))

    mf = pd.read_csv(BASE/"data"/"demo_fundamentals.csv", parse_dates=["date"])
    db = mf[["date","ticker","pe_ttm","pb"]].copy()
    db["turnover_rate"] = 1.2
    db["total_mv"] = 1_000_000
    db["circ_mv"] = 750_000

    f = mf[["date","ticker","roe","debt_ratio","profit_growth"]].rename(columns={"date":"report_date"})
    f["ann_date"] = f["report_date"] + pd.Timedelta(days=45)
    f["roe_dt"] = f["roe"]

    ticks = sorted(prices["ticker"].unique())
    month_ends = pd.date_range(prices["date"].min(), prices["date"].max(), freq="ME")
    membership = pd.DataFrame(
        [(d,t,100/len(ticks)) for d in month_ends for t in ticks],
        columns=["effective_date","ticker","weight"]
    )
    membership["index_code"] = "DEMO300"

    meta = pd.DataFrame({
        "ticker":ticks,
        "name":[f"DemoStock{i+1:02d}" for i in range(len(ticks))],
        "industry":["DemoIndustry"]*len(ticks)
    })
    return prices, db, f, membership, meta

def main():
    prices, db, f, membership, meta = build_demo_inputs()
    as_of = prices["date"].max()
    top, full = build_latest_selection(
        prices, db, f, membership, meta,
        as_of=as_of,
        top_n=10,
        momentum_lookback_days=126
    )
    tag = pd.Timestamp(as_of).strftime("%Y-%m-%d")
    top.to_csv(OUT/f"latest_selection_{tag}.csv", index=False, encoding="utf-8-sig")
    full.to_csv(OUT/f"latest_cross_section_{tag}.csv", index=False, encoding="utf-8-sig")
    build_freshness_summary(full).to_csv(OUT/f"data_freshness_{tag}.csv", index=False, encoding="utf-8-sig")
    print(top[["rank","ticker","name","report_period","ann_date","composite_score"]].to_string(index=False))

if __name__ == "__main__":
    main()
