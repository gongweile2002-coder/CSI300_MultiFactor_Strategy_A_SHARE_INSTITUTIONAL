
import pandas as pd
import numpy as np
from src.latest_selection import build_latest_selection

def test_latest_uses_latest_announced_not_future_report():
    dates = pd.bdate_range("2024-01-01", periods=150)
    tickers = [f"{i:06d}.SZ" for i in range(1, 13)]
    price_rows = []
    for j,t in enumerate(tickers):
        for i,d in enumerate(dates):
            price_rows.append((d,t,10+j*0.1+i*0.01,10+j*0.1+i*0.01))
    prices = pd.DataFrame(price_rows, columns=["date","ticker","open","close"])

    as_of = dates[-1]
    db = pd.DataFrame({
        "date":[as_of]*len(tickers),
        "ticker":tickers,
        "pe_ttm":[10+i for i in range(len(tickers))],
        "pb":[1+i*0.05 for i in range(len(tickers))],
        "turnover_rate":[1.0]*len(tickers),
        "total_mv":[1000000]*len(tickers),
        "circ_mv":[800000]*len(tickers),
    })

    fund_rows = []
    for t in tickers:
        fund_rows.append((t, as_of-pd.Timedelta(days=50), as_of-pd.Timedelta(days=80), 12, 12, 8, 45))
        # Future announcement must not be used
        fund_rows.append((t, as_of+pd.Timedelta(days=5), as_of-pd.Timedelta(days=10), 99, 99, 99, 1))
    f = pd.DataFrame(fund_rows, columns=[
        "ticker","ann_date","report_date","roe","roe_dt","profit_growth","debt_ratio"
    ])

    mem = pd.DataFrame({
        "index_code":["399300.SZ"]*len(tickers),
        "ticker":tickers,
        "effective_date":[as_of-pd.Timedelta(days=20)]*len(tickers),
        "weight":[100/len(tickers)]*len(tickers)
    })

    top, full = build_latest_selection(
        prices, db, f, mem,
        as_of=as_of,
        top_n=5,
        momentum_lookback_days=20
    )
    assert len(top) == 5
    assert (pd.to_datetime(full["ann_date"]) <= as_of).all()
    assert (full["quality_roe"] < 99).all()
