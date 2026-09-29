
from pathlib import Path
import numpy as np
import pandas as pd

from src.research_v4 import (
    build_point_in_time_panel_v4,
    run_realistic_backtest_v4,
    walk_forward_factor_validation,
)
from src.research_v2 import performance_summary

BASE=Path(__file__).resolve().parent
OUT=BASE/"outputs"/"demo_v4"
OUT.mkdir(parents=True,exist_ok=True)

def build_inputs():
    prices=pd.read_csv(BASE/"data"/"demo_prices.csv",parse_dates=["date"]).sort_values(["ticker","date"])
    prev=prices.groupby("ticker")["close"].shift(1)
    rng=np.random.default_rng(22)
    prices["open"]=prev.fillna(prices["close"])*(1+rng.normal(0,0.002,len(prices)))
    prices["high"]=prices[["open","close"]].max(axis=1)*1.003
    prices["low"]=prices[["open","close"]].min(axis=1)*0.997
    prices["amount"]=prices["close"]*prices["volume"]

    mf=pd.read_csv(BASE/"data"/"demo_fundamentals.csv",parse_dates=["date"])
    db=mf[["date","ticker","pe_ttm","pb"]].copy()
    db["turnover_rate"]=1.0
    # varying market cap so neutralization can be exercised
    size_map={t:500000+50000*i for i,t in enumerate(sorted(prices["ticker"].unique()))}
    db["total_mv"]=db["ticker"].map(size_map)
    db["circ_mv"]=db["total_mv"]*0.75

    f=mf[["date","ticker","roe","debt_ratio","profit_growth"]].rename(columns={"date":"report_date"})
    f["ann_date"]=f["report_date"]+pd.Timedelta(days=45)
    f["roe_dt"]=f["roe"]

    ticks=sorted(prices["ticker"].unique())
    months=pd.date_range(prices["date"].min(),prices["date"].max(),freq="ME")
    membership=pd.DataFrame(
        [(d,t,100/len(ticks)) for d in months for t in ticks],
        columns=["effective_date","ticker","weight"]
    )
    membership["index_code"]="DEMO300"

    meta=pd.DataFrame({
        "ticker":ticks,
        "name":[f"DemoStock{i+1:02d}" for i in range(len(ticks))],
        "market":["主板"]*len(ticks),
        "exchange":["SZSE" if t.endswith("SZ") else "SSE" for t in ticks],
        "list_date":[pd.Timestamp("2010-01-01")]*len(ticks),
    })

    # PIT industry classification
    inds=["Bank","Industrial","Consumer","Technology","Healthcare"]
    ind_rows=[]
    for i,t in enumerate(ticks):
        ind_rows.append((t,inds[i%len(inds)],pd.Timestamp("2010-01-01"),pd.NaT))
    industry=pd.DataFrame(ind_rows,columns=["ticker","l1_name","in_date","out_date"])

    # Mark one demo stock as ST on a few signal dates.
    st_rows=[]
    for d in months[10:14]:
        st_rows.append((ticks[0],"*ST Demo",d))
    st=pd.DataFrame(st_rows,columns=["ticker","name","trade_date"])

    # Price limits: normal wide limits, with one artificial opening limit-up on a date.
    lim=prices[["date","ticker","open"]].copy()
    lim["up_limit"]=lim["open"]*1.20
    lim["down_limit"]=lim["open"]*0.80
    if len(lim)>100:
        lim.loc[lim.index[100],"up_limit"]=lim.loc[lim.index[100],"open"]

    return prices,db,f,membership,meta,industry,st,lim

def main():
    prices,db,f,membership,meta,industry,st,lim=build_inputs()
    panel=build_point_in_time_panel_v4(
        prices,db,f,membership,
        metadata=meta,
        industry_membership=industry,
        st_status=st,
        start_date="2023-08-01",
        end_date="2025-12-31",
        momentum_lookback_days=126,
        exclude_st=True,
        min_listing_trading_days=120,
        liquidity_lookback_days=20,
        liquidity_min_quantile=0.10,
        neutralize_market_cap=True,
        neutralize_industry=True,
    )
    panel.to_csv(OUT/"factor_panel_v4.csv",index=False)

    ret,holdings,trades=run_realistic_backtest_v4(
        prices,panel,lim,top_n=10,
        commission_bps=3,slippage_bps=2,use_price_limits=True
    )
    holdings.to_csv(OUT/"holdings_v4.csv",index=False)
    trades.to_csv(OUT/"trade_costs_v4.csv",index=False)

    p=prices.pivot(index="date",columns="ticker",values="close").sort_index()
    bench=p.pct_change().mean(axis=1).fillna(0).reindex(ret.index).fillna(0)
    active=ret.index[ret.ne(0)]
    if len(active):
        ret=ret.loc[active.min():]
        bench=bench.loc[active.min():]
    perf=pd.DataFrame([performance_summary(ret,bench)])
    perf.to_csv(OUT/"performance_summary_v4.csv",index=False)

    folds,testic=walk_forward_factor_validation(prices,panel,train_months=12,test_months=4)
    folds.to_csv(OUT/"walk_forward_folds.csv",index=False)
    testic.to_csv(OUT/"walk_forward_test_ic.csv",index=False)

    print("Demo V4 completed.")
    print(perf.round(4).to_string(index=False))
    if not folds.empty:
        print("\nWalk-forward:")
        print(folds.round(4).to_string(index=False))

if __name__=="__main__":
    main()
