
from pathlib import Path
import numpy as np
import pandas as pd

from src.research_v4 import (
    build_point_in_time_panel_v4,
    walk_forward_factor_validation,
)
from src.research_v2 import performance_summary, factor_diagnostics
from src.portfolio_v5 import (
    build_optimized_targets,
    run_optimized_backtest_v5,
    portfolio_diagnostics,
)
from src.reporting_v6 import generate_html_report

BASE = Path(__file__).resolve().parent
OUT = BASE/"outputs"/"demo_v6"
OUT.mkdir(parents=True, exist_ok=True)

def build_inputs():
    prices=pd.read_csv(BASE/"data"/"demo_prices.csv",parse_dates=["date"]).sort_values(["ticker","date"])
    prev=prices.groupby("ticker")["close"].shift(1)
    rng=np.random.default_rng(66)
    prices["open"]=prev.fillna(prices["close"])*(1+rng.normal(0,0.002,len(prices)))
    prices["high"]=prices[["open","close"]].max(axis=1)*1.003
    prices["low"]=prices[["open","close"]].min(axis=1)*0.997
    prices["amount"]=prices["close"]*prices["volume"]

    mf=pd.read_csv(BASE/"data"/"demo_fundamentals.csv",parse_dates=["date"])
    db=mf[["date","ticker","pe_ttm","pb"]].copy()
    db["turnover_rate"]=1.2

    ticks=sorted(prices["ticker"].unique())
    size_map={t:500000+50000*i for i,t in enumerate(ticks)}
    db["total_mv"]=db["ticker"].map(size_map)
    db["circ_mv"]=db["total_mv"]*0.75

    f=mf[["date","ticker","roe","debt_ratio","profit_growth"]].rename(columns={"date":"report_date"})
    f["ann_date"]=f["report_date"]+pd.Timedelta(days=45)
    f["roe_dt"]=f["roe"]

    months=pd.date_range(prices["date"].min(),prices["date"].max(),freq="ME")
    membership=pd.DataFrame(
        [(d,t,100/len(ticks)) for d in months for t in ticks],
        columns=["effective_date","ticker","weight"]
    )
    membership["index_code"]="DEMO300"

    meta=pd.DataFrame({
        "ticker":ticks,
        "name":[f"DemoStock{i+1:02d}" for i in range(len(ticks))],
        "market":["Main"]*len(ticks),
        "exchange":["SZSE" if t.endswith("SZ") else "SSE" for t in ticks],
        "list_date":[pd.Timestamp("2010-01-01")]*len(ticks),
    })

    inds=["Bank","Industrial","Consumer","Technology","Healthcare"]
    industry=pd.DataFrame(
        [(t,inds[i%len(inds)],pd.Timestamp("2010-01-01"),pd.NaT) for i,t in enumerate(ticks)],
        columns=["ticker","l1_name","in_date","out_date"]
    )
    st=pd.DataFrame(columns=["ticker","name","trade_date"])

    lim=prices[["date","ticker","open"]].copy()
    lim["up_limit"]=lim["open"]*1.20
    lim["down_limit"]=lim["open"]*0.80

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
    panel.to_csv(OUT/"factor_panel_v6.csv",index=False)

    targets,risk=build_optimized_targets(
        prices,panel,
        risk_lookback_days=126,
        covariance_shrinkage=0.35,
        max_names=20,
        max_stock_weight=0.10,
        min_stock_weight=0.0,
        sector_active_limit=0.12,
        tracking_error_limit_annual=0.20,
        alpha_strength=1.0,
        risk_aversion=6.0,
        turnover_penalty=0.4,
    )
    targets.to_csv(OUT/"optimized_targets_v6.csv",index=False)
    risk.to_csv(OUT/"ex_ante_risk_v6.csv",index=False)
    portfolio_diagnostics(targets).to_csv(OUT/"portfolio_diagnostics_v6.csv",index=False)

    ret,holdings,trades=run_optimized_backtest_v5(
        prices,targets,lim,
        commission_bps=3,
        slippage_bps=2,
        use_price_limits=True
    )
    holdings.to_csv(OUT/"executed_holdings_v6.csv",index=False)
    trades.to_csv(OUT/"trade_costs_v6.csv",index=False)

    p=prices.pivot(index="date",columns="ticker",values="close").sort_index()
    bench=p.pct_change().mean(axis=1).fillna(0).reindex(ret.index).fillna(0)
    active=ret.index[ret.ne(0)]
    if len(active):
        ret=ret.loc[active.min():]
        bench=bench.loc[active.min():]

    perf=pd.DataFrame([performance_summary(ret,bench)])
    perf.to_csv(OUT/"performance_summary_v6.csv",index=False)

    nav=pd.DataFrame({
        "date":ret.index,
        "strategy_return":ret.values,
        "benchmark_return":bench.values,
        "strategy_nav":(1+ret).cumprod().values,
        "benchmark_nav":(1+bench).cumprod().values,
    })
    nav.to_csv(OUT/"nav_timeseries_v6.csv",index=False)

    # Factor diagnostics
    fr, ic_ts, quintile, ic_summary = factor_diagnostics(prices, panel)
    fr.to_csv(OUT/"forward_returns_v6.csv",index=False)
    ic_ts.to_csv(OUT/"factor_ic_timeseries_v6.csv",index=False)
    quintile.to_csv(OUT/"quintile_returns_v6.csv",index=False)
    ic_summary.to_csv(OUT/"ic_summary_v6.csv",index=False)

    # Walk-forward
    folds,test_ic=walk_forward_factor_validation(prices,panel,train_months=12,test_months=4)
    folds.to_csv(OUT/"walk_forward_folds_v6.csv",index=False)
    test_ic.to_csv(OUT/"walk_forward_test_ic_v6.csv",index=False)

    report = generate_html_report(
        OUT,
        OUT/"quant_research_report.html",
        title="CSI300 Multi-Factor Strategy v6 — Demo Research Report"
    )

    print("Demo V6 completed.")
    print(perf.round(4).to_string(index=False))
    print(f"Report: {report}")

if __name__=="__main__":
    main()
