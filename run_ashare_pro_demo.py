
from pathlib import Path
import json
import numpy as np
import pandas as pd

from src.research_v4 import build_point_in_time_panel_v4
from src.ashare_strategy_pro import build_adaptive_ashare_targets
from src.ashare_execution_rules import preferred_ashare_execution

BASE=Path(__file__).resolve().parent
OUT=BASE/"outputs"/"ashare_pro_demo"
OUT.mkdir(parents=True,exist_ok=True)

def build_demo():
    prices=pd.read_csv(BASE/"data"/"demo_prices.csv",parse_dates=["date"]).sort_values(["ticker","date"])
    prev=prices.groupby("ticker")["close"].shift(1)
    rng=np.random.default_rng(88)
    prices["open"]=prev.fillna(prices["close"])*(1+rng.normal(0,0.002,len(prices)))
    prices["high"]=prices[["open","close"]].max(axis=1)*1.004
    prices["low"]=prices[["open","close"]].min(axis=1)*0.996
    prices["amount"]=prices["close"]*prices["volume"]

    mf=pd.read_csv(BASE/"data"/"demo_fundamentals.csv",parse_dates=["date"])
    db=mf[["date","ticker","pe_ttm","pb"]].copy()
    db["turnover_rate"]=1.2
    db["turnover_rate_f"]=1.4
    db["volume_ratio"]=1.0
    db["dv_ttm"]=2.0
    db["limit_status"]=1

    ticks=sorted(prices["ticker"].unique())
    size={t:500000+50000*i for i,t in enumerate(ticks)}
    db["total_mv"]=db["ticker"].map(size)
    db["circ_mv"]=db["total_mv"]*0.75

    # Add deterministic cross-sectional variation.
    dv_map={t:0.5+(i%8)*0.6 for i,t in enumerate(ticks)}
    vr_map={t:0.8+(i%6)*0.25 for i,t in enumerate(ticks)}
    db["dv_ttm"]=db["ticker"].map(dv_map)
    db["volume_ratio"]=db["ticker"].map(vr_map)
    db["turnover_rate"]=db["ticker"].map({t:0.5+(i%10)*0.35 for i,t in enumerate(ticks)})

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
        [(t,inds[i%5],pd.Timestamp("2010-01-01"),pd.NaT) for i,t in enumerate(ticks)],
        columns=["ticker","l1_name","in_date","out_date"]
    )

    st=pd.DataFrame(columns=["ticker","name","trade_date"])
    bench=prices.groupby("date")["close"].mean().reset_index(name="close")
    return prices,db,f,membership,meta,industry,st,bench

def main():
    cfg=json.loads((BASE/"config.example.json").read_text(encoding="utf-8"))
    prices,db,f,membership,meta,industry,st,bench=build_demo()

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

    sig=pd.to_datetime(panel["signal_date"]).max()
    target,regime,scored,risk=build_adaptive_ashare_targets(
        prices,bench,db,panel,industry,sig,cfg
    )
    target.to_csv(OUT/"adaptive_targets.csv",index=False)
    scored.to_csv(OUT/"scored_cross_section.csv",index=False)
    pd.DataFrame([{
        k:(v if not isinstance(v,dict) else ";".join(f"{kk}:{vv}" for kk,vv in v.items()))
        for k,v in regime.items()
    }]).to_csv(OUT/"regime_summary.csv",index=False)

    mode=preferred_ashare_execution(sig,cfg)
    pd.DataFrame([{
        "signal_date":sig,
        "execution_mode":mode
    }]).to_csv(OUT/"execution_plan.csv",index=False)

    print("A-SHARE PRO demo completed.")
    print("\nRegime:")
    for k,v in regime.items():
        print(f"{k}: {v}")
    print("\nExecution:",mode)
    print("\nTop targets:")
    print(target[["ticker","industry_l1","target_weight","cash_weight","regime","composite_score"]].head(15).round(4).to_string(index=False))

if __name__=="__main__":
    main()
