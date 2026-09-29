
from pathlib import Path
import json
import numpy as np
import pandas as pd

from src.research_v4 import build_point_in_time_panel_v4
from src.research_v2 import factor_diagnostics
from src.ashare_strategy_pro import build_adaptive_ashare_targets
from src.ashare_factors import augment_ashare_factors
from src.strategy_zoo_elite import build_strategy_sleeves, combine_sleeves
from src.ml_ranker_elite import strict_walkforward_ml_scores
from src.validation_elite import oos_rank_ic_summary

BASE=Path(__file__).resolve().parent
OUT=BASE/"outputs"/"ashare_elite_demo"
OUT.mkdir(parents=True,exist_ok=True)

def inputs():
    prices=pd.read_csv(BASE/"data"/"demo_prices.csv",parse_dates=["date"]).sort_values(["ticker","date"])
    prev=prices.groupby("ticker")["close"].shift(1)
    rng=np.random.default_rng(123)
    prices["open"]=prev.fillna(prices["close"])*(1+rng.normal(0,0.002,len(prices)))
    prices["amount"]=prices["close"]*prices["volume"]

    mf=pd.read_csv(BASE/"data"/"demo_fundamentals.csv",parse_dates=["date"])
    db=mf[["date","ticker","pe_ttm","pb"]].copy()
    ticks=sorted(prices["ticker"].unique())
    db["turnover_rate"]=db["ticker"].map({t:0.6+(i%10)*0.3 for i,t in enumerate(ticks)})
    db["turnover_rate_f"]=db["turnover_rate"]*1.1
    db["volume_ratio"]=db["ticker"].map({t:0.8+(i%6)*0.2 for i,t in enumerate(ticks)})
    db["dv_ttm"]=db["ticker"].map({t:0.5+(i%8)*0.5 for i,t in enumerate(ticks)})
    db["limit_status"]=1
    size={t:500000+50000*i for i,t in enumerate(ticks)}
    db["total_mv"]=db["ticker"].map(size)
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
        "name":[f"Demo{i:02d}" for i in range(len(ticks))],
        "market":["Main"]*len(ticks),
        "exchange":["SZSE"]*len(ticks),
        "list_date":[pd.Timestamp("2010-01-01")]*len(ticks)
    })
    inds=["Bank","Industrial","Consumer","Technology","Healthcare"]
    industry=pd.DataFrame(
        [(t,inds[i%5],pd.Timestamp("2010-01-01"),pd.NaT) for i,t in enumerate(ticks)],
        columns=["ticker","l1_name","in_date","out_date"]
    )
    st=pd.DataFrame(columns=["ticker","name","trade_date"])
    return prices,db,f,membership,meta,industry,st

def main():
    cfg=json.loads((BASE/"config.example.json").read_text(encoding="utf-8"))
    prices,db,f,membership,meta,industry,st=inputs()

    panel=build_point_in_time_panel_v4(
        prices,db,f,membership,
        metadata=meta,industry_membership=industry,st_status=st,
        start_date="2023-08-01",end_date="2025-12-31",
        momentum_lookback_days=126,
        exclude_st=True,min_listing_trading_days=120,
        liquidity_lookback_days=20,liquidity_min_quantile=0.10,
        neutralize_market_cap=True,neutralize_industry=True
    )

    # Add A-share tactical factors month by month.
    rows=[]
    for sig,g in panel.groupby("signal_date"):
        ax=augment_ashare_factors(g,prices,db,industry,pd.Timestamp(sig),cfg)
        ax=build_strategy_sleeves(ax)
        rows.append(ax)
    rich=pd.concat(rows,ignore_index=True)

    # Attach forward returns from the already-tested PIT framework.
    from src.research_v4 import forward_returns_v4
    fr=forward_returns_v4(prices,panel)
    fwd=fr[["signal_date","ticker","forward_1m_return","forward_end_trade_date"]].copy()
    rich=rich.merge(fwd,on=["signal_date","ticker"],how="left")

    ml=strict_walkforward_ml_scores(
        rich,
        train_months=12,
        min_train_rows=150
    )
    if not ml.empty:
        ml_key=ml[["signal_date","ticker","ml_score"]].copy()
        rich=rich.merge(ml_key,on=["signal_date","ticker"],how="left")
    rich["ml_score"]=rich.get("ml_score",pd.Series(index=rich.index,dtype=float)).fillna(0.0)

    weights=cfg["ashare_elite"]["sleeves"]
    rich=combine_sleeves(rich,weights)
    rich.to_csv(OUT/"strategy_lab_panel.csv",index=False)

    summary=oos_rank_ic_summary(
        rich,
        [
            "sleeve_core_multifactor",
            "sleeve_sector_rotation",
            "sleeve_mean_reversion",
            "ml_score",
            "elite_score",
        ]
    )
    summary.to_csv(OUT/"oos_strategy_summary.csv",index=False)

    latest=rich[rich["signal_date"]==rich["signal_date"].max()].sort_values("elite_score",ascending=False)
    latest.head(30).to_csv(OUT/"latest_elite_ranking.csv",index=False)

    print("A-SHARE ELITE strategy lab completed.")
    print("\nOOS summary:")
    print(summary.round(4).to_string(index=False))
    print("\nLatest top 15:")
    cols=[c for c in ["ticker","industry_l1","elite_score","sleeve_core_multifactor","sleeve_sector_rotation","sleeve_mean_reversion","ml_score","new_buy_block"] if c in latest.columns]
    print(latest[cols].head(15).round(4).to_string(index=False))

if __name__=="__main__":
    main()
