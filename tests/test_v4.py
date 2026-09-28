
import numpy as np
import pandas as pd
from src.research_v4 import neutralize_factor, stamp_duty_rate, walk_forward_factor_validation

def test_stamp_duty_schedule():
    assert stamp_duty_rate("2023-08-27") == 0.001
    assert stamp_duty_rate("2023-08-28") == 0.0005
    assert stamp_duty_rate("2026-09-04") == 0.0005

def test_neutralization_removes_size_and_industry_mean():
    n=90
    rng=np.random.default_rng(1)
    df=pd.DataFrame({
        "total_mv":np.exp(rng.normal(10,1,n)),
        "industry_l1":np.repeat(["A","B","C"],n//3),
    })
    size=np.log(df["total_mv"])
    ind_effect=df["industry_l1"].map({"A":1.0,"B":-0.5,"C":0.3})
    df["factor"]=2.5*size+ind_effect+rng.normal(0,0.1,n)
    res=neutralize_factor(df,"factor",use_market_cap=True,use_industry=True)
    valid=res.notna()
    corr=np.corrcoef(res[valid],size[valid])[0,1]
    assert abs(corr)<1e-8
    means=pd.DataFrame({"r":res,"ind":df["industry_l1"]}).groupby("ind")["r"].mean()
    assert float(means.abs().max())<1e-8

def test_walk_forward_runs():
    dates=pd.date_range("2020-01-31",periods=20,freq="ME")
    ticks=[f"T{i:02d}" for i in range(15)]
    price_dates=pd.bdate_range("2019-01-01","2021-12-31")
    rows=[]
    for j,t in enumerate(ticks):
        for i,d in enumerate(price_dates):
            rows.append((d,t,10+j*0.1+i*0.001*(1+j/20)))
    prices=pd.DataFrame(rows,columns=["date","ticker","close"])

    panel_rows=[]
    rng=np.random.default_rng(2)
    for d in dates:
        for j,t in enumerate(ticks):
            v=rng.normal()
            q=rng.normal()
            m=rng.normal()
            panel_rows.append((d,t,v,q,m))
    panel=pd.DataFrame(panel_rows,columns=[
        "signal_date","ticker","value_score","quality_score","momentum_score"
    ])
    folds,ts=walk_forward_factor_validation(prices,panel,train_months=8,test_months=4)
    assert not folds.empty
    assert {"train_mean_rank_ic","test_mean_rank_ic"}.issubset(folds.columns)
