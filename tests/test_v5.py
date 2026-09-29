
import numpy as np
import pandas as pd
from src.portfolio_v5 import optimize_portfolio, portfolio_risk_stats, estimate_covariance

def test_optimizer_constraints_basic():
    n=12
    tickers=[f"T{i:02d}" for i in range(n)]
    g=pd.DataFrame({
        "ticker":tickers,
        "composite_score":np.linspace(2,-1,n),
        "weight":[100/n]*n,
        "industry_l1":["A"]*4+["B"]*4+["C"]*4,
    })
    rng=np.random.default_rng(5)
    A=rng.normal(size=(n,n))
    cov=pd.DataFrame((A@A.T)/1e5,index=tickers,columns=tickers)
    out,stats=optimize_portfolio(
        g,cov,max_names=12,max_stock_weight=0.15,
        sector_active_limit=0.20,tracking_error_limit_annual=0.50,
        risk_aversion=3.0,turnover_penalty=0.0
    )
    assert abs(out["target_weight"].sum()-1)<1e-8
    assert out["target_weight"].max() <= 0.150001
    assert (out["target_weight"] >= -1e-10).all()
    assert "ex_ante_vol_annual" in stats

def test_portfolio_risk_stats_nonnegative():
    tickers=["A","B","C"]
    cov=pd.DataFrame(
        [[0.0004,0.0001,0.00005],[0.0001,0.0003,0.00002],[0.00005,0.00002,0.0002]],
        index=tickers,columns=tickers
    )
    w=pd.Series([0.4,0.3,0.3],index=tickers)
    b=pd.Series([1/3]*3,index=tickers)
    s=portfolio_risk_stats(w,cov,b)
    assert s["ex_ante_vol_annual"] > 0
    assert s["ex_ante_tracking_error_annual"] >= 0

def test_covariance_shrinkage_runs():
    dates=pd.bdate_range("2024-01-01",periods=140)
    rows=[]
    rng=np.random.default_rng(7)
    for j,t in enumerate(["A","B","C","D"]):
        px=10*np.exp(np.cumsum(rng.normal(0,0.01,len(dates))))
        for d,p in zip(dates,px):
            rows.append((d,t,p))
    prices=pd.DataFrame(rows,columns=["date","ticker","close"])
    cov,end=estimate_covariance(prices,["A","B","C","D"],dates[-1],lookback_days=126,shrinkage=0.4)
    assert cov.shape==(4,4)
    assert end==dates[-1]
    eig=np.linalg.eigvalsh(cov.values)
    assert eig.min()>-1e-10
