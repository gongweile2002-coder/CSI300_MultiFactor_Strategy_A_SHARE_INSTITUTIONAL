
import pandas as pd
import numpy as np
from src.ashare_regime import compute_market_regime
from src.ashare_execution_rules import preferred_ashare_execution
from src.ashare_factors import compute_price_factors

def _prices():
    dates=pd.bdate_range("2025-01-01",periods=160)
    rows=[]
    for j,t in enumerate(["A","B","C","D","E","F","G","H","I","J"]):
        px=10*np.exp(np.cumsum(np.full(len(dates),0.0005+j*0.00001)))
        for d,p in zip(dates,px):
            rows.append((d,t,p,p*100000))
    return pd.DataFrame(rows,columns=["date","ticker","close","amount"])

def test_regime_engine_runs():
    p=_prices()
    b=p.groupby("date")["close"].mean().reset_index()
    r=compute_market_regime(p,b,p["date"].max(),p["ticker"].unique())
    assert r["regime"] in {"TREND_UP","HIGH_ROTATION","RANGE","RISK_OFF","PANIC"}
    assert "breadth_above_ma20" in r

def test_execution_rule_after_2026():
    cfg={"ashare_pro":{
        "post_close_fixed_price_from":"2026-07-06",
        "preferred_execution_after_2026_07_06":"next_day_post_close_fixed",
        "legacy_execution":"next_open"
    }}
    assert preferred_ashare_execution("2026-09-04",cfg)=="next_day_post_close_fixed"
    assert preferred_ashare_execution("2025-09-04",cfg)=="next_open"

def test_skip_month_momentum():
    p=_prices()
    out=compute_price_factors(
        p,p["date"].max(),p["ticker"].unique(),
        momentum_long_days=126,momentum_skip_days=20,
        short_reversal_days=5,low_vol_days=20
    )
    assert len(out)==10
    assert out["momentum_skip_raw"].notna().all()
    assert out["low_vol_raw"].notna().all()
