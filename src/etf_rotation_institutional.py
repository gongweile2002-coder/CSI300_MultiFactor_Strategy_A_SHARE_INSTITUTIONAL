from __future__ import annotations
import numpy as np
import pandas as pd

def _z(s):
    s=pd.to_numeric(s,errors="coerce"); sd=s.std(ddof=0)
    if not np.isfinite(sd) or sd==0: return pd.Series(0.0,index=s.index)
    return (s-s.mean())/sd

def etf_rotation_scores(etf_daily: pd.DataFrame, as_of, min_history_days=130, top_k=3):
    x=etf_daily.copy(); x["date"]=pd.to_datetime(x["date"],errors="coerce"); x=x[x["date"]<=pd.Timestamp(as_of)]
    close=x.pivot(index="date",columns="ticker",values="close").sort_index(); rows=[]
    for ticker in close.columns:
        s=close[ticker].dropna()
        if len(s)<int(min_history_days): continue
        r20=s.iloc[-1]/s.iloc[-21]-1; r60=s.iloc[-1]/s.iloc[-61]-1; r120=s.iloc[-1]/s.iloc[-121]-1
        vol20=s.pct_change().tail(20).std(ddof=0)*np.sqrt(252); ma60=s.tail(60).mean(); trend=float(s.iloc[-1]>ma60)
        g=x[x["ticker"]==ticker].sort_values("date").tail(20)
        amount20=pd.to_numeric(g.get("amount"),errors="coerce").mean() if "amount" in g.columns else np.nan
        rows.append({"ticker":ticker,"r20":r20,"r60":r60,"r120":r120,"vol20":vol20,"trend_ok":trend,"amount20":amount20})
    out=pd.DataFrame(rows)
    if out.empty: return out
    out["score"]=0.35*_z(out["r20"])+0.30*_z(out["r60"])+0.20*_z(out["r120"])+0.15*_z(-out["vol20"])
    out=out[(out["trend_ok"]>0)&(out["vol20"]>0)&(out["amount20"]>0)].copy()
    if out.empty:
        out["target_weight_etf_sleeve"]=pd.Series(dtype=float)
        return out
    selected=out.sort_values("score",ascending=False).head(int(top_k)).copy()
    inv=1/pd.to_numeric(selected["vol20"],errors="coerce").replace(0,np.nan); inv=inv.fillna(inv.median()).fillna(1.0)
    selected["target_weight_etf_sleeve"]=inv/inv.sum()
    return selected
