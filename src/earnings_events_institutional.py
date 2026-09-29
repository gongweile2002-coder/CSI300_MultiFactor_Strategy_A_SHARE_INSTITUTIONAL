from __future__ import annotations
import numpy as np
import pandas as pd
from .research_v4 import winsorize, zscore

TYPE_SCORE = {"预增":1.0,"扭亏":0.9,"略增":0.5,"续盈":0.3,"略减":-0.4,"预减":-0.8,"首亏":-1.0,"续亏":-0.9}

def _z(s):
    s = pd.to_numeric(s, errors="coerce")
    if s.notna().sum() < 3:
        return s.fillna(0.0)
    return zscore(winsorize(s,0.025,0.975)).fillna(0.0)

def latest_forecast_signal(forecast, signal_date, tickers, max_age_days=120):
    if forecast is None or forecast.empty:
        return pd.DataFrame({"ticker":list(map(str,tickers)),"forecast_event_raw":0.0})
    f = forecast.copy()
    f["ann_date"] = pd.to_datetime(f["ann_date"], errors="coerce")
    if "report_date" not in f.columns and "end_date" in f.columns:
        f["report_date"] = f["end_date"]
    f["report_date"] = pd.to_datetime(f["report_date"], errors="coerce")
    f = f[(f["ann_date"] <= pd.Timestamp(signal_date)) & f["ticker"].isin(set(map(str,tickers)))]
    rows=[]
    for ticker,g in f.groupby("ticker"):
        g=g.sort_values(["report_date","ann_date"])
        latest_period=g["report_date"].max()
        h=g[g["report_date"]==latest_period].sort_values("ann_date")
        last=h.iloc[-1]
        lo=pd.to_numeric(pd.Series([last.get("p_change_min")]),errors="coerce").iloc[0]
        hi=pd.to_numeric(pd.Series([last.get("p_change_max")]),errors="coerce").iloc[0]
        midpoint=np.nanmean([lo,hi]) if not (pd.isna(lo) and pd.isna(hi)) else np.nan
        revision=0.0
        if len(h)>=2:
            prev=h.iloc[-2]
            plo=pd.to_numeric(pd.Series([prev.get("p_change_min")]),errors="coerce").iloc[0]
            phi=pd.to_numeric(pd.Series([prev.get("p_change_max")]),errors="coerce").iloc[0]
            pmid=np.nanmean([plo,phi]) if not (pd.isna(plo) and pd.isna(phi)) else np.nan
            if pd.notna(midpoint) and pd.notna(pmid): revision=midpoint-pmid
        age=max(0,(pd.Timestamp(signal_date)-pd.Timestamp(last["ann_date"])).days)
        decay=max(0.0,1.0-age/max(int(max_age_days),1))
        rows.append({"ticker":str(ticker),"forecast_ann_date":last["ann_date"],"forecast_report_date":latest_period,
                     "forecast_growth_mid":midpoint,"forecast_revision":revision,
                     "forecast_type_score":TYPE_SCORE.get(str(last.get("type","")),0.0),"forecast_decay":decay})
    out=pd.DataFrame(rows)
    if out.empty:
        return pd.DataFrame({"ticker":list(map(str,tickers)),"forecast_event_raw":0.0})
    out["forecast_event_raw"]=(0.55*_z(out["forecast_growth_mid"])+0.25*_z(out["forecast_revision"])+0.20*out["forecast_type_score"].fillna(0.0))*out["forecast_decay"]
    return out

def latest_express_signal(express, signal_date, tickers, max_age_days=120):
    if express is None or express.empty:
        return pd.DataFrame({"ticker":list(map(str,tickers)),"express_event_raw":0.0})
    e=express.copy(); e["ann_date"]=pd.to_datetime(e["ann_date"],errors="coerce")
    if "report_date" not in e.columns and "end_date" in e.columns: e["report_date"]=e["end_date"]
    e["report_date"]=pd.to_datetime(e["report_date"],errors="coerce")
    e=e[(e["ann_date"]<=pd.Timestamp(signal_date)) & e["ticker"].isin(set(map(str,tickers)))]
    if e.empty:
        return pd.DataFrame({"ticker":list(map(str,tickers)),"express_event_raw":0.0})
    e=e.sort_values(["ticker","ann_date"]).groupby("ticker").tail(1).copy()
    for c in ["yoy_net_profit","yoy_sales","diluted_roe"]:
        if c not in e.columns: e[c]=np.nan
    age=(pd.Timestamp(signal_date)-e["ann_date"]).dt.days.clip(lower=0)
    decay=(1-age/max(int(max_age_days),1)).clip(lower=0)
    e["express_event_raw"]=(0.55*_z(e["yoy_net_profit"])+0.25*_z(e["yoy_sales"])+0.20*_z(e["diluted_roe"]))*decay
    return e[["ticker","ann_date","report_date","express_event_raw"]].rename(columns={"ann_date":"express_ann_date","report_date":"express_report_date"})

def add_earnings_event_factor(cross_section, forecast, express, signal_date, max_age_days=120):
    x=cross_section.copy(); tickers=x["ticker"].astype(str).tolist()
    f=latest_forecast_signal(forecast,signal_date,tickers,max_age_days)
    e=latest_express_signal(express,signal_date,tickers,max_age_days)
    x=x.merge(f,on="ticker",how="left").merge(e,on="ticker",how="left")
    x["forecast_event_raw"]=pd.to_numeric(x.get("forecast_event_raw"),errors="coerce").fillna(0.0)
    x["express_event_raw"]=pd.to_numeric(x.get("express_event_raw"),errors="coerce").fillna(0.0)
    has_exp=x["express_ann_date"].notna() if "express_ann_date" in x.columns else pd.Series(False,index=x.index)
    x["earnings_event_score"]=np.where(has_exp,0.65*x["express_event_raw"]+0.35*x["forecast_event_raw"],x["forecast_event_raw"])
    x["earnings_event_score"]=_z(x["earnings_event_score"])
    return x
