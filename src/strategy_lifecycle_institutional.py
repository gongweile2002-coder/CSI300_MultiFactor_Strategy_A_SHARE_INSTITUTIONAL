from __future__ import annotations
import pandas as pd

def evaluate_strategy_lifecycle(ic_timeseries: pd.DataFrame, config: dict, date_col="signal_date", strategy_col="strategy", ic_col="rank_ic"):
    lc=config["institutional"]["strategy_lifecycle"]; short_n=int(lc["short_window_months"]); long_n=int(lc["long_window_months"]); min_n=int(lc["min_months_for_decision"])
    x=ic_timeseries.copy(); x[date_col]=pd.to_datetime(x[date_col]); rows=[]
    for strategy,g in x.groupby(strategy_col):
        g=g.sort_values(date_col); s=pd.to_numeric(g[ic_col],errors="coerce").dropna()
        if len(s)==0: continue
        sm=s.tail(short_n).mean(); lm=s.tail(long_n).mean(); hit=float((s.tail(short_n)>0).mean())
        if len(s)<min_n: status,mult="WATCH",0.50
        elif sm<=float(lc["pause_ic_threshold"]) and lm<=0: status,mult="PAUSE",0.0
        elif sm<=float(lc["reduce_ic_threshold"]): status,mult="REDUCE",0.5
        elif sm>=float(lc["reactivate_ic_threshold"]): status,mult="ACTIVE",1.0
        else: status,mult="WATCH",0.75
        rows.append({"strategy":strategy,"status":status,"allocation_multiplier":mult,"short_mean_ic":sm,"long_mean_ic":lm,"short_hit_rate":hit,"months":len(s),"last_date":g[date_col].max()})
    return pd.DataFrame(rows)

def dynamic_sleeve_weights(base_weights: dict, lifecycle: pd.DataFrame, config: dict):
    lc=config["institutional"]["strategy_lifecycle"]
    mult=lifecycle.set_index("strategy")["allocation_multiplier"].to_dict() if lifecycle is not None and not lifecycle.empty else {}
    w={k:max(float(v),0.0)*float(mult.get(k,1.0)) for k,v in base_weights.items()}
    w["core_multifactor"]=max(w.get("core_multifactor",0.0),float(lc.get("core_floor",0.30)))
    w["ml_rank"]=min(w.get("ml_rank",0.0),float(lc.get("ml_cap",0.20)))
    total=sum(w.values())
    w={k:v/total for k,v in w.items()} if total>0 else {"core_multifactor":1.0}
    cap=float(lc.get("ml_cap",.20));floor=float(lc.get("core_floor",.30))
    if not 0<=cap<=1 or not 0<=floor<=1: raise ValueError('策略仓位约束非法')
    excess=max(0,w.get("ml_rank",0)-cap)
    w["ml_rank"]=w.get("ml_rank",0)-excess;w["core_multifactor"]=w.get("core_multifactor",0)+excess
    if w["core_multifactor"]<floor:
        other=1-w["core_multifactor"]
        for k in w:
            if k!="core_multifactor":w[k]*=(1-floor)/other
        w["core_multifactor"]=floor
    return w
