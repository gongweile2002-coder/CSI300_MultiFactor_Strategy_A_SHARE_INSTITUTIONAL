from __future__ import annotations
import pandas as pd
from .strategy_zoo_elite import build_strategy_sleeves
from .fundamental_extension_institutional import add_cash_quality_factors
from .earnings_events_institutional import add_earnings_event_factor

def institutional_enhance_cross_section(scored_cross_section,cashflow,forecast,express,signal_date,config):
    x=scored_cross_section.copy()
    x=add_cash_quality_factors(x,cashflow,signal_date)
    x=add_earnings_event_factor(x,forecast,express,signal_date,max_age_days=config["institutional"]["event_signal_max_age_days"])
    if "sleeve_core_multifactor" not in x.columns:
        x=build_strategy_sleeves(x)
    fw=config["institutional"]["fundamental_extension_weights"]
    ext=(float(fw["fcf_yield"])*pd.to_numeric(x["fcf_yield_score"],errors="coerce").fillna(0.0)
        +float(fw["cash_quality"])*pd.to_numeric(x["cash_quality_score"],errors="coerce").fillna(0.0)
        +float(fw["earnings_event"])*pd.to_numeric(x["earnings_event_score"],errors="coerce").fillna(0.0))
    x["sleeve_core_multifactor_original"]=x["sleeve_core_multifactor"]
    x["sleeve_core_multifactor"]=0.70*x["sleeve_core_multifactor"]+0.30*ext
    return x
