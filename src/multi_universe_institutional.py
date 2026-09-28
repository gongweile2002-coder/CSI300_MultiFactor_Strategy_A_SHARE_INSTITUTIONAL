from __future__ import annotations
import pandas as pd
import numpy as np

UNIVERSE_PRIORITY = {"CSI300": 0, "CSI500": 1, "CSI1000": 2}

def universe_weights_for_regime(regime: str, config: dict) -> dict:
    w = config["institutional"]["universe_weights_by_regime"].get(regime, {})
    total = sum(max(float(v), 0.0) for v in w.values())
    if total <= 0:
        return {"CSI300":1.0}
    return {k:max(float(v),0.0)/total for k,v in w.items()}

def active_multi_universe(membership: pd.DataFrame, signal_date) -> pd.DataFrame:
    x = membership.copy()
    if "universe" not in x.columns:
        raise ValueError("membership must include universe")
    x["effective_date"] = pd.to_datetime(x["effective_date"], errors="coerce")
    x["priority"] = x["universe"].map(UNIVERSE_PRIORITY).fillna(999)
    d = pd.Timestamp(signal_date)
    rows = []
    for universe, g in x.groupby("universe"):
        g = g[g["effective_date"] <= d]
        if g.empty:
            continue
        eff = g["effective_date"].max()
        rows.append(g[g["effective_date"] == eff].copy())
    if not rows:
        return pd.DataFrame()
    a = pd.concat(rows, ignore_index=True)
    a = a.sort_values(["ticker","priority"]).drop_duplicates("ticker", keep="first")
    return a.drop(columns=["priority"], errors="ignore")

def allocate_candidate_slots(total_names: int, regime: str, config: dict) -> dict:
    weights = universe_weights_for_regime(regime, config)
    raw = {k:total_names*v for k,v in weights.items()}
    slots = {k:int(np.floor(v)) for k,v in raw.items()}
    remainder = int(total_names - sum(slots.values()))
    order = sorted(raw, key=lambda k: raw[k]-slots[k], reverse=True)
    for k in order[:remainder]:
        slots[k] += 1
    return slots

def select_multi_universe_candidates(scored: pd.DataFrame, regime: str, config: dict, score_col="elite_score", total_names=60):
    if "universe" not in scored.columns:
        raise ValueError("scored cross-section needs universe tag")
    slots = allocate_candidate_slots(total_names, regime, config)
    out = []
    for universe, n in slots.items():
        if n <= 0:
            continue
        g = scored[scored["universe"] == universe].sort_values(score_col, ascending=False).head(n)
        out.append(g)
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()
