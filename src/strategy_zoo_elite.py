
from __future__ import annotations
import numpy as np
import pandas as pd


def _z(s):
    s = pd.to_numeric(s, errors="coerce")
    if s.notna().sum() < 3:
        return s.fillna(0.0)
    lo, hi = s.quantile([0.025, 0.975])
    s = s.clip(lo, hi)
    sd = s.std(ddof=0)
    if not np.isfinite(sd) or sd == 0:
        return pd.Series(0.0, index=s.index)
    return (s-s.mean())/sd


def build_strategy_sleeves(scored_cross_section: pd.DataFrame) -> pd.DataFrame:
    """
    Convert the A-SHARE PRO factor table into several deliberately different sleeves.

    The point is diversification of alpha logic, not adding dozens of correlated
    technical indicators.
    """
    x = scored_cross_section.copy()

    # Core: quality/value + medium-term momentum + defensive terms.
    core_cols = {
        "quality_score": 0.30,
        "value_score": 0.20,
        "momentum_skip_score": 0.20,
        "low_vol_score": 0.10,
        "dividend_score": 0.10,
        "crowding_score": 0.10,
    }
    core = pd.Series(0.0, index=x.index)
    used = 0.0
    for c,w in core_cols.items():
        if c in x.columns:
            core += w*pd.to_numeric(x[c], errors="coerce").fillna(0.0)
            used += abs(w)
    x["sleeve_core_multifactor"] = core/max(used, 1e-9)

    # Sector rotation: industry trend, but stock still needs decent quality and no crowding.
    x["sleeve_sector_rotation"] = (
        0.55*pd.to_numeric(x.get("sector_rotation_score", 0), errors="coerce").fillna(0.0)
        +0.20*pd.to_numeric(x.get("momentum_skip_score", 0), errors="coerce").fillna(0.0)
        +0.15*pd.to_numeric(x.get("quality_score", 0), errors="coerce").fillna(0.0)
        +0.10*pd.to_numeric(x.get("crowding_score", 0), errors="coerce").fillna(0.0)
    )

    # Mean reversion: tactical only. Oversold + quality + low-vol, explicitly not "buy any loser".
    x["sleeve_mean_reversion"] = (
        0.45*pd.to_numeric(x.get("short_reversal_score", 0), errors="coerce").fillna(0.0)
        +0.25*pd.to_numeric(x.get("quality_score", 0), errors="coerce").fillna(0.0)
        +0.15*pd.to_numeric(x.get("low_vol_score", 0), errors="coerce").fillna(0.0)
        +0.15*pd.to_numeric(x.get("crowding_score", 0), errors="coerce").fillna(0.0)
    )

    for c in [
        "sleeve_core_multifactor",
        "sleeve_sector_rotation",
        "sleeve_mean_reversion",
    ]:
        x[c] = _z(x[c])

    return x


def combine_sleeves(x: pd.DataFrame, weights: dict, ml_score_col="ml_score") -> pd.DataFrame:
    y = x.copy()
    mapping = {
        "core_multifactor":"sleeve_core_multifactor",
        "sector_rotation":"sleeve_sector_rotation",
        "mean_reversion":"sleeve_mean_reversion",
        "ml_rank":ml_score_col,
    }
    score = pd.Series(0.0, index=y.index)
    used = 0.0
    for name,w in weights.items():
        col = mapping.get(name)
        if col and col in y.columns and float(w) != 0:
            score += float(w)*pd.to_numeric(y[col], errors="coerce").fillna(0.0)
            used += abs(float(w))
    y["elite_score"] = score/max(used, 1e-9)
    return y
