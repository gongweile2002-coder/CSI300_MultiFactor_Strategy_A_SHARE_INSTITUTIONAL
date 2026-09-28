
import numpy as np
import pandas as pd

def winsorize_series(s: pd.Series, lower=0.025, upper=0.975):
    if s.notna().sum() < 3:
        return s
    lo, hi = s.quantile([lower, upper])
    return s.clip(lo, hi)

def zscore_series(s: pd.Series):
    std = s.std(ddof=0)
    if std == 0 or np.isnan(std):
        return pd.Series(0.0, index=s.index)
    return (s - s.mean()) / std

def cross_sectional_clean(df, cols):
    out = df.copy()
    for c in cols:
        out[c] = winsorize_series(out[c])
        out[c] = zscore_series(out[c])
    return out

def add_momentum(prices: pd.DataFrame, lookback_days=126):
    p = prices.pivot(index="date", columns="ticker", values="close").sort_index()
    mom = p / p.shift(lookback_days) - 1.0
    out = mom.stack().rename("momentum").reset_index()
    return out

def build_factor_scores(
    prices: pd.DataFrame,
    fundamentals: pd.DataFrame,
    value_weight=1/3,
    quality_weight=1/3,
    momentum_weight=1/3,
    momentum_lookback_days=126
):
    """
    因子方向：
    - Value: PE、PB 越低越好
    - Quality: ROE/利润增长越高越好，负债率越低越好
    - Momentum: 过去约6个月收益越高越好
    """
    mom = add_momentum(prices, momentum_lookback_days)
    df = fundamentals.merge(mom, on=["date", "ticker"], how="left")

    results = []
    for dt, g in df.groupby("date", sort=True):
        g = g.copy()
        raw_cols = ["pe_ttm","pb","roe","debt_ratio","profit_growth","momentum"]
        g = g.dropna(subset=raw_cols)
        if len(g) < 5:
            continue

        # 先标准化原始变量
        g = cross_sectional_clean(g, raw_cols)

        # value: 低PE低PB => 高分
        g["value_score"] = (-g["pe_ttm"] - g["pb"]) / 2

        # quality: 高ROE、高增长、低负债 => 高分
        g["quality_score"] = (g["roe"] + g["profit_growth"] - g["debt_ratio"]) / 3

        g["momentum_score"] = g["momentum"]

        # 再把三大因子标准化一次，便于可比
        g = cross_sectional_clean(g, ["value_score","quality_score","momentum_score"])

        total_w = value_weight + quality_weight + momentum_weight
        vw, qw, mw = value_weight/total_w, quality_weight/total_w, momentum_weight/total_w

        g["composite_score"] = (
            vw*g["value_score"] +
            qw*g["quality_score"] +
            mw*g["momentum_score"]
        )
        g["rank"] = g["composite_score"].rank(ascending=False, method="first")
        results.append(g)

    if not results:
        return pd.DataFrame()
    return pd.concat(results, ignore_index=True)
