
from __future__ import annotations

import numpy as np
import pandas as pd


def _nearest_on_or_before(index, d):
    idx = pd.DatetimeIndex(index)
    pos = idx.searchsorted(pd.Timestamp(d), side="right") - 1
    return idx[pos] if pos >= 0 else None


def _max_drawdown(s: pd.Series):
    if len(s) == 0:
        return np.nan
    nav = s / s.iloc[0]
    return float((nav/nav.cummax()-1).min())


def compute_market_regime(
    prices: pd.DataFrame,
    benchmark: pd.DataFrame,
    as_of,
    universe_tickers=None,
    breadth_ma_days=20,
    turnover_short_days=20,
    turnover_long_days=60,
):
    """
    A-share daily regime classifier.

    Signals:
    - benchmark trend (20/60/120D)
    - 20D realized volatility
    - 60D max drawdown
    - cross-sectional breadth above 20D MA
    - market turnover short/long ratio
    - cross-sectional 5D return dispersion

    Designed to separate:
      TREND_UP / HIGH_ROTATION / RANGE / RISK_OFF / PANIC
    """
    p = prices.copy()
    p["date"] = pd.to_datetime(p["date"], errors="coerce")
    b = benchmark.copy()
    b["date"] = pd.to_datetime(b["date"], errors="coerce")

    b_close_col = "close" if "close" in b.columns else "benchmark_close"
    b = b.dropna(subset=["date", b_close_col]).sort_values("date").set_index("date")
    d = _nearest_on_or_before(b.index, as_of)
    if d is None:
        raise ValueError("No benchmark data on/before as_of")

    bs = b.loc[:d, b_close_col].astype(float)
    if len(bs) < 61:
        raise ValueError("Insufficient benchmark history for regime detection")

    ret20 = float(bs.iloc[-1]/bs.iloc[-21]-1) if len(bs) >= 21 else np.nan
    ret60 = float(bs.iloc[-1]/bs.iloc[-61]-1) if len(bs) >= 61 else np.nan
    ma20 = float(bs.tail(20).mean())
    ma60 = float(bs.tail(60).mean())
    ma120 = float(bs.tail(120).mean()) if len(bs) >= 120 else ma60
    last = float(bs.iloc[-1])
    daily = bs.pct_change().dropna()
    vol20 = float(daily.tail(20).std(ddof=0)*np.sqrt(252))
    dd60 = _max_drawdown(bs.tail(60))

    x = p[p["date"] <= d].copy()
    if universe_tickers is not None:
        x = x[x["ticker"].isin(set(map(str, universe_tickers)))]
    close = x.pivot(index="date", columns="ticker", values="close").sort_index()

    common_date = _nearest_on_or_before(close.index, d)
    breadth = np.nan
    dispersion5 = np.nan
    if common_date is not None and len(close.loc[:common_date]) >= max(breadth_ma_days, 6):
        c = close.loc[:common_date]
        ma = c.tail(int(breadth_ma_days)).mean()
        last_cs = c.iloc[-1]
        valid = last_cs.notna() & ma.notna()
        if valid.sum() > 0:
            breadth = float((last_cs[valid] > ma[valid]).mean())
        r5 = c.iloc[-1]/c.iloc[-6]-1
        dispersion5 = float(r5.std(ddof=0))

    turnover_ratio = np.nan
    if "amount" in x.columns:
        market_amount = x.groupby("date")["amount"].sum().sort_index()
        if len(market_amount) >= turnover_long_days:
            short = market_amount.tail(int(turnover_short_days)).mean()
            long = market_amount.tail(int(turnover_long_days)).mean()
            turnover_ratio = float(short/long) if long > 0 else np.nan

    # Classification: prioritize severe states first.
    if (
        (pd.notna(dd60) and dd60 <= -0.10) or
        (pd.notna(ret20) and ret20 <= -0.07 and vol20 >= 0.25) or
        (pd.notna(breadth) and breadth <= 0.25)
    ):
        regime = "PANIC"
    elif (
        last < ma60 and
        (pd.isna(breadth) or breadth < 0.45) and
        (pd.isna(ret20) or ret20 < 0)
    ):
        regime = "RISK_OFF"
    elif (
        pd.notna(turnover_ratio) and turnover_ratio >= 1.10 and
        pd.notna(dispersion5) and dispersion5 >= 0.035 and
        (pd.isna(breadth) or 0.35 <= breadth <= 0.65)
    ):
        regime = "HIGH_ROTATION"
    elif (
        last > ma60 and ma20 >= ma60 and
        ret20 > 0.02 and
        (pd.isna(breadth) or breadth >= 0.55)
    ):
        regime = "TREND_UP"
    else:
        regime = "RANGE"

    return {
        "as_of": pd.Timestamp(d),
        "regime": regime,
        "benchmark_close": last,
        "ret20": ret20,
        "ret60": ret60,
        "ma20": ma20,
        "ma60": ma60,
        "ma120": ma120,
        "vol20_annual": vol20,
        "drawdown60": dd60,
        "breadth_above_ma20": breadth,
        "turnover_20d_vs_60d": turnover_ratio,
        "dispersion_5d": dispersion5,
    }


def regime_weight_map(regime: str, config: dict):
    ash = config.get("ashare_pro", {})
    gross = ash.get("gross_exposure_by_regime", {}).get(regime, 0.75)
    weights = ash.get("factor_weights_by_regime", {}).get(regime, {})
    return float(gross), dict(weights)
