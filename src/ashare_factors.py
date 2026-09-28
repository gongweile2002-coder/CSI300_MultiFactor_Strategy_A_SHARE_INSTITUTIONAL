
from __future__ import annotations

import numpy as np
import pandas as pd

from .research_v4 import winsorize, zscore, neutralize_factor, industry_for_date


def _nearest(index, d):
    idx = pd.DatetimeIndex(index)
    pos = idx.searchsorted(pd.Timestamp(d), side="right") - 1
    return idx[pos] if pos >= 0 else None


def _safe_z(s):
    s = pd.to_numeric(s, errors="coerce")
    s = winsorize(s, 0.025, 0.975)
    return zscore(s)


def compute_price_factors(
    prices: pd.DataFrame,
    signal_date,
    tickers,
    momentum_long_days=126,
    momentum_skip_days=20,
    short_reversal_days=5,
    low_vol_days=20,
):
    p = prices.copy()
    p["date"] = pd.to_datetime(p["date"], errors="coerce")
    p = p[p["ticker"].isin(set(map(str, tickers)))]
    close = p.pivot(index="date", columns="ticker", values="close").sort_index()
    d = _nearest(close.index, signal_date)
    if d is None:
        return pd.DataFrame(columns=["ticker"])

    hist = close.loc[:d]
    loc = len(hist)-1

    def ret_between(a, b):
        if loc < a:
            return pd.Series(np.nan, index=close.columns)
        return hist.iloc[-1-b] / hist.iloc[-1-a] - 1 if b < a else pd.Series(np.nan, index=close.columns)

    # 6m momentum excluding last ~1m:
    # Price(t-20) / Price(t-126) - 1
    if len(hist) > momentum_long_days:
        mom_skip = hist.iloc[-1-momentum_skip_days] / hist.iloc[-1-momentum_long_days] - 1
    else:
        mom_skip = pd.Series(np.nan, index=hist.columns)

    if len(hist) > short_reversal_days:
        ret_short = hist.iloc[-1]/hist.iloc[-1-short_reversal_days]-1
        short_reversal = -ret_short
    else:
        ret_short = pd.Series(np.nan, index=hist.columns)
        short_reversal = ret_short.copy()

    ret = hist.pct_change()
    vol = ret.tail(int(low_vol_days)).std(ddof=0) * np.sqrt(252)
    low_vol = -vol

    if len(hist) > 20:
        ret20 = hist.iloc[-1]/hist.iloc[-21]-1
    else:
        ret20 = pd.Series(np.nan, index=hist.columns)
    if len(hist) > 60:
        ret60 = hist.iloc[-1]/hist.iloc[-61]-1
    else:
        ret60 = pd.Series(np.nan, index=hist.columns)

    out = pd.DataFrame({
        "ticker": hist.columns.astype(str),
        "momentum_skip_raw": mom_skip.reindex(hist.columns).values,
        "short_reversal_raw": short_reversal.reindex(hist.columns).values,
        "low_vol_raw": low_vol.reindex(hist.columns).values,
        "ret5": ret_short.reindex(hist.columns).values,
        "ret20": ret20.reindex(hist.columns).values,
        "ret60": ret60.reindex(hist.columns).values,
    })
    return out


def compute_latest_daily_basic_features(daily_basic, signal_date, tickers):
    d = daily_basic.copy()
    d["date"] = pd.to_datetime(d["date"], errors="coerce")
    d = d[(d["date"] <= pd.Timestamp(signal_date)) & (d["ticker"].isin(set(map(str,tickers))))]
    if d.empty:
        return pd.DataFrame(columns=["ticker"])

    rows = []
    for ticker, g in d.groupby("ticker"):
        g = g.sort_values("date")
        last = g.iloc[-1]
        turnover = pd.to_numeric(g.get("turnover_rate"), errors="coerce")
        t60 = turnover.tail(60)
        if len(t60.dropna()) >= 10 and t60.std(ddof=0) > 0:
            turnover_z = (float(last.get("turnover_rate", np.nan))-float(t60.mean()))/float(t60.std(ddof=0))
        else:
            turnover_z = np.nan

        rows.append({
            "ticker": str(ticker),
            "dv_ttm": last.get("dv_ttm", np.nan),
            "volume_ratio": last.get("volume_ratio", np.nan),
            "turnover_rate": last.get("turnover_rate", np.nan),
            "turnover_z60": turnover_z,
            "limit_status": last.get("limit_status", np.nan),
            "total_mv": last.get("total_mv", np.nan),
        })
    return pd.DataFrame(rows)


def compute_sector_rotation(prices, industry_membership, signal_date, tickers, short_days=20, long_days=60):
    if industry_membership is None or industry_membership.empty:
        return pd.DataFrame({"ticker":list(map(str,tickers)), "sector_rotation_raw":0.0})

    ind = industry_for_date(industry_membership, signal_date)
    if ind.empty:
        return pd.DataFrame({"ticker":list(map(str,tickers)), "sector_rotation_raw":0.0})

    px = prices.copy()
    px["date"] = pd.to_datetime(px["date"], errors="coerce")
    px = px[px["ticker"].isin(set(map(str,tickers)))]
    close = px.pivot(index="date", columns="ticker", values="close").sort_index()
    d = _nearest(close.index, signal_date)
    if d is None:
        return pd.DataFrame(columns=["ticker","sector_rotation_raw"])

    hist = close.loc[:d]
    if len(hist) <= long_days:
        return pd.DataFrame({"ticker":list(map(str,tickers)), "sector_rotation_raw":0.0})

    r20 = hist.iloc[-1]/hist.iloc[-1-short_days]-1
    r60 = hist.iloc[-1]/hist.iloc[-1-long_days]-1
    stock = pd.DataFrame({"ticker":hist.columns.astype(str), "r20":r20.values, "r60":r60.values})
    stock = stock.merge(ind, on="ticker", how="left")
    stock["industry_l1"] = stock["industry_l1"].fillna("UNKNOWN")

    sector = stock.groupby("industry_l1")[["r20","r60"]].mean()
    sector["sector_rotation_raw"] = 0.6*_safe_z(sector["r20"]) + 0.4*_safe_z(sector["r60"])
    score_map = sector["sector_rotation_raw"].to_dict()
    stock["sector_rotation_raw"] = stock["industry_l1"].map(score_map).fillna(0.0)
    return stock[["ticker","sector_rotation_raw"]]


def augment_ashare_factors(
    base_cross_section: pd.DataFrame,
    prices: pd.DataFrame,
    daily_basic: pd.DataFrame,
    industry_membership,
    signal_date,
    config,
):
    """
    Adds A-share-specific tactical factors to an already PIT-cleaned base factor cross-section.

    The base cross-section is expected to contain:
      ticker, value_score, quality_score, total_mv, industry_l1
    """
    g = base_cross_section.copy()
    if g.empty:
        return g

    ash = config["ashare_pro"]
    tickers = g["ticker"].astype(str).tolist()

    pf = compute_price_factors(
        prices, signal_date, tickers,
        momentum_long_days=ash["momentum_long_days"],
        momentum_skip_days=ash["momentum_skip_days"],
        short_reversal_days=ash["short_reversal_days"],
        low_vol_days=ash["low_vol_days"],
    )
    db = compute_latest_daily_basic_features(daily_basic, signal_date, tickers)
    sr = compute_sector_rotation(
        prices, industry_membership, signal_date, tickers,
        short_days=ash["sector_mom_short_days"],
        long_days=ash["sector_mom_long_days"],
    )

    x = g.merge(pf, on="ticker", how="left").merge(db, on="ticker", how="left", suffixes=("","_db"))
    x = x.merge(sr, on="ticker", how="left")

    # dividend yield: higher is better; missing -> neutral
    x["dividend_raw"] = pd.to_numeric(x.get("dv_ttm"), errors="coerce")
    x["dividend_raw"] = x["dividend_raw"].fillna(x["dividend_raw"].median())

    # crowding: penalize extremely high turnover, but do not mechanically reward illiquidity.
    tz = pd.to_numeric(x.get("turnover_z60"), errors="coerce").fillna(0.0)
    vr = pd.to_numeric(x.get("volume_ratio"), errors="coerce").fillna(1.0)
    x["crowding_raw"] = -(tz.clip(lower=0)**2) - 0.25*((vr-1).clip(lower=0)**2)

    raw_map = {
        "momentum_skip":"momentum_skip_raw",
        "short_reversal":"short_reversal_raw",
        "low_vol":"low_vol_raw",
        "dividend":"dividend_raw",
        "sector_rotation":"sector_rotation_raw",
        "crowding":"crowding_raw",
    }

    # Neutralize A-share tactical factors by size + industry so the signal isn't
    # just a hidden sector/market-cap bet.
    for name, raw in raw_map.items():
        x[raw] = pd.to_numeric(x[raw], errors="coerce")
        x[raw] = x[raw].fillna(x[raw].median())
        x[raw] = _safe_z(x[raw])
        x[name+"_score"] = neutralize_factor(
            x, raw,
            market_cap_col="total_mv",
            industry_col="industry_l1",
            use_market_cap=True,
            use_industry=True,
        )
        x[name+"_score"] = x[name+"_score"].fillna(0.0)

    # A-share chase filters for new buys.
    limit_status = pd.to_numeric(x.get("limit_status"), errors="coerce")
    x["new_buy_block"] = False
    if ash.get("skip_limit_up_new_buys", True):
        # Tushare daily_basic: 2=limit-up, 3=one-price limit-up
        x.loc[limit_status.isin([2,3]), "new_buy_block"] = True

    volume_ratio = pd.to_numeric(x.get("volume_ratio"), errors="coerce")
    # Avoid chasing volume blow-offs after a sharp 5D rise.
    x.loc[
        (volume_ratio > float(ash.get("volume_ratio_buy_cap", 3.0))) &
        (pd.to_numeric(x["ret5"], errors="coerce") > 0.08),
        "new_buy_block"
    ] = True

    return x
