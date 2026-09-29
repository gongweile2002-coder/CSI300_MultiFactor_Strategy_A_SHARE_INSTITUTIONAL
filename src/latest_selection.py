
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

from .research_v2 import (
    winsorize, zscore, _asof_by_ticker
)


def _last_available_trade_date(prices: pd.DataFrame, as_of) -> pd.Timestamp:
    as_of = pd.Timestamp(as_of)
    dates = pd.to_datetime(prices["date"]).dropna()
    eligible = dates[dates <= as_of]
    if eligible.empty:
        raise ValueError(f"as_of={as_of.date()} 之前没有行情数据")
    return pd.Timestamp(eligible.max())


def _latest_membership_snapshot(membership: pd.DataFrame, as_of) -> pd.DataFrame:
    m = membership.copy()
    m["effective_date"] = pd.to_datetime(m["effective_date"], errors="coerce")
    m = m[m["effective_date"] <= pd.Timestamp(as_of)]
    if m.empty:
        raise ValueError("as_of 日期前没有沪深300历史成分股记录")
    latest_eff = m["effective_date"].max()
    return m[m["effective_date"] == latest_eff].copy()


def build_latest_selection(
    prices: pd.DataFrame,
    daily_basic: pd.DataFrame,
    fundamentals: pd.DataFrame,
    membership: pd.DataFrame,
    metadata: pd.DataFrame | None = None,
    as_of=None,
    top_n=30,
    momentum_lookback_days=126,
    winsor_lower=0.025,
    winsor_upper=0.975,
    factor_weights=None,
    min_turnover_rate=0.0,
    min_market_cap_cny_10k=0.0,
):
    """
    Build the latest cross-sectional stock selection as of a user-specified date.

    Important:
    - price/valuation date = latest available trading day <= as_of
    - financial record = latest ANNOUNCED record with ann_date <= as_of
    - index universe = latest known historical CSI300 membership effective <= as_of
    - never force all companies into a nominal quarter if the report was not public yet
    """
    factor_weights = factor_weights or {"value":1/3, "quality":1/3, "momentum":1/3}

    prices = prices.copy()
    daily_basic = daily_basic.copy()
    fundamentals = fundamentals.copy()
    membership = membership.copy()

    prices["date"] = pd.to_datetime(prices["date"], errors="coerce")
    daily_basic["date"] = pd.to_datetime(daily_basic["date"], errors="coerce")
    fundamentals["ann_date"] = pd.to_datetime(fundamentals["ann_date"], errors="coerce")
    if "report_date" in fundamentals.columns:
        fundamentals["report_date"] = pd.to_datetime(fundamentals["report_date"], errors="coerce")
    membership["effective_date"] = pd.to_datetime(membership["effective_date"], errors="coerce")

    as_of = pd.Timestamp(as_of) if as_of is not None else pd.Timestamp.today().normalize()
    trade_date = _last_available_trade_date(prices, as_of)

    mem = _latest_membership_snapshot(membership, as_of)
    universe = mem[["ticker","weight","effective_date"]].drop_duplicates("ticker").copy()
    universe["as_of"] = as_of
    universe["trade_date"] = trade_date

    # Latest valuation known by actual trade date
    val_left = universe[["ticker","trade_date"]].copy()
    val = daily_basic.rename(columns={"date":"valuation_date"}).copy()
    val_cols = [c for c in [
        "ticker","valuation_date","pe_ttm","pb","turnover_rate","total_mv","circ_mv"
    ] if c in val.columns]
    val = val[val_cols].dropna(subset=["valuation_date"])
    x = _asof_by_ticker(val_left, val, "trade_date", "valuation_date")

    # Latest published accounting data by as_of (not by report period label)
    fund_left = universe[["ticker","as_of"]].copy()
    fcols = [c for c in [
        "ticker","ann_date","report_date","roe","roe_dt","profit_growth","debt_ratio"
    ] if c in fundamentals.columns]
    from .data_contract_v8 import latest_financial_records
    f = latest_financial_records(fundamentals[fcols].dropna(subset=["ann_date"]), as_of)
    fx = _asof_by_ticker(fund_left, f, "as_of", "ann_date")

    x = x.merge(
        fx.drop(columns=["as_of"], errors="ignore"),
        on="ticker",
        how="left"
    )
    x = x.merge(
        universe[["ticker","weight","effective_date","as_of"]],
        on="ticker",
        how="left"
    )

    # 126 trading-day momentum ending on trade_date
    close = prices.pivot(index="date", columns="ticker", values="close").sort_index()
    if trade_date not in close.index:
        raise ValueError("latest trade_date 不在收盘价矩阵中")
    loc = close.index.get_loc(trade_date)
    if isinstance(loc, slice):
        loc = loc.stop - 1
    if loc < momentum_lookback_days:
        raise ValueError("行情历史不足，无法计算 momentum")
    past_date = close.index[loc - momentum_lookback_days]
    mom = close.loc[trade_date] / close.loc[past_date] - 1.0
    x["momentum_start_date"] = past_date
    x["momentum"] = x["ticker"].map(mom.to_dict())

    # Filters
    if "turnover_rate" in x.columns:
        x = x[x["turnover_rate"].fillna(0) >= float(min_turnover_rate)]
    if "total_mv" in x.columns and float(min_market_cap_cny_10k) > 0:
        x = x[x["total_mv"].fillna(0) >= float(min_market_cap_cny_10k)]

    if "roe_dt" in x.columns:
        x["quality_roe"] = x["roe_dt"].fillna(x.get("roe"))
    else:
        x["quality_roe"] = x.get("roe")

    x.loc[x["pe_ttm"] <= 0, "pe_ttm"] = np.nan
    x.loc[x["pb"] <= 0, "pb"] = np.nan

    required = ["pe_ttm","pb","quality_roe","profit_growth","debt_ratio","momentum"]
    x = x.dropna(subset=[c for c in required if c in x.columns]).copy()
    if len(x) < 10:
        raise RuntimeError("有效股票数量过少，请检查数据完整性/接口权限")

    # factor engineering
    x["inv_pe_signal"] = -np.log(x["pe_ttm"])
    x["inv_pb_signal"] = -np.log(x["pb"])

    raw_cols = [
        "inv_pe_signal","inv_pb_signal","quality_roe",
        "profit_growth","debt_ratio","momentum"
    ]
    for c in raw_cols:
        x[c] = winsorize(x[c], winsor_lower, winsor_upper)
        x[c] = zscore(x[c])

    x["value_score"] = (x["inv_pe_signal"] + x["inv_pb_signal"]) / 2
    x["quality_score"] = (x["quality_roe"] + x["profit_growth"] - x["debt_ratio"]) / 3
    x["momentum_score"] = x["momentum"]

    for c in ["value_score","quality_score","momentum_score"]:
        x[c] = winsorize(x[c], winsor_lower, winsor_upper)
        x[c] = zscore(x[c])

    wv = factor_weights.get("value", 1/3)
    wq = factor_weights.get("quality", 1/3)
    wm = factor_weights.get("momentum", 1/3)
    total = wv+wq+wm
    wv, wq, wm = wv/total, wq/total, wm/total

    x["composite_score"] = (
        wv*x["value_score"] +
        wq*x["quality_score"] +
        wm*x["momentum_score"]
    )
    x["rank"] = x["composite_score"].rank(ascending=False, method="first").astype(int)

    # Metadata
    if metadata is not None and not metadata.empty:
        meta = metadata.copy()
        keep = [c for c in ["ticker","name","industry","market","exchange","list_date","delist_date"] if c in meta.columns]
        if keep:
            x = x.merge(meta[keep].drop_duplicates("ticker"), on="ticker", how="left")

    # Freshness labels
    if "report_date" in x.columns:
        x["report_period"] = pd.to_datetime(x["report_date"], errors="coerce").dt.strftime("%Y-%m-%d")
    x["financial_age_days"] = (as_of - pd.to_datetime(x["ann_date"], errors="coerce")).dt.days
    x["valuation_age_days"] = (trade_date - pd.to_datetime(x["valuation_date"], errors="coerce")).dt.days

    front = [
        "rank","ticker","name","industry","as_of","trade_date",
        "effective_date","report_period","ann_date","valuation_date",
        "value_score","quality_score","momentum_score","composite_score",
        "pe_ttm","pb","quality_roe","profit_growth","debt_ratio",
        "momentum","weight","total_mv","circ_mv","turnover_rate",
        "financial_age_days","valuation_age_days"
    ]
    cols = [c for c in front if c in x.columns] + [c for c in x.columns if c not in front]
    x = x[cols].sort_values("rank").reset_index(drop=True)

    return x.head(int(top_n)).copy(), x


def build_freshness_summary(full_cross_section: pd.DataFrame) -> pd.DataFrame:
    if full_cross_section.empty:
        return pd.DataFrame()

    rows = []
    if "report_period" in full_cross_section.columns:
        counts = full_cross_section["report_period"].value_counts(dropna=False)
        for period, n in counts.items():
            rows.append({
                "dimension":"report_period",
                "value":str(period),
                "count":int(n),
                "share":float(n/len(full_cross_section))
            })

    if "valuation_date" in full_cross_section.columns:
        counts = pd.to_datetime(full_cross_section["valuation_date"], errors="coerce").dt.strftime("%Y-%m-%d").value_counts(dropna=False)
        for val, n in counts.items():
            rows.append({
                "dimension":"valuation_date",
                "value":str(val),
                "count":int(n),
                "share":float(n/len(full_cross_section))
            })
    return pd.DataFrame(rows)
