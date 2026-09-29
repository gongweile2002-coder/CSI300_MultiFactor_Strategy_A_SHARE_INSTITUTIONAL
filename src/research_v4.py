
from __future__ import annotations

import itertools
import numpy as np
import pandas as pd

from .research_v2 import (
    winsorize, zscore, _asof_by_ticker,
    trading_month_ends, membership_for_date,
    performance_summary
)


def stamp_duty_rate(trade_date) -> float:
    """
    China A-share seller-side securities transaction stamp duty.
    The rate was halved effective 2023-08-28.
    This function is parameter-free so historical backtests do not apply
    the post-2023 rate to the whole sample.
    """
    d = pd.Timestamp(trade_date)
    return 0.0005 if d >= pd.Timestamp("2023-08-28") else 0.0010


def industry_for_date(industry_membership: pd.DataFrame | None, signal_date) -> pd.DataFrame:
    if industry_membership is None or industry_membership.empty:
        return pd.DataFrame(columns=["ticker","industry_l1"])

    m = industry_membership.copy()
    for c in ["in_date","out_date"]:
        if c in m.columns:
            m[c] = pd.to_datetime(m[c], errors="coerce")

    d = pd.Timestamp(signal_date)
    active = m[
        (m["in_date"].fillna(pd.Timestamp("1900-01-01")) <= d) &
        (m["out_date"].isna() | (m["out_date"] >= d))
    ].copy()

    if active.empty:
        return pd.DataFrame(columns=["ticker","industry_l1"])

    if "l1_name" in active.columns:
        active["industry_l1"] = active["l1_name"]
    elif "industry" in active.columns:
        active["industry_l1"] = active["industry"]
    else:
        active["industry_l1"] = "UNKNOWN"

    # If overlapping historical classifications exist, prefer the latest in_date.
    active = active.sort_values(["ticker","in_date"]).drop_duplicates("ticker", keep="last")
    return active[["ticker","industry_l1"]]


def st_tickers_for_date(st_status: pd.DataFrame | None, signal_date) -> set[str]:
    if st_status is None or st_status.empty:
        return set()

    s = st_status.copy()
    s["trade_date"] = pd.to_datetime(s["trade_date"], errors="coerce")
    d = pd.Timestamp(signal_date)
    today = s[s["trade_date"] == d]
    return set(today["ticker"].dropna().astype(str))


def add_trailing_liquidity(prices: pd.DataFrame, signal_date, lookback_days=20) -> pd.DataFrame:
    p = prices.copy()
    p["date"] = pd.to_datetime(p["date"], errors="coerce")
    p = p[p["date"] <= pd.Timestamp(signal_date)].sort_values(["ticker","date"])

    # Tushare daily amount is the preferred liquidity field.
    if "amount" in p.columns:
        p["_liq"] = pd.to_numeric(p["amount"], errors="coerce")
    elif {"close","volume"}.issubset(p.columns):
        p["_liq"] = pd.to_numeric(p["close"], errors="coerce") * pd.to_numeric(p["volume"], errors="coerce")
    else:
        return pd.DataFrame(columns=["ticker","avg_amount_20d","listing_trading_days"])

    rows = []
    for t, g in p.groupby("ticker"):
        g = g.dropna(subset=["_liq"])
        if g.empty:
            continue
        tail = g.tail(int(lookback_days))
        rows.append({
            "ticker": t,
            "avg_amount_20d": float(tail["_liq"].mean()),
            "listing_trading_days": int(len(g)),
        })
    return pd.DataFrame(rows)


def neutralize_factor(
    df: pd.DataFrame,
    factor_col: str,
    market_cap_col="total_mv",
    industry_col="industry_l1",
    use_market_cap=True,
    use_industry=True,
) -> pd.Series:
    """
    OLS residualization against log market cap and industry dummies.
    Returned residuals are z-scored cross-sectionally.
    """
    x = df.copy()
    y = pd.to_numeric(x[factor_col], errors="coerce")

    regressors = [pd.Series(1.0, index=x.index, name="intercept")]

    if use_market_cap and market_cap_col in x.columns:
        mc = pd.to_numeric(x[market_cap_col], errors="coerce")
        log_mc = np.log(mc.where(mc > 0))
        if log_mc.notna().sum() >= 5 and log_mc.std(ddof=0) > 0:
            regressors.append(log_mc.rename("log_market_cap"))

    if use_industry and industry_col in x.columns:
        ind = x[industry_col].fillna("UNKNOWN").astype(str)
        dummies = pd.get_dummies(ind, prefix="ind", drop_first=True, dtype=float)
        for c in dummies.columns:
            regressors.append(dummies[c])

    X = pd.concat(regressors, axis=1)
    valid = y.notna() & X.notna().all(axis=1)
    out = pd.Series(np.nan, index=x.index, dtype=float)

    if valid.sum() < max(5, X.shape[1] + 1):
        out.loc[valid] = zscore(y.loc[valid])
        return out

    beta, *_ = np.linalg.lstsq(
        X.loc[valid].to_numpy(dtype=float),
        y.loc[valid].to_numpy(dtype=float),
        rcond=None
    )
    resid = y.loc[valid].to_numpy(dtype=float) - X.loc[valid].to_numpy(dtype=float) @ beta
    resid = pd.Series(resid, index=x.index[valid])
    out.loc[valid] = zscore(resid)
    return out


def build_point_in_time_panel_v4(
    prices: pd.DataFrame,
    daily_basic: pd.DataFrame,
    fundamentals: pd.DataFrame,
    membership: pd.DataFrame,
    metadata: pd.DataFrame | None = None,
    industry_membership: pd.DataFrame | None = None,
    st_status: pd.DataFrame | None = None,
    start_date=None,
    end_date=None,
    momentum_lookback_days=126,
    winsor_lower=0.025,
    winsor_upper=0.975,
    factor_weights=None,
    min_turnover_rate=0.0,
    min_market_cap_cny_10k=0.0,
    exclude_st=True,
    min_listing_trading_days=120,
    liquidity_lookback_days=20,
    liquidity_min_quantile=0.10,
    neutralize_market_cap=True,
    neutralize_industry=True,
    signal_dates_override=None,
):
    factor_weights = factor_weights or {"value":1/3, "quality":1/3, "momentum":1/3}

    prices = prices.copy()
    daily_basic = daily_basic.copy()
    fundamentals = fundamentals.copy()
    membership = membership.copy()

    for df, cols in [
        (prices, ["date"]),
        (daily_basic, ["date"]),
        (fundamentals, ["ann_date","report_date"]),
        (membership, ["effective_date"]),
    ]:
        for c in cols:
            if c in df.columns:
                df[c] = pd.to_datetime(df[c], errors="coerce")

    if metadata is not None and not metadata.empty:
        metadata = metadata.copy()
        if "list_date" in metadata.columns:
            metadata["list_date"] = pd.to_datetime(metadata["list_date"], errors="coerce")

    signal_dates = list(pd.to_datetime(signal_dates_override)) if signal_dates_override is not None else trading_month_ends(prices)
    if start_date is not None:
        signal_dates = [d for d in signal_dates if d >= pd.Timestamp(start_date)]
    if end_date is not None:
        signal_dates = [d for d in signal_dates if d <= pd.Timestamp(end_date)]

    close = prices.pivot(index="date", columns="ticker", values="close").sort_index()
    rows = []

    for signal_date in signal_dates:
        mem = membership_for_date(membership, signal_date)
        if mem.empty:
            continue

        universe = mem[["ticker","weight"]].drop_duplicates("ticker").copy()
        universe["signal_date"] = pd.Timestamp(signal_date)

        # Historical ST exclusion.
        if exclude_st:
            st_set = st_tickers_for_date(st_status, signal_date)
            if st_set:
                universe = universe[~universe["ticker"].isin(st_set)]

        # Historical industry classification (SW level-1 if downloaded).
        ind = industry_for_date(industry_membership, signal_date)
        universe = universe.merge(ind, on="ticker", how="left")

        # Listing age and trailing 20D liquidity.
        liq = add_trailing_liquidity(prices, signal_date, liquidity_lookback_days)
        universe = universe.merge(liq, on="ticker", how="left")
        if int(min_listing_trading_days) > 0:
            universe = universe[
                universe["listing_trading_days"].fillna(0) >= int(min_listing_trading_days)
            ]

        if not universe.empty and float(liquidity_min_quantile) > 0:
            cutoff = universe["avg_amount_20d"].quantile(float(liquidity_min_quantile))
            universe = universe[universe["avg_amount_20d"] >= cutoff]

        if universe.empty:
            continue

        # Point-in-time valuation
        left = universe[["ticker","signal_date"]].copy()
        val = daily_basic.rename(columns={"date":"valuation_date"}).copy()
        cols = [c for c in [
            "ticker","valuation_date","pe_ttm","pb","turnover_rate","total_mv","circ_mv"
        ] if c in val.columns]
        val = val[cols].dropna(subset=["valuation_date"])
        left = _asof_by_ticker(left, val, "signal_date", "valuation_date")

        # Point-in-time accounting data by ANNOUNCEMENT date.
        fcols = [c for c in [
            "ticker","ann_date","report_date","roe","roe_dt","profit_growth","debt_ratio"
        ] if c in fundamentals.columns]
        from .data_contract_v8 import latest_financial_records
        f = latest_financial_records(fundamentals[fcols].dropna(subset=["ann_date"]), signal_date)
        left = _asof_by_ticker(left, f, "signal_date", "ann_date")

        left = left.merge(
            universe[["ticker","weight","industry_l1","avg_amount_20d","listing_trading_days"]],
            on="ticker", how="left"
        )

        if metadata is not None and not metadata.empty:
            keep = [c for c in ["ticker","name","market","exchange","list_date"] if c in metadata.columns]
            if keep:
                left = left.merge(metadata[keep].drop_duplicates("ticker"), on="ticker", how="left")

        # Momentum
        if signal_date not in close.index:
            continue
        loc = close.index.get_loc(signal_date)
        if isinstance(loc, slice):
            loc = loc.stop - 1
        if loc < momentum_lookback_days:
            continue
        past_date = close.index[loc - momentum_lookback_days]
        mom = close.loc[signal_date] / close.loc[past_date] - 1.0
        left["momentum"] = left["ticker"].map(mom.to_dict())
        left["momentum_start_date"] = past_date

        if "turnover_rate" in left.columns:
            left = left[left["turnover_rate"].fillna(0) >= float(min_turnover_rate)]
        if "total_mv" in left.columns and float(min_market_cap_cny_10k) > 0:
            left = left[left["total_mv"].fillna(0) >= float(min_market_cap_cny_10k)]

        if "roe_dt" in left.columns:
            left["quality_roe"] = left["roe_dt"].fillna(left.get("roe"))
        else:
            left["quality_roe"] = left.get("roe")

        left.loc[left["pe_ttm"] <= 0, "pe_ttm"] = np.nan
        left.loc[left["pb"] <= 0, "pb"] = np.nan

        needed = ["pe_ttm","pb","quality_roe","profit_growth","debt_ratio","momentum"]
        left = left.dropna(subset=needed).copy()
        if len(left) < 10:
            continue

        left["inv_pe_signal"] = -np.log(left["pe_ttm"])
        left["inv_pb_signal"] = -np.log(left["pb"])

        raw_cols = [
            "inv_pe_signal","inv_pb_signal","quality_roe",
            "profit_growth","debt_ratio","momentum"
        ]
        for c in raw_cols:
            left[c] = winsorize(left[c], winsor_lower, winsor_upper)
            left[c] = zscore(left[c])

        left["value_score_raw"] = (left["inv_pe_signal"] + left["inv_pb_signal"]) / 2
        left["quality_score_raw"] = (
            left["quality_roe"] + left["profit_growth"] - left["debt_ratio"]
        ) / 3
        left["momentum_score_raw"] = left["momentum"]

        # Size + industry neutralization.
        left["value_score"] = neutralize_factor(
            left, "value_score_raw",
            use_market_cap=neutralize_market_cap,
            use_industry=neutralize_industry
        )
        left["quality_score"] = neutralize_factor(
            left, "quality_score_raw",
            use_market_cap=neutralize_market_cap,
            use_industry=neutralize_industry
        )
        left["momentum_score"] = neutralize_factor(
            left, "momentum_score_raw",
            use_market_cap=neutralize_market_cap,
            use_industry=neutralize_industry
        )

        left = left.dropna(subset=["value_score","quality_score","momentum_score"]).copy()
        if len(left) < 10:
            continue

        wv = factor_weights.get("value", 1/3)
        wq = factor_weights.get("quality", 1/3)
        wm = factor_weights.get("momentum", 1/3)
        tw = wv+wq+wm
        wv, wq, wm = wv/tw, wq/tw, wm/tw

        left["composite_score"] = (
            wv*left["value_score"] +
            wq*left["quality_score"] +
            wm*left["momentum_score"]
        )
        left["rank"] = left["composite_score"].rank(ascending=False, method="first")
        rows.append(left)

    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True).sort_values(["signal_date","rank"])


def _limit_lookup(stock_limits: pd.DataFrame | None):
    if stock_limits is None or stock_limits.empty:
        return {}
    x = stock_limits.copy()
    x["date"] = pd.to_datetime(x["date"], errors="coerce")
    out = {}
    for _, r in x.iterrows():
        out[(pd.Timestamp(r["date"]), str(r["ticker"]))] = (
            r.get("up_limit", np.nan),
            r.get("down_limit", np.nan)
        )
    return out


def _at_limit(price, limit_price, tol=1e-6):
    if pd.isna(price) or pd.isna(limit_price):
        return False
    return float(price) >= float(limit_price) - tol


def _at_down_limit(price, limit_price, tol=1e-6):
    if pd.isna(price) or pd.isna(limit_price):
        return False
    return float(price) <= float(limit_price) + tol


def run_realistic_backtest_v4(
    prices: pd.DataFrame,
    panel: pd.DataFrame,
    stock_limits: pd.DataFrame | None = None,
    top_n=30,
    commission_bps=3.0,
    slippage_bps=2.0,
    use_price_limits=True,
):
    """
    Monthly signal at month-end close, execute next trading day open.

    Simplified but more realistic execution:
    - New buys at opening limit-up are skipped.
    - Existing holdings at opening limit-down cannot be sold and are retained.
    - Commission and slippage apply both directions.
    - Seller stamp duty uses a historical date schedule.
    """
    prices = prices.copy()
    prices["date"] = pd.to_datetime(prices["date"], errors="coerce")
    dates = pd.DatetimeIndex(sorted(prices["date"].dropna().unique()))
    open_w = prices.pivot(index="date", columns="ticker", values="open").sort_index()
    close_w = prices.pivot(index="date", columns="ticker", values="close").sort_index()
    limit_map = _limit_lookup(stock_limits)

    signal_dates = sorted(pd.to_datetime(panel["signal_date"].unique()))
    daily_ret = pd.Series(0.0, index=dates)
    holdings_rows, trade_rows = [], []

    prev_holdings = set()

    for i, sig in enumerate(signal_dates):
        pos = dates.searchsorted(sig, side="right")
        if pos >= len(dates):
            break
        exec_date = dates[pos]

        next_exec = None
        if i+1 < len(signal_dates):
            np_ = dates.searchsorted(signal_dates[i+1], side="right")
            if np_ < len(dates):
                next_exec = dates[np_]

        g = panel[panel["signal_date"] == sig].sort_values("composite_score", ascending=False)
        desired = g.head(int(top_n))["ticker"].astype(str).tolist()

        # Positions we want to exit but cannot because they open at down-limit.
        blocked_sells = set()
        if use_price_limits:
            for t in prev_holdings - set(desired):
                if t not in open_w.columns or exec_date not in open_w.index:
                    blocked_sells.add(t)
                    continue
                op = open_w.loc[exec_date, t]
                _, dn = limit_map.get((exec_date, t), (np.nan, np.nan))
                if pd.isna(op) or _at_down_limit(op, dn):
                    blocked_sells.add(t)

        # New candidate buys: skip suspension / opening limit-up.
        tradable_buys = []
        for t in desired:
            if t in prev_holdings:
                tradable_buys.append(t)
                continue
            if t not in open_w.columns or exec_date not in open_w.index:
                continue
            op = open_w.loc[exec_date, t]
            if pd.isna(op):
                continue
            if use_price_limits:
                up, _ = limit_map.get((exec_date, t), (np.nan, np.nan))
                if _at_limit(op, up):
                    continue
            tradable_buys.append(t)

        new_holdings = set(tradable_buys) | blocked_sells
        if not new_holdings:
            continue

        # Equal-weight approximation after practical trading constraints.
        n_new = len(new_holdings)
        new_weights = {t: 1.0/n_new for t in new_holdings}
        prev_weights = {t: 1.0/len(prev_holdings) for t in prev_holdings} if prev_holdings else {}

        all_t = set(prev_weights) | set(new_weights)
        buys = sum(max(new_weights.get(t,0)-prev_weights.get(t,0), 0) for t in all_t)
        sells = sum(max(prev_weights.get(t,0)-new_weights.get(t,0), 0) for t in all_t)

        commission = (buys+sells) * float(commission_bps) / 10000.0
        slippage = (buys+sells) * float(slippage_bps) / 10000.0
        stamp = sells * stamp_duty_rate(exec_date)
        total_cost = commission + slippage + stamp

        if next_exec is None:
            hdates = dates[dates >= exec_date]
        else:
            hdates = dates[(dates >= exec_date) & (dates < next_exec)]
        if len(hdates) == 0:
            continue

        held = sorted(new_holdings)

        # First day: open -> close, with execution cost.
        first = close_w.loc[exec_date, held] / open_w.loc[exec_date, held] - 1.0
        first = first.replace([np.inf,-np.inf], np.nan).dropna()
        if len(first):
            daily_ret.loc[exec_date] = float(first.mean()) - total_cost

        # Subsequent close-to-close.
        for j in range(1, len(hdates)):
            d, pd_ = hdates[j], hdates[j-1]
            rr = close_w.loc[d, held] / close_w.loc[pd_, held] - 1.0
            rr = rr.replace([np.inf,-np.inf], np.nan).dropna()
            if len(rr):
                daily_ret.loc[d] = float(rr.mean())

        score_map = g.set_index("ticker")["composite_score"].to_dict()
        for t in held:
            holdings_rows.append({
                "signal_date": sig,
                "execution_date": exec_date,
                "ticker": t,
                "weight": 1.0/n_new,
                "composite_score": score_map.get(t, np.nan),
                "blocked_sell_retained": t in blocked_sells,
            })

        trade_rows.append({
            "signal_date": sig,
            "execution_date": exec_date,
            "n_holdings": n_new,
            "buy_turnover": buys,
            "sell_turnover": sells,
            "gross_turnover": buys+sells,
            "commission_cost": commission,
            "slippage_cost": slippage,
            "stamp_duty_cost": stamp,
            "total_cost": total_cost,
            "blocked_sell_count": len(blocked_sells),
            "skipped_buy_count": max(0, len(desired)-len(tradable_buys)),
        })
        prev_holdings = new_holdings

    return daily_ret, pd.DataFrame(holdings_rows), pd.DataFrame(trade_rows)


def forward_returns_v4(prices: pd.DataFrame, panel: pd.DataFrame):
    """
    One-period forward returns between adjacent signal dates.
    If a signal date is not itself a trading day, map it to the last available
    trading date on or before the signal date. This makes diagnostics robust
    to calendar month-end labels while preserving point-in-time ordering.
    """
    px = prices.copy()
    px["date"] = pd.to_datetime(px["date"], errors="coerce")
    close = px.pivot(index="date", columns="ticker", values="close").sort_index()
    trading_dates = pd.DatetimeIndex(close.index)
    sigs = sorted(pd.to_datetime(panel["signal_date"].unique()))

    def resolve(d):
        pos = trading_dates.searchsorted(pd.Timestamp(d), side="right") - 1
        return trading_dates[pos] if pos >= 0 else None

    rows = []
    for i, s in enumerate(sigs[:-1]):
        e = sigs[i+1]
        s_trade = resolve(s)
        e_trade = resolve(e)
        if s_trade is None or e_trade is None or e_trade <= s_trade:
            continue
        fr = close.loc[e_trade] / close.loc[s_trade] - 1.0
        g = panel[panel["signal_date"] == s].copy()
        g["forward_1m_return"] = g["ticker"].map(fr.to_dict())
        g["forward_start_trade_date"] = s_trade
        g["forward_end_trade_date"] = e_trade
        rows.append(g)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def _mean_monthly_rank_ic(df, weights):
    wv, wq, wm = weights
    ics = []
    for _, g in df.groupby("signal_date"):
        comp = wv*g["value_score"] + wq*g["quality_score"] + wm*g["momentum_score"]
        tmp = pd.DataFrame({"s":comp, "r":g["forward_1m_return"]}).dropna()
        if len(tmp) >= 10:
            ics.append(tmp["s"].corr(tmp["r"], method="spearman"))
    return float(np.nanmean(ics)) if ics else np.nan


def walk_forward_factor_validation(
    prices: pd.DataFrame,
    panel: pd.DataFrame,
    train_months=24,
    test_months=6,
    weight_grid=None,
):
    """
    Expanding sequence of non-overlapping test folds.
    Candidate factor weights are selected only on the training window by mean Rank IC.
    """
    fr = forward_returns_v4(prices, panel)
    if fr.empty:
        return pd.DataFrame(), pd.DataFrame()

    dates = sorted(pd.to_datetime(fr["signal_date"].unique()))
    weight_grid = weight_grid or [
        (1/3,1/3,1/3),
        (0.50,0.25,0.25),
        (0.25,0.50,0.25),
        (0.25,0.25,0.50),
        (0.40,0.40,0.20),
        (0.20,0.40,0.40),
        (0.40,0.20,0.40),
    ]

    fold_rows = []
    test_rows = []
    start = int(train_months)
    fold_id = 1

    while start < len(dates):
        train_dates = dates[max(0, start-int(train_months)):start]
        test_dates = dates[start:min(len(dates), start+int(test_months))]
        if len(train_dates) < int(train_months) or len(test_dates) == 0:
            break

        train = fr[fr["signal_date"].isin(train_dates) & (pd.to_datetime(fr["forward_end_trade_date"]) < min(test_dates))]
        scored = []
        for w in weight_grid:
            scored.append((w, _mean_monthly_rank_ic(train, w)))
        scored = [x for x in scored if not pd.isna(x[1])]
        if not scored:
            break
        best_w, train_ic = max(scored, key=lambda x: x[1])

        test = fr[fr["signal_date"].isin(test_dates)].copy()
        test_ic = _mean_monthly_rank_ic(test, best_w)

        fold_rows.append({
            "fold": fold_id,
            "train_start": min(train_dates),
            "train_end": max(train_dates),
            "last_training_label_end": train["forward_end_trade_date"].max(),
            "test_start": min(test_dates),
            "test_end": max(test_dates),
            "value_weight": best_w[0],
            "quality_weight": best_w[1],
            "momentum_weight": best_w[2],
            "train_mean_rank_ic": train_ic,
            "test_mean_rank_ic": test_ic,
        })

        for d, g in test.groupby("signal_date"):
            comp = (
                best_w[0]*g["value_score"] +
                best_w[1]*g["quality_score"] +
                best_w[2]*g["momentum_score"]
            )
            tmp = pd.DataFrame({"score":comp, "ret":g["forward_1m_return"]}).dropna()
            ic = tmp["score"].corr(tmp["ret"], method="spearman") if len(tmp) >= 10 else np.nan
            test_rows.append({
                "fold": fold_id,
                "signal_date": d,
                "rank_ic": ic,
                "value_weight": best_w[0],
                "quality_weight": best_w[1],
                "momentum_weight": best_w[2],
            })

        fold_id += 1
        start += int(test_months)

    return pd.DataFrame(fold_rows), pd.DataFrame(test_rows)
