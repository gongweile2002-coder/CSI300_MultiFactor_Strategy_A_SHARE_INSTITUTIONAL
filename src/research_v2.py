
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pandas as pd


def winsorize(s: pd.Series, lower=0.025, upper=0.975):
    s = pd.to_numeric(s, errors="coerce")
    valid = s.dropna()
    if len(valid) < 5:
        return s
    lo, hi = valid.quantile([lower, upper])
    return s.clip(lo, hi)


def zscore(s: pd.Series):
    s = pd.to_numeric(s, errors="coerce")
    std = s.std(ddof=0)
    if pd.isna(std) or std == 0:
        return pd.Series(0.0, index=s.index)
    return (s - s.mean()) / std


def _asof_by_ticker(left: pd.DataFrame, right: pd.DataFrame, left_on: str, right_on: str):
    """
    Strict point-in-time merge:
    for each ticker, use only right-side information whose timestamp <= left timestamp.
    """
    pieces = []
    right_groups = {k: g.sort_values(right_on) for k, g in right.groupby("ticker")}
    for ticker, lg in left.groupby("ticker"):
        rg = right_groups.get(ticker)
        if rg is None or rg.empty:
            tmp = lg.copy()
            for c in right.columns:
                if c not in tmp.columns and c != "ticker":
                    tmp[c] = np.nan
            pieces.append(tmp)
            continue
        tmp = pd.merge_asof(
            lg.sort_values(left_on),
            rg.drop(columns=["ticker"]).sort_values(right_on),
            left_on=left_on,
            right_on=right_on,
            direction="backward",
            allow_exact_matches=True
        )
        tmp["ticker"] = ticker
        pieces.append(tmp)
    return pd.concat(pieces, ignore_index=True) if pieces else left.iloc[0:0].copy()


def trading_month_ends(prices: pd.DataFrame):
    d = pd.DataFrame({"date": sorted(pd.to_datetime(prices["date"]).dropna().unique())})
    d["month"] = d["date"].dt.to_period("M")
    return d.groupby("month")["date"].max().tolist()


def membership_for_date(membership: pd.DataFrame, signal_date):
    m = membership[membership["effective_date"] <= signal_date]
    if m.empty:
        return pd.DataFrame(columns=membership.columns)
    latest = m["effective_date"].max()
    return m[m["effective_date"] == latest].copy()


def build_point_in_time_panel(
    prices: pd.DataFrame,
    daily_basic: pd.DataFrame,
    fundamentals: pd.DataFrame,
    membership: pd.DataFrame,
    start_date=None,
    end_date=None,
    momentum_lookback_days=126,
    winsor_lower=0.025,
    winsor_upper=0.975,
    factor_weights=None,
    min_turnover_rate=0.0,
    min_market_cap_cny_10k=0.0,
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

    signal_dates = trading_month_ends(prices)
    if start_date is not None:
        signal_dates = [d for d in signal_dates if d >= pd.Timestamp(start_date)]
    if end_date is not None:
        signal_dates = [d for d in signal_dates if d <= pd.Timestamp(end_date)]

    # Precompute wide close for momentum.
    close = prices.pivot(index="date", columns="ticker", values="close").sort_index()

    rows = []
    for signal_date in signal_dates:
        mem = membership_for_date(membership, signal_date)
        if mem.empty:
            continue

        universe = mem[["ticker","weight"]].drop_duplicates("ticker").copy()
        universe["signal_date"] = pd.Timestamp(signal_date)

        # Point-in-time valuation: latest daily_basic <= signal date.
        left = universe[["ticker","signal_date"]].copy()
        val = daily_basic.rename(columns={"date":"valuation_date"}).copy()
        cols = [c for c in ["ticker","valuation_date","pe_ttm","pb","turnover_rate","total_mv","circ_mv"] if c in val.columns]
        val = val[cols].dropna(subset=["valuation_date"])
        left = _asof_by_ticker(left, val, "signal_date", "valuation_date")

        # Point-in-time accounting data: latest ANNOUNCEMENT date <= signal date.
        fcols = [c for c in [
            "ticker","ann_date","report_date","roe","roe_dt","profit_growth","debt_ratio"
        ] if c in fundamentals.columns]
        f = fundamentals[fcols].dropna(subset=["ann_date"])
        left = _asof_by_ticker(left, f, "signal_date", "ann_date")

        # Add index weight.
        left = left.merge(universe[["ticker","weight"]], on="ticker", how="left")

        # Momentum based only on prices known by signal_date.
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

        # Basic filters.
        if "turnover_rate" in left.columns:
            left = left[(left["turnover_rate"].fillna(0) >= float(min_turnover_rate))]
        if "total_mv" in left.columns and float(min_market_cap_cny_10k) > 0:
            left = left[left["total_mv"].fillna(0) >= float(min_market_cap_cny_10k)]

        # Prefer ROE after extraordinary items if available, else plain ROE.
        if "roe_dt" in left.columns:
            left["quality_roe"] = left["roe_dt"].fillna(left.get("roe"))
        else:
            left["quality_roe"] = left.get("roe")

        # PE/PB must be positive for valuation interpretation.
        left.loc[left["pe_ttm"] <= 0, "pe_ttm"] = np.nan
        left.loc[left["pb"] <= 0, "pb"] = np.nan

        needed = ["pe_ttm","pb","quality_roe","profit_growth","debt_ratio","momentum"]
        left = left.dropna(subset=[c for c in needed if c in left.columns])
        if len(left) < 10:
            continue

        # Log valuation ratios reduce extreme skew.
        left["inv_pe_signal"] = -np.log(left["pe_ttm"])
        left["inv_pb_signal"] = -np.log(left["pb"])

        raw_cols = [
            "inv_pe_signal","inv_pb_signal","quality_roe",
            "profit_growth","debt_ratio","momentum"
        ]
        for c in raw_cols:
            left[c] = winsorize(left[c], winsor_lower, winsor_upper)
            left[c] = zscore(left[c])

        left["value_score"] = (left["inv_pe_signal"] + left["inv_pb_signal"]) / 2
        left["quality_score"] = (
            left["quality_roe"] + left["profit_growth"] - left["debt_ratio"]
        ) / 3
        left["momentum_score"] = left["momentum"]

        for c in ["value_score","quality_score","momentum_score"]:
            left[c] = winsorize(left[c], winsor_lower, winsor_upper)
            left[c] = zscore(left[c])

        wv = factor_weights.get("value", 1/3)
        wq = factor_weights.get("quality", 1/3)
        wm = factor_weights.get("momentum", 1/3)
        total_w = wv + wq + wm
        wv, wq, wm = wv/total_w, wq/total_w, wm/total_w

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


def _next_trading_date(all_dates, d):
    all_dates = pd.DatetimeIndex(all_dates)
    pos = all_dates.searchsorted(pd.Timestamp(d), side="right")
    return all_dates[pos] if pos < len(all_dates) else None


def run_next_open_backtest(
    prices: pd.DataFrame,
    panel: pd.DataFrame,
    top_n=30,
    transaction_cost_bps=10,
):
    """
    Signal at month-end close -> trade at next available market trading day's OPEN.
    If a selected stock has no open on execution date (e.g. suspension), it is excluded.
    First holding-day return uses close/open; subsequent days use close/previous close.
    """
    prices = prices.copy()
    prices["date"] = pd.to_datetime(prices["date"])
    all_dates = pd.DatetimeIndex(sorted(prices["date"].dropna().unique()))

    open_w = prices.pivot(index="date", columns="ticker", values="open").sort_index()
    close_w = prices.pivot(index="date", columns="ticker", values="close").sort_index()

    daily_ret = pd.Series(0.0, index=all_dates)
    holdings = []
    turnover_records = []
    prev_weights = pd.Series(dtype=float)

    signal_dates = sorted(pd.to_datetime(panel["signal_date"].unique()))

    for i, signal_date in enumerate(signal_dates):
        exec_date = _next_trading_date(all_dates, signal_date)
        if exec_date is None:
            break

        next_exec = None
        if i + 1 < len(signal_dates):
            next_exec = _next_trading_date(all_dates, signal_dates[i+1])

        g = panel[panel["signal_date"] == signal_date].sort_values("composite_score", ascending=False)
        picks = g.head(int(top_n))["ticker"].tolist()

        # Tradability at next open.
        tradable = [
            t for t in picks
            if t in open_w.columns and exec_date in open_w.index and pd.notna(open_w.loc[exec_date, t])
        ]
        if not tradable:
            continue

        w = pd.Series(1.0/len(tradable), index=tradable)
        all_names = prev_weights.index.union(w.index)
        turnover = (
            w.reindex(all_names, fill_value=0) -
            prev_weights.reindex(all_names, fill_value=0)
        ).abs().sum()
        cost = turnover * float(transaction_cost_bps) / 10000.0

        if next_exec is None:
            hdates = all_dates[all_dates >= exec_date]
        else:
            hdates = all_dates[(all_dates >= exec_date) & (all_dates < next_exec)]
        if len(hdates) == 0:
            continue

        # Portfolio return: first day close/open; later close/prev close.
        first_cross = (close_w.loc[exec_date, tradable] / open_w.loc[exec_date, tradable] - 1.0)
        first_cross = first_cross.replace([np.inf,-np.inf], np.nan).dropna()
        if len(first_cross):
            daily_ret.loc[exec_date] = float(first_cross.mean()) - cost

        for j in range(1, len(hdates)):
            d = hdates[j]
            pd_ = hdates[j-1]
            rr = (close_w.loc[d, tradable] / close_w.loc[pd_, tradable] - 1.0)
            rr = rr.replace([np.inf,-np.inf], np.nan).dropna()
            if len(rr):
                daily_ret.loc[d] = float(rr.mean())

        score_map = g.set_index("ticker")["composite_score"].to_dict()
        for t in tradable:
            holdings.append({
                "signal_date": signal_date,
                "execution_date": exec_date,
                "ticker": t,
                "weight": float(w[t]),
                "composite_score": float(score_map.get(t, np.nan))
            })

        turnover_records.append({
            "signal_date": signal_date,
            "execution_date": exec_date,
            "turnover": float(turnover),
            "transaction_cost": float(cost),
            "n_holdings": len(tradable)
        })
        prev_weights = w

    return daily_ret, pd.DataFrame(holdings), pd.DataFrame(turnover_records)


def forward_returns(prices: pd.DataFrame, panel: pd.DataFrame):
    close = prices.pivot(index="date", columns="ticker", values="close").sort_index()
    dates = close.index
    records = []

    sigs = sorted(pd.to_datetime(panel["signal_date"].unique()))
    for i, s in enumerate(sigs[:-1]):
        e = sigs[i+1]
        if s not in dates or e not in dates:
            continue
        fr = close.loc[e] / close.loc[s] - 1.0
        g = panel[panel["signal_date"] == s].copy()
        g["forward_1m_return"] = g["ticker"].map(fr.to_dict())
        records.append(g)
    return pd.concat(records, ignore_index=True) if records else pd.DataFrame()


def factor_diagnostics(prices: pd.DataFrame, panel: pd.DataFrame):
    fr = forward_returns(prices, panel)
    if fr.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    factors = ["value_score","quality_score","momentum_score","composite_score"]
    ic_rows = []
    quintile_rows = []

    for d, g in fr.groupby("signal_date"):
        for f in factors:
            tmp = g[[f,"forward_1m_return"]].dropna()
            if len(tmp) >= 10:
                ic = tmp[f].corr(tmp["forward_1m_return"], method="spearman")
                ic_rows.append({"signal_date":d, "factor":f, "rank_ic":ic})

                # quintiles: 1=lowest score, 5=highest score
                try:
                    q = pd.qcut(tmp[f].rank(method="first"), 5, labels=False) + 1
                    qret = tmp.assign(quintile=q).groupby("quintile")["forward_1m_return"].mean()
                    for qn, val in qret.items():
                        quintile_rows.append({
                            "signal_date":d,
                            "factor":f,
                            "quintile":int(qn),
                            "mean_forward_return":float(val)
                        })
                except ValueError:
                    pass

    ic_ts = pd.DataFrame(ic_rows)
    if ic_ts.empty:
        ic_summary = pd.DataFrame()
    else:
        ic_summary = ic_ts.groupby("factor")["rank_ic"].agg(["mean","std","count"]).reset_index()
        ic_summary["icir"] = ic_summary["mean"] / ic_summary["std"].replace(0, np.nan)

    return fr, ic_ts, pd.DataFrame(quintile_rows), ic_summary


def performance_summary(strategy_ret: pd.Series, benchmark_ret: pd.Series | None = None, rf=0.0):
    r = strategy_ret.dropna()
    nav = (1+r).cumprod()
    years = len(r)/252
    ann_ret = nav.iloc[-1]**(1/years)-1 if len(nav) and years > 0 else np.nan
    ann_vol = r.std(ddof=0)*np.sqrt(252)
    dd = nav/nav.cummax()-1

    out = {
        "total_return": float(nav.iloc[-1]-1) if len(nav) else np.nan,
        "annualized_return": float(ann_ret),
        "annualized_volatility": float(ann_vol),
        "sharpe": float((ann_ret-rf)/ann_vol) if ann_vol and not np.isnan(ann_vol) else np.nan,
        "max_drawdown": float(dd.min()) if len(dd) else np.nan,
        "positive_day_ratio": float((r>0).mean()) if len(r) else np.nan,
    }

    if benchmark_ret is not None:
        x = pd.concat([r.rename("strategy"), benchmark_ret.rename("benchmark")], axis=1).dropna()
        if len(x) > 2:
            active = x["strategy"] - x["benchmark"]
            te = active.std(ddof=0)*np.sqrt(252)
            out["tracking_error"] = float(te)
            out["information_ratio"] = float(active.mean()*252/te) if te > 0 else np.nan
            varb = x["benchmark"].var()
            if varb > 0:
                beta = x.cov().loc["strategy","benchmark"]/varb
                alpha_d = x["strategy"].mean() - beta*x["benchmark"].mean()
                out["beta"] = float(beta)
                out["alpha_annualized"] = float(alpha_d*252)
    return out


def load_real_data(data_dir):
    data_dir = Path(data_dir)
    prices = pd.read_csv(data_dir/"prices.csv", parse_dates=["date"])
    daily_basic = pd.read_csv(data_dir/"daily_basic.csv", parse_dates=["date"])
    fundamentals = pd.read_csv(data_dir/"fundamentals_raw.csv", parse_dates=["ann_date","report_date"])
    membership = pd.read_csv(data_dir/"index_membership.csv", parse_dates=["effective_date"])
    benchmark = pd.read_csv(data_dir/"benchmark.csv", parse_dates=["date"])
    return prices, daily_basic, fundamentals, membership, benchmark
