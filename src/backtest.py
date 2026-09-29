
import pandas as pd
import numpy as np

def _month_end_trading_dates(dates):
    s = pd.Series(pd.to_datetime(sorted(pd.unique(dates))))
    return s.groupby(s.dt.to_period("M")).max().tolist()

def run_monthly_topn_backtest(
    prices: pd.DataFrame,
    factor_scores: pd.DataFrame,
    top_n=10,
    transaction_cost_bps=10
):
    """
    简化但严格避免同日未来信息：
    月末t观察因子，下一交易日开盘逻辑近似为使用下一交易日close-to-close收益开始持有。
    这里使用日收盘价序列，实际项目可进一步替换为next-open执行。
    """
    px = prices.pivot(index="date", columns="ticker", values="close").sort_index()
    ret = px.pct_change().fillna(0)

    score_dates = sorted(pd.to_datetime(factor_scores["date"].unique()))
    trading_dates = list(px.index)

    daily_strategy = pd.Series(0.0, index=px.index)
    weight_history = []
    prev_weights = pd.Series(0.0, index=px.columns)

    for i, score_dt in enumerate(score_dates):
        eligible_next = [d for d in trading_dates if d > score_dt]
        if not eligible_next:
            break
        start_dt = eligible_next[0]

        if i + 1 < len(score_dates):
            next_score_dt = score_dates[i+1]
            eligible_end = [d for d in trading_dates if d > next_score_dt]
            end_dt = eligible_end[0] if eligible_end else trading_dates[-1]
        else:
            end_dt = trading_dates[-1]

        g = factor_scores[factor_scores["date"] == score_dt].sort_values("composite_score", ascending=False)
        picks = [t for t in g.head(top_n)["ticker"].tolist() if t in px.columns]
        if not picks:
            continue

        w = pd.Series(0.0, index=px.columns)
        w.loc[picks] = 1 / len(picks)

        turnover = (w - prev_weights).abs().sum()
        cost = turnover * transaction_cost_bps / 10000.0

        holding_dates = [d for d in trading_dates if start_dt <= d < end_dt] if end_dt != trading_dates[-1] else [d for d in trading_dates if start_dt <= d <= end_dt]
        if not holding_dates:
            continue

        port_ret = ret.loc[holding_dates, picks].mean(axis=1)
        port_ret.iloc[0] -= cost
        daily_strategy.loc[holding_dates] = port_ret

        for t in picks:
            weight_history.append({
                "signal_date": score_dt,
                "start_date": start_dt,
                "ticker": t,
                "weight": 1/len(picks),
                "composite_score": float(g.set_index("ticker").loc[t, "composite_score"])
            })
        prev_weights = w

    return daily_strategy, pd.DataFrame(weight_history)
