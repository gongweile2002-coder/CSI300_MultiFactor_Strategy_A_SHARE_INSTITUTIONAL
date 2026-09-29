
import numpy as np
import pandas as pd

TRADING_DAYS = 252

def max_drawdown(nav: pd.Series):
    peak = nav.cummax()
    dd = nav / peak - 1
    return float(dd.min())

def performance_metrics(strategy_returns: pd.Series, benchmark_returns: pd.Series | None = None, rf=0.0):
    r = strategy_returns.dropna()
    if len(r) == 0:
        raise ValueError("策略收益序列为空")

    nav = (1+r).cumprod()
    years = len(r) / TRADING_DAYS
    total_return = nav.iloc[-1] - 1
    ann_return = nav.iloc[-1] ** (1/years) - 1 if years > 0 else np.nan
    ann_vol = r.std(ddof=0) * np.sqrt(TRADING_DAYS)
    sharpe = (ann_return - rf) / ann_vol if ann_vol > 0 else np.nan

    out = {
        "total_return": total_return,
        "annualized_return": ann_return,
        "annualized_volatility": ann_vol,
        "sharpe": sharpe,
        "max_drawdown": max_drawdown(nav),
    }

    if benchmark_returns is not None:
        b = benchmark_returns.reindex(r.index).dropna()
        aligned = pd.concat([r, b], axis=1).dropna()
        aligned.columns = ["strategy", "benchmark"]
        if len(aligned) > 2 and aligned["benchmark"].var() > 0:
            beta = aligned.cov().loc["strategy","benchmark"] / aligned["benchmark"].var()
            alpha_daily = aligned["strategy"].mean() - beta*aligned["benchmark"].mean()
            out["beta"] = float(beta)
            out["alpha_annualized"] = float(alpha_daily * TRADING_DAYS)

    return out
