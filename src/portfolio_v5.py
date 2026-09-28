
from __future__ import annotations

import math
from dataclasses import dataclass
import numpy as np
import pandas as pd

try:
    from scipy.optimize import minimize
except Exception:
    minimize = None


TRADING_DAYS = 252


def _nearest_trade_date(index: pd.DatetimeIndex, d, side="left"):
    d = pd.Timestamp(d)
    if side == "left":
        pos = index.searchsorted(d, side="right") - 1
        return index[pos] if pos >= 0 else None
    pos = index.searchsorted(d, side="right")
    return index[pos] if pos < len(index) else None


def estimate_covariance(
    prices: pd.DataFrame,
    tickers: list[str],
    signal_date,
    lookback_days=126,
    shrinkage=0.35,
) -> tuple[pd.DataFrame, pd.Timestamp | None]:
    """
    Daily return covariance with simple diagonal shrinkage:
        Sigma = (1-s)*sample_cov + s*diag(sample_cov)
    """
    p = prices.copy()
    p["date"] = pd.to_datetime(p["date"], errors="coerce")
    close = p.pivot(index="date", columns="ticker", values="close").sort_index()
    close = close[[t for t in tickers if t in close.columns]]

    if close.empty:
        return pd.DataFrame(), None

    end = _nearest_trade_date(pd.DatetimeIndex(close.index), signal_date, "left")
    if end is None:
        return pd.DataFrame(), None

    sub = close.loc[:end].tail(int(lookback_days) + 1)
    ret = sub.pct_change().dropna(how="all")
    ret = ret.dropna(axis=1, thresh=max(10, int(len(ret) * 0.8)))
    if ret.shape[1] < 2:
        return pd.DataFrame(), end

    cov = ret.cov().fillna(0.0)
    diag = pd.DataFrame(np.diag(np.diag(cov.values)), index=cov.index, columns=cov.columns)
    s = float(np.clip(shrinkage, 0.0, 1.0))
    shrunk = (1.0 - s) * cov + s * diag

    # Small ridge for numerical stability.
    shrunk += np.eye(len(shrunk)) * 1e-8
    return shrunk, end


def benchmark_weights_from_panel(g: pd.DataFrame, tickers: list[str]) -> pd.Series:
    if "weight" not in g.columns:
        return pd.Series(1.0/len(tickers), index=tickers)

    b = g.set_index("ticker")["weight"].reindex(tickers).fillna(0.0).astype(float)
    # Tushare index_weight is usually percentage points, so normalize regardless of scale.
    if b.sum() <= 0:
        b[:] = 1.0
    b = b / b.sum()
    return b


def sector_matrix(g: pd.DataFrame, tickers: list[str]) -> tuple[pd.DataFrame, list[str]]:
    if "industry_l1" not in g.columns:
        return pd.DataFrame(index=tickers), []

    ind = g.set_index("ticker")["industry_l1"].reindex(tickers).fillna("UNKNOWN").astype(str)
    sectors = sorted(ind.unique().tolist())
    mat = pd.DataFrame(0.0, index=tickers, columns=sectors)
    for t in tickers:
        mat.loc[t, ind.loc[t]] = 1.0
    return mat, sectors


def portfolio_risk_stats(
    weights: pd.Series,
    cov_daily: pd.DataFrame,
    benchmark_weights: pd.Series | None = None,
):
    names = [t for t in weights.index if t in cov_daily.index]
    if not names:
        return {}

    w = weights.reindex(names).fillna(0.0).to_numpy(float)
    cov = cov_daily.loc[names, names].to_numpy(float)
    port_var = float(w @ cov @ w)
    port_vol_ann = math.sqrt(max(port_var, 0.0) * TRADING_DAYS)

    out = {"ex_ante_vol_annual": port_vol_ann}

    if benchmark_weights is not None:
        b = benchmark_weights.reindex(names).fillna(0.0).to_numpy(float)
        active = w - b
        te_var = float(active @ cov @ active)
        out["ex_ante_tracking_error_annual"] = math.sqrt(max(te_var, 0.0) * TRADING_DAYS)

    # Marginal and component risk contribution.
    if port_var > 0:
        mrc = cov @ w
        crc = w * mrc / port_var
        out["risk_contribution"] = pd.Series(crc, index=names)
    return out


def _smooth_abs(x, eps=1e-8):
    return np.sqrt(x*x + eps)


def optimize_portfolio(
    cross_section: pd.DataFrame,
    cov_daily: pd.DataFrame,
    prev_weights: pd.Series | None = None,
    max_names=30,
    max_stock_weight=0.08,
    min_stock_weight=0.0,
    sector_active_limit=0.08,
    tracking_error_limit_annual=0.12,
    alpha_strength=1.0,
    risk_aversion=8.0,
    turnover_penalty=0.8,
):
    """
    Benchmark-aware long-only optimization.

    Objective:
      maximize alpha_strength * alpha'w
               - risk_aversion * w'Σw
               - turnover_penalty * smooth_L1(w-prev)

    Constraints:
      sum(w)=1
      min_weight <= w_i <= max_weight
      industry active weight within +/- sector_active_limit
      ex-ante tracking error <= tracking_error_limit_annual

    Candidate universe is pre-screened to top `max_names` by composite score.
    """
    if minimize is None:
        raise ImportError("scipy 未安装，请运行 pip install -r requirements.txt")

    g = cross_section.sort_values("composite_score", ascending=False).head(int(max_names)).copy()
    tickers = [t for t in g["ticker"].astype(str).tolist() if t in cov_daily.index]
    if len(tickers) < 2:
        raise RuntimeError("优化器可用股票数量不足")

    g = g[g["ticker"].isin(tickers)].copy()
    g = g.drop_duplicates("ticker").set_index("ticker").reindex(tickers)

    alpha = pd.to_numeric(g["composite_score"], errors="coerce").fillna(0.0)
    # Cross-sectional standardization prevents arbitrary score scale from dominating.
    if alpha.std(ddof=0) > 0:
        alpha = (alpha-alpha.mean())/alpha.std(ddof=0)

    cov = cov_daily.loc[tickers, tickers].to_numpy(float)
    b = benchmark_weights_from_panel(g.reset_index(), tickers)

    prev = pd.Series(0.0, index=tickers)
    if prev_weights is not None and len(prev_weights):
        prev = prev_weights.reindex(tickers).fillna(0.0).astype(float)
        if prev.sum() > 0:
            # Keep prior weights on the candidate subset in their actual scale,
            # do not renormalize: selling names outside this subset is turnover.
            pass

    sector_mat, sectors = sector_matrix(g.reset_index(), tickers)
    A = sector_mat.to_numpy(float) if len(sectors) else np.empty((len(tickers), 0))
    bvec = b.to_numpy(float)

    n = len(tickers)
    lo = float(min_stock_weight)
    hi = float(max_stock_weight)

    if not (0 <= lo <= hi <= 1) or hi*n < 1-1e-10 or lo*n > 1+1e-10:
        raise ValueError('权重约束不可行；不会自动放宽单股上限')

    x0 = b.copy()
    if x0.sum() <= 0:
        x0[:] = 1.0/n
    x0 = x0.clip(lower=lo, upper=hi)
    if x0.sum() == 0:
        x0[:] = 1.0/n
    x0 = x0 / x0.sum()

    def objective(w):
        alpha_term = float(alpha.to_numpy() @ w)
        risk_term = float(w @ cov @ w) * TRADING_DAYS
        turnover = float(_smooth_abs(w - prev.to_numpy()).sum())
        return (
            -float(alpha_strength)*alpha_term
            + float(risk_aversion)*risk_term
            + float(turnover_penalty)*turnover
        )

    constraints = [{"type":"eq", "fun":lambda w: np.sum(w)-1.0}]

    # Sector active limits relative to benchmark within candidate set.
    if A.shape[1] > 0 and sector_active_limit is not None:
        bench_sector = A.T @ bvec
        lim = float(sector_active_limit)
        for j in range(A.shape[1]):
            a = A[:, j].copy()
            bs = float(bench_sector[j])
            constraints.append({"type":"ineq", "fun":lambda w, a=a, bs=bs: lim - (a@w - bs)})
            constraints.append({"type":"ineq", "fun":lambda w, a=a, bs=bs: lim + (a@w - bs)})

    if tracking_error_limit_annual is not None and float(tracking_error_limit_annual) > 0:
        te2 = (float(tracking_error_limit_annual) ** 2) / TRADING_DAYS
        constraints.append({
            "type":"ineq",
            "fun":lambda w: te2 - float((w-bvec) @ cov @ (w-bvec))
        })

    result = minimize(
        objective,
        x0.to_numpy(float),
        method="SLSQP",
        bounds=[(lo, hi)]*n,
        constraints=constraints,
        options={"maxiter":1000, "ftol":1e-10, "disp":False},
    )

    if not result.success:
        raise RuntimeError(f'优化失败: {result.message}')
    w = np.asarray(result.x, dtype=float)
    if not np.isfinite(w).all() or (w < lo-1e-7).any() or (w > hi+1e-7).any():
        raise RuntimeError('优化输出不满足权重约束')
    for constraint in constraints:
        value = float(constraint['fun'](w))
        if not np.isfinite(value) or (abs(value)>1e-7 if constraint['type']=='eq' else value < -1e-7):
            raise RuntimeError('优化输出未通过全部约束复核')
    status = 'optimized'

    out = pd.DataFrame({
        "ticker":tickers,
        "target_weight":w,
        "benchmark_weight":b.reindex(tickers).values,
        "composite_score":g["composite_score"].values,
        "industry_l1":g["industry_l1"].values if "industry_l1" in g.columns else "UNKNOWN",
    })
    out["active_weight"] = out["target_weight"] - out["benchmark_weight"]
    out["optimizer_status"] = status
    out = out.sort_values("target_weight", ascending=False).reset_index(drop=True)

    stats = portfolio_risk_stats(
        out.set_index("ticker")["target_weight"],
        cov_daily,
        out.set_index("ticker")["benchmark_weight"],
    )
    return out, stats


def build_optimized_targets(
    prices: pd.DataFrame,
    panel: pd.DataFrame,
    risk_lookback_days=126,
    covariance_shrinkage=0.35,
    max_names=30,
    max_stock_weight=0.08,
    min_stock_weight=0.0,
    sector_active_limit=0.08,
    tracking_error_limit_annual=0.12,
    alpha_strength=1.0,
    risk_aversion=8.0,
    turnover_penalty=0.8,
):
    all_rows = []
    risk_rows = []
    prev = pd.Series(dtype=float)

    for sig in sorted(pd.to_datetime(panel["signal_date"].unique())):
        g = panel[panel["signal_date"] == sig].copy()
        candidates = g.sort_values("composite_score", ascending=False).head(int(max_names))
        cov, cov_end = estimate_covariance(
            prices,
            candidates["ticker"].astype(str).tolist(),
            sig,
            lookback_days=risk_lookback_days,
            shrinkage=covariance_shrinkage
        )
        if cov.empty or len(cov) < 2:
            raise RuntimeError(f"{sig}: 协方差数据不足")

        try:
            target, stats = optimize_portfolio(
                g,
                cov,
                prev_weights=prev,
                max_names=max_names,
                max_stock_weight=max_stock_weight,
                min_stock_weight=min_stock_weight,
                sector_active_limit=sector_active_limit,
                tracking_error_limit_annual=tracking_error_limit_annual,
                alpha_strength=alpha_strength,
                risk_aversion=risk_aversion,
                turnover_penalty=turnover_penalty,
            )
        except Exception as exc:
            raise RuntimeError(f"{sig}: 优化中止") from exc

        target["signal_date"] = sig
        target["covariance_end_date"] = cov_end
        all_rows.append(target)

        risk_rows.append({
            "signal_date":sig,
            "covariance_end_date":cov_end,
            "ex_ante_vol_annual":stats.get("ex_ante_vol_annual", np.nan),
            "ex_ante_tracking_error_annual":stats.get("ex_ante_tracking_error_annual", np.nan),
            "optimizer_status":target["optimizer_status"].iloc[0] if not target.empty else "none",
            "n_names":len(target),
        })

        prev = target.set_index("ticker")["target_weight"]

    targets = pd.concat(all_rows, ignore_index=True) if all_rows else pd.DataFrame()
    risks = pd.DataFrame(risk_rows)
    return targets, risks


def run_optimized_backtest_v5(prices, targets, stock_limits=None, commission_bps=3.0,
                              slippage_bps=2.0, use_price_limits=True):
    """Compatibility entry: cash/holdings accounting, next-session open execution."""
    from .accounting_backtest_v8 import run_accounting_backtest
    return run_accounting_backtest(prices, targets, stock_limits, commission_bps,
                                   slippage_bps, use_price_limits)


def portfolio_diagnostics(targets: pd.DataFrame):
    if targets.empty:
        return pd.DataFrame()

    rows = []
    for sig, g in targets.groupby("signal_date"):
        w = g["target_weight"].astype(float)
        active = g["active_weight"].astype(float) if "active_weight" in g.columns else pd.Series(0.0, index=g.index)
        rows.append({
            "signal_date":sig,
            "n_names":int((w > 1e-8).sum()),
            "max_weight":float(w.max()),
            "effective_n":float(1.0/(w.pow(2).sum())) if w.pow(2).sum() > 0 else np.nan,
            "active_share_candidate_set":float(0.5*active.abs().sum()),
            "top5_weight":float(w.nlargest(min(5, len(w))).sum()),
        })
    return pd.DataFrame(rows)
