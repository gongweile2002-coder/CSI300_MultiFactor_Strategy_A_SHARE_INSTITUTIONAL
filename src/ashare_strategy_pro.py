
from __future__ import annotations

import numpy as np
import pandas as pd

from .ashare_regime import compute_market_regime, regime_weight_map
from .ashare_factors import augment_ashare_factors
from .portfolio_v5 import estimate_covariance, optimize_portfolio


def _factor_col(name):
    if name in {"value","quality"}:
        return name+"_score"
    return name+"_score"


def adaptive_score_cross_section(x: pd.DataFrame, weights: dict):
    g = x.copy()
    score = pd.Series(0.0, index=g.index)
    used = 0.0
    for factor, w in weights.items():
        col = _factor_col(factor)
        if col in g.columns and float(w) != 0:
            v = pd.to_numeric(g[col], errors="coerce").fillna(0.0)
            score = score + float(w)*v
            used += abs(float(w))
    if used == 0:
        g["ashare_score"] = 0.0
    else:
        g["ashare_score"] = score / used
    return g


def build_adaptive_ashare_targets(
    prices: pd.DataFrame,
    benchmark: pd.DataFrame,
    daily_basic: pd.DataFrame,
    panel: pd.DataFrame,
    industry_membership,
    signal_date,
    config,
    prev_weights=None,
):
    """
    Build a regime-adaptive, A-share-specific target portfolio for one signal date.

    Returns:
      targets, regime_summary, scored_cross_section, risk_stats
    """
    g = panel[pd.to_datetime(panel["signal_date"]) == pd.Timestamp(signal_date)].copy()
    if g.empty:
        raise ValueError(f"No factor cross-section for {signal_date}")

    tickers = g["ticker"].astype(str).tolist()
    regime = compute_market_regime(
        prices,
        benchmark,
        signal_date,
        universe_tickers=tickers,
        breadth_ma_days=config["ashare_pro"]["breadth_ma_days"],
        turnover_short_days=config["ashare_pro"]["turnover_short_days"],
        turnover_long_days=config["ashare_pro"]["turnover_long_days"],
    )
    gross, weights = regime_weight_map(regime["regime"], config)

    x = augment_ashare_factors(
        g, prices, daily_basic, industry_membership, signal_date, config
    )
    x = adaptive_score_cross_section(x, weights)

    # Do not initiate names that are already limit-up or obvious volume blow-offs.
    investable = x[~x["new_buy_block"]].copy()
    if len(investable) < 10:
        investable = x.copy()

    # Portfolio optimizer expects composite_score.
    investable["composite_score_original"] = investable.get("composite_score", np.nan)
    investable["composite_score"] = investable["ashare_score"]

    cfg = config
    candidates = investable.sort_values("composite_score", ascending=False).head(int(cfg["max_names"]))
    cov, cov_end = estimate_covariance(
        prices,
        candidates["ticker"].astype(str).tolist(),
        signal_date,
        lookback_days=cfg["risk_lookback_days"],
        shrinkage=cfg["covariance_shrinkage"],
    )
    if cov.empty:
        raise RuntimeError("Covariance matrix unavailable")

    target, risk = optimize_portfolio(
        investable,
        cov,
        prev_weights=prev_weights,
        max_names=cfg["max_names"],
        max_stock_weight=cfg["max_stock_weight"],
        min_stock_weight=cfg["min_stock_weight"],
        sector_active_limit=cfg["sector_active_limit"],
        tracking_error_limit_annual=cfg["tracking_error_limit_annual"],
        alpha_strength=cfg["alpha_strength"],
        risk_aversion=cfg["risk_aversion"],
        turnover_penalty=cfg["turnover_penalty"],
    )

    # Dynamic cash exposure is a deliberate A-share risk control.
    target["target_weight_full_invested"] = target["target_weight"]
    target["target_weight"] = target["target_weight"] * gross
    target["cash_weight"] = 1.0 - target["target_weight"].sum()
    target["regime"] = regime["regime"]
    target["gross_exposure_target"] = gross
    target["signal_date"] = pd.Timestamp(signal_date)
    target["covariance_end_date"] = cov_end

    # Recompute active weight relative to the original benchmark weights only as a display aid.
    if "benchmark_weight" in target.columns:
        target["active_weight_scaled"] = target["target_weight"] - target["benchmark_weight"]

    regime["gross_exposure_target"] = gross
    regime["factor_weights"] = weights
    regime["covariance_end_date"] = cov_end
    return target, regime, x, risk


def build_adaptive_history(
    prices, benchmark, daily_basic, panel, industry_membership, config
):
    targets = []
    regimes = []
    scored = []
    prev = pd.Series(dtype=float)

    for sig in sorted(pd.to_datetime(panel["signal_date"].unique())):
        try:
            t, r, x, risk = build_adaptive_ashare_targets(
                prices, benchmark, daily_basic, panel, industry_membership,
                sig, config, prev_weights=prev
            )
        except Exception:
            continue

        targets.append(t)
        rr = dict(r)
        rr["factor_weights"] = json_safe_dict(rr.get("factor_weights", {}))
        regimes.append(rr)
        x = x.copy()
        x["regime"] = r["regime"]
        scored.append(x)
        prev = t.set_index("ticker")["target_weight"]

    return (
        pd.concat(targets, ignore_index=True) if targets else pd.DataFrame(),
        pd.DataFrame(regimes),
        pd.concat(scored, ignore_index=True) if scored else pd.DataFrame(),
    )


def json_safe_dict(d):
    return ";".join(f"{k}:{float(v):.4f}" for k,v in d.items())
