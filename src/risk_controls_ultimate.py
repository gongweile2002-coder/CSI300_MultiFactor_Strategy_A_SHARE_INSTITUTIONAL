
from __future__ import annotations

from pathlib import Path
import math
import numpy as np
import pandas as pd


def load_restricted_list(path):
    path = Path(path)
    if not path.exists():
        return set()
    df = pd.read_csv(path)
    if "ticker" not in df.columns:
        raise ValueError("restricted list must contain ticker")
    return set(df["ticker"].dropna().astype(str))


def validate_price_freshness(price_df: pd.DataFrame, as_of, max_staleness_days=3):
    x = price_df.copy()
    if "date" not in x.columns:
        return {"status":"BLOCK", "reason":"price_data_missing_date"}
    x["date"] = pd.to_datetime(x["date"], errors="coerce")
    latest = x["date"].max()
    if pd.isna(latest):
        return {"status":"BLOCK", "reason":"price_dates_invalid"}
    gap = (pd.Timestamp(as_of).normalize() - latest.normalize()).days
    return {
        "status":"PASS" if 0 <= gap <= int(max_staleness_days) else "BLOCK",
        "reason":"" if 0 <= gap <= int(max_staleness_days) else f"stale_price_data_{gap}d",
        "latest_price_date":latest,
        "staleness_days":gap,
    }


def portfolio_target_checks(
    targets: pd.DataFrame,
    nav_value: float,
    restricted_tickers=None,
    max_single_weight=0.10,
    max_sector_weight=0.35,
    max_gross_exposure=1.00,
):
    restricted_tickers = set(restricted_tickers or set())
    g = targets.copy()
    g["ticker"] = g["ticker"].astype(str)
    g["target_weight"] = pd.to_numeric(g["target_weight"], errors="coerce").fillna(0.0)

    rows = []
    gross = float(g["target_weight"].abs().sum())
    rows.append({
        "check":"gross_exposure",
        "status":"PASS" if gross <= float(max_gross_exposure)+1e-10 else "BLOCK",
        "value":gross,
        "limit":float(max_gross_exposure),
        "details":"",
    })

    max_w = float(g["target_weight"].max()) if len(g) else 0.0
    rows.append({
        "check":"max_single_weight",
        "status":"PASS" if max_w <= float(max_single_weight)+1e-10 else "BLOCK",
        "value":max_w,
        "limit":float(max_single_weight),
        "details":"",
    })

    restricted_hit = sorted(set(g.loc[g["target_weight"]>0, "ticker"]) & restricted_tickers)
    rows.append({
        "check":"restricted_list",
        "status":"PASS" if not restricted_hit else "BLOCK",
        "value":len(restricted_hit),
        "limit":0,
        "details":"|".join(restricted_hit),
    })

    if "industry_l1" in g.columns:
        sec = g.groupby(g["industry_l1"].fillna("UNKNOWN"))["target_weight"].sum()
        max_sec = float(sec.max()) if len(sec) else 0.0
        rows.append({
            "check":"max_sector_weight",
            "status":"PASS" if max_sec <= float(max_sector_weight)+1e-10 else "BLOCK",
            "value":max_sec,
            "limit":float(max_sector_weight),
            "details":sec.idxmax() if len(sec) else "",
        })

    return pd.DataFrame(rows)


def estimate_adv(prices: pd.DataFrame, as_of, lookback_days=20):
    x = prices.copy()
    x["date"] = pd.to_datetime(x["date"], errors="coerce")
    x = x[x["date"] <= pd.Timestamp(as_of)].sort_values(["ticker","date"])
    if "amount" not in x.columns:
        if {"close","volume"}.issubset(x.columns):
            x["amount"] = pd.to_numeric(x["close"], errors="coerce") * pd.to_numeric(x["volume"], errors="coerce")
        else:
            raise ValueError("prices require amount or close*volume")
    out = (
        x.groupby("ticker", group_keys=False)
         .tail(int(lookback_days))
         .groupby("ticker")["amount"]
         .mean()
         .rename("adv20")
         .reset_index()
    )
    return out


def order_capacity_checks(
    orders: pd.DataFrame,
    prices: pd.DataFrame,
    as_of,
    max_adv_participation_pct=0.05,
    nav_value=None,
    max_order_notional_pct_nav=0.10,
):
    if orders is None or orders.empty:
        return pd.DataFrame(columns=[
            "order_id","ticker","notional","adv20","adv_participation",
            "nav_fraction","status","issues"
        ])

    adv = estimate_adv(prices, as_of, 20).set_index("ticker")["adv20"].to_dict()
    rows = []
    for _, o in orders.iterrows():
        ticker = str(o["ticker"])
        notional = abs(float(o.get("qty",0))*float(o.get("reference_price",0)))
        a = float(adv.get(ticker, np.nan))
        part = notional/a if pd.notna(a) and a > 0 else np.nan
        nav_frac = notional/float(nav_value) if nav_value and nav_value > 0 else np.nan
        issues = []
        if pd.isna(part):
            issues.append("missing_adv")
        elif part > float(max_adv_participation_pct):
            issues.append("adv_participation_limit")
        if pd.notna(nav_frac) and nav_frac > float(max_order_notional_pct_nav):
            issues.append("order_notional_limit")

        rows.append({
            "order_id":o.get("order_id",""),
            "ticker":ticker,
            "notional":notional,
            "adv20":a,
            "adv_participation":part,
            "nav_fraction":nav_frac,
            "status":"PASS" if not issues else "BLOCK",
            "issues":"|".join(issues),
        })
    return pd.DataFrame(rows)


def deterministic_stress(targets: pd.DataFrame, market_shocks=(-0.03,-0.05,-0.10), sector_shock=-0.15):
    g = targets.copy()
    g["target_weight"] = pd.to_numeric(g["target_weight"], errors="coerce").fillna(0.0)
    rows = []

    for shock in market_shocks:
        pnl = float(g["target_weight"].sum())*float(shock)
        rows.append({
            "scenario":f"market_{int(abs(shock)*100)}pct_down",
            "estimated_portfolio_return":pnl,
            "details":"uniform market shock"
        })

    if "industry_l1" in g.columns and len(g):
        sec = (
            g.groupby(g["industry_l1"].fillna("UNKNOWN"))["target_weight"]
             .sum()
             .sort_values(ascending=False)
        )
        for sector, w in sec.head(5).items():
            rows.append({
                "scenario":f"{sector}_sector_shock",
                "estimated_portfolio_return":float(w)*float(sector_shock),
                "details":f"{sector} weight={float(w):.4f}, shock={float(sector_shock):.4f}",
            })
    return pd.DataFrame(rows)


def historical_var_cvar(prices: pd.DataFrame, targets: pd.DataFrame, as_of, lookback_days=252, confidence=0.95):
    p = prices.copy()
    p["date"] = pd.to_datetime(p["date"], errors="coerce")
    tickers = targets["ticker"].astype(str).tolist()
    weights = targets.set_index("ticker")["target_weight"].astype(float)

    close = p.pivot(index="date", columns="ticker", values="close").sort_index()
    cols = [t for t in tickers if t in close.columns]
    if len(cols) < 2:
        return {"var":np.nan, "cvar":np.nan, "n_obs":0}

    ret = close.loc[:pd.Timestamp(as_of), cols].tail(int(lookback_days)+1).pct_change().dropna(how="all")
    ret = ret.fillna(0.0)
    w = weights.reindex(cols).fillna(0.0)
    # Preserve actual equity exposure; residual cash has zero return.
    port = ret.to_numpy() @ w.to_numpy()
    if len(port) < 20:
        return {"var":np.nan, "cvar":np.nan, "n_obs":len(port)}

    alpha = 1-float(confidence)
    q = float(np.quantile(port, alpha))
    cvar = float(port[port <= q].mean()) if np.any(port <= q) else q
    return {"var":q, "cvar":cvar, "n_obs":len(port)}


def bootstrap_var_cvar(prices: pd.DataFrame, targets: pd.DataFrame, as_of, scenarios=2000, confidence=0.95, seed=123):
    p = prices.copy()
    p["date"] = pd.to_datetime(p["date"], errors="coerce")
    tickers = targets["ticker"].astype(str).tolist()
    weights = targets.set_index("ticker")["target_weight"].astype(float)
    close = p.pivot(index="date", columns="ticker", values="close").sort_index()

    cols = [t for t in tickers if t in close.columns]
    if len(cols) < 2:
        return {"bootstrap_var":np.nan, "bootstrap_cvar":np.nan, "scenarios":0}

    ret = close.loc[:pd.Timestamp(as_of), cols].tail(253).pct_change().dropna(how="all").fillna(0.0)
    w = weights.reindex(cols).fillna(0.0)
    # Preserve actual equity exposure; residual cash has zero return.

    daily = ret.to_numpy() @ w.to_numpy()
    if len(daily) < 20:
        return {"bootstrap_var":np.nan, "bootstrap_cvar":np.nan, "scenarios":0}

    rng = np.random.default_rng(seed)
    sim = rng.choice(daily, size=int(scenarios), replace=True)
    q = float(np.quantile(sim, 1-float(confidence)))
    cvar = float(sim[sim <= q].mean()) if np.any(sim <= q) else q
    return {"bootstrap_var":q, "bootstrap_cvar":cvar, "scenarios":int(scenarios)}


def combine_risk_reports(
    targets,
    prices,
    as_of,
    market_shocks=(-0.03,-0.05,-0.10),
    sector_shock=-0.15,
    confidence=0.95,
    scenarios=2000,
):
    stress = deterministic_stress(targets, market_shocks, sector_shock)
    hist = historical_var_cvar(prices, targets, as_of, 252, confidence)
    boot = bootstrap_var_cvar(prices, targets, as_of, scenarios, confidence)
    summary = pd.DataFrame([{
        "as_of":pd.Timestamp(as_of),
        "historical_var_1d":hist["var"],
        "historical_cvar_1d":hist["cvar"],
        "bootstrap_var_1d":boot["bootstrap_var"],
        "bootstrap_cvar_1d":boot["bootstrap_cvar"],
        "historical_observations":hist["n_obs"],
        "bootstrap_scenarios":boot["scenarios"],
    }])
    return summary, stress
