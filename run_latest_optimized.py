
import argparse
import json
from pathlib import Path
import pandas as pd

from src.research_v2 import load_real_data
from src.latest_selection import build_latest_selection
from src.portfolio_v5 import estimate_covariance, optimize_portfolio

BASE=Path(__file__).resolve().parent

def main():
    p=argparse.ArgumentParser(description="Latest CSI300 optimized portfolio")
    p.add_argument("--as-of",default=None)
    p.add_argument("--top-n",type=int,default=None)
    args=p.parse_args()

    cfg=json.loads((BASE/"config.example.json").read_text(encoding="utf-8"))
    as_of=args.as_of or cfg.get("latest_as_of") or cfg["end_date"]
    top_n=args.top_n or cfg["max_names"]

    prices,daily_basic,fundamentals,membership,benchmark=load_real_data(BASE/"data"/"real")
    meta_path=BASE/"data"/"real"/"stock_metadata.csv"
    meta=pd.read_csv(meta_path) if meta_path.exists() else pd.DataFrame()

    # Latest factor cross-section from v3 logic.
    _,full=build_latest_selection(
        prices,daily_basic,fundamentals,membership,meta,
        as_of=as_of,
        top_n=max(100,top_n),
        momentum_lookback_days=cfg["momentum_lookback_days"],
        winsor_lower=cfg["winsor_lower"],
        winsor_upper=cfg["winsor_upper"],
        factor_weights=cfg["factor_weights"],
        min_turnover_rate=cfg["min_turnover_rate"],
        min_market_cap_cny_10k=cfg["min_market_cap_cny_10k"],
    )

    candidates=full.sort_values("composite_score",ascending=False).head(int(top_n))
    cov,cov_end=estimate_covariance(
        prices,candidates["ticker"].astype(str).tolist(),as_of,
        lookback_days=cfg["risk_lookback_days"],
        shrinkage=cfg["covariance_shrinkage"],
    )

    # If latest-selection output has no PIT industry column, fall back to metadata industry.
    if "industry_l1" not in candidates.columns:
        if "industry" in candidates.columns:
            candidates=candidates.rename(columns={"industry":"industry_l1"})
        else:
            candidates["industry_l1"]="UNKNOWN"

    portfolio,stats=optimize_portfolio(
        candidates,cov,
        prev_weights=None,
        max_names=top_n,
        max_stock_weight=cfg["max_stock_weight"],
        min_stock_weight=cfg["min_stock_weight"],
        sector_active_limit=cfg["sector_active_limit"],
        tracking_error_limit_annual=cfg["tracking_error_limit_annual"],
        alpha_strength=cfg["alpha_strength"],
        risk_aversion=cfg["risk_aversion"],
        turnover_penalty=0.0,
    )

    name_map=candidates.set_index("ticker")["name"].to_dict() if "name" in candidates.columns else {}
    portfolio["name"]=portfolio["ticker"].map(name_map)
    portfolio["as_of"]=pd.Timestamp(as_of)
    portfolio["covariance_end_date"]=cov_end
    portfolio["ex_ante_vol_annual"]=stats.get("ex_ante_vol_annual")
    portfolio["ex_ante_tracking_error_annual"]=stats.get("ex_ante_tracking_error_annual")

    out=BASE/"outputs"/"latest_v5"
    out.mkdir(parents=True,exist_ok=True)
    tag=pd.Timestamp(as_of).strftime("%Y-%m-%d")
    path=out/f"latest_optimized_portfolio_{tag}.csv"
    portfolio.to_csv(path,index=False,encoding="utf-8-sig")

    print(f"Saved: {path}")
    cols=[c for c in [
        "ticker","name","industry_l1","target_weight","benchmark_weight",
        "active_weight","composite_score","optimizer_status"
    ] if c in portfolio.columns]
    print(portfolio[cols].to_string(index=False))
    print()
    print("Ex-ante annual vol:",round(float(stats.get("ex_ante_vol_annual",float("nan"))),4))
    print("Ex-ante annual TE:",round(float(stats.get("ex_ante_tracking_error_annual",float("nan"))),4))

if __name__=="__main__":
    main()
