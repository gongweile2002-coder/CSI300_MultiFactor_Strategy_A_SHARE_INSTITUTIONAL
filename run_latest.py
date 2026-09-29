
import argparse
import json
from pathlib import Path
import pandas as pd

from src.research_v2 import load_real_data
from src.latest_selection import build_latest_selection, build_freshness_summary

BASE = Path(__file__).resolve().parent

def main():
    parser = argparse.ArgumentParser(description="Generate latest CSI300 multi-factor stock selection.")
    parser.add_argument("--as-of", default=None, help="YYYY-MM-DD; defaults to config latest_as_of")
    parser.add_argument("--top-n", type=int, default=None)
    args = parser.parse_args()

    cfg = json.loads((BASE/"config.example.json").read_text(encoding="utf-8"))
    as_of = args.as_of or cfg.get("latest_as_of") or cfg["end_date"]
    top_n = args.top_n or cfg.get("latest_top_n", cfg["top_n"])

    prices, daily_basic, fundamentals, membership, benchmark = load_real_data(BASE/"data"/"real")

    metadata_path = BASE/"data"/"real"/"stock_metadata.csv"
    metadata = pd.read_csv(metadata_path) if metadata_path.exists() else pd.DataFrame()

    top, full = build_latest_selection(
        prices=prices,
        daily_basic=daily_basic,
        fundamentals=fundamentals,
        membership=membership,
        metadata=metadata,
        as_of=as_of,
        top_n=top_n,
        momentum_lookback_days=cfg["momentum_lookback_days"],
        winsor_lower=cfg["winsor_lower"],
        winsor_upper=cfg["winsor_upper"],
        factor_weights=cfg["factor_weights"],
        min_turnover_rate=cfg["min_turnover_rate"],
        min_market_cap_cny_10k=cfg["min_market_cap_cny_10k"],
    )

    out = BASE/"outputs"/"latest"
    out.mkdir(parents=True, exist_ok=True)

    date_tag = pd.Timestamp(as_of).strftime("%Y-%m-%d")
    top_path = out/f"latest_selection_{date_tag}.csv"
    full_path = out/f"latest_cross_section_{date_tag}.csv"
    fresh_path = out/f"data_freshness_{date_tag}.csv"

    top.to_csv(top_path, index=False, encoding="utf-8-sig")
    full.to_csv(full_path, index=False, encoding="utf-8-sig")
    build_freshness_summary(full).to_csv(fresh_path, index=False, encoding="utf-8-sig")

    print(f"Latest selection generated: {top_path}")
    print(f"Cross-section: {full_path}")
    print(f"Freshness report: {fresh_path}")
    print()
    show_cols = [c for c in [
        "rank","ticker","name","industry","report_period","ann_date",
        "value_score","quality_score","momentum_score","composite_score"
    ] if c in top.columns]
    print(top[show_cols].to_string(index=False))

if __name__ == "__main__":
    main()
