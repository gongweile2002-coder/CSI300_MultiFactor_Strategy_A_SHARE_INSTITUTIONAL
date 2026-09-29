
import argparse
import json
import os
from pathlib import Path
import pandas as pd

from src.tushare_provider import TushareDownloader

BASE = Path(__file__).resolve().parent

def load_dotenv_simple(path):
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip())

def main():
    parser = argparse.ArgumentParser(description="Refresh recent real data for latest stock selection.")
    parser.add_argument("--as-of", default=None)
    parser.add_argument("--lookback-days", type=int, default=260)
    parser.add_argument("--sleep", type=float, default=0.06)
    args = parser.parse_args()

    cfg = json.loads((BASE/"config.example.json").read_text(encoding="utf-8"))
    as_of = pd.Timestamp(args.as_of or cfg.get("latest_as_of") or cfg["end_date"])
    start = as_of - pd.Timedelta(days=max(args.lookback_days, 230))

    load_dotenv_simple(BASE/".env")
    token = os.getenv("TUSHARE_TOKEN", "")

    dl = TushareDownloader(
        token=token,
        output_dir=BASE/"data"/"real",
        sleep_seconds=args.sleep
    )

    # Pull membership over a broader recent window, then infer the active constituent set.
    membership = dl.fetch_index_membership(
        cfg["index_code"],
        start - pd.Timedelta(days=120),
        as_of
    )
    dl.fetch_stock_metadata()

    latest_eff = membership.loc[membership["effective_date"] <= as_of, "effective_date"].max()
    active = sorted(
        membership.loc[membership["effective_date"] == latest_eff, "ticker"]
        .dropna().unique().tolist()
    )

    # Need extra history for momentum.
    px_start = as_of - pd.Timedelta(days=260)
    dl.fetch_prices_for_tickers(active, px_start, as_of)
    dl.fetch_daily_basic_for_tickers(active, px_start, as_of)

    # Financial announcement history: use a longer window so every stock has a latest published report.
    dl.fetch_fundamentals_for_tickers(active, as_of - pd.Timedelta(days=800), as_of)
    dl.fetch_benchmark(cfg["index_code"], px_start, as_of)

    print(f"Refreshed latest dataset for {len(active)} constituents as of {as_of.date()}.")

if __name__ == "__main__":
    main()
