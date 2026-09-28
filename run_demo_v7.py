
from pathlib import Path
import pandas as pd
import numpy as np

from src.paper_trading_v7 import (
    init_account, generate_orders, pre_trade_checks,
    execute_orders, reconcile, save_account
)

BASE = Path(__file__).resolve().parent
OUT = BASE/"outputs"/"demo_v7"
OUT.mkdir(parents=True, exist_ok=True)

def latest_targets():
    path = BASE/"outputs"/"demo_v6"/"optimized_targets_v6.csv"
    x = pd.read_csv(path, parse_dates=["signal_date"])
    latest = x["signal_date"].max()
    return x[x["signal_date"]==latest].copy(), latest

def demo_prices(trade_date):
    prices = pd.read_csv(BASE/"data"/"demo_prices.csv", parse_dates=["date"])
    prices = prices.sort_values(["ticker","date"])
    # Use last available close as reference/open proxy for offline paper demo.
    eligible = prices[prices["date"] <= trade_date]
    use_date = eligible["date"].max()
    snap = eligible[eligible["date"]==use_date].copy()
    return dict(zip(snap["ticker"], snap["close"])), snap, use_date

def main():
    targets, signal_date = latest_targets()
    prices, snap, price_date = demo_prices(pd.Timestamp("2025-12-31"))

    account = init_account(500000, "2025-12-31", "DEMO-PAPER")
    save_account(account, OUT/"paper_account_before.json")

    orders = generate_orders(
        account, targets, prices, "2025-12-31",
        lot_size=100,
        cash_buffer_pct=0.01,
        max_single_weight=0.10,
        enforce_t_plus_one=True,
        allow_short=False,
    )
    checks = pre_trade_checks(
        account, orders, prices,
        max_single_weight=0.10,
        allow_short=False,
        enforce_t_plus_one=True,
    )
    orders = orders.merge(checks, on=["order_id","ticker","side","qty"], how="left")
    orders.to_csv(OUT/"orders.csv", index=False)

    executable = orders[(orders["status"]=="NEW") & (orders["check_status"]=="PASS")].copy()
    account, fills = execute_orders(
        account, executable, prices, "2025-12-31",
        commission_bps=3,
        min_commission_cny=5,
        slippage_bps=2,
        enforce_t_plus_one=True,
    )
    fills.to_csv(OUT/"fills.csv", index=False)
    save_account(account, OUT/"paper_account_after.json")

    summary, positions = reconcile(account, prices, "2025-12-31")
    summary.to_csv(OUT/"account_summary.csv", index=False)
    positions.to_csv(OUT/"positions.csv", index=False)

    print("Demo V7 paper trading completed.")
    print("\nOrders:")
    print(orders.head(20).to_string(index=False))
    print("\nFills:")
    print(fills.head(20).to_string(index=False))
    print("\nAccount:")
    print(summary.to_string(index=False))
    print("\nPositions:")
    print(positions.head(20).to_string(index=False))

if __name__=="__main__":
    main()
