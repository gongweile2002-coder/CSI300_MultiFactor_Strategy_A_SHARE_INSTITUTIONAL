
from pathlib import Path
import json
import pandas as pd
import numpy as np

from src.paper_trading_v7 import (
    init_account, generate_orders, pre_trade_checks, execute_orders, reconcile
)
from src.ops_ultimate import (
    ensure_no_duplicate_orders, approve_orders, build_run_manifest, save_manifest
)
from src.risk_controls_ultimate import (
    portfolio_target_checks, order_capacity_checks, combine_risk_reports
)
from src.ledger_ultimate import Ledger

BASE = Path(__file__).resolve().parent
OUT = BASE/"outputs"/"ultimate_demo"
OUT.mkdir(parents=True, exist_ok=True)

def main():
    cfg = json.loads((BASE/"config.example.json").read_text(encoding="utf-8"))

    targets = pd.read_csv(BASE/"outputs"/"demo_v6"/"optimized_targets_v6.csv", parse_dates=["signal_date"])
    targets = targets[targets["signal_date"]==targets["signal_date"].max()].copy()

    prices = pd.read_csv(BASE/"data"/"demo_prices.csv", parse_dates=["date"])
    prices = prices.sort_values(["ticker","date"])
    # Add amount for capacity checks if needed.
    if "amount" not in prices.columns:
        prices["amount"] = prices["close"]*prices["volume"]

    as_of = prices["date"].max()
    snap = prices[prices["date"]==as_of].copy()
    price_map = dict(zip(snap["ticker"], snap["close"]))

    account = init_account(500000, as_of, "ULTIMATE-DEMO")
    summary0, _ = reconcile(account, price_map, as_of)

    target_checks = portfolio_target_checks(
        targets, 500000,
        restricted_tickers=set(),
        max_single_weight=0.10,
        max_sector_weight=0.60,  # demo universe is intentionally small
        max_gross_exposure=1.05
    )
    target_checks.to_csv(OUT/"target_risk_checks.csv", index=False)

    orders = generate_orders(
        account, targets, price_map, as_of,
        lot_size=100, cash_buffer_pct=0.01,
        max_single_weight=0.10,
        enforce_t_plus_one=True, allow_short=False
    )
    pre = pre_trade_checks(
        account, orders, price_map,
        max_single_weight=0.10, allow_short=False, enforce_t_plus_one=True
    )
    orders = orders.merge(pre, on=["order_id","ticker","side","qty"], how="left")

    capacity = order_capacity_checks(
        orders, prices, as_of,
        max_adv_participation_pct=0.25,  # loose only for synthetic demo
        nav_value=500000,
        max_order_notional_pct_nav=0.12
    )
    orders = orders.merge(
        capacity[["order_id","status","issues"]].rename(
            columns={"status":"capacity_status","issues":"capacity_issues"}
        ),
        on="order_id", how="left"
    )
    orders = ensure_no_duplicate_orders(orders)
    orders.to_csv(OUT/"proposed_orders.csv", index=False)

    # Manual approval simulation in demo.
    approved = approve_orders(orders, "demo_user", "offline demo approval")
    approved.to_csv(OUT/"approved_orders.csv", index=False)

    executable = approved[
        (approved["status"]=="NEW") &
        (approved["check_status"]=="PASS") &
        (approved["capacity_status"]=="PASS")
    ].copy()

    account, fills = execute_orders(
        account, executable, price_map, as_of,
        commission_bps=3, min_commission_cny=5,
        slippage_bps=2, enforce_t_plus_one=True
    )
    fills.to_csv(OUT/"fills.csv", index=False)

    summary, positions = reconcile(account, price_map, as_of)
    summary.to_csv(OUT/"reconciliation_summary.csv", index=False)
    positions.to_csv(OUT/"reconciliation_positions.csv", index=False)

    risk_summary, stress = combine_risk_reports(
        targets, prices, as_of,
        market_shocks=(-0.03,-0.05,-0.10),
        sector_shock=-0.15,
        confidence=0.95,
        scenarios=1000
    )
    risk_summary.to_csv(OUT/"risk_summary.csv", index=False)
    stress.to_csv(OUT/"stress_report.csv", index=False)

    manifest = build_run_manifest(
        BASE,
        config_path=BASE/"config.example.json",
        input_files=[BASE/"data"/"demo_prices.csv", BASE/"outputs"/"demo_v6"/"optimized_targets_v6.csv"],
        extra={"mode":"ultimate_demo","as_of":str(as_of.date()),"live_submission":False}
    )
    save_manifest(manifest, OUT/"run_manifest.json")

    ledger = Ledger(OUT/"ledger.sqlite3")
    ledger.record_run("ultimate-demo", manifest)
    ledger.record_orders(approved)
    ledger.record_fills(fills)
    ledger.record_reconciliation(summary)
    ledger.close()

    print("ULTIMATE demo completed.")
    print("\nRisk summary:")
    print(risk_summary.round(4).to_string(index=False))
    print("\nTarget checks:")
    print(target_checks.to_string(index=False))
    print("\nOrders:", len(orders), "Approved:", len(approved), "Executed:", len(fills))
    print("\nAccount:")
    print(summary.round(4).to_string(index=False))
    print("\nStress:")
    print(stress.round(4).to_string(index=False))

if __name__=="__main__":
    main()
