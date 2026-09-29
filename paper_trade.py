
import argparse
import json
from pathlib import Path
import pandas as pd

from src.paper_trading_v7 import (
    init_account, save_account, load_account,
    generate_orders, execute_orders, pre_trade_checks, reconcile
)

BASE = Path(__file__).resolve().parent
PAPER_DIR = BASE/"paper"
PAPER_DIR.mkdir(exist_ok=True)
ACCOUNT = PAPER_DIR/"paper_account.json"

def read_prices(path, date=None):
    df = pd.read_csv(path)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        if date is not None:
            d = pd.Timestamp(date)
            eligible = df[df["date"] <= d]
            if eligible.empty:
                raise ValueError("指定日期前无价格")
            use_date = eligible["date"].max()
            df = df[df["date"] == use_date]
    if "ticker" not in df.columns:
        raise ValueError("价格文件缺少 ticker")
    price_col = "open" if "open" in df.columns else ("close" if "close" in df.columns else None)
    if price_col is None:
        raise ValueError("价格文件需要 open 或 close")
    return dict(zip(df["ticker"].astype(str), df[price_col].astype(float))), df

def main():
    p = argparse.ArgumentParser(description="CSI300 v7 paper trading engine")
    sub = p.add_subparsers(dest="cmd", required=True)

    a = sub.add_parser("init")
    a.add_argument("--cash", type=float, default=500000)
    a.add_argument("--date", required=True)

    a = sub.add_parser("orders")
    a.add_argument("--targets", required=True)
    a.add_argument("--prices", required=True)
    a.add_argument("--date", required=True)
    a.add_argument("--output", default=str(PAPER_DIR/"orders.csv"))

    a = sub.add_parser("execute")
    a.add_argument("--orders", required=True)
    a.add_argument("--prices", required=True)
    a.add_argument("--date", required=True)
    a.add_argument("--fills", default=str(PAPER_DIR/"fills.csv"))

    a = sub.add_parser("status")
    a.add_argument("--prices", required=True)
    a.add_argument("--date", required=True)

    args = p.parse_args()
    cfg = json.loads((BASE/"config.example.json").read_text(encoding="utf-8"))

    if args.cmd == "init":
        acc = init_account(args.cash, args.date)
        save_account(acc, ACCOUNT)
        print(f"Initialized paper account: {ACCOUNT}")
        print(json.dumps(acc, ensure_ascii=False, indent=2))
        return

    acc = load_account(ACCOUNT)
    prices, price_df = read_prices(args.prices, args.date)

    if args.cmd == "orders":
        targets = pd.read_csv(args.targets)
        if "signal_date" in targets.columns:
            targets["signal_date"] = pd.to_datetime(targets["signal_date"], errors="coerce")
            latest = targets["signal_date"].max()
            targets = targets[targets["signal_date"] == latest].copy()

        orders = generate_orders(
            acc, targets, prices, args.date,
            lot_size=cfg["paper_lot_size"],
            cash_buffer_pct=cfg["paper_cash_buffer_pct"],
            max_single_weight=cfg["paper_max_single_weight"],
            enforce_t_plus_one=cfg["paper_enforce_t_plus_one"],
            allow_short=cfg["paper_allow_short"],
        )
        checks = pre_trade_checks(
            acc, orders, prices,
            max_single_weight=cfg["paper_max_single_weight"],
            allow_short=cfg["paper_allow_short"],
            enforce_t_plus_one=cfg["paper_enforce_t_plus_one"],
        )
        orders = orders.merge(checks, on=["order_id","ticker","side","qty"], how="left")
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        orders.to_csv(args.output, index=False, encoding="utf-8-sig")
        print(f"Orders saved: {args.output}")
        print(orders.to_string(index=False))
        return

    if args.cmd == "execute":
        orders = pd.read_csv(args.orders)
        # Only execute orders that passed checks if the columns exist.
        if "check_status" in orders.columns:
            orders = orders[orders["check_status"]=="PASS"].copy()
        acc, fills = execute_orders(
            acc, orders, prices, args.date,
            commission_bps=cfg["paper_commission_bps"],
            min_commission_cny=cfg["paper_min_commission_cny"],
            slippage_bps=cfg["paper_slippage_bps"],
            enforce_t_plus_one=cfg["paper_enforce_t_plus_one"],
        )
        save_account(acc, ACCOUNT)
        fills.to_csv(args.fills, index=False, encoding="utf-8-sig")
        print(f"Fills saved: {args.fills}")
        print(fills.to_string(index=False))
        summary, snap = reconcile(acc, prices, args.date)
        summary.to_csv(PAPER_DIR/"account_summary.csv", index=False, encoding="utf-8-sig")
        snap.to_csv(PAPER_DIR/"positions.csv", index=False, encoding="utf-8-sig")
        print("\nAccount summary:")
        print(summary.to_string(index=False))
        return

    if args.cmd == "status":
        summary, snap = reconcile(acc, prices, args.date)
        print(summary.to_string(index=False))
        print()
        print(snap.to_string(index=False) if not snap.empty else "No positions.")

if __name__=="__main__":
    main()
