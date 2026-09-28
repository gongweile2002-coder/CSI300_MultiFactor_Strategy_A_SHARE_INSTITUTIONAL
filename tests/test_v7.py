
import pandas as pd
import numpy as np

from src.paper_trading_v7 import (
    init_account, generate_orders, execute_orders,
    sellable_qty, reconcile
)

def test_buy_rounds_to_board_lot():
    acc = init_account(100000, "2026-09-01")
    targets = pd.DataFrame({"ticker":["000001.SZ"],"target_weight":[0.50]})
    orders = generate_orders(
        acc, targets, {"000001.SZ":12.34}, "2026-09-01",
        lot_size=100, cash_buffer_pct=0.0, max_single_weight=0.8
    )
    buy = orders[orders["side"]=="BUY"].iloc[0]
    assert int(buy["qty"]) % 100 == 0

def test_t_plus_one_blocks_same_day_sell():
    acc = init_account(100000, "2026-09-01")
    orders = pd.DataFrame([{
        "order_id":"1","trade_date":"2026-09-01","ticker":"000001.SZ",
        "side":"BUY","qty":100,"reference_price":10.0,"status":"NEW","reason":""
    }])
    acc, fills = execute_orders(
        acc, orders, {"000001.SZ":10.0}, "2026-09-01",
        commission_bps=3, min_commission_cny=5, slippage_bps=0,
        enforce_t_plus_one=True
    )
    assert sellable_qty(acc, "000001.SZ", "2026-09-01", True) == 0
    assert sellable_qty(acc, "000001.SZ", "2026-09-02", True) == 100

def test_sell_updates_cash_and_position():
    acc = init_account(100000, "2026-09-01")
    buy = pd.DataFrame([{
        "order_id":"1","trade_date":"2026-09-01","ticker":"000001.SZ",
        "side":"BUY","qty":100,"reference_price":10.0,"status":"NEW","reason":""
    }])
    acc, _ = execute_orders(
        acc, buy, {"000001.SZ":10.0}, "2026-09-01",
        commission_bps=0, min_commission_cny=0, slippage_bps=0,
        enforce_t_plus_one=True
    )
    cash_after_buy = acc["cash"]

    sell = pd.DataFrame([{
        "order_id":"2","trade_date":"2026-09-02","ticker":"000001.SZ",
        "side":"SELL","qty":100,"reference_price":11.0,"status":"NEW","reason":""
    }])
    acc, fills = execute_orders(
        acc, sell, {"000001.SZ":11.0}, "2026-09-02",
        commission_bps=0, min_commission_cny=0, slippage_bps=0,
        enforce_t_plus_one=True
    )
    assert acc["cash"] > cash_after_buy
    assert sellable_qty(acc, "000001.SZ", "2026-09-03", True) == 0
    assert len(fills) == 1

def test_reconcile_nav():
    acc = init_account(100000, "2026-09-01")
    buy = pd.DataFrame([{
        "order_id":"1","trade_date":"2026-09-01","ticker":"000001.SZ",
        "side":"BUY","qty":100,"reference_price":10.0,"status":"NEW","reason":""
    }])
    acc, _ = execute_orders(
        acc, buy, {"000001.SZ":10.0}, "2026-09-01",
        commission_bps=0, min_commission_cny=0, slippage_bps=0,
        enforce_t_plus_one=True
    )
    summary, pos = reconcile(acc, {"000001.SZ":11.0}, "2026-09-02")
    assert abs(float(summary.iloc[0]["nav"]) - 100100) < 1e-6
    assert int(pos.iloc[0]["qty"]) == 100
