
from pathlib import Path
import pandas as pd
import numpy as np
import pytest

from src.ops_ultimate import (
    attach_order_fingerprints, ensure_no_duplicate_orders,
    set_kill_switch, kill_switch_is_on
)
from src.risk_controls_ultimate import (
    portfolio_target_checks, deterministic_stress,
    historical_var_cvar, order_capacity_checks
)
from src.broker_adapter_ultimate import LiveBrokerDisabled
from src.ledger_ultimate import Ledger

def test_order_fingerprint_stable():
    x = pd.DataFrame([{
        "trade_date":"2026-09-04","ticker":"000001.SZ","side":"BUY",
        "qty":100,"reference_price":10.0
    }])
    a = attach_order_fingerprints(x)["order_fingerprint"].iloc[0]
    b = attach_order_fingerprints(x)["order_fingerprint"].iloc[0]
    assert a == b

def test_duplicate_orders_block():
    x = pd.DataFrame([
        {"trade_date":"2026-09-04","ticker":"000001.SZ","side":"BUY","qty":100,"reference_price":10.0},
        {"trade_date":"2026-09-04","ticker":"000001.SZ","side":"BUY","qty":100,"reference_price":10.0},
    ])
    with pytest.raises(ValueError):
        ensure_no_duplicate_orders(x)

def test_kill_switch(tmp_path):
    set_kill_switch(tmp_path, True, "ops/KILL_SWITCH", "test")
    assert kill_switch_is_on(tmp_path, "ops/KILL_SWITCH")
    set_kill_switch(tmp_path, False, "ops/KILL_SWITCH")
    assert not kill_switch_is_on(tmp_path, "ops/KILL_SWITCH")

def test_restricted_list_blocks():
    t = pd.DataFrame({
        "ticker":["A","B"],
        "target_weight":[0.5,0.5],
        "industry_l1":["X","Y"]
    })
    c = portfolio_target_checks(
        t, 100000, restricted_tickers={"A"},
        max_single_weight=0.6, max_sector_weight=0.6, max_gross_exposure=1.0
    )
    row = c[c["check"]=="restricted_list"].iloc[0]
    assert row["status"] == "BLOCK"

def test_stress_negative():
    t = pd.DataFrame({
        "ticker":["A","B"],
        "target_weight":[0.5,0.5],
        "industry_l1":["X","Y"]
    })
    s = deterministic_stress(t, (-0.05,), -0.10)
    assert (s["estimated_portfolio_return"] <= 0).all()

def test_historical_var_runs():
    dates = pd.bdate_range("2024-01-01", periods=260)
    rng = np.random.default_rng(1)
    rows = []
    for ticker in ["A","B","C"]:
        px = 10*np.exp(np.cumsum(rng.normal(0,0.01,len(dates))))
        for d,p in zip(dates,px):
            rows.append((d,ticker,p))
    prices = pd.DataFrame(rows, columns=["date","ticker","close"])
    targets = pd.DataFrame({
        "ticker":["A","B","C"],
        "target_weight":[0.4,0.3,0.3]
    })
    r = historical_var_cvar(prices, targets, dates[-1], 252, 0.95)
    assert r["n_obs"] >= 200
    assert r["var"] <= 0.02

def test_live_broker_disabled():
    b = LiveBrokerDisabled()
    with pytest.raises(RuntimeError):
        b.submit_orders(pd.DataFrame())

def test_ledger_roundtrip(tmp_path):
    ledger = Ledger(tmp_path/"x.sqlite3")
    orders = pd.DataFrame([{
        "order_fingerprint":"abc",
        "order_id":"1",
        "trade_date":"2026-09-04",
        "ticker":"A","side":"BUY","qty":100,
        "reference_price":10.0,"status":"NEW",
        "approved":True,"approved_by":"tester"
    }])
    ledger.record_orders(orders)
    out = ledger.table("orders")
    ledger.close()
    assert len(out) == 1
    assert out.iloc[0]["ticker"] == "A"
