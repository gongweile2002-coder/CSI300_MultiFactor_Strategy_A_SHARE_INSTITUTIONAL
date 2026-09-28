"""v9.2 live-safety regressions. Synthetic only; never contacts a broker."""
import sqlite3
from dataclasses import replace

import pandas as pd
import pytest

from src.live_v8 import RiskConfig, TradingBlocked, TradingCalendar, make_plan, verify_plan
from src.order_journal_v8 import OrderJournal, submit_reserved


def base_inputs(two=False):
    now = pd.Timestamp('2026-09-11T10:00:00+08:00')
    account = {
        'source': 'broker', 'risk_reference_date': '2026-09-10', 'account_id': 'TEST_ONLY',
        'as_of': now.isoformat(), 'cash_available': 100000., 'cash_total': 100000., 'total_asset': 100000.,
        'prior_close_nav_adjusted': 100000., 'high_water_nav_adjusted': 100000., 'open_order_count': 0,
        'positions': []
    }
    tickers = ['600000.SH', '000001.SZ'] if two else ['600000.SH']
    targets = pd.DataFrame([
        {'ticker': t, 'target_weight': .05, 'signal_date': '2026-09-10', 'signal_close_raw': 10., 'source': 'real'}
        for t in tickers
    ])
    quotes = pd.DataFrame([
        {'ticker': t, 'last': 10., 'bid': 9.99, 'ask': 10.01, 'up_limit': 11., 'down_limit': 9.,
         'quote_time': now.isoformat(), 'price_basis': 'raw', 'status': 'TRADING', 'adv20_cny': 1e8,
         'industry_l1': 'BANK', 'is_st': False, 'is_delisting': False}
        for t in tickers
    ])
    calendar = TradingCalendar(pd.DataFrame({'date': pd.date_range('2026-09-09', '2026-09-12'), 'is_open': [1, 1, 1, 0]}))
    return account, targets, quotes, calendar, now


def test_v92_plan_has_expiry_and_new_schema():
    p = make_plan(*base_inputs())
    assert p['schema_version'] == 9
    assert p['expires_at'] == '2026-09-11T10:10:00+08:00'
    assert RiskConfig().version == 'A_SHARE_V9_2_2026_09_28'


def test_plan_expiry_is_enforced_only_when_consumed():
    p = make_plan(*base_inputs())
    verify_plan(p)
    verify_plan(p, '2026-09-11T10:09:59+08:00', require_fresh=True)
    with pytest.raises(TradingBlocked, match='有效期'):
        verify_plan(p, '2026-09-11T10:10:01+08:00', require_fresh=True)


def test_batch_order_count_is_fail_closed():
    args = base_inputs(two=True)
    cfg = replace(RiskConfig(), max_orders_per_batch=1)
    with pytest.raises(TradingBlocked, match='单批委托数量'):
        make_plan(*args, cfg)


def test_journal_reserves_and_releases_buy_cash(tmp_path):
    p = make_plan(*base_inputs())
    key = p['orders'][0]['client_order_id']
    j = OrderJournal(tmp_path/'journal.db')
    try:
        j.reserve(p)
        row = j.rows()[0]
        assert row['status'] == 'APPROVED'
        first = j.reservations('TEST_ONLY')['reserved_cash']
        assert first >= p['orders'][0]['notional']
        j.update(key, 'SUBMITTED', '123')
        j.update(key, 'PARTIAL', '123', 100, 10.0)
        assert 0 < j.reservations('TEST_ONLY')['reserved_cash'] < first
        j.update(key, 'CANCELLED', '123', 100, 10.0)
        assert j.reservations('TEST_ONLY')['reserved_cash'] == 0
        assert j.reservations('TEST_ONLY')['active_order_count'] == 0
    finally:
        j.close()


def test_journal_reserves_sellable_shares(tmp_path):
    account, targets, quotes, calendar, now = base_inputs()
    account['positions'] = [{'ticker': '600000.SH', 'qty': 500, 'sellable_qty': 200, 'last_price': 10.}]
    account['cash_total'] = account['cash_available'] = 95000.
    targets.loc[0, 'target_weight'] = 0
    p = make_plan(account, targets, quotes, calendar, now)
    assert p['orders'][0]['side'] == 'SELL' and p['orders'][0]['qty'] == 200
    j = OrderJournal(tmp_path/'journal.db')
    try:
        j.reserve(p)
        assert j.reservations('TEST_ONLY')['reserved_sell_shares'] == {'600000.SH': 200}
    finally:
        j.close()


def test_illegal_state_regression_is_blocked(tmp_path):
    p = make_plan(*base_inputs())
    key = p['orders'][0]['client_order_id']
    j = OrderJournal(tmp_path/'journal.db')
    try:
        j.reserve(p)
        j.update(key, 'SUBMITTED', '123')
        with pytest.raises(TradingBlocked, match='非法状态迁移'):
            j.update(key, 'SUBMITTING')
    finally:
        j.close()


def test_kill_switch_blocks_before_broker_call(tmp_path):
    p = make_plan(*base_inputs(two=True))
    calls = []
    class Broker:
        def submit_checked(self, order):
            calls.append(order)
            return 123
    j = OrderJournal(tmp_path/'journal.db')
    try:
        rows = submit_reserved(p, j, Broker(), kill_check=lambda: True)
        selected = [r for r in rows if r['plan_hash'] == p['plan_hash']]
        assert calls == []
        assert selected and all(r['status'] == 'BLOCKED' for r in selected)
        assert j.reservations('TEST_ONLY')['active_order_count'] == 0
    finally:
        j.close()


def test_existing_v8_journal_migrates_in_place(tmp_path):
    path = tmp_path/'legacy.db'
    db = sqlite3.connect(path)
    db.executescript('''
      CREATE TABLE orders_v8 (
        client_order_id TEXT PRIMARY KEY,account_id TEXT NOT NULL,strategy_id TEXT NOT NULL,trade_date TEXT NOT NULL,
        ticker TEXT NOT NULL,side TEXT NOT NULL,qty INTEGER NOT NULL,limit_price REAL NOT NULL,plan_hash TEXT NOT NULL,
        status TEXT NOT NULL,broker_order_id TEXT,filled_qty INTEGER NOT NULL DEFAULT 0,avg_fill_price REAL NOT NULL DEFAULT 0,updated_at TEXT NOT NULL
      );
      CREATE TABLE events_v8(event_id INTEGER PRIMARY KEY AUTOINCREMENT,client_order_id TEXT,event TEXT NOT NULL,detail TEXT NOT NULL,created_at TEXT NOT NULL);
    ''')
    db.commit(); db.close()
    j = OrderJournal(path)
    try:
        cols = {r['name'] for r in j.db.execute('PRAGMA table_info(orders_v8)')}
        assert {'estimated_fees', 'reserved_cash', 'reserved_shares', 'expires_at'} <= cols
    finally:
        j.close()


def test_expire_only_unsubmitted_approved_orders(tmp_path):
    p = make_plan(*base_inputs())
    j = OrderJournal(tmp_path/'journal.db')
    try:
        j.reserve(p)
        assert j.expire_approved('2026-09-11T10:09:00+08:00') == 0
        assert j.expire_approved('2026-09-11T10:11:00+08:00') == 1
        assert j.rows()[0]['status'] == 'EXPIRED'
        assert j.reservations('TEST_ONLY')['active_order_count'] == 0
    finally:
        j.close()


def test_malformed_broker_acceptance_becomes_unknown(tmp_path):
    p = make_plan(*base_inputs())
    class Broker:
        def submit_checked(self, order):
            return 'NOT_AN_ORDER_ID'
    j = OrderJournal(tmp_path/'journal.db')
    try:
        rows = submit_reserved(p, j, Broker())
        assert rows[0]['status'] == 'UNKNOWN'
        with pytest.raises(TradingBlocked):
            j.context('TEST_ONLY', '2026-09-11')
    finally:
        j.close()


def test_old_risk_config_version_is_rejected():
    old = RiskConfig().__dict__.copy()
    old['version'] = 'A_SHARE_V9_1_2026_09_26'
    with pytest.raises(TradingBlocked, match='代码版本'):
        RiskConfig.from_dict(old)
