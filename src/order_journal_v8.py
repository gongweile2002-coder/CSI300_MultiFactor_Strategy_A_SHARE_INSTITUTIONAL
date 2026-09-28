"""Durable order lifecycle and idempotency for the live-planning path.

Broker acceptance is not a fill. A timeout is UNKNOWN. Existing v8 SQLite
journals are migrated in place and must not be deleted when upgrading.
"""
from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from .live_v8 import finite, integer, require, verify_plan

TERMINAL = {'FILLED', 'CANCELLED', 'REJECTED', 'EXPIRED', 'BLOCKED'}
ACTIVE = {'APPROVED', 'SUBMITTING', 'UNKNOWN', 'SUBMITTED', 'PARTIAL'}
ALL_STATUSES = ACTIVE | TERMINAL

# Direct broker/manual reports can legitimately skip some local intermediate
# states, but regressions (for example SUBMITTED -> SUBMITTING) are rejected.
TRANSITIONS = {
    'APPROVED': {'APPROVED', 'SUBMITTING', 'SUBMITTED', 'PARTIAL', 'FILLED', 'CANCELLED', 'REJECTED', 'EXPIRED', 'BLOCKED', 'UNKNOWN'},
    'SUBMITTING': {'SUBMITTING', 'SUBMITTED', 'PARTIAL', 'FILLED', 'CANCELLED', 'REJECTED', 'UNKNOWN', 'BLOCKED'},
    'UNKNOWN': {'UNKNOWN', 'SUBMITTED', 'PARTIAL', 'FILLED', 'CANCELLED', 'REJECTED'},
    'SUBMITTED': {'SUBMITTED', 'PARTIAL', 'FILLED', 'CANCELLED', 'REJECTED'},
    'PARTIAL': {'PARTIAL', 'FILLED', 'CANCELLED'},
    'FILLED': {'FILLED'},
    'CANCELLED': {'CANCELLED'},
    'REJECTED': {'REJECTED'},
    'EXPIRED': {'EXPIRED'},
    'BLOCKED': {'BLOCKED'},
}


def _utcnow():
    return datetime.now(timezone.utc).isoformat()


class OrderJournal:
    def __init__(self, path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(path, timeout=10)
        self.db.row_factory = sqlite3.Row
        self.db.execute('PRAGMA journal_mode=WAL')
        self.db.execute('PRAGMA synchronous=FULL')
        self.db.executescript('''
        CREATE TABLE IF NOT EXISTS orders_v8 (
          client_order_id TEXT PRIMARY KEY,
          account_id TEXT NOT NULL,
          strategy_id TEXT NOT NULL,
          trade_date TEXT NOT NULL,
          ticker TEXT NOT NULL,
          side TEXT NOT NULL,
          qty INTEGER NOT NULL,
          limit_price REAL NOT NULL,
          plan_hash TEXT NOT NULL,
          status TEXT NOT NULL,
          broker_order_id TEXT,
          filled_qty INTEGER NOT NULL DEFAULT 0,
          avg_fill_price REAL NOT NULL DEFAULT 0,
          updated_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS events_v8(
          event_id INTEGER PRIMARY KEY AUTOINCREMENT,
          client_order_id TEXT,
          event TEXT NOT NULL,
          detail TEXT NOT NULL,
          created_at TEXT NOT NULL
        );
        ''')
        self._migrate_columns()
        self.db.commit()

    def _migrate_columns(self):
        """Non-destructive v9.2 migration of an existing v8 journal."""
        cols = {r['name'] for r in self.db.execute('PRAGMA table_info(orders_v8)')}
        additions = {
            'estimated_fees': 'REAL NOT NULL DEFAULT 0',
            'reserved_cash': 'REAL NOT NULL DEFAULT 0',
            'reserved_shares': 'INTEGER NOT NULL DEFAULT 0',
            'expires_at': 'TEXT',
        }
        for name, ddl in additions.items():
            if name not in cols:
                self.db.execute(f'ALTER TABLE orders_v8 ADD COLUMN {name} {ddl}')

    def close(self):
        self.db.close()

    def rows(self, account_id=None):
        query = 'SELECT * FROM orders_v8' + (' WHERE account_id=?' if account_id is not None else '') + ' ORDER BY rowid'
        args = (account_id,) if account_id is not None else ()
        return [dict(r) for r in self.db.execute(query, args)]

    def active_rows(self, account_id=None):
        return [r for r in self.rows(account_id) if r['status'] in ACTIVE]

    def reservations(self, account_id=None):
        """Outstanding local reservations; useful after restart and before reconciliation."""
        active = self.active_rows(account_id)
        shares = {}
        for r in active:
            if r['reserved_shares']:
                shares[r['ticker']] = shares.get(r['ticker'], 0) + int(r['reserved_shares'])
        return {
            'active_order_count': len(active),
            'reserved_cash': round(sum(float(r['reserved_cash']) for r in active), 2),
            'reserved_sell_shares': shares,
            'statuses': {s: sum(r['status'] == s for r in active) for s in sorted(ACTIVE)},
        }

    def context(self, account_id, trade_date):
        rows = self.rows(account_id)
        require(not any(r['status'] in ACTIVE for r in rows), '存在未完成/状态未知委托，先 reconcile，不能盲目重发')
        today = [r for r in rows if r['trade_date'] == str(trade_date)]
        used = {r['client_order_id'] for r in today}
        turnover = sum(r['filled_qty'] * max(r['avg_fill_price'], r['limit_price']) for r in today)
        return used, turnover

    def expire_approved(self, now=None, account_id=None):
        """Expire locally approved orders that were never submitted before plan expiry."""
        if now is None:
            current = datetime.now(timezone.utc)
        elif isinstance(now, datetime):
            current = now if now.tzinfo is not None else now.replace(tzinfo=timezone.utc)
        else:
            current = datetime.fromisoformat(str(now))
            if current.tzinfo is None:
                current = current.replace(tzinfo=timezone.utc)
        keys = []
        for r in self.rows(account_id):
            if r['status'] != 'APPROVED' or not r.get('expires_at'):
                continue
            expiry = datetime.fromisoformat(str(r['expires_at']))
            if expiry.tzinfo is None:
                expiry = expiry.replace(tzinfo=timezone.utc)
            if expiry <= current:
                keys.append(r['client_order_id'])
        for key in keys:
            self.update(key, 'EXPIRED')
        return len(keys)

    def reserve(self, plan):
        """Persist a reviewed plan before any manual or API submission.

        The row enters APPROVED, not SUBMITTED. For live API submission,
        submit_reserved moves each row to SUBMITTING immediately before the SDK call.
        """
        verify_plan(plan)
        require(plan['mode'] != 'DEMO', '演示计划不能登记为真实委托')
        now = _utcnow()
        try:
            self.db.execute('BEGIN IMMEDIATE')
            require(not any(r['status'] in ACTIVE for r in self.rows(plan['account_id'])), '账户存在未完成/未知委托')
            for o in plan['orders']:
                est_fee = finite(o.get('estimated_fees', 0), '预计费用')
                reserved_cash = o['qty'] * o['limit_price'] + est_fee if o['side'] == 'BUY' else 0.0
                reserved_shares = o['qty'] if o['side'] == 'SELL' else 0
                self.db.execute('''
                INSERT INTO orders_v8(
                  client_order_id,account_id,strategy_id,trade_date,ticker,side,qty,limit_price,plan_hash,
                  status,broker_order_id,filled_qty,avg_fill_price,updated_at,
                  estimated_fees,reserved_cash,reserved_shares,expires_at
                ) VALUES(?,?,?,?,?,?,?,?,?,'APPROVED',NULL,0,0,?,?,?,?,?)
                ''', (
                    o['client_order_id'], plan['account_id'], plan['strategy_id'], plan['trade_date'],
                    o['ticker'], o['side'], o['qty'], o['limit_price'], plan['plan_hash'], now,
                    est_fee, reserved_cash, reserved_shares, plan.get('expires_at')
                ))
                for event in ('created', 'validated', 'approved'):
                    self.db.execute(
                        'INSERT INTO events_v8(client_order_id,event,detail,created_at) VALUES(?,?,?,?)',
                        (o['client_order_id'], event, plan['plan_hash'], now)
                    )
            self.db.commit()
        except Exception:
            self.db.rollback()
            raise

    def update(self, key, status, broker_order_id=None, filled_qty=0, avg_fill_price=0):
        require(status in ALL_STATUSES, '非法委托状态')
        filled_qty = integer(filled_qty, '累计成交数')
        avg_fill_price = finite(avg_fill_price, '成交均价')
        now = _utcnow()
        try:
            self.db.execute('BEGIN IMMEDIATE')
            r = self.db.execute('SELECT * FROM orders_v8 WHERE client_order_id=?', (key,)).fetchone()
            require(r is not None, '回报没有对应委托')
            require(status in TRANSITIONS.get(r['status'], set()), f"非法状态迁移: {r['status']} -> {status}")
            require(r['filled_qty'] <= filled_qty <= r['qty'], '累计成交量倒退或超量')
            require(not filled_qty or avg_fill_price > 0, '成交缺少价格')
            if filled_qty:
                require(
                    avg_fill_price <= r['limit_price'] + .005 if r['side'] == 'BUY' else avg_fill_price >= r['limit_price'] - .005,
                    '成交价违反委托限价'
                )
            if status == 'FILLED':
                require(filled_qty == r['qty'], 'FILLED 必须为全量成交')
            if status == 'PARTIAL':
                require(0 < filled_qty < r['qty'], 'PARTIAL 数量非法')
            if status in {'REJECTED', 'EXPIRED', 'BLOCKED'}:
                require(filled_qty == 0, f'{status} 不能带成交数量')

            bid = str(broker_order_id).strip() if broker_order_id is not None else r['broker_order_id']
            if status in {'SUBMITTED', 'PARTIAL', 'FILLED', 'CANCELLED'} or filled_qty:
                require(bool(bid) and str(bid).lower() not in {'nan', 'none'}, '券商受理/成交/撤单回报缺少委托号')
            if r['broker_order_id'] and bid:
                require(bid == r['broker_order_id'], '券商委托号变化')

            if r['status'] in TERMINAL:
                require(
                    status == r['status'] and filled_qty == r['filled_qty'] and abs(avg_fill_price - r['avg_fill_price']) < 1e-8,
                    '终态回报矛盾，请核对券商'
                )

            remaining = max(0, int(r['qty']) - filled_qty)
            if status in TERMINAL:
                reserved_cash = 0.0
                reserved_shares = 0
            elif r['side'] == 'BUY':
                reserved_cash = remaining * float(r['limit_price']) + (float(r['estimated_fees']) if remaining else 0.0)
                reserved_shares = 0
            else:
                reserved_cash = 0.0
                reserved_shares = remaining

            self.db.execute('''
              UPDATE orders_v8
              SET status=?,broker_order_id=?,filled_qty=?,avg_fill_price=?,reserved_cash=?,reserved_shares=?,updated_at=?
              WHERE client_order_id=?
            ''', (status, bid, filled_qty, avg_fill_price, reserved_cash, reserved_shares, now, key))
            self.db.execute(
                'INSERT INTO events_v8(client_order_id,event,detail,created_at) VALUES(?,?,?,?)',
                (key, status, json.dumps({'broker_order_id': bid, 'filled_qty': filled_qty, 'avg_fill_price': avg_fill_price}), now)
            )
            self.db.commit()
        except Exception:
            self.db.rollback()
            raise


def submit_reserved(plan, journal, broker, kill_check=lambda: False):
    """Reserve first, then submit each order with a durable pre-call state."""
    journal.reserve(plan)
    stopped = False
    for o in plan['orders']:
        key = o['client_order_id']
        if stopped or kill_check():
            journal.update(key, 'BLOCKED')
            stopped = True
            continue
        journal.update(key, 'SUBMITTING')
        try:
            bid = broker.submit_checked(o)
            accepted = bid is not None and int(bid) > 0
        except Exception:
            # The SDK/network may have accepted the order before throwing, or may have
            # returned a malformed acceptance token. Never retry blindly.
            journal.update(key, 'UNKNOWN')
            stopped = True
            continue
        if not accepted:
            journal.update(key, 'REJECTED')
            stopped = True
        else:
            journal.update(key, 'SUBMITTED', str(bid))
    return journal.rows(plan['account_id'])
