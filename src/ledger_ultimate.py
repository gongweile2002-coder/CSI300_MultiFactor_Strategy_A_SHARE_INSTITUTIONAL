
from __future__ import annotations

from pathlib import Path
import sqlite3
import json
from datetime import datetime, timezone
import pandas as pd


SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    run_id TEXT PRIMARY KEY,
    created_utc TEXT NOT NULL,
    manifest_json TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS orders (
    order_fingerprint TEXT PRIMARY KEY,
    order_id TEXT,
    trade_date TEXT,
    ticker TEXT,
    side TEXT,
    qty INTEGER,
    reference_price REAL,
    status TEXT,
    approved INTEGER DEFAULT 0,
    approved_by TEXT,
    created_utc TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS fills (
    fill_id TEXT PRIMARY KEY,
    order_id TEXT,
    trade_date TEXT,
    ticker TEXT,
    side TEXT,
    qty INTEGER,
    fill_price REAL,
    notional REAL,
    commission REAL,
    stamp_duty REAL,
    total_fee REAL,
    created_utc TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS reconciliations (
    as_of TEXT,
    cash REAL,
    market_value REAL,
    nav REAL,
    realized_pnl REAL,
    fees_paid REAL,
    n_positions INTEGER,
    created_utc TEXT NOT NULL
);
"""


class Ledger:
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.path)
        self.conn.executescript(SCHEMA)
        self.conn.commit()

    def close(self):
        self.conn.close()

    def record_run(self, run_id, manifest):
        self.conn.execute(
            "INSERT OR REPLACE INTO runs(run_id,created_utc,manifest_json) VALUES(?,?,?)",
            (run_id, datetime.now(timezone.utc).isoformat(), json.dumps(manifest, ensure_ascii=False, default=str))
        )
        self.conn.commit()

    def record_orders(self, orders: pd.DataFrame):
        now = datetime.now(timezone.utc).isoformat()
        for _, r in orders.iterrows():
            self.conn.execute(
                """INSERT OR REPLACE INTO orders(
                    order_fingerprint,order_id,trade_date,ticker,side,qty,
                    reference_price,status,approved,approved_by,created_utc
                ) VALUES(?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    str(r.get("order_fingerprint","")),
                    str(r.get("order_id","")),
                    str(r.get("trade_date","")),
                    str(r.get("ticker","")),
                    str(r.get("side","")),
                    int(r.get("qty",0)),
                    float(r.get("reference_price",0.0)),
                    str(r.get("status","")),
                    int(bool(r.get("approved",False))),
                    str(r.get("approved_by","")),
                    now
                )
            )
        self.conn.commit()

    def record_fills(self, fills: pd.DataFrame):
        now = datetime.now(timezone.utc).isoformat()
        for _, r in fills.iterrows():
            self.conn.execute(
                """INSERT OR REPLACE INTO fills(
                    fill_id,order_id,trade_date,ticker,side,qty,fill_price,notional,
                    commission,stamp_duty,total_fee,created_utc
                ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    str(r.get("fill_id","")),
                    str(r.get("order_id","")),
                    str(r.get("trade_date","")),
                    str(r.get("ticker","")),
                    str(r.get("side","")),
                    int(r.get("qty",0)),
                    float(r.get("fill_price",0.0)),
                    float(r.get("notional",0.0)),
                    float(r.get("commission",0.0)),
                    float(r.get("stamp_duty",0.0)),
                    float(r.get("total_fee",0.0)),
                    now
                )
            )
        self.conn.commit()

    def record_reconciliation(self, summary: pd.DataFrame):
        now = datetime.now(timezone.utc).isoformat()
        for _, r in summary.iterrows():
            self.conn.execute(
                """INSERT INTO reconciliations(
                    as_of,cash,market_value,nav,realized_pnl,fees_paid,n_positions,created_utc
                ) VALUES(?,?,?,?,?,?,?,?)""",
                (
                    str(r.get("as_of","")),
                    float(r.get("cash",0.0)),
                    float(r.get("market_value",0.0)),
                    float(r.get("nav",0.0)),
                    float(r.get("realized_pnl",0.0)),
                    float(r.get("fees_paid",0.0)),
                    int(r.get("n_positions",0)),
                    now
                )
            )
        self.conn.commit()

    def table(self, name):
        if name not in {"runs","orders","fills","reconciliations"}:
            raise ValueError("unsupported table")
        return pd.read_sql_query(f"SELECT * FROM {name}", self.conn)
