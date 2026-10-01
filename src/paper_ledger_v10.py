from __future__ import annotations

import hashlib
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import pandas as pd


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)


def _frame_records(frame: pd.DataFrame | None) -> list[dict[str, Any]]:
    if frame is None or frame.empty:
        return []
    x = frame.copy()
    for col in x.columns:
        if pd.api.types.is_datetime64_any_dtype(x[col]):
            x[col] = pd.to_datetime(x[col], errors="raise").dt.strftime("%Y-%m-%d")
    return json.loads(x.to_json(orient="records", date_format="iso", double_precision=15))


def signal_bundle_hash(frame: pd.DataFrame, signal_date) -> str:
    x = frame.copy()
    if "ticker" in x.columns:
        x["ticker"] = x["ticker"].astype(str)
    sort_cols = [c for c in ["ticker", "rank"] if c in x.columns]
    if sort_cols:
        x = x.sort_values(sort_cols, kind="stable")
    x = x.reindex(sorted(x.columns), axis=1).reset_index(drop=True)
    payload = {
        "signal_date": str(pd.Timestamp(signal_date).normalize().date()),
        "rows": _frame_records(x),
    }
    return hashlib.sha256(_json_dumps(payload).encode("utf-8")).hexdigest()


def _atomic_text(path: Path, text: str, encoding: str = "utf-8") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding=encoding)
    tmp.replace(path)


def _atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(tmp, index=False, encoding="utf-8-sig")
    tmp.replace(path)


class PaperLedger:
    """
    SQLite is the authoritative paper-trading state.

    JSON/CSV files under paper/live are exports only. A crash before COMMIT rolls
    back account, pending intents, orders, fills and snapshots together.
    """

    def __init__(self, db_path):
        self.path = Path(db_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path, timeout=30.0, isolation_level=None)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys=ON")
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=FULL")
        return conn

    def _init_schema(self) -> None:
        with self._connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS account_state (
                    account_id TEXT PRIMARY KEY,
                    payload_json TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS paper_state (
                    account_id TEXT PRIMARY KEY,
                    payload_json TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS pending_intents (
                    account_id TEXT NOT NULL,
                    order_id TEXT NOT NULL,
                    source_signal_date TEXT NOT NULL,
                    execution_date TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    PRIMARY KEY (account_id, order_id)
                );

                CREATE TABLE IF NOT EXISTS runs (
                    account_id TEXT NOT NULL,
                    signal_date TEXT NOT NULL,
                    signal_hash TEXT NOT NULL,
                    executed_signal_date TEXT,
                    status TEXT NOT NULL,
                    committed_at TEXT NOT NULL,
                    PRIMARY KEY (account_id, signal_date)
                );

                CREATE TABLE IF NOT EXISTS records (
                    account_id TEXT NOT NULL,
                    kind TEXT NOT NULL,
                    record_key TEXT NOT NULL,
                    as_of TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    PRIMARY KEY (account_id, kind, record_key)
                );

                CREATE TABLE IF NOT EXISTS corporate_action_events (
                    account_id TEXT NOT NULL,
                    event_key TEXT NOT NULL,
                    action_id TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    event_date TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    PRIMARY KEY (account_id, event_key)
                );

                CREATE INDEX IF NOT EXISTS idx_records_kind_asof
                    ON records(account_id, kind, as_of);
                """
            )

    def load_account(self, account_id: str) -> dict[str, Any] | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT payload_json FROM account_state WHERE account_id=?",
                (account_id,),
            ).fetchone()
        return json.loads(row["payload_json"]) if row else None

    def load_state(self, account_id: str) -> dict[str, Any]:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT payload_json FROM paper_state WHERE account_id=?",
                (account_id,),
            ).fetchone()
        return json.loads(row["payload_json"]) if row else {}

    def load_pending(self, account_id: str) -> pd.DataFrame:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT payload_json
                FROM pending_intents
                WHERE account_id=?
                ORDER BY order_id
                """,
                (account_id,),
            ).fetchall()
        return pd.DataFrame([json.loads(r["payload_json"]) for r in rows])

    def get_run(self, account_id: str, signal_date) -> dict[str, Any] | None:
        d = str(pd.Timestamp(signal_date).normalize().date())
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT account_id, signal_date, signal_hash,
                       executed_signal_date, status, committed_at
                FROM runs
                WHERE account_id=? AND signal_date=?
                """,
                (account_id, d),
            ).fetchone()
        return dict(row) if row else None

    def position_qty_on(self, account_id: str, ticker: str, as_of) -> int | None:
        day = str(pd.Timestamp(as_of).normalize().date())
        with self._connect() as conn:
            run = conn.execute(
                "SELECT 1 FROM runs WHERE account_id=? AND signal_date=?",
                (account_id, day),
            ).fetchone()
            if not run:
                return None
            row = conn.execute(
                """
                SELECT payload_json
                FROM records
                WHERE account_id=? AND kind='position' AND record_key=?
                """,
                (account_id, f"{day}|{ticker}"),
            ).fetchone()
        if not row:
            return 0
        payload = json.loads(row["payload_json"])
        return int(payload.get("qty", 0))

    def applied_corporate_action_ids(self, account_id: str) -> set[str]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT DISTINCT action_id
                FROM corporate_action_events
                WHERE account_id=? AND event_type='ENTITLEMENT'
                """,
                (account_id,),
            ).fetchall()
        return {str(r["action_id"]) for r in rows}

    def last_nav(self, account_id: str) -> float | None:
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT payload_json
                FROM records
                WHERE account_id=? AND kind='nav'
                ORDER BY as_of DESC
                LIMIT 1
                """,
                (account_id,),
            ).fetchone()
        if not row:
            return None
        payload = json.loads(row["payload_json"])
        value = payload.get("nav")
        return float(value) if value is not None else None

    def _insert_records(
        self,
        conn: sqlite3.Connection,
        account_id: str,
        kind: str,
        frame: pd.DataFrame | None,
        as_of: str,
        key_fn: Callable[[dict[str, Any]], str],
    ) -> None:
        for row in _frame_records(frame):
            key = key_fn(row)
            conn.execute(
                """
                INSERT INTO records(account_id, kind, record_key, as_of, payload_json)
                VALUES (?, ?, ?, ?, ?)
                """,
                (account_id, kind, key, as_of, _json_dumps(row)),
            )

    def commit_day(
        self,
        *,
        account_id: str,
        signal_date,
        signal_hash: str,
        executed_signal_date: str | None,
        account: dict[str, Any],
        state: dict[str, Any],
        pending_intents: pd.DataFrame,
        candidates: pd.DataFrame,
        orders: pd.DataFrame,
        fills: pd.DataFrame,
        summary: pd.DataFrame,
        positions: pd.DataFrame,
        corporate_action_events: pd.DataFrame | None = None,
        fault_point: str | None = None,
    ) -> str:
        signal_day = str(pd.Timestamp(signal_date).normalize().date())
        conn = self._connect()
        try:
            conn.execute("BEGIN IMMEDIATE")
            existing = conn.execute(
                "SELECT signal_hash FROM runs WHERE account_id=? AND signal_date=?",
                (account_id, signal_day),
            ).fetchone()
            if existing:
                if existing["signal_hash"] == signal_hash:
                    conn.rollback()
                    return "IDEMPOTENT"
                conn.rollback()
                raise ValueError(
                    "同一 signal_date 已提交不同 signal bundle，禁止静默覆盖"
                )

            now = _utc_now()
            conn.execute(
                """
                INSERT INTO account_state(account_id, payload_json, updated_at)
                VALUES (?, ?, ?)
                ON CONFLICT(account_id) DO UPDATE SET
                    payload_json=excluded.payload_json,
                    updated_at=excluded.updated_at
                """,
                (account_id, _json_dumps(account), now),
            )
            if fault_point == "after_account":
                raise RuntimeError("simulated crash after_account")

            conn.execute(
                """
                INSERT INTO paper_state(account_id, payload_json, updated_at)
                VALUES (?, ?, ?)
                ON CONFLICT(account_id) DO UPDATE SET
                    payload_json=excluded.payload_json,
                    updated_at=excluded.updated_at
                """,
                (account_id, _json_dumps(state), now),
            )

            conn.execute(
                "DELETE FROM pending_intents WHERE account_id=?",
                (account_id,),
            )
            for row in _frame_records(pending_intents):
                order_id = str(row.get("order_id", ""))
                if not order_id:
                    raise ValueError("pending intent 缺少 order_id")
                conn.execute(
                    """
                    INSERT INTO pending_intents(
                        account_id, order_id, source_signal_date,
                        execution_date, payload_json
                    )
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (
                        account_id,
                        order_id,
                        str(row.get("source_signal_date", signal_day)),
                        str(row.get("expected_execution_date", row.get("trade_date", "")))[:10],
                        _json_dumps(row),
                    ),
                )

            self._insert_records(
                conn, account_id, "candidate", candidates, signal_day,
                lambda r: f"{signal_day}|{r.get('ticker', '')}",
            )
            self._insert_records(
                conn, account_id, "order", orders, signal_day,
                lambda r: str(r["order_id"]),
            )
            self._insert_records(
                conn, account_id, "fill", fills, signal_day,
                lambda r: str(r["fill_id"]),
            )
            self._insert_records(
                conn, account_id, "nav", summary, signal_day,
                lambda r: signal_day,
            )
            self._insert_records(
                conn, account_id, "position", positions, signal_day,
                lambda r: f"{signal_day}|{r.get('ticker', '')}",
            )

            for row in _frame_records(corporate_action_events):
                event_key = str(row.get("event_key", ""))
                action_id = str(row.get("action_id", ""))
                event_type = str(row.get("event_type", ""))
                event_date = str(row.get("event_date", signal_day))[:10]
                if not event_key or not action_id or not event_type:
                    raise ValueError("corporate action event 缺少关键字段")
                conn.execute(
                    """
                    INSERT INTO corporate_action_events(
                        account_id, event_key, action_id,
                        event_type, event_date, payload_json
                    )
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        account_id,
                        event_key,
                        action_id,
                        event_type,
                        event_date,
                        _json_dumps(row),
                    ),
                )

            conn.execute(
                """
                INSERT INTO runs(
                    account_id, signal_date, signal_hash,
                    executed_signal_date, status, committed_at
                )
                VALUES (?, ?, ?, ?, 'COMMITTED', ?)
                """,
                (
                    account_id,
                    signal_day,
                    signal_hash,
                    executed_signal_date,
                    now,
                ),
            )
            if fault_point == "before_commit":
                raise RuntimeError("simulated crash before_commit")
            conn.commit()
            # Keep the main database file self-contained after each committed
            # paper day. WAL remains enabled for crash safety during the run.
            conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
            return "OK"
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _records_frame(
        self,
        account_id: str,
        kind: str,
        as_of: str | None = None,
    ) -> pd.DataFrame:
        sql = """
            SELECT payload_json
            FROM records
            WHERE account_id=? AND kind=?
        """
        args: list[Any] = [account_id, kind]
        if as_of is not None:
            sql += " AND as_of=?"
            args.append(as_of)
        sql += " ORDER BY as_of, record_key"
        with self._connect() as conn:
            rows = conn.execute(sql, args).fetchall()
        return pd.DataFrame([json.loads(r["payload_json"]) for r in rows])

    def _corporate_action_frame(self, account_id: str) -> pd.DataFrame:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT payload_json
                FROM corporate_action_events
                WHERE account_id=?
                ORDER BY event_date, event_key
                """,
                (account_id,),
            ).fetchall()
        return pd.DataFrame([json.loads(r["payload_json"]) for r in rows])

    def export(self, state_dir, account_id: str, signal_date) -> None:
        state_dir = Path(state_dir)
        signal_day = str(pd.Timestamp(signal_date).normalize().date())
        account = self.load_account(account_id)
        state = self.load_state(account_id)
        pending = self.load_pending(account_id)

        if account is not None:
            _atomic_text(
                state_dir / "paper_account.json",
                json.dumps(account, ensure_ascii=False, indent=2),
            )
        _atomic_text(
            state_dir / "state.json",
            json.dumps(state, ensure_ascii=False, indent=2),
        )
        _atomic_csv(state_dir / "pending_intents.csv", pending)

        history_map = {
            "candidate": "candidate_history.csv",
            "order": "order_history.csv",
            "fill": "fill_history.csv",
            "nav": "nav_history.csv",
            "position": "position_history.csv",
        }
        for kind, name in history_map.items():
            frame = self._records_frame(account_id, kind)
            if not frame.empty:
                _atomic_csv(state_dir / name, frame)

        corporate_actions = self._corporate_action_frame(account_id)
        if not corporate_actions.empty:
            _atomic_csv(
                state_dir / "corporate_action_history.csv",
                corporate_actions,
            )

        run_dir = state_dir / "runs" / signal_day
        run_dir.mkdir(parents=True, exist_ok=True)
        run_map = {
            "candidate": "candidates.csv",
            "order": "orders.csv",
            "fill": "fills.csv",
            "nav": "account_summary.csv",
            "position": "positions.csv",
        }
        for kind, name in run_map.items():
            frame = self._records_frame(account_id, kind, signal_day)
            _atomic_csv(run_dir / name, frame)
        _atomic_csv(run_dir / "pending_intents.csv", pending)
        if not corporate_actions.empty and "event_date" in corporate_actions.columns:
            day_actions = corporate_actions[
                corporate_actions["event_date"].astype(str).str[:10].eq(signal_day)
            ].copy()
            _atomic_csv(run_dir / "corporate_actions.csv", day_actions)
