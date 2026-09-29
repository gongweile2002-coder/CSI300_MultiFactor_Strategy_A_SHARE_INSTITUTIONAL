from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from .paper_ledger_v10 import PaperLedger, signal_bundle_hash
from .paper_trading_v7 import (
    execute_orders,
    generate_orders,
    init_account,
    load_account,
    pre_trade_checks,
    reconcile,
)


def _date(value) -> pd.Timestamp:
    return pd.Timestamp(value).normalize()


def next_open_session(trade_calendar: pd.DataFrame, after_date) -> pd.Timestamp:
    required = {"date", "is_open"}
    missing = required - set(trade_calendar.columns)
    if missing:
        raise ValueError(f"trade_calendar 缺少列: {sorted(missing)}")
    cal = trade_calendar.copy()
    cal["date"] = pd.to_datetime(cal["date"], errors="raise").dt.normalize()
    cal["is_open"] = pd.to_numeric(cal["is_open"], errors="raise").astype(int)
    future = cal[(cal["date"] > _date(after_date)) & (cal["is_open"] == 1)]
    if future.empty:
        raise ValueError("交易日历中找不到下一开放交易日")
    return pd.Timestamp(future["date"].min()).normalize()


def _atomic_json(path: Path, obj: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    tmp.replace(path)


def _append_history(path: Path, frame: pd.DataFrame, subset: list[str]) -> None:
    if frame is None or frame.empty:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    out = frame.copy()
    if path.exists():
        old = pd.read_csv(path)
        out = pd.concat([old, out], ignore_index=True)
    out = out.drop_duplicates(subset=subset, keep="last")
    out.to_csv(path, index=False, encoding="utf-8-sig")


def _ensure_raw_prices(frame: pd.DataFrame) -> pd.DataFrame:
    required = {"date", "ticker", "open", "close"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"raw_prices 缺少列: {sorted(missing)}")
    x = frame.copy()
    x["date"] = pd.to_datetime(x["date"], errors="raise").dt.normalize()
    x["ticker"] = x["ticker"].astype(str)
    for col in ["open", "close"]:
        x[col] = pd.to_numeric(x[col], errors="raise")
    if "price_basis" in x.columns and not x["price_basis"].eq("raw").all():
        raise ValueError("Paper Trading 只能使用未复权 raw_prices")
    if x.duplicated(["date", "ticker"]).any():
        raise ValueError("raw_prices 存在重复 date/ticker")
    return x.sort_values(["date", "ticker"]).reset_index(drop=True)


def exact_price_map(raw_prices: pd.DataFrame, trade_date, field: str) -> dict[str, float]:
    x = _ensure_raw_prices(raw_prices)
    d = _date(trade_date)
    day = x[x["date"] == d]
    if field not in day.columns:
        raise ValueError(f"raw_prices 缺少 {field}")
    return {
        str(r["ticker"]): float(r[field])
        for _, r in day.iterrows()
        if pd.notna(r[field]) and float(r[field]) > 0
    }


def latest_close_map(raw_prices: pd.DataFrame, trade_date) -> dict[str, float]:
    x = _ensure_raw_prices(raw_prices)
    d = _date(trade_date)
    eligible = x[x["date"] <= d]
    if eligible.empty:
        raise ValueError("指定交易日前没有可用 raw close")
    latest = eligible.sort_values(["ticker", "date"]).groupby("ticker", as_index=False).tail(1)
    return {
        str(r["ticker"]): float(r["close"])
        for _, r in latest.iterrows()
        if pd.notna(r["close"]) and float(r["close"]) > 0
    }


def stock_limits_for_date(stock_limits: pd.DataFrame | None, trade_date) -> pd.DataFrame:
    if stock_limits is None or stock_limits.empty:
        return pd.DataFrame(columns=["date", "ticker", "up_limit", "down_limit"])
    x = stock_limits.copy()
    required = {"date", "ticker", "up_limit", "down_limit"}
    missing = required - set(x.columns)
    if missing:
        raise ValueError(f"stock_limits 缺少列: {sorted(missing)}")
    x["date"] = pd.to_datetime(x["date"], errors="raise").dt.normalize()
    x["ticker"] = x["ticker"].astype(str)
    for col in ["up_limit", "down_limit"]:
        x[col] = pd.to_numeric(x[col], errors="coerce")
    day = x[x["date"] == _date(trade_date)].copy()
    if day.duplicated("ticker").any():
        raise ValueError("stock_limits 存在重复 date/ticker")
    return day


def price_limit_map_for_date(
    stock_limits: pd.DataFrame | None,
    trade_date,
) -> dict[str, tuple[float, float]]:
    day = stock_limits_for_date(stock_limits, trade_date)
    out: dict[str, tuple[float, float]] = {}
    for _, r in day.iterrows():
        up, down = r["up_limit"], r["down_limit"]
        if pd.isna(up) or pd.isna(down):
            continue
        up, down = float(up), float(down)
        if up <= 0 or down <= 0 or down > up:
            continue
        out[str(r["ticker"])] = (up, down)
    return out


def _apply_open_constraints(
    orders: pd.DataFrame,
    exact_open: dict[str, float],
    stock_limits: pd.DataFrame | None,
    trade_date,
) -> pd.DataFrame:
    if orders is None or orders.empty:
        return pd.DataFrame() if orders is None else orders.copy()

    out = orders.copy()
    limit_map = price_limit_map_for_date(stock_limits, trade_date)

    for idx, row in out.iterrows():
        if str(row.get("status", "")) != "NEW":
            continue
        ticker = str(row["ticker"])
        side = str(row["side"])
        if ticker not in exact_open:
            out.loc[idx, "status"] = "REJECTED"
            out.loc[idx, "reason"] = "no_exact_open_trade_bar"
            continue
        if ticker not in limit_map:
            out.loc[idx, "status"] = "REJECTED"
            out.loc[idx, "reason"] = "missing_or_invalid_stock_limit_data"
            continue

        px = float(exact_open[ticker])
        up, down = limit_map[ticker]
        if px > up + 1e-8 or px < down - 1e-8:
            out.loc[idx, "status"] = "REJECTED"
            out.loc[idx, "reason"] = "open_outside_price_limits"
        elif side == "BUY" and px >= up - 1e-8:
            out.loc[idx, "status"] = "REJECTED"
            out.loc[idx, "reason"] = "limit_up"
        elif side == "SELL" and px <= down + 1e-8:
            out.loc[idx, "status"] = "REJECTED"
            out.loc[idx, "reason"] = "limit_down"

    return out


def _candidate_frame(targets: pd.DataFrame, signal_date) -> pd.DataFrame:
    required = {"ticker", "target_weight"}
    if not required <= set(targets):
        raise ValueError("targets 缺少 ticker/target_weight")
    out = targets.copy()
    out["ticker"] = out["ticker"].astype(str)
    out["signal_date"] = str(_date(signal_date).date())
    out["target_weight"] = pd.to_numeric(out["target_weight"], errors="raise")
    score_col = "composite_score" if "composite_score" in out.columns else None
    if score_col:
        out["rank"] = out[score_col].rank(method="first", ascending=False).astype(int)
    else:
        out["rank"] = range(1, len(out) + 1)
    keep = [c for c in [
        "signal_date", "rank", "ticker", "target_weight", "industry_l1",
        "composite_score", "signal_close_raw", "strategy_version", "source"
    ] if c in out.columns]
    return out[keep].sort_values(["rank", "ticker"]).reset_index(drop=True)


def _paper_cfg(config: dict[str, Any]) -> dict[str, Any]:
    return {
        "lot_size": int(config.get("paper_lot_size", 100)),
        "cash_buffer_pct": float(config.get("paper_cash_buffer_pct", 0.01)),
        "max_single_weight": float(config.get("paper_max_single_weight", 0.10)),
        "enforce_t_plus_one": bool(config.get("paper_enforce_t_plus_one", True)),
        "allow_short": bool(config.get("paper_allow_short", False)),
        "commission_bps": float(config.get("paper_commission_bps", 3.0)),
        "min_commission_cny": float(config.get("paper_min_commission_cny", 5.0)),
        "slippage_bps": float(config.get("paper_slippage_bps", 2.0)),
    }


def run_paper_day(
    state_dir,
    current_targets: pd.DataFrame,
    raw_prices: pd.DataFrame,
    stock_limits: pd.DataFrame | None,
    signal_report: dict[str, Any],
    config: dict[str, Any],
    trade_calendar: pd.DataFrame,
    initial_cash: float = 500000.0,
    fault_point: str | None = None,
) -> dict[str, Any]:
    """
    Post-close paper session.

    If yesterday's targets are staged, they are simulated at today's real raw OPEN.
    Today's new targets are only staged for the next completed trading session.
    NAV is marked at today's real raw CLOSE. This separation prevents same-day
    close signals from receiving an impossible same-day open fill.
    """
    state_dir = Path(state_dir)
    state_dir.mkdir(parents=True, exist_ok=True)
    account_id = "PAPER-CSI300-LIVE"
    ledger = PaperLedger(state_dir / "paper.sqlite3")

    signal_date = _date(signal_report["signal_date"])
    candidates = _candidate_frame(current_targets, signal_date)
    target_dates = pd.to_datetime(current_targets.get("signal_date", signal_date), errors="coerce")
    if hasattr(target_dates, "dropna") and not target_dates.dropna().empty:
        if not target_dates.dropna().dt.normalize().eq(signal_date).all():
            raise ValueError("targets 的 signal_date 与 signal_report 不一致")
    if "source" in current_targets.columns and not current_targets["source"].astype(str).eq("real").all():
        raise ValueError("每日 Paper Trading 只接受 source=real 的候选组合")

    raw = _ensure_raw_prices(raw_prices)
    if raw["date"].max() != signal_date:
        raise ValueError("raw_prices 未更新到当前 signal_date")
    next_session = next_open_session(trade_calendar, signal_date)
    signal_hash = signal_bundle_hash(current_targets, signal_date)

    existing_run = ledger.get_run(account_id, signal_date)
    if existing_run is not None:
        if existing_run["signal_hash"] != signal_hash:
            raise ValueError(
                "同一 signal_date 已提交不同 signal bundle，禁止静默覆盖"
            )
        ledger.export(state_dir, account_id, signal_date)
        return {
            "status": "IDEMPOTENT",
            "signal_date": str(signal_date.date()),
            "orders": 0,
            "fills": 0,
            "nav": ledger.last_nav(account_id),
        }

    state = ledger.load_state(account_id)
    cfg = _paper_cfg(config)
    account = ledger.load_account(account_id)
    if account is None:
        account = init_account(
            float(initial_cash),
            signal_date,
            account_id=account_id,
        )
    pending = ledger.load_pending(account_id)

    close_map = latest_close_map(raw, signal_date)
    exact_open = exact_price_map(raw, signal_date, "open")
    risk_prices = close_map.copy()
    risk_prices.update(exact_open)

    orders = pd.DataFrame()
    fills = pd.DataFrame()
    executed_signal_date = None

    if not pending.empty and state.get("pending_signal_date"):
        pending_signal_date = _date(state["pending_signal_date"])
        if pending_signal_date >= signal_date:
            raise ValueError("待执行信号日期必须早于当前交易日")
        if not state.get("pending_execution_date"):
            raise ValueError("pending state 缺少 expected execution date")

        expected_execution = _date(state["pending_execution_date"])
        if signal_date != expected_execution:
            raise ValueError(
                f"漏跑或错位交易日：expected={expected_execution.date()}, "
                f"actual={signal_date.date()}"
            )

        orders = pending.copy()
        if not orders.empty:
            orders["trade_date"] = pd.to_datetime(
                orders["trade_date"], errors="raise"
            ).dt.normalize()
            if not orders["trade_date"].eq(signal_date).all():
                raise ValueError("pending intent 的 trade_date 与当前执行日不一致")
        executed_signal_date = str(pending_signal_date.date())

        orders = _apply_open_constraints(
            orders,
            exact_open,
            stock_limits,
            signal_date,
        )

        if not orders.empty:
            checks = pre_trade_checks(
                account,
                orders,
                risk_prices,
                max_single_weight=cfg["max_single_weight"],
                allow_short=cfg["allow_short"],
                enforce_t_plus_one=cfg["enforce_t_plus_one"],
            )
            orders = orders.merge(
                checks,
                on=["order_id", "ticker", "side", "qty"],
                how="left",
                validate="one_to_one",
            )
            passed = orders[orders["check_status"] == "PASS"].copy()
            account, fills = execute_orders(
                account,
                passed,
                exact_open,
                signal_date,
                commission_bps=cfg["commission_bps"],
                min_commission_cny=cfg["min_commission_cny"],
                slippage_bps=cfg["slippage_bps"],
                enforce_t_plus_one=cfg["enforce_t_plus_one"],
                price_limits=price_limit_map_for_date(stock_limits, signal_date),
            )

    summary, positions = reconcile(account, close_map, signal_date)

    pending_intents = generate_orders(
        account,
        current_targets,
        close_map,
        next_session,
        lot_size=cfg["lot_size"],
        cash_buffer_pct=cfg["cash_buffer_pct"],
        max_single_weight=cfg["max_single_weight"],
        enforce_t_plus_one=cfg["enforce_t_plus_one"],
        allow_short=cfg["allow_short"],
        stock_limits=None,
    )
    pending_intents["source_signal_date"] = str(signal_date.date())
    pending_intents["expected_execution_date"] = str(next_session.date())

    state = {
        "schema_version": 2,
        "mode": "REAL_DATA_PAPER_ONLY",
        "last_completed_signal_date": str(signal_date.date()),
        "pending_signal_date": str(signal_date.date()),
        "pending_execution_date": str(next_session.date()),
        "pending_intent_count": int(len(pending_intents)),
        "executed_signal_date": executed_signal_date,
        "account_id": account["account_id"],
        "real_broker_submission": False,
    }

    commit_status = ledger.commit_day(
        account_id=account_id,
        signal_date=signal_date,
        signal_hash=signal_hash,
        executed_signal_date=executed_signal_date,
        account=account,
        state=state,
        pending_intents=pending_intents,
        candidates=candidates,
        orders=orders,
        fills=fills,
        summary=summary,
        positions=positions,
        fault_point=fault_point,
    )
    if commit_status != "OK":
        raise RuntimeError(f"unexpected ledger commit status: {commit_status}")
    ledger.export(state_dir, account_id, signal_date)

    return {
        "status": "OK",
        "signal_date": str(signal_date.date()),
        "executed_signal_date": executed_signal_date,
        "orders": int(len(orders)),
        "fills": int(len(fills)),
        "nav": float(summary.iloc[0]["nav"]),
        "cash": float(summary.iloc[0]["cash"]),
        "positions": int(summary.iloc[0]["n_positions"]),
        "pending_intents": int(len(pending_intents)),
        "pending_execution_date": str(next_session.date()),
        "state_dir": str(state_dir),
    }
