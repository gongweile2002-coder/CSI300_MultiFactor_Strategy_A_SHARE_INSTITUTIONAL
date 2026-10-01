from __future__ import annotations

import hashlib
import json
import math
from typing import Any, Callable

import pandas as pd


_DATE_COLS = [
    "end_date",
    "ann_date",
    "record_date",
    "ex_date",
    "pay_date",
    "div_listdate",
    "imp_ann_date",
    "base_date",
]


def _date(value) -> pd.Timestamp:
    return pd.Timestamp(value).normalize()


def _number(value, default=0.0) -> float:
    if value is None or pd.isna(value):
        return float(default)
    return float(value)


def normalize_corporate_actions(frame: pd.DataFrame | None) -> pd.DataFrame:
    columns = [
        "ticker",
        "end_date",
        "ann_date",
        "div_proc",
        "stk_div",
        "stk_bo_rate",
        "stk_co_rate",
        "cash_div",
        "cash_div_tax",
        "record_date",
        "ex_date",
        "pay_date",
        "div_listdate",
        "imp_ann_date",
        "base_date",
        "base_share",
        "action_id",
    ]
    if frame is None or frame.empty:
        return pd.DataFrame(columns=columns)

    x = frame.copy().rename(columns={"ts_code": "ticker"})
    required = {"ticker", "div_proc", "record_date", "ex_date"}
    missing = required - set(x.columns)
    if missing:
        raise ValueError(f"corporate_actions 缺少列: {sorted(missing)}")

    x["ticker"] = x["ticker"].astype(str)
    for col in _DATE_COLS:
        if col in x.columns:
            x[col] = pd.to_datetime(x[col], errors="coerce").dt.normalize()
        else:
            x[col] = pd.NaT

    for col in [
        "stk_div",
        "stk_bo_rate",
        "stk_co_rate",
        "cash_div",
        "cash_div_tax",
        "base_share",
    ]:
        if col not in x.columns:
            x[col] = 0.0
        x[col] = pd.to_numeric(x[col], errors="coerce")

    # Only implemented distributions have enforceable record/ex/payment dates.
    x = x[x["div_proc"].astype(str).str.strip().eq("实施")].copy()
    x = x.dropna(subset=["record_date", "ex_date"])
    x = x[x["record_date"] <= x["ex_date"]].copy()

    def make_id(row: pd.Series) -> str:
        fields = {
            "ticker": str(row["ticker"]),
            "end_date": str(row["end_date"].date()) if pd.notna(row["end_date"]) else "",
            "record_date": str(row["record_date"].date()),
            "ex_date": str(row["ex_date"].date()),
            "pay_date": str(row["pay_date"].date()) if pd.notna(row["pay_date"]) else "",
            "div_listdate": (
                str(row["div_listdate"].date()) if pd.notna(row["div_listdate"]) else ""
            ),
            "imp_ann_date": (
                str(row["imp_ann_date"].date()) if pd.notna(row["imp_ann_date"]) else ""
            ),
            "stk_div": _number(row["stk_div"]),
            "cash_div": _number(row["cash_div"]),
            "cash_div_tax": _number(row["cash_div_tax"]),
        }
        raw = json.dumps(fields, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    x["action_id"] = x.apply(make_id, axis=1)
    x = x.drop_duplicates("action_id", keep="last")
    for col in columns:
        if col not in x.columns:
            x[col] = pd.NA
    return x[columns].sort_values(["ex_date", "ticker", "action_id"]).reset_index(drop=True)


def _pending_share_qty(account: dict[str, Any], ticker: str) -> int:
    total = 0
    for item in account.get("share_receivables", {}).values():
        if str(item.get("ticker")) == str(ticker):
            total += int(item.get("qty", 0))
    return int(total)


def _economic_tickers(account: dict[str, Any]) -> set[str]:
    tickers = set(account.get("positions", {}).keys())
    tickers |= {
        str(v.get("ticker"))
        for v in account.get("share_receivables", {}).values()
        if v.get("ticker")
    }
    return tickers


def settle_due_corporate_actions(
    account: dict[str, Any],
    trade_date,
) -> list[dict[str, Any]]:
    day = _date(trade_date)
    events: list[dict[str, Any]] = []

    cash_receivables = account.setdefault("cash_receivables", {})
    for action_id, item in list(cash_receivables.items()):
        pay_date = pd.to_datetime(item.get("pay_date"), errors="coerce")
        if pd.isna(pay_date) or _date(pay_date) > day:
            continue
        amount = float(item.get("amount", 0.0))
        account["cash"] = float(account.get("cash", 0.0)) + amount
        events.append({
            "event_key": f"{action_id}|CASH_PAID",
            "action_id": action_id,
            "event_type": "CASH_PAID",
            "event_date": str(day.date()),
            "ticker": str(item.get("ticker", "")),
            "amount": amount,
            "qty": 0,
        })
        del cash_receivables[action_id]

    share_receivables = account.setdefault("share_receivables", {})
    for action_id, item in list(share_receivables.items()):
        list_date = pd.to_datetime(item.get("list_date"), errors="coerce")
        if pd.isna(list_date) or _date(list_date) > day:
            continue
        ticker = str(item["ticker"])
        qty = int(item.get("qty", 0))
        if qty > 0:
            pos = account.setdefault("positions", {}).setdefault(ticker, {"lots": []})
            pos.setdefault("lots", []).append({
                "qty": qty,
                "price": 0.0,
                "acquired_date": str(item.get("ex_date")),
                "sellable_from": str(_date(list_date).date()),
                "source": "corporate_action",
                "action_id": action_id,
            })
        events.append({
            "event_key": f"{action_id}|SHARES_LISTED",
            "action_id": action_id,
            "event_type": "SHARES_LISTED",
            "event_date": str(day.date()),
            "ticker": ticker,
            "amount": 0.0,
            "qty": qty,
        })
        del share_receivables[action_id]

    return events


def apply_corporate_actions(
    account: dict[str, Any],
    corporate_actions: pd.DataFrame | None,
    trade_date,
    *,
    account_id: str,
    applied_action_ids: set[str],
    position_qty_at: Callable[[str, str, Any], int | None],
) -> tuple[dict[str, Any], pd.DataFrame]:
    """
    Apply ex-date entitlements using the record-date paper position snapshot.

    Cash dividends use Tushare's cash_div (after-tax) field and are carried as an
    economic receivable from ex-date until pay_date. Stock/bonus shares are
    carried as non-tradable share receivables from ex-date until div_listdate.
    """
    day = _date(trade_date)
    event_rows = settle_due_corporate_actions(account, day)
    actions = normalize_corporate_actions(corporate_actions)
    if actions.empty:
        return account, pd.DataFrame(event_rows)

    relevant = actions[actions["ex_date"].eq(day)].copy()
    economic_tickers = _economic_tickers(account)
    relevant = relevant[relevant["ticker"].isin(economic_tickers)]

    for _, row in relevant.iterrows():
        action_id = str(row["action_id"])
        if action_id in applied_action_ids:
            continue

        record_date = _date(row["record_date"])
        eligible_qty = position_qty_at(account_id, str(row["ticker"]), record_date)
        if eligible_qty is None:
            raise ValueError(
                f"{row['ticker']}: 缺少股权登记日 {record_date.date()} 的已提交持仓快照"
            )
        eligible_qty = int(eligible_qty)
        if eligible_qty <= 0:
            continue

        cash_per_share = _number(row.get("cash_div"), 0.0)
        stock_per_share = _number(row.get("stk_div"), 0.0)
        cash_amount = max(0.0, eligible_qty * cash_per_share)
        stock_float = max(0.0, eligible_qty * stock_per_share)
        stock_qty = int(round(stock_float))
        if not math.isclose(stock_float, stock_qty, abs_tol=1e-8):
            raise ValueError(
                f"{row['ticker']}: 送转股产生非整数权益 {stock_float}，需券商舍入规则"
            )

        if cash_amount > 0:
            if pd.isna(row["pay_date"]):
                raise ValueError(f"{row['ticker']}: 现金分红缺少 pay_date")
            account.setdefault("cash_receivables", {})[action_id] = {
                "ticker": str(row["ticker"]),
                "amount": float(cash_amount),
                "pay_date": str(_date(row["pay_date"]).date()),
                "ex_date": str(day.date()),
            }

        if stock_qty > 0:
            if pd.isna(row["div_listdate"]):
                raise ValueError(f"{row['ticker']}: 送转股缺少 div_listdate")
            account.setdefault("share_receivables", {})[action_id] = {
                "ticker": str(row["ticker"]),
                "qty": int(stock_qty),
                "list_date": str(_date(row["div_listdate"]).date()),
                "ex_date": str(day.date()),
            }

        event_rows.append({
            "event_key": f"{action_id}|ENTITLEMENT",
            "action_id": action_id,
            "event_type": "ENTITLEMENT",
            "event_date": str(day.date()),
            "ticker": str(row["ticker"]),
            "record_date": str(record_date.date()),
            "eligible_qty": eligible_qty,
            "cash_per_share": cash_per_share,
            "cash_amount": cash_amount,
            "stock_per_share": stock_per_share,
            "qty": stock_qty,
            "pay_date": (
                str(_date(row["pay_date"]).date()) if pd.notna(row["pay_date"]) else ""
            ),
            "div_listdate": (
                str(_date(row["div_listdate"]).date())
                if pd.notna(row["div_listdate"])
                else ""
            ),
        })

    # Same-day pay/list dates should settle in the same atomic paper run.
    event_rows.extend(settle_due_corporate_actions(account, day))
    return account, pd.DataFrame(event_rows)
