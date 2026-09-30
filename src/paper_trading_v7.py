
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import math
import uuid
import numpy as np
import pandas as pd

from .research_v4 import stamp_duty_rate


def _ts(x):
    return pd.Timestamp(x).normalize()


def load_account(path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"账户文件不存在: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def save_account(account, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(account, ensure_ascii=False, indent=2), encoding="utf-8")


def init_account(initial_cash, as_of, account_id="PAPER-CSI300"):
    return {
        "account_id": account_id,
        "currency": "CNY",
        "created_at": str(_ts(as_of).date()),
        "as_of": str(_ts(as_of).date()),
        "cash": float(initial_cash),
        "cash_receivables": {},
        "share_receivables": {},
        "positions": {},
        "realized_pnl": 0.0,
        "fees_paid": 0.0,
        "trade_seq": 0,
    }


def _position_qty(account, ticker):
    p = account.get("positions", {}).get(ticker, {})
    return int(sum(int(l.get("qty", 0)) for l in p.get("lots", [])))


def _pending_share_qty(account, ticker):
    return int(sum(
        int(v.get("qty", 0))
        for v in account.get("share_receivables", {}).values()
        if str(v.get("ticker")) == str(ticker)
    ))


def _economic_position_qty(account, ticker):
    return int(_position_qty(account, ticker) + _pending_share_qty(account, ticker))


def _cash_receivable_value(account):
    return float(sum(
        float(v.get("amount", 0.0))
        for v in account.get("cash_receivables", {}).values()
    ))


def _economic_tickers(account):
    out = set(account.get("positions", {}).keys())
    out |= {
        str(v.get("ticker"))
        for v in account.get("share_receivables", {}).values()
        if v.get("ticker")
    }
    return out


def sellable_qty(account, ticker, trade_date, enforce_t_plus_one=True):
    p = account.get("positions", {}).get(ticker, {})
    total = 0
    td = _ts(trade_date)
    for lot in p.get("lots", []):
        q = int(lot.get("qty", 0))
        if q <= 0:
            continue
        sellable_from = lot.get("sellable_from")
        if sellable_from and _ts(sellable_from) > td:
            continue
        if not enforce_t_plus_one:
            total += q
        else:
            acquired = _ts(lot.get("acquired_date"))
            if acquired < td:
                total += q
    return int(total)


def position_cost_basis(account, ticker):
    p = account.get("positions", {}).get(ticker, {})
    lots = [l for l in p.get("lots", []) if int(l.get("qty", 0)) > 0]
    settled_qty = sum(int(l["qty"]) for l in lots)
    economic_qty = settled_qty + _pending_share_qty(account, ticker)
    if economic_qty <= 0:
        return np.nan
    total_cost = sum(int(l["qty"])*float(l["price"]) for l in lots)
    return float(total_cost / economic_qty)


def market_value(account, price_map):
    mv = 0.0
    for ticker in _economic_tickers(account):
        qty = _economic_position_qty(account, ticker)
        px = price_map.get(ticker, np.nan)
        if qty > 0:
            if pd.isna(px) or not np.isfinite(float(px)) or float(px)<=0:
                raise ValueError(f"{ticker}: 持仓估值缺失")
            mv += qty * float(px)
    return float(mv)


def nav(account, price_map):
    return (
        float(account.get("cash", 0.0))
        + _cash_receivable_value(account)
        + market_value(account, price_map)
    )


def account_snapshot(account, price_map, as_of):
    rows = []
    total_nav = nav(account, price_map)
    for ticker in sorted(_economic_tickers(account)):
        settled_qty = _position_qty(account, ticker)
        pending_share_qty = _pending_share_qty(account, ticker)
        qty = settled_qty + pending_share_qty
        if qty <= 0:
            continue
        px = price_map.get(ticker, np.nan)
        mv = qty * float(px) if pd.notna(px) else np.nan
        basis = position_cost_basis(account, ticker)
        rows.append({
            "as_of": _ts(as_of),
            "ticker": ticker,
            "qty": qty,
            "settled_qty": settled_qty,
            "pending_share_qty": pending_share_qty,
            "sellable_qty": sellable_qty(account, ticker, as_of, True),
            "last_price": px,
            "market_value": mv,
            "weight": (mv/total_nav if total_nav > 0 and pd.notna(mv) else np.nan),
            "avg_cost": basis,
            "unrealized_pnl": ((float(px)-basis)*qty if pd.notna(px) and pd.notna(basis) else np.nan),
        })
    return pd.DataFrame(rows)


def _floor_lot(qty, lot_size):
    if qty <= 0:
        return 0
    return int(math.floor(qty / lot_size) * lot_size)


def _target_shares_from_weight(nav_value, weight, price, lot_size):
    if price <= 0 or weight <= 0:
        return 0
    raw = nav_value * float(weight) / float(price)
    return _floor_lot(raw, lot_size)


def _current_weights(account, prices):
    total_nav = nav(account, prices)
    out = {}
    if total_nav <= 0:
        return out
    for ticker in account.get("positions", {}):
        q = _position_qty(account, ticker)
        px = prices.get(ticker, np.nan)
        if q > 0 and pd.notna(px):
            out[ticker] = q*float(px)/total_nav
    return out


def _limit_flags(stock_limits, trade_date):
    if stock_limits is None or stock_limits.empty:
        return {}
    x = stock_limits.copy()
    x["date"] = pd.to_datetime(x["date"], errors="coerce")
    x = x[x["date"] == _ts(trade_date)]
    return {
        str(r["ticker"]): (r.get("up_limit", np.nan), r.get("down_limit", np.nan))
        for _, r in x.iterrows()
    }


def generate_orders(
    account,
    targets: pd.DataFrame,
    prices: dict,
    trade_date,
    lot_size=100,
    cash_buffer_pct=0.01,
    max_single_weight=0.10,
    enforce_t_plus_one=True,
    allow_short=False,
    stock_limits: pd.DataFrame | None = None,
):
    """
    Convert target portfolio weights into executable paper orders.

    Buy quantities are rounded DOWN to board lots.
    Sell quantities can include an odd-lot residual only when fully closing a position.
    No short selling by default.
    """
    trade_date = _ts(trade_date)
    g = targets.copy()
    if "target_weight" not in g.columns or "ticker" not in g.columns:
        raise ValueError("targets 必须包含 ticker 和 target_weight")

    g = g[["ticker","target_weight"]].drop_duplicates("ticker").copy()
    g["ticker"] = g["ticker"].astype(str)
    g["target_weight"] = pd.to_numeric(g["target_weight"], errors="coerce").fillna(0.0)
    g["target_weight"] = g["target_weight"].clip(lower=0.0, upper=float(max_single_weight))

    total_nav = nav(account, prices)
    investable_nav = total_nav * (1.0 - float(cash_buffer_pct))
    limit_map = _limit_flags(stock_limits, trade_date)

    current_tickers = _economic_tickers(account)
    target_tickers = set(g["ticker"])
    all_tickers = sorted(current_tickers | target_tickers)

    target_weight_map = g.set_index("ticker")["target_weight"].to_dict()

    order_rows = []
    for ticker in all_tickers:
        px = prices.get(ticker, np.nan)
        if pd.isna(px) or float(px) <= 0:
            order_rows.append({
                "order_id": str(uuid.uuid4())[:8],
                "trade_date": trade_date,
                "ticker": ticker,
                "side": "NONE",
                "qty": 0,
                "reference_price": np.nan,
                "target_weight": float(target_weight_map.get(ticker, 0.0)),
                "status": "REJECTED",
                "reason": "missing_price",
            })
            continue

        current_qty = _economic_position_qty(account, ticker)
        target_weight = float(target_weight_map.get(ticker, 0.0))
        target_qty = _target_shares_from_weight(
            investable_nav, target_weight, float(px), int(lot_size)
        )
        diff = target_qty - current_qty

        if diff > 0:
            qty = _floor_lot(diff, int(lot_size))
            if qty <= 0:
                continue
            up, _ = limit_map.get(ticker, (np.nan, np.nan))
            if pd.notna(up) and float(px) >= float(up) - 1e-8:
                status, reason = "REJECTED", "limit_up"
            else:
                status, reason = "NEW", ""
            side = "BUY"

        elif diff < 0:
            desired_sell = abs(diff)
            available = sellable_qty(account, ticker, trade_date, enforce_t_plus_one)

            # If fully closing, allow selling the exact odd-lot residual.
            if target_qty == 0:
                qty = min(current_qty, available)
            else:
                qty = min(_floor_lot(desired_sell, int(lot_size)), available)

            if qty <= 0:
                order_rows.append({
                    "order_id": str(uuid.uuid4())[:8],
                    "trade_date": trade_date,
                    "ticker": ticker,
                    "side": "SELL",
                    "qty": 0,
                    "reference_price": float(px),
                    "target_weight": target_weight,
                    "status": "REJECTED",
                    "reason": "t_plus_one_or_no_sellable_qty",
                })
                continue

            _, dn = limit_map.get(ticker, (np.nan, np.nan))
            if pd.notna(dn) and float(px) <= float(dn) + 1e-8:
                status, reason = "REJECTED", "limit_down"
            else:
                status, reason = "NEW", ""
            side = "SELL"
        else:
            continue

        if not allow_short and side == "SELL":
            qty = min(qty, current_qty)

        order_rows.append({
            "order_id": str(uuid.uuid4())[:8],
            "trade_date": trade_date,
            "ticker": ticker,
            "side": side,
            "qty": int(qty),
            "reference_price": float(px),
            "target_weight": target_weight,
            "status": status,
            "reason": reason,
        })

    orders = pd.DataFrame(order_rows)

    # Cash-aware buy scaling after sell orders are considered.
    if not orders.empty:
        cash = float(account.get("cash", 0.0))
        expected_sell = (
            orders[(orders["side"]=="SELL") & (orders["status"]=="NEW")]
            .assign(v=lambda x: x["qty"]*x["reference_price"])["v"].sum()
        )
        buying_power = cash + expected_sell

        new_buys = orders[(orders["side"]=="BUY") & (orders["status"]=="NEW")].copy()
        est_buy = float((new_buys["qty"]*new_buys["reference_price"]).sum())

        if est_buy > buying_power and est_buy > 0:
            scale = max(0.0, buying_power/est_buy)
            for idx in new_buys.index:
                old_qty = int(orders.loc[idx, "qty"])
                scaled_qty = _floor_lot(old_qty*scale, int(lot_size))
                orders.loc[idx, "qty"] = scaled_qty
                if scaled_qty <= 0:
                    orders.loc[idx, "status"] = "REJECTED"
                    orders.loc[idx, "reason"] = "insufficient_cash_after_scaling"

    return orders.reset_index(drop=True)


def _commission(notional, commission_bps, min_commission):
    if notional <= 0:
        return 0.0
    return max(float(min_commission), notional*float(commission_bps)/10000.0)


def _transfer_fee(notional, transfer_fee_bps):
    if notional <= 0:
        return 0.0
    return float(notional) * float(transfer_fee_bps) / 10000.0


def _consume_fifo_lots(account, ticker, qty_to_sell, sell_price, trade_date, enforce_t_plus_one):
    p = account["positions"].setdefault(ticker, {"lots":[]})
    lots = p.get("lots", [])
    remaining = int(qty_to_sell)
    realized = 0.0
    td = _ts(trade_date)

    for lot in lots:
        if remaining <= 0:
            break
        q = int(lot.get("qty",0))
        if q <= 0:
            continue
        sellable_from = lot.get("sellable_from")
        if sellable_from and _ts(sellable_from) > td:
            continue
        if enforce_t_plus_one and _ts(lot["acquired_date"]) >= td:
            continue

        take = min(q, remaining)
        realized += take*(float(sell_price)-float(lot["price"]))
        lot["qty"] = q-take
        remaining -= take

    p["lots"] = [l for l in lots if int(l.get("qty",0))>0]
    if not p["lots"]:
        account["positions"].pop(ticker, None)

    return int(qty_to_sell-remaining), float(realized)


def execute_orders(
    account,
    orders: pd.DataFrame,
    execution_prices: dict,
    trade_date,
    commission_bps=3.0,
    min_commission_cny=5.0,
    slippage_bps=2.0,
    transfer_fee_bps=0.0,
    enforce_t_plus_one=True,
    price_limits=None,
):
    """
    Paper execution only. It does NOT send orders to a broker.

    SELL orders are processed first to free cash, then BUY orders.
    """
    trade_date = _ts(trade_date)
    if orders is None or orders.empty:
        return account, pd.DataFrame()

    fills = []
    work = orders.copy()
    work = work[work["status"]=="NEW"].copy()
    used=set(account.get("executed_order_ids",[]))
    if work["order_id"].duplicated().any() or set(work["order_id"]) & used:
        raise ValueError("重复模拟委托，禁止重复入账")

    # SELL first, then BUY
    side_order = pd.Categorical(work["side"], categories=["SELL","BUY"], ordered=True)
    work = work.assign(_side_order=side_order).sort_values("_side_order")

    for _, o in work.iterrows():
        ticker = str(o["ticker"])
        side = str(o["side"])
        qty = int(o["qty"])
        if qty <= 0:
            continue

        ref = execution_prices.get(ticker, o.get("reference_price", np.nan))
        if pd.isna(ref) or float(ref) <= 0:
            continue

        raw_execution_price = float(ref)
        slip = float(slippage_bps)/10000.0
        proposed_fill_price = raw_execution_price*(1+slip if side=="BUY" else 1-slip)
        price_limit_bounded = False
        fill_price = proposed_fill_price

        if price_limits is not None:
            limits = price_limits.get(ticker)
            if limits is None:
                # Defensive fail-closed. The daily-paper coordinator should
                # already reject this order before reaching the executor.
                continue
            up_limit, down_limit = limits
            up_limit, down_limit = float(up_limit), float(down_limit)
            if (
                not np.isfinite(up_limit)
                or not np.isfinite(down_limit)
                or up_limit <= 0
                or down_limit <= 0
                or down_limit > up_limit
                or raw_execution_price > up_limit + 1e-8
                or raw_execution_price < down_limit - 1e-8
            ):
                continue
            bounded = min(max(proposed_fill_price, down_limit), up_limit)
            price_limit_bounded = abs(bounded - proposed_fill_price) > 1e-12
            fill_price = bounded

        notional = qty*fill_price
        commission = _commission(notional, commission_bps, min_commission_cny)
        transfer_fee = _transfer_fee(notional, transfer_fee_bps)
        stamp = notional*stamp_duty_rate(trade_date) if side=="SELL" else 0.0
        total_fee = commission + transfer_fee + stamp

        if side == "SELL":
            available = sellable_qty(account, ticker, trade_date, enforce_t_plus_one)
            qty_exec = min(qty, available)
            if qty_exec <= 0:
                continue
            notional = qty_exec*fill_price
            commission = _commission(notional, commission_bps, min_commission_cny)
            transfer_fee = _transfer_fee(notional, transfer_fee_bps)
            stamp = notional*stamp_duty_rate(trade_date)
            total_fee = commission + transfer_fee + stamp

            sold, realized = _consume_fifo_lots(
                account, ticker, qty_exec, fill_price, trade_date, enforce_t_plus_one
            )
            if sold <= 0:
                continue
            notional = sold*fill_price
            account["cash"] += notional-total_fee
            account["realized_pnl"] = float(account.get("realized_pnl",0.0))+realized-total_fee
            account["fees_paid"] = float(account.get("fees_paid",0.0))+total_fee
            qty_exec = sold

        else:
            total_cash_needed = notional + commission + transfer_fee
            if total_cash_needed > float(account["cash"]) + 1e-8:
                # Reduce to maximum affordable board lots.
                lot_size = 100
                affordable = int(float(account["cash"]) / max(fill_price, 1e-9))
                qty_exec = int(math.floor(affordable/lot_size)*lot_size)
                if qty_exec <= 0:
                    continue
                notional = qty_exec*fill_price
                commission = _commission(notional, commission_bps, min_commission_cny)
                transfer_fee = _transfer_fee(notional, transfer_fee_bps)
                total_cash_needed = notional + commission + transfer_fee
                if total_cash_needed > float(account["cash"]) + 1e-8:
                    continue
            else:
                qty_exec = qty

            account["cash"] -= total_cash_needed
            total_fee = commission + transfer_fee
            account["fees_paid"] = float(account.get("fees_paid",0.0)) + total_fee
            p = account["positions"].setdefault(ticker, {"lots":[]})
            p["lots"].append({
                "qty": int(qty_exec),
                "price": float(fill_price),
                "acquired_date": str(trade_date.date()),
            })
            stamp = 0.0

        account["trade_seq"] = int(account.get("trade_seq",0))+1
        fills.append({
            "fill_id": f"F{account['trade_seq']:06d}",
            "order_id": o["order_id"],
            "trade_date": trade_date,
            "ticker": ticker,
            "side": side,
            "qty": int(qty_exec),
            "reference_price": float(raw_execution_price),
            "fill_price": float(fill_price),
            "price_limit_bounded": bool(price_limit_bounded),
            "notional": float(qty_exec*fill_price),
            "commission": float(commission),
            "transfer_fee": float(transfer_fee),
            "stamp_duty": float(stamp),
            "total_fee": float(total_fee),
        })

    account["as_of"] = str(trade_date.date())
    account["executed_order_ids"]=sorted(used | {f["order_id"] for f in fills})
    return account, pd.DataFrame(fills)


def pre_trade_checks(
    account,
    orders: pd.DataFrame,
    prices: dict,
    max_single_weight=0.10,
    allow_short=False,
    enforce_t_plus_one=True,
):
    rows = []
    total_nav = nav(account, prices)

    for _, o in orders.iterrows():
        ticker = str(o["ticker"])
        side = str(o["side"])
        qty = int(o["qty"])
        px = prices.get(ticker, o.get("reference_price", np.nan))
        current_qty = _economic_position_qty(account, ticker)
        available = sellable_qty(account, ticker, o["trade_date"], enforce_t_plus_one)

        issues = []
        if side == "SELL" and not allow_short and qty > current_qty:
            issues.append("sell_exceeds_position")
        if side == "SELL" and qty > available:
            issues.append("sell_exceeds_sellable_qty")
        if side == "BUY" and pd.notna(px) and total_nav > 0:
            post_value = (current_qty+qty)*float(px)
            if post_value/total_nav > float(max_single_weight)+1e-8:
                issues.append("single_name_weight_limit")
        if o.get("status") != "NEW":
            issues.append(str(o.get("reason","rejected")))

        rows.append({
            "order_id": o["order_id"],
            "ticker": ticker,
            "side": side,
            "qty": qty,
            "check_status": "PASS" if not issues else "BLOCK",
            "issues": "|".join(issues),
        })
    return pd.DataFrame(rows)


def reconcile(account, prices, as_of):
    snap = account_snapshot(account, prices, as_of)
    total_mv = float(snap["market_value"].sum()) if not snap.empty else 0.0
    cash_receivable = _cash_receivable_value(account)
    total_nav = float(account.get("cash",0.0)) + cash_receivable + total_mv

    summary = pd.DataFrame([{
        "as_of": _ts(as_of),
        "cash": float(account.get("cash",0.0)),
        "cash_receivable": cash_receivable,
        "market_value": total_mv,
        "nav": total_nav,
        "realized_pnl": float(account.get("realized_pnl",0.0)),
        "fees_paid": float(account.get("fees_paid",0.0)),
        "n_positions": int(len(snap)),
    }])
    return summary, snap
