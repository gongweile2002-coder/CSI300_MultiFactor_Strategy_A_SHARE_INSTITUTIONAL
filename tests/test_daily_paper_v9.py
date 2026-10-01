import json
from pathlib import Path

import pandas as pd
import pytest

from src.daily_paper_v9 import run_paper_day


def _config():
    return {
        "paper_lot_size": 100,
        "paper_commission_bps": 0,
        "paper_min_commission_cny": 0,
        "paper_transfer_fee_bps": 0,
        "paper_slippage_bps": 0,
        "paper_cash_buffer_pct": 0,
        "paper_max_single_weight": 0.8,
        "paper_allow_short": False,
        "paper_enforce_t_plus_one": True,
    }


def _raw():
    return pd.DataFrame([
        {"date": "2026-09-01", "ticker": "000001.SZ", "open": 10.0, "close": 10.0, "price_basis": "raw"},
        {"date": "2026-09-02", "ticker": "000001.SZ", "open": 11.0, "close": 12.0, "price_basis": "raw"},
        {"date": "2026-09-03", "ticker": "000001.SZ", "open": 13.0, "close": 13.5, "price_basis": "raw"},
    ])


def _calendar():
    return pd.DataFrame([
        {"date": "2026-09-01", "is_open": 1},
        {"date": "2026-09-02", "is_open": 1},
        {"date": "2026-09-03", "is_open": 1},
        {"date": "2026-09-04", "is_open": 1},
    ])


def _limits():
    return pd.DataFrame([
        {
            "date": "2026-09-02",
            "ticker": "000001.SZ",
            "up_limit": 20.0,
            "down_limit": 5.0,
        },
        {
            "date": "2026-09-03",
            "ticker": "000001.SZ",
            "up_limit": 20.0,
            "down_limit": 5.0,
        },
    ])


def test_first_day_stages_candidates_without_same_day_fill(tmp_path):
    targets = pd.DataFrame([{
        "ticker": "000001.SZ",
        "target_weight": 0.5,
        "signal_date": "2026-09-01",
        "source": "real",
        "composite_score": 1.2,
    }])
    result = run_paper_day(
        tmp_path,
        targets,
        _raw()[lambda x: x["date"] == "2026-09-01"],
        pd.DataFrame(),
        {"signal_date": "2026-09-01"},
        _config(),
        _calendar(),
        initial_cash=100000,
    )
    assert result["status"] == "OK"
    assert result["fills"] == 0
    state = json.loads((tmp_path / "state.json").read_text())
    assert state["pending_signal_date"] == "2026-09-01"
    assert state["pending_execution_date"] == "2026-09-02"
    assert (tmp_path / "pending_intents.csv").exists()


def test_next_day_executes_prior_signal_at_real_open_and_marks_close(tmp_path):
    day1 = pd.DataFrame([{
        "ticker": "000001.SZ",
        "target_weight": 0.5,
        "signal_date": "2026-09-01",
        "source": "real",
        "composite_score": 1.2,
    }])
    run_paper_day(
        tmp_path, day1,
        _raw()[lambda x: x["date"] == "2026-09-01"],
        pd.DataFrame(),
        {"signal_date": "2026-09-01"},
        _config(), _calendar(), initial_cash=100000,
    )

    day2 = day1.copy()
    day2["signal_date"] = "2026-09-02"
    raw_to_day2 = _raw()[lambda x: x["date"].isin(["2026-09-01", "2026-09-02"])]
    result = run_paper_day(
        tmp_path, day2, raw_to_day2, _limits(),
        {"signal_date": "2026-09-02"},
        _config(), _calendar(), initial_cash=100000,
    )
    assert result["fills"] == 1
    fills = pd.read_csv(tmp_path / "fill_history.csv")
    assert float(fills.iloc[0]["fill_price"]) == 11.0
    assert int(fills.iloc[0]["qty"]) == 5000
    nav = pd.read_csv(tmp_path / "nav_history.csv")
    assert str(nav.iloc[-1]["as_of"]) == "2026-09-02"
    assert float(nav.iloc[-1]["nav"]) > 100000


def test_same_signal_date_is_idempotent(tmp_path):
    targets = pd.DataFrame([{
        "ticker": "000001.SZ",
        "target_weight": 0.5,
        "signal_date": "2026-09-01",
        "source": "real",
    }])
    first = run_paper_day(
        tmp_path, targets,
        _raw()[lambda x: x["date"] == "2026-09-01"],
        pd.DataFrame(),
        {"signal_date": "2026-09-01"},
        _config(), _calendar(), initial_cash=100000,
    )
    second = run_paper_day(
        tmp_path, targets,
        _raw()[lambda x: x["date"] == "2026-09-01"],
        pd.DataFrame(),
        {"signal_date": "2026-09-01"},
        _config(), _calendar(), initial_cash=100000,
    )
    assert first["status"] == "OK"
    assert second["status"] == "IDEMPOTENT"


def test_rejects_non_real_targets(tmp_path):
    targets = pd.DataFrame([{
        "ticker": "000001.SZ",
        "target_weight": 0.5,
        "signal_date": "2026-09-01",
        "source": "synthetic",
    }])
    try:
        run_paper_day(
            tmp_path, targets,
            _raw()[lambda x: x["date"] == "2026-09-01"],
            pd.DataFrame(),
            {"signal_date": "2026-09-01"},
            _config(), _calendar(), initial_cash=100000,
        )
    except ValueError as exc:
        assert "source=real" in str(exc)
    else:
        raise AssertionError("synthetic targets must be rejected")


def test_share_quantity_is_fixed_before_next_open(tmp_path):
    day1 = pd.DataFrame([{
        "ticker": "000001.SZ",
        "target_weight": 0.5,
        "signal_date": "2026-09-01",
        "source": "real",
    }])
    run_paper_day(
        tmp_path,
        day1,
        _raw()[lambda x: x["date"] == "2026-09-01"],
        pd.DataFrame(),
        {"signal_date": "2026-09-01"},
        _config(),
        _calendar(),
        initial_cash=100000,
    )

    pending = pd.read_csv(tmp_path / "pending_intents.csv")
    assert int(pending.loc[pending["side"] == "BUY", "qty"].iloc[0]) == 5000

    day2_raw = _raw()[lambda x: x["date"].isin(["2026-09-01", "2026-09-02"])].copy()
    day2_raw.loc[day2_raw["date"] == "2026-09-02", "open"] = 15.0
    day2 = day1.copy()
    day2["signal_date"] = "2026-09-02"
    run_paper_day(
        tmp_path,
        day2,
        day2_raw,
        _limits(),
        {"signal_date": "2026-09-02"},
        _config(),
        _calendar(),
        initial_cash=100000,
    )
    fills = pd.read_csv(tmp_path / "fill_history.csv")
    assert int(fills.iloc[0]["qty"]) == 5000


def test_missed_expected_session_does_not_fill_at_later_open(tmp_path):
    day1 = pd.DataFrame([{
        "ticker": "000001.SZ",
        "target_weight": 0.5,
        "signal_date": "2026-09-01",
        "source": "real",
    }])
    run_paper_day(
        tmp_path,
        day1,
        _raw()[lambda x: x["date"] == "2026-09-01"],
        pd.DataFrame(),
        {"signal_date": "2026-09-01"},
        _config(),
        _calendar(),
        initial_cash=100000,
    )

    day3 = day1.copy()
    day3["signal_date"] = "2026-09-03"
    raw_to_day3 = _raw()[lambda x: x["date"].isin(["2026-09-01", "2026-09-03"])]
    with pytest.raises(ValueError, match="expected=2026-09-02"):
        run_paper_day(
            tmp_path,
            day3,
            raw_to_day3,
            pd.DataFrame(),
            {"signal_date": "2026-09-03"},
            _config(),
            _calendar(),
            initial_cash=100000,
        )


def test_missing_limit_data_fails_closed(tmp_path):
    day1 = pd.DataFrame([{
        "ticker": "000001.SZ",
        "target_weight": 0.5,
        "signal_date": "2026-09-01",
        "source": "real",
    }])
    run_paper_day(
        tmp_path,
        day1,
        _raw()[lambda x: x["date"] == "2026-09-01"],
        pd.DataFrame(),
        {"signal_date": "2026-09-01"},
        _config(),
        _calendar(),
        initial_cash=100000,
    )

    day2 = day1.copy()
    day2["signal_date"] = "2026-09-02"
    raw_to_day2 = _raw()[lambda x: x["date"].isin(["2026-09-01", "2026-09-02"])]
    result = run_paper_day(
        tmp_path,
        day2,
        raw_to_day2,
        pd.DataFrame(),
        {"signal_date": "2026-09-02"},
        _config(),
        _calendar(),
        initial_cash=100000,
    )

    assert result["fills"] == 0
    orders = pd.read_csv(tmp_path / "order_history.csv")
    assert "missing_or_invalid_stock_limit_data" in set(orders["reason"].astype(str))


def test_slippage_fill_is_bounded_by_up_limit(tmp_path):
    cfg = _config()
    cfg["paper_slippage_bps"] = 100.0
    day1 = pd.DataFrame([{
        "ticker": "000001.SZ",
        "target_weight": 0.5,
        "signal_date": "2026-09-01",
        "source": "real",
    }])
    run_paper_day(
        tmp_path,
        day1,
        _raw()[lambda x: x["date"] == "2026-09-01"],
        pd.DataFrame(),
        {"signal_date": "2026-09-01"},
        cfg,
        _calendar(),
        initial_cash=100000,
    )

    day2_raw = _raw()[lambda x: x["date"].isin(["2026-09-01", "2026-09-02"])].copy()
    day2_raw.loc[day2_raw["date"] == "2026-09-02", "open"] = 10.99
    day2 = day1.copy()
    day2["signal_date"] = "2026-09-02"
    tight_limits = pd.DataFrame([{
        "date": "2026-09-02",
        "ticker": "000001.SZ",
        "up_limit": 11.0,
        "down_limit": 9.0,
    }])
    result = run_paper_day(
        tmp_path,
        day2,
        day2_raw,
        tight_limits,
        {"signal_date": "2026-09-02"},
        cfg,
        _calendar(),
        initial_cash=100000,
    )

    assert result["fills"] == 1
    fills = pd.read_csv(tmp_path / "fill_history.csv")
    assert float(fills.iloc[0]["reference_price"]) == pytest.approx(10.99)
    assert float(fills.iloc[0]["fill_price"]) == pytest.approx(11.0)
    assert str(fills.iloc[0]["price_limit_bounded"]).lower() in {"true", "1"}


def test_changed_signal_same_date_is_conflict(tmp_path):
    targets = pd.DataFrame([{
        "ticker": "000001.SZ",
        "target_weight": 0.5,
        "signal_date": "2026-09-01",
        "source": "real",
    }])
    run_paper_day(
        tmp_path,
        targets,
        _raw()[lambda x: x["date"] == "2026-09-01"],
        pd.DataFrame(),
        {"signal_date": "2026-09-01"},
        _config(),
        _calendar(),
        initial_cash=100000,
    )

    changed = targets.copy()
    changed["target_weight"] = 0.6
    with pytest.raises(ValueError, match="不同 signal bundle"):
        run_paper_day(
            tmp_path,
            changed,
            _raw()[lambda x: x["date"] == "2026-09-01"],
            pd.DataFrame(),
            {"signal_date": "2026-09-01"},
            _config(),
            _calendar(),
            initial_cash=100000,
        )


def test_crash_before_sqlite_commit_does_not_double_book(tmp_path):
    day1 = pd.DataFrame([{
        "ticker": "000001.SZ",
        "target_weight": 0.5,
        "signal_date": "2026-09-01",
        "source": "real",
    }])
    run_paper_day(
        tmp_path,
        day1,
        _raw()[lambda x: x["date"] == "2026-09-01"],
        pd.DataFrame(),
        {"signal_date": "2026-09-01"},
        _config(),
        _calendar(),
        initial_cash=100000,
    )

    day2 = day1.copy()
    day2["signal_date"] = "2026-09-02"
    raw_to_day2 = _raw()[lambda x: x["date"].isin(["2026-09-01", "2026-09-02"])]

    with pytest.raises(RuntimeError, match="simulated crash before_commit"):
        run_paper_day(
            tmp_path,
            day2,
            raw_to_day2,
            _limits(),
            {"signal_date": "2026-09-02"},
            _config(),
            _calendar(),
            initial_cash=100000,
            fault_point="before_commit",
        )

    result = run_paper_day(
        tmp_path,
        day2,
        raw_to_day2,
        _limits(),
        {"signal_date": "2026-09-02"},
        _config(),
        _calendar(),
        initial_cash=100000,
    )
    assert result["fills"] == 1
    fills = pd.read_csv(tmp_path / "fill_history.csv")
    assert len(fills) == 1
    assert fills["fill_id"].nunique() == 1


def test_transfer_fee_is_charged_and_audited_on_buy(tmp_path):
    cfg = _config()
    cfg["paper_transfer_fee_bps"] = 1.0

    day1 = pd.DataFrame([{
        "ticker": "000001.SZ",
        "target_weight": 0.5,
        "signal_date": "2026-09-01",
        "source": "real",
    }])
    run_paper_day(
        tmp_path,
        day1,
        _raw()[lambda x: x["date"] == "2026-09-01"],
        pd.DataFrame(),
        {"signal_date": "2026-09-01"},
        cfg,
        _calendar(),
        initial_cash=100000,
    )

    day2 = day1.copy()
    day2["signal_date"] = "2026-09-02"
    raw_to_day2 = _raw()[lambda x: x["date"].isin(["2026-09-01", "2026-09-02"])]
    result = run_paper_day(
        tmp_path,
        day2,
        raw_to_day2,
        _limits(),
        {"signal_date": "2026-09-02"},
        cfg,
        _calendar(),
        initial_cash=100000,
    )

    fills = pd.read_csv(tmp_path / "fill_history.csv")
    fill = fills.iloc[0]
    assert float(fill["notional"]) == pytest.approx(55000.0)
    assert float(fill["transfer_fee"]) == pytest.approx(5.5)
    assert float(fill["total_fee"]) == pytest.approx(5.5)
    assert result["cash"] == pytest.approx(44994.5)


def test_corporate_action_entitlement_uses_record_date_snapshot(tmp_path):
    cfg = _config()
    day1 = pd.DataFrame([{
        "ticker": "000001.SZ",
        "target_weight": 0.5,
        "signal_date": "2026-09-01",
        "source": "real",
    }])
    run_paper_day(
        tmp_path,
        day1,
        _raw()[lambda x: x["date"] == "2026-09-01"],
        pd.DataFrame(),
        {"signal_date": "2026-09-01"},
        cfg,
        _calendar(),
        initial_cash=100000,
    )

    day2 = day1.copy()
    day2["signal_date"] = "2026-09-02"
    run_paper_day(
        tmp_path,
        day2,
        _raw()[lambda x: x["date"].isin(["2026-09-01", "2026-09-02"])],
        _limits(),
        {"signal_date": "2026-09-02"},
        cfg,
        _calendar(),
        initial_cash=100000,
    )

    action = pd.DataFrame([{
        "ticker": "000001.SZ",
        "end_date": "2025-12-31",
        "ann_date": "2026-08-20",
        "div_proc": "实施",
        "stk_div": 0.0,
        "stk_bo_rate": 0.0,
        "stk_co_rate": 0.0,
        "cash_div": 0.1,
        "cash_div_tax": 0.12,
        "record_date": "2026-09-02",
        "ex_date": "2026-09-03",
        "pay_date": "2026-09-04",
        "div_listdate": None,
        "imp_ann_date": "2026-08-25",
        "base_date": "2026-09-02",
        "base_share": 100000.0,
    }])

    day3 = day1.copy()
    day3["signal_date"] = "2026-09-03"
    result = run_paper_day(
        tmp_path,
        day3,
        _raw(),
        _limits(),
        {"signal_date": "2026-09-03"},
        cfg,
        _calendar(),
        initial_cash=100000,
        corporate_actions=action,
    )

    assert result["status"] == "OK"
    account = json.loads((tmp_path / "paper_account.json").read_text(encoding="utf-8"))
    receivables = list(account["cash_receivables"].values())
    assert len(receivables) == 1
    assert float(receivables[0]["amount"]) == pytest.approx(500.0)

    history = pd.read_csv(tmp_path / "corporate_action_history.csv")
    entitlement = history[history["event_type"] == "ENTITLEMENT"].iloc[0]
    assert int(entitlement["eligible_qty"]) == 5000
    assert float(entitlement["cash_amount"]) == pytest.approx(500.0)
