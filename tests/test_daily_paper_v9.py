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
        tmp_path, day2, raw_to_day2, pd.DataFrame(),
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
        pd.DataFrame(),
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
