import json
from pathlib import Path

import pandas as pd

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
        initial_cash=100000,
    )
    assert result["status"] == "OK"
    assert result["fills"] == 0
    state = json.loads((tmp_path / "state.json").read_text())
    assert state["pending_signal_date"] == "2026-09-01"
    assert (tmp_path / "pending_targets.csv").exists()


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
        _config(), initial_cash=100000,
    )

    day2 = day1.copy()
    day2["signal_date"] = "2026-09-02"
    result = run_paper_day(
        tmp_path, day2, _raw(), pd.DataFrame(),
        {"signal_date": "2026-09-02"},
        _config(), initial_cash=100000,
    )
    assert result["fills"] == 1
    fills = pd.read_csv(tmp_path / "fill_history.csv")
    assert float(fills.iloc[0]["fill_price"]) == 11.0
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
        _config(), initial_cash=100000,
    )
    second = run_paper_day(
        tmp_path, targets,
        _raw()[lambda x: x["date"] == "2026-09-01"],
        pd.DataFrame(),
        {"signal_date": "2026-09-01"},
        _config(), initial_cash=100000,
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
            _config(), initial_cash=100000,
        )
    except ValueError as exc:
        assert "source=real" in str(exc)
    else:
        raise AssertionError("synthetic targets must be rejected")
