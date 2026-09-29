import pandas as pd

from update_live_data import merge_frame, raw_to_adjusted


def test_merge_frame_replaces_duplicate_key_with_new_value():
    old = pd.DataFrame([{"date": "2026-09-01", "ticker": "000001.SZ", "close": 10.0}])
    new = pd.DataFrame([{"date": "2026-09-01", "ticker": "000001.SZ", "close": 10.5}])
    out = merge_frame(old, new, ["date", "ticker"], ["date", "ticker"])
    assert len(out) == 1
    assert float(out.iloc[0]["close"]) == 10.5


def test_raw_to_adjusted_preserves_raw_prices_and_marks_research_basis():
    raw = pd.DataFrame([{
        "ticker": "000001.SZ",
        "date": "2026-09-01",
        "open": 10.0,
        "high": 11.0,
        "low": 9.0,
        "close": 10.5,
        "adj_factor": 2.0,
    }])
    out = raw_to_adjusted(raw)
    assert float(out.iloc[0]["raw_open"]) == 10.0
    assert float(out.iloc[0]["open"]) == 20.0
    assert out.iloc[0]["price_basis"] == "adjusted_research_only"
