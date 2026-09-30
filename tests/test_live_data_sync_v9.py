import pandas as pd

from update_live_data import (
    _corporate_actions_by_ex_date,
    _financial_rows_by_announcement,
    merge_frame,
    raw_to_adjusted,
    required_market_data_universe,
)


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


class _FakePro:
    def __init__(self):
        self.calls = []

    def fina_indicator(self, **kwargs):
        self.calls.append(kwargs)
        if kwargs.get("ann_date") == "20260927":
            return pd.DataFrame([{
                "ts_code": kwargs["ts_code"],
                "ann_date": "20260927",
                "end_date": "20260630",
                "roe": 8.5,
                "roe_dt": 8.1,
                "netprofit_yoy": 12.0,
                "debt_to_assets": 45.0,
            }])
        return pd.DataFrame()


class _FakeDownloader:
    def __init__(self):
        self.pro = _FakePro()
        self.pause_count = 0

    def _pause(self):
        self.pause_count += 1


def test_incremental_financial_fetch_uses_announcement_date_not_report_range():
    dl = _FakeDownloader()
    out = _financial_rows_by_announcement(
        dl,
        ["000001.SZ"],
        pd.Timestamp("2026-09-26"),
        pd.Timestamp("2026-09-28"),
    )

    assert [c["ann_date"] for c in dl.pro.calls] == [
        "20260926",
        "20260927",
        "20260928",
    ]
    assert all("start_date" not in c and "end_date" not in c for c in dl.pro.calls)
    assert len(out) == 1
    assert out.iloc[0]["ticker"] == "000001.SZ"
    assert str(out.iloc[0]["ann_date"].date()) == "2026-09-27"


def test_incremental_financial_fetch_includes_weekend_announcements():
    dl = _FakeDownloader()
    out = _financial_rows_by_announcement(
        dl,
        ["000001.SZ"],
        "2026-09-25",
        "2026-09-28",
    )

    assert "20260927" in {c["ann_date"] for c in dl.pro.calls}
    assert not out.empty


def test_market_data_universe_keeps_removed_holdings_and_pending_names():
    current = {"000001.SZ", "000002.SZ"}
    prior = {"000001.SZ", "600000.SH"}
    paper_required = {"600000.SH", "300001.SZ"}

    out = required_market_data_universe(current, prior, paper_required)

    assert out == {
        "000001.SZ",
        "000002.SZ",
        "600000.SH",
        "300001.SZ",
    }


class _FakeDividendPro:
    def __init__(self):
        self.calls = []

    def dividend(self, **kwargs):
        self.calls.append(kwargs)
        if kwargs.get("ex_date") != "20260902":
            return pd.DataFrame()
        return pd.DataFrame([
            {
                "ts_code": "000001.SZ",
                "end_date": "20251231",
                "ann_date": "20260820",
                "div_proc": "实施",
                "stk_div": 0.0,
                "stk_bo_rate": 0.0,
                "stk_co_rate": 0.0,
                "cash_div": 0.1,
                "cash_div_tax": 0.12,
                "record_date": "20260901",
                "ex_date": "20260902",
                "pay_date": "20260904",
                "div_listdate": None,
                "imp_ann_date": "20260825",
                "base_date": "20260901",
                "base_share": 100000.0,
            },
            {
                "ts_code": "600000.SH",
                "end_date": "20251231",
                "ann_date": "20260820",
                "div_proc": "实施",
                "stk_div": 0.0,
                "stk_bo_rate": 0.0,
                "stk_co_rate": 0.0,
                "cash_div": 0.2,
                "cash_div_tax": 0.24,
                "record_date": "20260901",
                "ex_date": "20260902",
                "pay_date": "20260904",
                "div_listdate": None,
                "imp_ann_date": "20260825",
                "base_date": "20260901",
                "base_share": 100000.0,
            },
        ])


class _FakeDividendDownloader:
    def __init__(self):
        self.pro = _FakeDividendPro()
        self.pause_count = 0

    def _pause(self):
        self.pause_count += 1


def test_incremental_corporate_actions_fetch_by_ex_date_and_filter_universe():
    dl = _FakeDividendDownloader()
    out = _corporate_actions_by_ex_date(
        dl,
        [pd.Timestamp("2026-09-02")],
        {"000001.SZ"},
    )

    assert len(dl.pro.calls) == 1
    assert dl.pro.calls[0]["ex_date"] == "20260902"
    assert set(out["ticker"]) == {"000001.SZ"}
    assert out.iloc[0]["div_proc"] == "实施"
    assert float(out.iloc[0]["cash_div"]) == 0.1
    assert pd.Timestamp(out.iloc[0]["record_date"]) == pd.Timestamp("2026-09-01")
