"""Synthetic correctness checks; no test claims market performance."""
from dataclasses import replace
import json

import numpy as np
import pandas as pd
import pytest

from src.live_v8 import RiskConfig, TradingBlocked
from src.strategy_lab_v9_4 import (
    LabConfig, STRATEGIES, build_lab_targets, compare_strategies, demo_inputs,
    price_features, save_lab_results, select_research_portfolio, summary_metrics,
)


def cross_section(count=60):
    return pd.DataFrame({"ticker": [f"{600001+i:06d}.SH" for i in range(count)],
                         "industry_l1": [f"S{i%5}" for i in range(count)],
                         "composite_score": np.arange(count, 0, -1, dtype=float)})


def test_buffer_prevents_rank_churn_and_bounds_voluntary_exits():
    panel = cross_section()
    old = list(panel.ticker.iloc[:30])
    panel.loc[:29, "composite_score"] -= 35
    selected, audit = select_research_portfolio(
        panel, .8, RiskConfig(), LabConfig(rank_buffer=0, max_replacements=3),
        old, {t: 20 for t in old}, buffered=True)
    assert audit["voluntary_replacements"] == 3
    assert audit["removed_names"] == 3
    assert len(selected) == 30
    assert set(selected.ticker).intersection(old) == set(old[:27])


def test_rank_buffer_and_minimum_holding_age_retain_names():
    panel = cross_section()
    old = list(panel.ticker.iloc[:30])
    panel.loc[29, "composite_score"] = 26.5  # rank 34, inside K+10
    selected, audit = select_research_portfolio(
        panel, .8, RiskConfig(), LabConfig(), old, {t: 20 for t in old}, True)
    assert set(selected.ticker) == set(old) and audit["removed_names"] == 0
    panel.loc[29, "composite_score"] = -100
    selected, audit = select_research_portfolio(
        panel, .8, RiskConfig(), LabConfig(), old, {t: 1 for t in old}, True)
    assert set(selected.ticker) == set(old) and audit["voluntary_replacements"] == 0


def test_safety_exits_override_zero_replacement_budget():
    panel = cross_section()
    old = list(panel.ticker.iloc[:30])
    panel = panel[~panel.ticker.isin(old[-5:])]
    selected, audit = select_research_portfolio(
        panel, .8, RiskConfig(), LabConfig(max_replacements=0), old, {t: 20 for t in old}, True)
    assert len(selected) == 30 and not set(selected.ticker).intersection(old[-5:])
    assert audit["ineligible_exits"] == 5 and audit["voluntary_replacements"] == 0


def test_cap_override_is_audited_and_cash_not_inflated():
    panel = cross_section()
    panel.loc[:29, "industry_l1"] = "concentrated"
    old = list(panel.ticker.iloc[:30])
    risk = replace(RiskConfig(), max_single_weight=.01)
    selected, audit = select_research_portfolio(
        panel, .8, risk, LabConfig(max_replacements=0), old, {t: 20 for t in old}, True)
    assert selected.target_weight.sum() == pytest.approx(.3)
    assert selected.target_weight.max() <= .01
    assert audit["cash_target"] == pytest.approx(.7)
    selected, audit = select_research_portfolio(
        panel, .8, RiskConfig(), LabConfig(max_replacements=0), old, {t: 20 for t in old}, True)
    assert selected.groupby("industry_l1").target_weight.sum().max() <= .3+1e-12
    assert audit["constraint_exits"] > 0


def test_tied_scores_are_deterministic_and_duplicates_block():
    panel = cross_section()
    panel.composite_score = 1.
    a, _ = select_research_portfolio(panel, .8, RiskConfig(), LabConfig())
    b, _ = select_research_portfolio(panel.sample(frac=1, random_state=3), .8, RiskConfig(), LabConfig())
    pd.testing.assert_frame_equal(a, b)
    with pytest.raises(TradingBlocked, match="重复"):
        select_research_portfolio(pd.concat([panel, panel.iloc[[0]]]), .8, RiskConfig(), LabConfig())


def test_features_exclude_future_and_recent_month_from_skip_momentum():
    inputs = demo_inputs()
    p = inputs["prices"].copy()
    dates = pd.DatetimeIndex(sorted(p.date.unique()))
    signal = dates[210]
    a = price_features(p, [signal])
    p.loc[p.date > signal, "close"] *= 100
    pd.testing.assert_frame_equal(a, price_features(p, [signal]))
    ticker = p.ticker.iloc[0]
    history = p[p.ticker == ticker].set_index("date").close
    assert a[a.ticker == ticker].momentum_skip_raw.iloc[0] == pytest.approx(
        history.loc[dates[189]]/history.loc[dates[84]]-1)
    p.loc[p.date > dates[189], "close"] *= 2
    changed = price_features(p, [signal])
    pd.testing.assert_series_equal(a.momentum_skip_raw, changed.momentum_skip_raw)
    assert not a.lowvol_raw.equals(changed.lowvol_raw)


def test_missing_bar_does_not_get_forward_filled_into_volatility():
    p = demo_inputs()["prices"]
    dates = pd.DatetimeIndex(sorted(p.date.unique()))
    ticker = p.ticker.iloc[0]
    p = p[~((p.ticker == ticker) & (p.date == dates[190]))]
    features = price_features(p, [dates[210]])
    assert pd.isna(features[features.ticker == ticker].lowvol_raw.iloc[0])


def test_first_day_loss_is_included_in_drawdown_and_alignment_is_strict():
    dates = pd.bdate_range("2024-01-01", periods=2)
    r = pd.Series([-.10, 0.], index=dates)
    b = pd.Series([0., 0.], index=dates)
    assert summary_metrics(r, b)["max_drawdown"] == pytest.approx(-.10)
    with pytest.raises(TradingBlocked, match="对齐"):
        summary_metrics(r, b.iloc[:1])


@pytest.fixture(scope="module")
def comparison():
    return compare_strategies(demo_inputs(), LabConfig(frequency="monthly"))


def test_comparison_has_common_dates_cost_scenarios_and_no_real_targets(comparison):
    assert len(comparison["comparison"]) == len(STRATEGIES)*2*2
    dates = comparison["daily_returns"].groupby(["strategy", "scenario"]).date.apply(list)
    assert all(value == dates.iloc[0] for value in dates)
    assert comparison["targets"].source.eq("research_only").all()
    weights = comparison["targets"].groupby(["strategy", "signal_date"]).target_weight.sum()
    assert (weights <= .8+1e-12).all()
    costs = comparison["trade_costs"]
    assert (pd.to_datetime(costs.execution_date) > pd.to_datetime(costs.signal_date)).all()
    assert (costs.total_cost >= 0).all()
    assert (costs.blocked_buy_count >= 0).all()
    assert comparison["factor_audit"].ann_date.lt(comparison["factor_audit"].signal_date).all()


def test_future_financial_revision_cannot_change_prior_targets():
    inputs = demo_inputs()
    dates = pd.DatetimeIndex(sorted(inputs["prices"].date.unique()))
    start, end = dates[200], dates[225]
    a, _, _, _ = build_lab_targets(inputs, LabConfig(frequency="monthly"), RiskConfig(), start, end)
    future = inputs["fundamentals_raw"].tail(60).copy()
    future.ann_date = end+pd.Timedelta(days=1)
    future.report_date = start-pd.Timedelta(days=60)
    future.roe_dt = 1e8
    inputs["fundamentals_raw"] = pd.concat([inputs["fundamentals_raw"], future])
    b, _, _, _ = build_lab_targets(inputs, LabConfig(frequency="monthly"), RiskConfig(), start, end)
    pd.testing.assert_frame_equal(a, b)


def test_missing_valuation_day_blocks_instead_of_skipping():
    inputs = demo_inputs()
    dates = pd.DatetimeIndex(sorted(inputs["prices"].date.unique()))
    inputs["daily_basic"] = inputs["daily_basic"][inputs["daily_basic"].date != dates[180]]
    with pytest.raises(TradingBlocked, match="禁止跳过"):
        build_lab_targets(inputs, LabConfig(), RiskConfig())


def test_output_manifest_provenance_and_existing_run_protection(comparison, tmp_path):
    provenance = {"source": "synthetic", "notice": "SYNTHETIC TEST ONLY"}
    manifest = save_lab_results(comparison, tmp_path, LabConfig(), RiskConfig(), provenance)
    assert manifest["kind"] == "strategy_lab_research_only"
    assert manifest["performance_validated"] is False and manifest["real_broker_submission"] is False
    assert json.loads((tmp_path/"research_manifest.json").read_text())["provenance"] == provenance
    assert not (tmp_path/"signal_manifest_v9.json").exists()
    with pytest.raises(TradingBlocked, match="非空"):
        save_lab_results(comparison, tmp_path, LabConfig(), RiskConfig(), provenance)


@pytest.mark.parametrize("kwargs", [{"max_replacements": -1}, {"rank_buffer": 1.5},
                                   {"min_hold_sessions": True}, {"holdout_fraction": 1},
                                   {"initial_cash": np.inf}])
def test_invalid_parameters_rejected(kwargs):
    with pytest.raises(TradingBlocked):
        LabConfig(**kwargs).validated()
