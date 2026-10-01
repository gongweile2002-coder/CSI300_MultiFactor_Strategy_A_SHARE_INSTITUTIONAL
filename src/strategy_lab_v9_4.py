"""Fixed, PIT research comparisons. Never publishes a broker signal bundle.

Name buffering refers to prior *target* membership, not actual broker holdings.
Execution uses the existing adjusted-unit accounting research model, not the
fixed-share, corporate-action-aware v9.3 forward Paper ledger.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import sys

import numpy as np
import pandas as pd

from .accounting_backtest_v8 import run_accounting_backtest
from .data_contract_v8 import load_verified_dataset
from .live_v8 import RiskConfig, board, require
from .ops_v9 import audit_dataset, atomic_json, sha256
from .research_v2 import membership_for_date, winsorize
from .research_v4 import build_point_in_time_panel_v4, neutralize_factor


STRATEGIES = ("core", "skip_momentum", "quality_lowvol", "buffered_core")


@dataclass(frozen=True)
class LabConfig:
    top_k: int = 30
    rank_buffer: int = 10
    max_replacements: int = 3
    min_hold_sessions: int = 5
    frequency: str = "weekly"
    initial_cash: float = 500000.0
    holdout_fraction: float = 0.30

    def validated(self):
        for field in ("top_k", "rank_buffer", "max_replacements", "min_hold_sessions"):
            value = getattr(self, field)
            require(isinstance(value, int) and not isinstance(value, bool) and value >= 0,
                    f"{field} 必须为非负整数")
        require(15 <= self.top_k <= 30, "top_k 须在 15–30 之间")
        require(self.max_replacements <= self.top_k, "替换数超过持仓数")
        require(self.frequency in {"daily", "weekly", "monthly"}, "调仓频率非法")
        require(np.isfinite(self.initial_cash) and self.initial_cash > 0, "初始资金非法")
        require(np.isfinite(self.holdout_fraction) and 0 < self.holdout_fraction < 1,
                "holdout_fraction 须在 0–1 之间")
        return self


def price_features(prices, signal_dates):
    """Trading-session offsets, no forward fill across missing/suspended bars."""
    p = prices.copy()
    p["date"] = pd.to_datetime(p["date"], errors="raise")
    require(not p.duplicated(["date", "ticker"]).any(), "研究行情重复主键")
    close = p.pivot(index="date", columns="ticker", values="close").sort_index()
    require(np.isfinite(close.stack()).all() and (close.stack() > 0).all(), "研究价格非法")
    # P(D-21)/P(D-126)-1: six-month momentum with most recent month excluded.
    skip = close.shift(21) / close.shift(126) - 1
    vol = close.pct_change(fill_method=None).rolling(60, min_periods=60).std(ddof=0)
    pieces = []
    for day in pd.to_datetime(signal_dates):
        require(day in close.index, "信号日缺少行情")
        loc = close.index.get_loc(day)
        require(loc >= 180, "信号日历史不足 180 个交易日")
        x = pd.DataFrame({"ticker": close.columns, "momentum_skip_raw": skip.loc[day].values,
                          "lowvol_raw": -vol.loc[day].values, "signal_date": day})
        x["momentum_skip_start"] = close.index[loc - 126]
        x["momentum_skip_end"] = close.index[loc - 21]
        x["volatility_start"] = close.index[loc - 60]
        x["volatility_end"] = day
        pieces.append(x)
    require(bool(pieces), "没有可用信号日")
    return pd.concat(pieces, ignore_index=True)


def research_scores(panel):
    x = panel.copy()
    for feature, score in [("momentum_skip_raw", "momentum_skip_score"),
                           ("lowvol_raw", "lowvol_score")]:
        require(np.isfinite(x[feature]).all(), f"{feature} 缺失或非法")
        x[feature] = winsorize(x[feature])
        x[score] = neutralize_factor(x, feature)
    x["score_core"] = .30*x.value_score + .40*x.quality_score + .30*x.momentum_score
    x["score_skip_momentum"] = .30*x.value_score + .40*x.quality_score + .30*x.momentum_skip_score
    x["score_quality_lowvol"] = .20*x.value_score + .50*x.quality_score + .30*x.lowvol_score
    x["score_buffered_core"] = x["score_core"]
    require(np.isfinite(x[["score_"+s for s in STRATEGIES]]).all().all(), "策略评分非法")
    return x


def select_research_portfolio(panel, gross, risk, cfg, previous=None, ages=None, buffered=False):
    """Equal slot budgets and caps. Safety exits override voluntary name budget."""
    cfg.validated()
    risk.validated()
    require(len(panel) >= 15 and not panel.ticker.duplicated().any(), "合格证券不足或重复")
    require(np.isfinite(panel.composite_score).all(), "选股评分非法")
    require(panel.industry_l1.notna().all() and
            not panel.industry_l1.isin(["", "UNKNOWN"]).any(), "行业缺失")
    require(np.isfinite(gross) and 0 < gross <= risk.max_gross_weight, "目标仓位非法")
    ranked = panel.sort_values(["composite_score", "ticker"], ascending=[False, True]).reset_index(drop=True)
    ranks = {t: i+1 for i, t in enumerate(ranked.ticker)}
    old = set(previous or [])
    ages = ages or {}
    retained = sorted(old & set(ranks), key=lambda t: ranks[t])[:cfg.top_k]
    voluntary = []
    if buffered and retained:
        challengers = [t for t in ranked.ticker if t not in retained]
        # Retain ranks within K+buffer. Minimum holding age is in market sessions.
        droppable = [t for t in reversed(retained)
                     if ranks[t] > cfg.top_k + cfg.rank_buffer and
                     ages.get(t, 0) >= cfg.min_hold_sessions]
        for loser, winner in zip(droppable[:cfg.max_replacements], challengers):
            if ranks[winner] >= ranks[loser]:
                break
            retained.remove(loser)
            retained.append(winner)
            voluntary.append(loser)
        priority = retained + [t for t in ranked.ticker if t not in retained and t not in voluntary]
    else:
        priority = list(ranked.ticker)
    lookup = ranked.set_index("ticker")
    budget = min(gross/min(cfg.top_k, len(ranked)), risk.max_single_weight)
    sectors, rows, weights = {}, [], []
    for ticker in priority:
        row = lookup.loc[ticker]
        sector = row.industry_l1
        weight = min(budget, max(0., risk.max_sector_weight-sectors.get(sector, 0.)),
                     max(0., gross-sum(weights)))
        if weight <= 1e-12:
            continue
        rows.append(ticker)
        weights.append(weight)
        sectors[sector] = sectors.get(sector, 0.) + weight
        if len(rows) == cfg.top_k:
            break
    require(len(rows) >= 15, "风险约束后不足 15 只，研究停止")
    selected = lookup.loc[rows].reset_index()
    selected["target_weight"] = weights
    now = set(rows)
    audit = {"selected_names": len(now), "retained_names": len(now & old),
             "new_names": len(now-old), "removed_names": len(old-now),
             "ineligible_exits": len(old-set(ranks)),
             "voluntary_replacements": len(set(voluntary)-now),
             "constraint_exits": len((old & set(ranks))-now-set(voluntary)),
             "gross_target": float(sum(weights)), "cash_target": float(1-sum(weights))}
    return selected, audit


def signal_schedule(prices, cfg, start=None, end=None):
    cfg.validated()
    dates = pd.DatetimeIndex(sorted(pd.to_datetime(prices.date).unique()))
    require(len(dates) > 181, "至少需要 182 个行情交易日")
    start = pd.Timestamp(start) if start is not None else dates[180]
    end = pd.Timestamp(end) if end is not None else dates[-1]
    require(start >= dates[180] and end <= dates[-1] and start < end, "研究日期或历史长度非法")
    active = dates[(dates >= start) & (dates <= end)]
    require(len(active) >= 20, "比较区间至少需要 20 个交易日")
    # Leave final session for execution/valuation; signals never need future prices.
    candidates = active[:-1]
    if cfg.frequency == "weekly":
        candidates = candidates[::5]
    elif cfg.frequency == "monthly":
        candidates = pd.DatetimeIndex(pd.Series(candidates, index=candidates).groupby(candidates.to_period("M")).max())
    return candidates, active[-1]


def build_lab_targets(inputs, cfg, risk, start=None, end=None):
    dates, end = signal_schedule(inputs["prices"], cfg, start, end)
    prices = inputs["prices"].copy()
    prices["date"] = pd.to_datetime(prices.date)
    # Historical snapshot intervals must cover every requested rebalance date.
    members = inputs["index_membership"].copy()
    members["effective_date"] = pd.to_datetime(members.effective_date)
    for day in dates:
        snapshot = membership_for_date(members, day)
        require(not snapshot.empty and (day-snapshot.effective_date.max()).days <= 45,
                f"{day.date()}: 缺少及时的成分快照")
    base = build_point_in_time_panel_v4(
        prices, inputs["daily_basic"], inputs["fundamentals_raw"], members,
        metadata=inputs["stock_metadata"], industry_membership=inputs["industry_membership"],
        st_status=inputs["st_status"], signal_dates_override=dates, momentum_lookback_days=126,
        factor_weights={"value": .30, "quality": .40, "momentum": .30},
        min_listing_trading_days=180, liquidity_min_quantile=.10, exclude_st=True)
    require(not base.empty, "没有足够的 PIT 财报/估值/行情")
    base = base.merge(price_features(prices, dates), on=["signal_date", "ticker"], validate="one_to_one")
    base = base[(base.valuation_date == base.signal_date) &
                ((base.signal_date-base.ann_date).dt.days <= 200) &
                base.ticker.map(lambda t: board(t) in risk.allowed_buy_boards)].copy()
    raw = inputs["raw_prices"].copy()
    raw["date"] = pd.to_datetime(raw.date)
    require(not raw.duplicated(["date", "ticker"]).any(), "原始行情重复主键")
    current = raw[["date", "ticker", "close", "volume"]].rename(
        columns={"date": "signal_date", "close": "signal_close_raw", "volume": "signal_volume"})
    base = base.merge(current, on=["signal_date", "ticker"], validate="one_to_one")
    base = base[(base.signal_volume > 0) & base.industry_l1.notna() &
                ~base.industry_l1.isin(["", "UNKNOWN"])].copy()
    # Common complete feature universe prevents comparing differently filtered pools.
    base = base.replace([np.inf, -np.inf], np.nan).dropna(
        subset=["value_score", "quality_score", "momentum_score", "momentum_skip_raw", "lowvol_raw"])
    require(set(dates) == set(pd.DatetimeIndex(base.signal_date.unique())),
            "有信号日缺少共同合格股票池，禁止跳过")
    benchmark = inputs["benchmark"].copy()
    benchmark["date"] = pd.to_datetime(benchmark.date)
    require(not benchmark.date.duplicated().any(), "基准重复日期")
    closes = benchmark.set_index("date").close.sort_index()
    sessions = pd.DatetimeIndex(sorted(prices.date.unique()))
    session_ids = {day: i for i, day in enumerate(sessions)}
    targets, audits, factors = [], [], []
    previous = {s: set() for s in STRATEGIES}
    entered = {s: {} for s in STRATEGIES}
    for day, cross in base.groupby("signal_date", sort=True):
        cross = research_scores(cross)
        history = closes.loc[:day].tail(120)
        require(len(history) == 120 and history.index[-1] == day and
                np.isfinite(history).all() and (history > 0).all(), "基准历史不足/过期/非法")
        trend = history.iloc[-1] >= history.mean()
        gross = min(risk.max_gross_weight, .80 if trend else .40)
        for strategy in STRATEGIES:
            scored = cross.copy()
            scored["composite_score"] = scored["score_"+strategy]
            ages = {t: session_ids[day]-i for t, i in entered[strategy].items()}
            selected, audit = select_research_portfolio(
                scored, gross, risk, cfg, sorted(previous[strategy]), ages,
                buffered=strategy == "buffered_core")
            selected["strategy"] = strategy
            selected["source"] = "research_only"
            targets.append(selected[["signal_date", "ticker", "target_weight", "strategy", "source",
                                     "industry_l1", "composite_score", "signal_close_raw"]])
            names = set(selected.ticker)
            entered[strategy] = {t: entered[strategy].get(t, session_ids[day]) for t in names}
            previous[strategy] = names
            audits.append({"signal_date": day, "strategy": strategy, "eligible_names": len(cross),
                           "risk_regime": "ABOVE_MA120" if trend else "BELOW_MA120", **audit})
        factors.append(cross)
    return pd.concat(targets, ignore_index=True), pd.DataFrame(audits), pd.concat(factors, ignore_index=True), end


def summary_metrics(returns, benchmark):
    require(not returns.empty and returns.index.equals(benchmark.index), "策略/基准日期必须完全对齐")
    require(np.isfinite(returns).all() and np.isfinite(benchmark).all() and
            (returns > -1).all() and (benchmark > -1).all(), "收益序列非法")
    nav = (1+returns).cumprod()
    vol = returns.std(ddof=0)
    active = returns-benchmark
    te = active.std(ddof=0)*np.sqrt(252)
    return {"sessions": len(returns), "total_return": float(nav.iloc[-1]-1),
            "annualized_return": float(nav.iloc[-1]**(252/len(returns))-1),
            "annualized_volatility": float(vol*np.sqrt(252)),
            "sharpe": float(returns.mean()/vol*np.sqrt(252)) if vol > 0 else None,
            # Include initial NAV=1 so a loss on the first day is not hidden.
            "max_drawdown": float((nav/nav.cummax().clip(lower=1)-1).min()),
            "benchmark_return": float((1+benchmark).prod()-1),
            "relative_total_return": float(nav.iloc[-1]/(1+benchmark).prod()-1),
            "tracking_error": float(te),
            "information_ratio": float(active.mean()*252/te) if te > 0 else None}


def compare_strategies(inputs, cfg=None, risk=None, start=None, end=None):
    cfg = (cfg or LabConfig()).validated()
    risk = (risk or RiskConfig()).validated()
    targets, audit, factors, end = build_lab_targets(inputs, cfg, risk, start, end)
    first = pd.Timestamp(targets.signal_date.min())
    p = inputs["prices"].copy()
    p["date"] = pd.to_datetime(p.date)
    # Keep pre-sample rows for trailing ADV; no positions exist before first signal.
    p = p[p.date <= end]
    benchmark = inputs["benchmark"].copy()
    benchmark["date"] = pd.to_datetime(benchmark.date)
    b = benchmark.set_index("date").close.sort_index().pct_change(fill_method=None)
    rows, returns_out, holdings_out, trades_out = [], [], [], []
    for strategy in STRATEGIES:
        chosen = targets[targets.strategy == strategy]
        for scenario, multiplier in [("base", 1.), ("fees_x2", 2.)]:
            r, holdings, trades = run_accounting_backtest(
                p, chosen, inputs["stock_limits"], initial_cash=cfg.initial_cash,
                commission_bps=risk.commission_bps*multiplier,
                minimum_commission=risk.minimum_commission*multiplier,
                slippage_bps=2.*multiplier, transfer_fee_bps=risk.transfer_fee_bps*multiplier,
                max_adv_participation=risk.max_adv_participation, use_price_limits=True)
            r = r.loc[first:end]
            aligned_b = b.reindex(r.index)
            cut = int(len(r)*(1-cfg.holdout_fraction))
            require(0 < cut < len(r), "无法建立后段留出区间")
            for period, section in [("full", r), ("holdout", r.iloc[cut:])]:
                costs = trades[pd.to_datetime(trades.execution_date).isin(section.index)]
                rows.append({"strategy": strategy, "scenario": scenario, "period": period,
                             "start": str(section.index[0].date()), "end": str(section.index[-1].date()),
                             **summary_metrics(section, aligned_b.loc[section.index]),
                             "gross_turnover_sum": float(costs.gross_turnover.sum()),
                             "mean_rebalance_turnover": float(costs.gross_turnover.mean()) if len(costs) else 0.,
                             "cost_nav_fraction_sum": float(costs.total_cost.sum()),
                             "blocked_buys": int(costs.blocked_buy_count.sum()),
                             "blocked_sells": int(costs.blocked_sell_count.sum()),
                             "stale_valuation_days_full": r.attrs.get("stale_valuation_days", 0)})
            returns_out.append(pd.DataFrame({"date": r.index, "strategy": strategy, "scenario": scenario,
                                            "return": r.values, "benchmark_return": aligned_b.values}))
            for frame, bucket in [(holdings, holdings_out), (trades, trades_out)]:
                frame = frame.copy()
                frame["strategy"], frame["scenario"] = strategy, scenario
                bucket.append(frame)
    return {"comparison": pd.DataFrame(rows), "targets": targets, "selection_audit": audit,
            "factor_audit": factors, "daily_returns": pd.concat(returns_out, ignore_index=True),
            "holdings": pd.concat(holdings_out, ignore_index=True),
            "trade_costs": pd.concat(trades_out, ignore_index=True)}


INPUT_FILES = ("prices", "raw_prices", "daily_basic", "fundamentals_raw", "index_membership",
               "stock_metadata", "industry_membership", "st_status", "benchmark", "stock_limits")


def load_lab_inputs(directory):
    directory = Path(directory)
    manifest, calendar = load_verified_dataset(directory)
    require(manifest['source']=='tushare',
            '原版策略对比需要完整历史股票池及原财务口径；免费版用于前向Paper，不能伪装成原版历史回测')
    manifest_hash = sha256(directory/"data_manifest_v8.json")
    require(not manifest.get("fixture_notice"), "合成测试清单不能作为真实研究数据")
    now = pd.Timestamp(manifest["as_of"]+"T18:00:00+08:00")
    audit = audit_dataset(directory, now)
    require(audit["status"] == "PASS", "数据审计未通过: "+str(audit["checks"]))
    frames = {name: pd.read_csv(directory/(name+".csv"), dtype={"ticker": str}) for name in INPUT_FILES}
    dates = pd.to_datetime(frames["prices"].date)
    actual = set(pd.DatetimeIndex(dates.unique()))
    expected = {d for d, opened in calendar.days.items() if opened and dates.min() <= d <= dates.max()}
    require(actual == expected, "历史行情漏掉整场交易日，不能将后日开盘替代预期执行日")
    benchmark_dates = set(pd.to_datetime(frames["benchmark"].date))
    require(actual <= benchmark_dates, "基准漏交易日")
    limits = frames["stock_limits"]
    values = limits[["up_limit", "down_limit"]].apply(pd.to_numeric, errors="raise")
    require(np.isfinite(values).all().all() and (values > 0).all().all() and
            (values.down_limit <= values.up_limit).all(), "涨跌停边界非法")
    load_verified_dataset(directory)
    require(sha256(directory/"data_manifest_v8.json") == manifest_hash, "读取期间数据版本变化")
    provenance = {"source": "tushare", "data_manifest_hash": manifest_hash,
                  "as_of": manifest["as_of"], "membership_basis": manifest.get("membership_basis"),
                  "financial_vintages": manifest.get("financial_vintages"),
                  "st_history_status": manifest.get("st_history_status", "unknown"),
                  "input_files": manifest["files"], "data_audit": audit}
    return frames, provenance


def save_lab_results(results, directory, cfg, risk, provenance):
    """Publish a completion manifest last; reject replacement of an existing run."""
    out = Path(directory)
    require(not out.exists() or not any(out.iterdir()), "输出目录非空，请使用新的研究运行目录")
    out.mkdir(parents=True, exist_ok=True)
    for name, frame in results.items():
        frame.to_csv(out/(name+".csv"), index=False)
    manifest = {"schema_version": 1, "kind": "strategy_lab_research_only", "performance_validated": False,
                "real_broker_submission": False, "strategies": list(STRATEGIES),
                "config": asdict(cfg), "risk_config": asdict(risk), "provenance": provenance,
                "execution_settings": {"base_slippage_bps": 2., "use_price_limits": True,
                                       "engine": "accounting_backtest_v8"},
                "runtime": {"python": sys.version.split()[0], "numpy": np.__version__, "pandas": pd.__version__},
                "code_hashes": {name: sha256(Path(__file__).parent/name) for name in
                                ("strategy_lab_v9_4.py", "accounting_backtest_v8.py",
                                 "research_v4.py", "data_contract_v8.py")},
                "files": {name+".csv": sha256(out/(name+".csv")) for name in results},
                "limitations": [
                    "Adjusted economic units with gross distribution reinvestment; not v9.3 fixed-share Paper.",
                    "Research target sizing uses execution open; not a pre-open fixed-share intent.",
                    "No dividend tax/payment dates, rights elections, order book or delisting recovery.",
                    "Buffer holds prior target names, not actual executed positions; constraints override name budget.",
                    "Holdout is a fixed chronological tail with carried positions; no fitted model or winner promotion.",
                    "fees_x2 doubles commission/minimum commission/slippage/transfer; statutory stamp duty unchanged.",
                    "Summed cost/NAV fractions are diagnostics, not the portfolio's gross-minus-net return.",
                    "Vendor financial revisions and approximate membership/ST history can limit historical PIT fidelity."]}
    atomic_json(out/"research_manifest.json", manifest)
    return manifest


def demo_inputs(seed=94):
    """Explicit synthetic sample; never writes a Tushare or real-signal manifest."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2024-01-02", periods=320)
    names = [f"{600001+i:06d}.SH" for i in range(60)]
    common = rng.normal(.0002, .006, len(dates))
    rows = []
    for j, ticker in enumerate(names):
        returns = common + rng.normal(.0001, .005+.0001*j, len(dates))
        close = (12+j/4)*np.exp(np.cumsum(returns))
        prev = np.r_[close[0], close[:-1]]
        op = prev*np.exp(rng.normal(0, .002, len(dates)))
        for day, o, c in zip(dates, op, close):
            rows.append((day, ticker, o, max(o,c)*1.002, min(o,c)*.998, c, 1., 2e6, c*2e6))
    raw = pd.DataFrame(rows, columns=["date", "ticker", "open", "high", "low", "close", "adj_factor", "volume", "amount"])
    raw["price_basis"], raw["amount_unit"], raw["volume_unit"] = "raw", "CNY", "share"
    prices = raw.copy()
    prices["price_basis"] = "adjusted_research_only"
    for field in ("open", "high", "low", "close"):
        prices["raw_"+field] = prices[field]
    db = raw[["date", "ticker"]].copy()
    index = db.ticker.map({t: j for j, t in enumerate(names)})
    db["pe_ttm"], db["pb"] = 8+index%19, 1+(index%11)/10
    db["turnover_rate"], db["total_mv"], db["circ_mv"] = 1., 100000+index*1300, 80000+index*1000
    f = pd.DataFrame([(t, d-pd.Timedelta(days=10), d-pd.Timedelta(days=45),
                       5+j%13, 5+j%13, 2+j%23, 20+j%37)
                      for d in dates[::63] for j, t in enumerate(names)],
                     columns=["ticker", "ann_date", "report_date", "roe", "roe_dt", "profit_growth", "debt_ratio"])
    members = pd.DataFrame([(d, t, 100/len(names)) for d in dates[::20] for t in names],
                           columns=["effective_date", "ticker", "weight"])
    meta = pd.DataFrame({"ticker": names, "name": ["Synthetic"+str(j) for j in range(60)], "list_date": "2000-01-01"})
    industry = pd.DataFrame({"ticker": names, "l1_name": ["Sector"+str(j%5) for j in range(60)],
                             "in_date": "2000-01-01", "out_date": pd.NaT})
    limits = raw[["date", "ticker"]].copy()
    prev = raw.groupby("ticker").close.shift().fillna(raw.close)
    limits["up_limit"], limits["down_limit"] = prev*1.1, prev*.9
    benchmark = pd.DataFrame({"date": dates, "close": 1000*np.exp(np.cumsum(common))})
    return dict(zip(INPUT_FILES, [prices, raw, db, f, members, meta, industry,
                                 pd.DataFrame(columns=["ticker", "trade_date"]), benchmark, limits]))
