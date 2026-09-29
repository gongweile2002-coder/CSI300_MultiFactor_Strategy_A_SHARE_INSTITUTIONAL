
import json
from pathlib import Path
import pandas as pd

from src.data_loader_v4 import load_real_data_v4
from src.research_v4 import (
    build_point_in_time_panel_v4,
    walk_forward_factor_validation,
)
from src.research_v2 import performance_summary, factor_diagnostics
from src.portfolio_v5 import (
    build_optimized_targets,
    run_optimized_backtest_v5,
    portfolio_diagnostics,
)
from src.reporting_v6 import generate_html_report

BASE=Path(__file__).resolve().parent

def main():
    cfg=json.loads((BASE/"config.example.json").read_text(encoding="utf-8"))
    OUT=BASE/"outputs"/"real_v6"
    OUT.mkdir(parents=True,exist_ok=True)

    (
        prices,daily_basic,fundamentals,membership,benchmark,
        metadata,industry,st_status,stock_limits
    )=load_real_data_v4(BASE/"data"/"real")

    panel=build_point_in_time_panel_v4(
        prices=prices,
        daily_basic=daily_basic,
        fundamentals=fundamentals,
        membership=membership,
        metadata=metadata,
        industry_membership=industry,
        st_status=st_status,
        start_date=cfg["start_date"],
        end_date=cfg["end_date"],
        momentum_lookback_days=cfg["momentum_lookback_days"],
        winsor_lower=cfg["winsor_lower"],
        winsor_upper=cfg["winsor_upper"],
        factor_weights=cfg["factor_weights"],
        min_turnover_rate=cfg["min_turnover_rate"],
        min_market_cap_cny_10k=cfg["min_market_cap_cny_10k"],
        exclude_st=cfg["exclude_st"],
        min_listing_trading_days=cfg["min_listing_trading_days"],
        liquidity_lookback_days=cfg["liquidity_lookback_days"],
        liquidity_min_quantile=cfg["liquidity_min_quantile"],
        neutralize_market_cap=cfg["neutralize_market_cap"],
        neutralize_industry=cfg["neutralize_industry"],
    )
    if panel.empty:
        raise RuntimeError("real v6 factor panel 为空")
    panel.to_csv(OUT/"factor_panel_v6.csv",index=False)

    targets,risk=build_optimized_targets(
        prices=prices,
        panel=panel,
        risk_lookback_days=cfg["risk_lookback_days"],
        covariance_shrinkage=cfg["covariance_shrinkage"],
        max_names=cfg["max_names"],
        max_stock_weight=cfg["max_stock_weight"],
        min_stock_weight=cfg["min_stock_weight"],
        sector_active_limit=cfg["sector_active_limit"],
        tracking_error_limit_annual=cfg["tracking_error_limit_annual"],
        alpha_strength=cfg["alpha_strength"],
        risk_aversion=cfg["risk_aversion"],
        turnover_penalty=cfg["turnover_penalty"],
    )
    if targets.empty:
        raise RuntimeError("real v6 optimized targets 为空")
    targets.to_csv(OUT/"optimized_targets_v6.csv",index=False)
    risk.to_csv(OUT/"ex_ante_risk_v6.csv",index=False)
    portfolio_diagnostics(targets).to_csv(OUT/"portfolio_diagnostics_v6.csv",index=False)

    ret,holdings,trades=run_optimized_backtest_v5(
        prices=prices,
        targets=targets,
        stock_limits=stock_limits,
        commission_bps=cfg["commission_bps"],
        slippage_bps=cfg["slippage_bps"],
        use_price_limits=cfg["use_price_limits"],
    )
    holdings.to_csv(OUT/"executed_holdings_v6.csv",index=False)
    trades.to_csv(OUT/"trade_costs_v6.csv",index=False)

    bench=benchmark.set_index("date")["close"].pct_change().fillna(0).reindex(ret.index).fillna(0)
    active=ret.index[ret.ne(0)]
    if len(active):
        ret=ret.loc[active.min():]
        bench=bench.loc[active.min():]

    perf=pd.DataFrame([performance_summary(ret,bench)])
    perf.to_csv(OUT/"performance_summary_v6.csv",index=False)

    nav=pd.DataFrame({
        "date":ret.index,
        "strategy_return":ret.values,
        "benchmark_return":bench.values,
        "strategy_nav":(1+ret).cumprod().values,
        "benchmark_nav":(1+bench).cumprod().values,
    })
    nav.to_csv(OUT/"nav_timeseries_v6.csv",index=False)

    fr,ic_ts,quintile,ic_summary=factor_diagnostics(prices,panel)
    fr.to_csv(OUT/"forward_returns_v6.csv",index=False)
    ic_ts.to_csv(OUT/"factor_ic_timeseries_v6.csv",index=False)
    quintile.to_csv(OUT/"quintile_returns_v6.csv",index=False)
    ic_summary.to_csv(OUT/"ic_summary_v6.csv",index=False)

    folds,test_ic=walk_forward_factor_validation(
        prices,panel,
        train_months=cfg["walk_forward_train_months"],
        test_months=cfg["walk_forward_test_months"],
    )
    folds.to_csv(OUT/"walk_forward_folds_v6.csv",index=False)
    test_ic.to_csv(OUT/"walk_forward_test_ic_v6.csv",index=False)

    report=generate_html_report(
        OUT,
        OUT/"quant_research_report.html",
        title="CSI300 Multi-Factor Strategy v6 — Quant Research Report"
    )

    print("Real V6 completed.")
    print(perf.round(4).to_string(index=False))
    print(f"Report: {report}")

if __name__=="__main__":
    main()
