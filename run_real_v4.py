
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.data_loader_v4 import load_real_data_v4
from src.research_v4 import (
    build_point_in_time_panel_v4,
    run_realistic_backtest_v4,
    walk_forward_factor_validation,
)
from src.research_v2 import performance_summary

BASE=Path(__file__).resolve().parent

def main():
    cfg=json.loads((BASE/"config.example.json").read_text(encoding="utf-8"))
    out=BASE/"outputs"/"real_v4"
    out.mkdir(parents=True,exist_ok=True)

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
        raise RuntimeError("v4 panel 为空")
    panel.to_csv(out/"factor_panel_v4.csv",index=False)

    ret,holdings,trades=run_realistic_backtest_v4(
        prices=prices,
        panel=panel,
        stock_limits=stock_limits,
        top_n=cfg["top_n"],
        commission_bps=cfg["commission_bps"],
        slippage_bps=cfg["slippage_bps"],
        use_price_limits=cfg["use_price_limits"],
    )
    holdings.to_csv(out/"holdings_v4.csv",index=False)
    trades.to_csv(out/"trade_costs_v4.csv",index=False)

    bench=benchmark.set_index("date")["close"].pct_change().fillna(0).reindex(ret.index).fillna(0)
    active=ret.index[ret.ne(0)]
    if len(active):
        ret=ret.loc[active.min():]
        bench=bench.loc[active.min():]

    perf=pd.DataFrame([performance_summary(ret,bench)])
    perf.to_csv(out/"performance_summary_v4.csv",index=False)

    folds,test_ic=walk_forward_factor_validation(
        prices,panel,
        train_months=cfg["walk_forward_train_months"],
        test_months=cfg["walk_forward_test_months"],
    )
    folds.to_csv(out/"walk_forward_folds.csv",index=False)
    test_ic.to_csv(out/"walk_forward_test_ic.csv",index=False)

    nav=(1+ret).cumprod()
    bnav=(1+bench).cumprod()

    plt.figure(figsize=(10,5))
    plt.plot(nav.index,nav.values,label="V4 strategy")
    plt.plot(bnav.index,bnav.values,label="CSI300")
    plt.title("CSI300 Multi-Factor V4")
    plt.xlabel("Date"); plt.ylabel("NAV"); plt.legend()
    plt.tight_layout()
    plt.savefig(out/"equity_curve_v4.png",dpi=180)
    plt.close()

    dd=nav/nav.cummax()-1
    plt.figure(figsize=(10,4))
    plt.plot(dd.index,dd.values)
    plt.title("V4 Drawdown")
    plt.xlabel("Date"); plt.ylabel("Drawdown")
    plt.tight_layout()
    plt.savefig(out/"drawdown_v4.png",dpi=180)
    plt.close()

    if not test_ic.empty:
        rolling=test_ic.set_index("signal_date")["rank_ic"].rolling(6,min_periods=2).mean()
        plt.figure(figsize=(10,4))
        plt.plot(rolling.index,rolling.values)
        plt.title("Walk-forward OOS Rank IC (6M rolling mean)")
        plt.xlabel("Date"); plt.ylabel("Rank IC")
        plt.tight_layout()
        plt.savefig(out/"walk_forward_oos_ic.png",dpi=180)
        plt.close()

    print("V4 real-data pipeline completed.")
    print(perf.round(4).to_string(index=False))
    if not folds.empty:
        print("\nWalk-forward folds:")
        print(folds.round(4).to_string(index=False))

if __name__=="__main__":
    main()
