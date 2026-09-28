
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.research_v2 import (
    load_real_data,
    build_point_in_time_panel,
    run_next_open_backtest,
    factor_diagnostics,
    performance_summary,
)

BASE = Path(__file__).resolve().parent

def main():
    cfg = json.loads((BASE/"config.example.json").read_text(encoding="utf-8"))
    out = BASE/"outputs"/"real"
    out.mkdir(parents=True, exist_ok=True)

    prices, daily_basic, fundamentals, membership, benchmark = load_real_data(BASE/"data"/"real")

    panel = build_point_in_time_panel(
        prices=prices,
        daily_basic=daily_basic,
        fundamentals=fundamentals,
        membership=membership,
        start_date=cfg["start_date"],
        end_date=cfg["end_date"],
        momentum_lookback_days=cfg["momentum_lookback_days"],
        winsor_lower=cfg["winsor_lower"],
        winsor_upper=cfg["winsor_upper"],
        factor_weights=cfg["factor_weights"],
        min_turnover_rate=cfg["min_turnover_rate"],
        min_market_cap_cny_10k=cfg["min_market_cap_cny_10k"],
    )
    if panel.empty:
        raise RuntimeError("Point-in-time panel 为空，请检查真实数据文件或日期范围。")
    panel.to_csv(out/"factor_panel.csv", index=False)

    strat_ret, holdings, turnover = run_next_open_backtest(
        prices, panel,
        top_n=cfg["top_n"],
        transaction_cost_bps=cfg["transaction_cost_bps"]
    )
    holdings.to_csv(out/"holdings.csv", index=False)
    turnover.to_csv(out/"turnover.csv", index=False)

    bench = benchmark.set_index("date")["close"].pct_change().fillna(0)
    bench = bench.reindex(strat_ret.index).fillna(0)

    active_dates = strat_ret.index[strat_ret.ne(0)]
    if len(active_dates):
        start = active_dates.min()
        strat_ret = strat_ret.loc[start:]
        bench = bench.loc[start:]

    summary = pd.DataFrame([performance_summary(strat_ret, bench)])
    summary.to_csv(out/"performance_summary.csv", index=False)

    fr, ic_ts, quintile, ic_summary = factor_diagnostics(prices, panel)
    fr.to_csv(out/"forward_returns.csv", index=False)
    ic_ts.to_csv(out/"factor_ic_timeseries.csv", index=False)
    quintile.to_csv(out/"quintile_returns.csv", index=False)
    ic_summary.to_csv(out/"ic_summary.csv", index=False)

    nav = (1+strat_ret).cumprod()
    bnav = (1+bench).cumprod()
    plt.figure(figsize=(10,5))
    plt.plot(nav.index, nav.values, label="Multi-factor strategy")
    plt.plot(bnav.index, bnav.values, label="CSI300 benchmark")
    plt.title("CSI300 Multi-Factor Strategy")
    plt.xlabel("Date")
    plt.ylabel("NAV")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out/"equity_curve.png", dpi=180)
    plt.close()

    dd = nav/nav.cummax()-1
    plt.figure(figsize=(10,4))
    plt.plot(dd.index, dd.values)
    plt.title("Strategy Drawdown")
    plt.xlabel("Date")
    plt.ylabel("Drawdown")
    plt.tight_layout()
    plt.savefig(out/"drawdown.png", dpi=180)
    plt.close()

    rolling = strat_ret.rolling(126).mean()/strat_ret.rolling(126).std(ddof=0)*np.sqrt(252)
    plt.figure(figsize=(10,4))
    plt.plot(rolling.index, rolling.values)
    plt.title("126D Rolling Sharpe")
    plt.xlabel("Date")
    plt.ylabel("Sharpe")
    plt.tight_layout()
    plt.savefig(out/"rolling_sharpe.png", dpi=180)
    plt.close()

    print("Real-data research pipeline completed.")
    print(summary.round(4).to_string(index=False))
    if not ic_summary.empty:
        print("\nIC Summary")
        print(ic_summary.round(4).to_string(index=False))

if __name__ == "__main__":
    main()
