
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

from src.data_loader import load_demo_data
from src.factors import build_factor_scores
from src.backtest import run_monthly_topn_backtest
from src.metrics import performance_metrics

BASE = Path(__file__).resolve().parent
DATA = BASE / "data"
OUTPUTS = BASE / "outputs"
OUTPUTS.mkdir(exist_ok=True)

def main():
    prices, fundamentals, benchmark = load_demo_data(DATA)

    scores = build_factor_scores(
        prices,
        fundamentals,
        value_weight=1/3,
        quality_weight=1/3,
        momentum_weight=1/3,
        momentum_lookback_days=126
    )
    scores.to_csv(OUTPUTS / "factor_scores.csv", index=False)

    strategy_ret, holdings = run_monthly_topn_backtest(
        prices,
        scores,
        top_n=10,
        transaction_cost_bps=10
    )
    holdings.to_csv(OUTPUTS / "holdings.csv", index=False)

    bench = benchmark.set_index("date")["benchmark_close"].pct_change().fillna(0)
    bench = bench.reindex(strategy_ret.index).fillna(0)

    # Only score periods where strategy has started
    active = strategy_ret.index[strategy_ret.ne(0)]
    if len(active):
        start = active.min()
        strategy_ret = strategy_ret.loc[start:]
        bench = bench.loc[start:]

    metrics = performance_metrics(strategy_ret, bench)
    metrics_df = pd.DataFrame([metrics])
    metrics_df.to_csv(OUTPUTS / "performance_summary.csv", index=False)

    nav = (1 + strategy_ret).cumprod()
    bench_nav = (1 + bench).cumprod()

    plt.figure(figsize=(10, 5))
    plt.plot(nav.index, nav.values, label="Multi-factor strategy")
    plt.plot(bench_nav.index, bench_nav.values, label="Benchmark")
    plt.title("CSI300 Multi-Factor Strategy - Demo Backtest")
    plt.xlabel("Date")
    plt.ylabel("Net Asset Value")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUTS / "equity_curve.png", dpi=160)
    plt.close()

    print("Backtest completed.")
    print(metrics_df.round(4).to_string(index=False))
    print(f"Outputs saved to: {OUTPUTS}")

if __name__ == "__main__":
    main()
