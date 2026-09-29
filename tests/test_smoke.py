
from pathlib import Path
from src.data_loader import load_demo_data
from src.factors import build_factor_scores
from src.backtest import run_monthly_topn_backtest

def test_pipeline_runs():
    base = Path(__file__).resolve().parents[1]
    prices, fundamentals, benchmark = load_demo_data(base / "data")
    scores = build_factor_scores(prices, fundamentals)
    assert not scores.empty
    rets, holdings = run_monthly_topn_backtest(prices, scores, top_n=5)
    assert len(rets) > 0
    assert not holdings.empty
