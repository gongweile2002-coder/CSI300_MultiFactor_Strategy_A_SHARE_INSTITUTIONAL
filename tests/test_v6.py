
from pathlib import Path
import pandas as pd

from src.reporting_v6 import load_run_folder, generate_html_report

def test_report_generation(tmp_path):
    folder = tmp_path/"run"
    folder.mkdir()

    pd.DataFrame([{
        "total_return":0.2,
        "annualized_return":0.1,
        "annualized_volatility":0.15,
        "sharpe":0.8,
        "max_drawdown":-0.12,
        "information_ratio":0.5,
        "beta":0.9,
        "alpha_annualized":0.03,
    }]).to_csv(folder/"performance_summary_v6.csv", index=False)

    pd.DataFrame({
        "date":pd.date_range("2024-01-01", periods=5),
        "strategy_nav":[1,1.01,1.00,1.04,1.06],
        "benchmark_nav":[1,1.005,1.01,1.015,1.02],
    }).to_csv(folder/"nav_timeseries_v6.csv", index=False)

    out = generate_html_report(folder, folder/"report.html", "Test")
    assert out.exists()
    text = out.read_text(encoding="utf-8")
    assert "Test" in text
    assert "Annualized Return" in text

def test_load_run_folder_prefers_v6(tmp_path):
    folder = tmp_path/"run"
    folder.mkdir()
    pd.DataFrame([{"sharpe":1.23}]).to_csv(folder/"performance_summary_v6.csv", index=False)
    data = load_run_folder(folder)
    assert not data["performance"].empty
    assert float(data["performance"].iloc[0]["sharpe"]) == 1.23
