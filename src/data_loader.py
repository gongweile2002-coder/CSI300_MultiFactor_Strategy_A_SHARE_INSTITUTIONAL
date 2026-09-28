
from pathlib import Path
import pandas as pd

REQUIRED_PRICE_COLS = {"date", "ticker", "close"}
REQUIRED_FUND_COLS = {"date", "ticker", "pe_ttm", "pb", "roe", "debt_ratio", "profit_growth"}
REQUIRED_BENCH_COLS = {"date", "benchmark_close"}

def _read_csv(path, required_cols):
    df = pd.read_csv(path, parse_dates=["date"])
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"{path} 缺少列: {sorted(missing)}")
    return df.sort_values(["date"] + (["ticker"] if "ticker" in df.columns else [])).reset_index(drop=True)

def load_demo_data(data_dir="data"):
    data_dir = Path(data_dir)
    prices = _read_csv(data_dir / "demo_prices.csv", REQUIRED_PRICE_COLS)
    fundamentals = _read_csv(data_dir / "demo_fundamentals.csv", REQUIRED_FUND_COLS)
    benchmark = _read_csv(data_dir / "demo_benchmark.csv", REQUIRED_BENCH_COLS)
    return prices, fundamentals, benchmark

def load_custom_data(price_csv, fundamental_csv, benchmark_csv):
    prices = _read_csv(price_csv, REQUIRED_PRICE_COLS)
    fundamentals = _read_csv(fundamental_csv, REQUIRED_FUND_COLS)
    benchmark = _read_csv(benchmark_csv, REQUIRED_BENCH_COLS)
    return prices, fundamentals, benchmark
