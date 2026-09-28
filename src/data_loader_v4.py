
from pathlib import Path
import pandas as pd


def _read_optional(path, date_cols=None):
    path = Path(path)
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    for c in (date_cols or []):
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="coerce")
    return df


def load_real_data_v4(data_dir):
    d = Path(data_dir)
    prices = pd.read_csv(d/"prices.csv", parse_dates=["date"])
    daily_basic = pd.read_csv(d/"daily_basic.csv", parse_dates=["date"])
    fundamentals = pd.read_csv(d/"fundamentals_raw.csv", parse_dates=["ann_date","report_date"])
    membership = pd.read_csv(d/"index_membership.csv", parse_dates=["effective_date"])
    benchmark = pd.read_csv(d/"benchmark.csv", parse_dates=["date"])

    metadata = _read_optional(d/"stock_metadata.csv", ["list_date","delist_date"])
    industry = _read_optional(d/"industry_membership.csv", ["in_date","out_date"])
    st_status = _read_optional(d/"st_status.csv", ["trade_date"])
    stock_limits = _read_optional(d/"stock_limits.csv", ["date"])

    return (
        prices, daily_basic, fundamentals, membership, benchmark,
        metadata, industry, st_status, stock_limits
    )
