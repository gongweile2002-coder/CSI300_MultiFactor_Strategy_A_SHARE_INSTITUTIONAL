
import pandas as pd
from src.research_v2 import _asof_by_ticker

def test_asof_does_not_use_future_announcement():
    left = pd.DataFrame({
        "ticker":["000001.SZ","000001.SZ"],
        "signal_date":pd.to_datetime(["2024-04-01","2024-05-01"])
    })
    right = pd.DataFrame({
        "ticker":["000001.SZ","000001.SZ"],
        "ann_date":pd.to_datetime(["2024-03-20","2024-04-20"]),
        "roe":[10.0,99.0]
    })
    out = _asof_by_ticker(left, right, "signal_date", "ann_date").sort_values("signal_date")
    assert out.iloc[0]["roe"] == 10.0
    assert out.iloc[1]["roe"] == 99.0
