
import pandas as pd
import numpy as np
from src.strategy_zoo_elite import build_strategy_sleeves, combine_sleeves
from src.validation_elite import multiple_testing_haircut, probabilistic_sharpe_ratio

def test_strategy_sleeves_build():
    n=30
    rng=np.random.default_rng(1)
    x=pd.DataFrame({
        "quality_score":rng.normal(size=n),
        "value_score":rng.normal(size=n),
        "momentum_skip_score":rng.normal(size=n),
        "short_reversal_score":rng.normal(size=n),
        "low_vol_score":rng.normal(size=n),
        "dividend_score":rng.normal(size=n),
        "sector_rotation_score":rng.normal(size=n),
        "crowding_score":rng.normal(size=n),
    })
    y=build_strategy_sleeves(x)
    assert {"sleeve_core_multifactor","sleeve_sector_rotation","sleeve_mean_reversion"}.issubset(y.columns)

def test_combine_sleeves():
    x=pd.DataFrame({
        "sleeve_core_multifactor":[1,0],
        "sleeve_sector_rotation":[0,1],
        "sleeve_mean_reversion":[0,0],
        "ml_score":[0.5,-0.5],
    })
    y=combine_sleeves(x,{"core_multifactor":0.5,"sector_rotation":0.3,"ml_rank":0.2})
    assert "elite_score" in y.columns
    assert y.loc[0,"elite_score"] > y.loc[1,"elite_score"]

def test_multiple_testing_haircut():
    x=multiple_testing_haircut({"a":2.0,"b":1.0,"c":0.5})
    assert len(x)==3
    assert (x["testing_count"]==3).all()

def test_psr_runs():
    rng=np.random.default_rng(2)
    r=pd.Series(rng.normal(0.0005,0.01,500))
    p=probabilistic_sharpe_ratio(r)
    assert 0 <= p <= 1
