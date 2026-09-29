"""Synthetic allocation regressions; no performance claims."""
from dataclasses import replace
import pandas as pd
import pytest
from src.signals_v8 import select_capped_portfolio
from src.live_v8 import RiskConfig,TradingBlocked

def panel(sectors):
    return pd.DataFrame({'ticker':[f'{600001+i}.SH' for i in range(len(sectors))],
                         'industry_l1':sectors,'composite_score':list(range(len(sectors),0,-1))})

def test_exhausted_sector_does_not_consume_positive_slots():
    x=panel(['A']*35+['B']*15+['C']*15)
    y=select_capped_portfolio(x,.8,RiskConfig())
    assert len(y)==30 and (y.target_weight>0).all()
    assert y.groupby('industry_l1').target_weight.sum().max()<=.3+1e-12
    assert y.target_weight.sum()<=.8+1e-12
    assert y.target_weight.max()<=.08
    assert set(y.industry_l1)=={'A','B','C'}

def test_too_few_positive_names_blocks():
    with pytest.raises(TradingBlocked,match='实际正权重'):
        select_capped_portfolio(panel(['A']*40),.8,RiskConfig())

def test_cash_is_retained_when_single_name_cap_binds():
    y=select_capped_portfolio(panel([f'S{i}' for i in range(20)]),.8,
                             replace(RiskConfig(),max_single_weight=.01))
    assert len(y)==20 and y.target_weight.sum()==pytest.approx(.2)

def test_tied_ranks_are_input_order_independent():
    x=panel(['A','B','C']*20);x.composite_score=1.
    a=select_capped_portfolio(x,.4,RiskConfig())
    b=select_capped_portfolio(x.sample(frac=1,random_state=4),.4,RiskConfig())
    pd.testing.assert_frame_equal(a,b)

def test_duplicate_tickers_block():
    x=panel(['A','B','C']*10);x.loc[1,'ticker']=x.loc[0,'ticker']
    with pytest.raises(TradingBlocked,match='重复证券'):
        select_capped_portfolio(x,.8,RiskConfig())


def test_duplicate_dataframe_indices_do_not_duplicate_holdings():
    x=panel(['A','B','C']*20);x.index=[0]*len(x)
    y=select_capped_portfolio(x,.8,RiskConfig())
    assert len(y)==30 and y.ticker.is_unique
    assert y.target_weight.sum()<=.8+1e-12
