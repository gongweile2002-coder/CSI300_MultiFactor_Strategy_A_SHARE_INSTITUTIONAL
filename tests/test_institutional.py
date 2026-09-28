import pandas as pd
import numpy as np
from src.multi_universe_institutional import allocate_candidate_slots
from src.fundamental_extension_institutional import add_cash_quality_factors
from src.earnings_events_institutional import add_earnings_event_factor
from src.etf_rotation_institutional import etf_rotation_scores
from src.strategy_lifecycle_institutional import evaluate_strategy_lifecycle, dynamic_sleeve_weights

CFG={'institutional':{'universe_weights_by_regime':{'TREND_UP':{'CSI300':0.55,'CSI500':0.30,'CSI1000':0.15}},'event_signal_max_age_days':120,
'strategy_lifecycle':{'short_window_months':6,'long_window_months':12,'pause_ic_threshold':-0.02,'reduce_ic_threshold':0.0,'reactivate_ic_threshold':0.02,'min_months_for_decision':6,'core_floor':0.30,'ml_cap':0.20}}}

def test_universe_slots_sum():
    s=allocate_candidate_slots(60,'TREND_UP',CFG); assert sum(s.values())==60 and s['CSI300']>s['CSI1000']

def test_cash_quality_factors():
    sig=pd.Timestamp('2026-09-04'); x=pd.DataFrame({'ticker':[f'T{i}' for i in range(20)],'total_mv':np.linspace(1e6,2e6,20),'industry_l1':['A']*10+['B']*10})
    cf=pd.DataFrame({'ticker':x['ticker'],'pit_ann_date':[sig-pd.Timedelta(days=10)]*20,'report_date':[pd.Timestamp('2026-06-30')]*20,'free_cashflow':np.linspace(1e8,3e8,20),'n_cashflow_act':np.linspace(1.2e8,3.2e8,20),'net_profit':np.linspace(1e8,2e8,20)})
    y=add_cash_quality_factors(x,cf,sig); assert y['fcf_yield_score'].notna().all() and y['cash_quality_score'].notna().all()

def test_earnings_event_no_future():
    sig=pd.Timestamp('2026-09-04'); x=pd.DataFrame({'ticker':['A','B','C']})
    f=pd.DataFrame({'ticker':['A','B','C'],'ann_date':[sig-pd.Timedelta(days=10),sig+pd.Timedelta(days=1),sig-pd.Timedelta(days=5)],'report_date':[pd.Timestamp('2026-09-30')]*3,'type':['预增','预增','预减'],'p_change_min':[20,100,-30],'p_change_max':[30,120,-20]})
    y=add_earnings_event_factor(x,f,pd.DataFrame(),sig,120); assert len(y)==3

def test_etf_rotation():
    dates=pd.bdate_range('2025-01-01',periods=150); rows=[]; rng=np.random.default_rng(1)
    for j,t in enumerate(['A','B','C','D']):
        px=np.exp(np.cumsum(rng.normal(0.0002+j*0.0001,0.01,len(dates))))
        for d,p in zip(dates,px): rows.append((d,t,p,1e8))
    y=etf_rotation_scores(pd.DataFrame(rows,columns=['date','ticker','close','amount']),dates[-1],top_k=2)
    assert 0 < len(y) <= 2 and abs(y['target_weight_etf_sleeve'].sum()-1)<1e-8
    assert (y['trend_ok']>0).all()

def test_strategy_lifecycle_pause():
    dates=pd.date_range('2025-01-31',periods=12,freq='ME'); ic=pd.DataFrame([(d,'ml_rank',-0.03) for d in dates],columns=['signal_date','strategy','rank_ic'])
    lc=evaluate_strategy_lifecycle(ic,CFG); assert lc.iloc[0]['status']=='PAUSE'
    w=dynamic_sleeve_weights({'core_multifactor':0.5,'ml_rank':0.5},lc,CFG); assert w.get('ml_rank',0)==0
