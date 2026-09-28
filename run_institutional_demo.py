from pathlib import Path
import json
import numpy as np
import pandas as pd
from src.institutional_engine import institutional_enhance_cross_section
from src.multi_universe_institutional import allocate_candidate_slots, select_multi_universe_candidates
from src.strategy_lifecycle_institutional import evaluate_strategy_lifecycle, dynamic_sleeve_weights
from src.etf_rotation_institutional import etf_rotation_scores

BASE=Path(__file__).resolve().parent
OUT=BASE/'outputs'/'institutional_demo'; OUT.mkdir(parents=True,exist_ok=True)

def main():
    cfg=json.loads((BASE/'config.example.json').read_text(encoding='utf-8'))
    rng=np.random.default_rng(2026); n=90
    tickers=[f'T{i:04d}' for i in range(n)]
    x=pd.DataFrame({
        'ticker':tickers,'universe':np.repeat(['CSI300','CSI500','CSI1000'],30),
        'industry_l1':np.tile(['Bank','Tech','Industrial','Consumer','Healthcare'],18),
        'total_mv':np.exp(rng.normal(12,0.8,n)),
        'value_score':rng.normal(size=n),'quality_score':rng.normal(size=n),
        'momentum_skip_score':rng.normal(size=n),'short_reversal_score':rng.normal(size=n),
        'low_vol_score':rng.normal(size=n),'dividend_score':rng.normal(size=n),
        'sector_rotation_score':rng.normal(size=n),'crowding_score':rng.normal(size=n),
        'sleeve_core_multifactor':rng.normal(size=n),'sleeve_sector_rotation':rng.normal(size=n),
        'sleeve_mean_reversion':rng.normal(size=n),'ml_score':rng.normal(size=n),'elite_score':rng.normal(size=n)})
    sig=pd.Timestamp('2026-09-04')
    cashflow=pd.DataFrame({'ticker':tickers,'pit_ann_date':[sig-pd.Timedelta(days=40)]*n,
        'report_date':[pd.Timestamp('2026-06-30')]*n,'free_cashflow':rng.normal(2e8,1e8,n),
        'n_cashflow_act':rng.normal(3e8,1e8,n),'net_profit':rng.normal(2.5e8,0.9e8,n)})
    forecast=pd.DataFrame({'ticker':tickers,'ann_date':[sig-pd.Timedelta(days=25)]*n,
        'report_date':[pd.Timestamp('2026-09-30')]*n,
        'type':['预增' if i%3==0 else '略增' if i%3==1 else '预减' for i in range(n)],
        'p_change_min':rng.normal(12,20,n),'p_change_max':rng.normal(25,20,n)})
    express=pd.DataFrame({'ticker':tickers,'ann_date':[sig-pd.Timedelta(days=15)]*n,
        'report_date':[pd.Timestamp('2026-06-30')]*n,
        'yoy_net_profit':rng.normal(15,25,n),'yoy_sales':rng.normal(10,15,n),'diluted_roe':rng.normal(10,3,n)})
    y=institutional_enhance_cross_section(x,cashflow,forecast,express,sig,cfg)
    y['institutional_score']=0.70*y['elite_score']+0.10*y['fcf_yield_score']+0.10*y['cash_quality_score']+0.10*y['earnings_event_score']
    cand=select_multi_universe_candidates(y,'HIGH_ROTATION',cfg,score_col='institutional_score',total_names=60)
    cand.to_csv(OUT/'multi_universe_candidates.csv',index=False)
    for regime in ['TREND_UP','HIGH_ROTATION','RANGE','RISK_OFF','PANIC']:
        slots=allocate_candidate_slots(60,regime,cfg); pd.DataFrame([{'regime':regime,**slots}]).to_csv(OUT/f'universe_slots_{regime}.csv',index=False)
    dates=pd.date_range('2025-01-31',periods=12,freq='ME'); rows=[]
    bases={'core_multifactor':0.04,'sector_rotation':0.025,'mean_reversion':0.015,'ml_rank':-0.03}
    for s,b in bases.items():
        for d in dates: rows.append((d,s,b+rng.normal(0,0.015)))
    ic=pd.DataFrame(rows,columns=['signal_date','strategy','rank_ic'])
    lc=evaluate_strategy_lifecycle(ic,cfg); lc.to_csv(OUT/'strategy_lifecycle.csv',index=False)
    dyn=dynamic_sleeve_weights(cfg['ashare_elite']['sleeves'],lc,cfg); pd.DataFrame([dyn]).to_csv(OUT/'dynamic_sleeve_weights.csv',index=False)
    dts=pd.bdate_range('2025-12-01',periods=180); erows=[]
    for j,t in enumerate(['ETF_A','ETF_B','ETF_C','ETF_D','ETF_E']):
        px=np.exp(np.cumsum(rng.normal(0.0002+j*0.00008,0.008+0.001*j,len(dts))))
        for d,p in zip(dts,px): erows.append((d,t,p,1e8*(1+j*0.2)))
    etf=pd.DataFrame(erows,columns=['date','ticker','close','amount']); er=etf_rotation_scores(etf,dts[-1],top_k=3); er.to_csv(OUT/'etf_rotation.csv',index=False)
    y.sort_values('institutional_score',ascending=False).head(30).to_csv(OUT/'latest_institutional_ranking.csv',index=False)
    print('A-SHARE INSTITUTIONAL demo completed.')
    print('\nDynamic sleeve weights:'); print(pd.DataFrame([dyn]).round(4).to_string(index=False))
    print('\nLifecycle:'); print(lc.round(4).to_string(index=False))
    print('\nETF sleeve:'); print(er.round(4).to_string(index=False))
    print('\nTop 10:'); print(y.sort_values('institutional_score',ascending=False)[['ticker','universe','institutional_score','fcf_yield_score','cash_quality_score','earnings_event_score']].head(10).round(4).to_string(index=False))

if __name__=='__main__': main()
