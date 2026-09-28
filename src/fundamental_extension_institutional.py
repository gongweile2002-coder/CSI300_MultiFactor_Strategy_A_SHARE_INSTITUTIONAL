from __future__ import annotations
import numpy as np
import pandas as pd
from .research_v4 import neutralize_factor, winsorize, zscore

def _safe_z(s):
    s=pd.to_numeric(s,errors='coerce')
    if s.notna().sum()<3:return pd.Series(0.,index=s.index)
    return zscore(winsorize(s,.025,.975)).fillna(0.)

def point_in_time_cashflow(cross_section,cashflow,signal_date):
    x=cross_section.copy();values=['free_cashflow','n_cashflow_act','net_profit']
    if cashflow is None or cashflow.empty:
        for c in values:x[c]=np.nan
        return x
    cf=cashflow.copy()
    for c in ['ann_date','f_ann_date','pit_ann_date','report_date']:
        if c in cf:cf[c]=pd.to_datetime(cf[c],errors='coerce')
    if 'pit_ann_date' not in cf:
        cf['pit_ann_date']=cf['f_ann_date'].fillna(cf['ann_date']) if 'f_ann_date' in cf else cf['ann_date']
    cutoff=pd.Timestamp(signal_date).normalize()
    cf=cf[(cf['pit_ann_date']<cutoff)&(cf['report_date']<=cutoff)]
    if 'report_type' in cf:cf=cf[pd.to_numeric(cf['report_type'],errors='coerce')==1]
    rows=[]
    for t,g in cf.groupby('ticker'):
        g=g.sort_values(['report_date','pit_ann_date']).drop_duplicates('report_date',keep='last').set_index('report_date')
        period=g.index.max();latest=g.loc[period];row={'ticker':t,'cashflow_report_date':period,'pit_ann_date':latest['pit_ann_date']}
        annual=pd.Timestamp(year=period.year-1,month=12,day=31);prior=period-pd.DateOffset(years=1)
        for c in values:
            v=pd.to_numeric(latest.get(c,np.nan),errors='coerce')
            if period.month!=12:
                v=v+pd.to_numeric(g.loc[annual].get(c,np.nan),errors='coerce')-pd.to_numeric(g.loc[prior].get(c,np.nan),errors='coerce') if annual in g.index and prior in g.index else np.nan
            row[c]=v
        rows.append(row)
    result=pd.DataFrame(rows,columns=['ticker','cashflow_report_date','pit_ann_date']+values)
    return x.drop(columns=values,errors='ignore').merge(result,on='ticker',how='left',validate='many_to_one')

def add_cash_quality_factors(cross_section,cashflow,signal_date):
    x=point_in_time_cashflow(cross_section,cashflow,signal_date)
    def numeric(c):return pd.to_numeric(x.get(c,pd.Series(np.nan,index=x.index)),errors='coerce')
    fcf,mv,ocf,npf=(numeric(c) for c in ['free_cashflow','total_mv','n_cashflow_act','net_profit'])
    x['fcf_yield_raw']=fcf/(mv*10000).where(mv>0)
    threshold=npf.abs().median()*.02 if npf.notna().any() else 0
    x['cash_quality_raw']=ocf/npf.where(npf>threshold)
    for raw,score in [('fcf_yield_raw','fcf_yield_score'),('cash_quality_raw','cash_quality_score')]:
        x[raw]=x[raw].replace([np.inf,-np.inf],np.nan)
        standardized=raw+'_standardized'
        x[standardized]=_safe_z(x[raw].fillna(x[raw].median() if x[raw].notna().any() else 0))
        x[score]=neutralize_factor(x,standardized).fillna(0.) if {'industry_l1','total_mv'}<=set(x) else x[standardized]
    return x
