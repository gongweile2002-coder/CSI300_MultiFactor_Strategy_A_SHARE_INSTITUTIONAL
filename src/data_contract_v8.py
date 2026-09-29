"""Units, provenance and conservative publication-date availability."""
import hashlib,json
from pathlib import Path
import numpy as np
import pandas as pd
from .live_v8 import require,TradingCalendar

def normalize_tushare_prices(daily,adj):
    keys=['ts_code','trade_date'];require(not daily.duplicated(keys).any() and not adj.duplicated(keys).any(),'行情/复权主键重复')
    x=daily.merge(adj[keys+['adj_factor']],on=keys,how='left',validate='one_to_one')
    require(x['adj_factor'].notna().all() and np.isfinite(x['adj_factor']).all() and (x['adj_factor']>0).all(),'复权因子缺失或非法')
    x=x.rename(columns={'ts_code':'ticker','trade_date':'date','vol':'volume'})
    x['date']=pd.to_datetime(x['date'],format='%Y%m%d',errors='raise')
    x['volume']=pd.to_numeric(x['volume'],errors='raise')*100;x['amount']=pd.to_numeric(x['amount'],errors='raise')*1000
    for c in ['open','high','low','close','volume','amount']:require(np.isfinite(x[c]).all() and (x[c]>=0).all(),f'行情非法 {c}')
    require((x[['open','high','low','close']]>0).all().all(),'价格必须为正')
    require((x['low']<=x[['open','close']].min(axis=1)).all() and (x['high']>=x[['open','close']].max(axis=1)).all(),'OHLC 关系错误')
    x['amount_unit']='CNY';x['volume_unit']='share';x['price_basis']='raw'
    return x.sort_values(['ticker','date']).reset_index(drop=True)

def latest_financial_records(frame,signal_date,announcement_col='ann_date',period_col='report_date'):
    x=frame.copy();x[announcement_col]=pd.to_datetime(x[announcement_col],errors='raise');x[period_col]=pd.to_datetime(x[period_col],errors='raise')
    cutoff=pd.Timestamp(signal_date).normalize()
    x=x[(x[announcement_col]<cutoff)&(x[period_col]<=cutoff)]
    return x.sort_values(['ticker',period_col,announcement_col]).drop_duplicates('ticker',keep='last')

def load_verified_dataset(directory):
    d=Path(directory);manifest=json.loads((d/'data_manifest_v8.json').read_text(encoding='utf-8'))
    require(manifest.get('schema_version')==8 and manifest.get('source')=='tushare','缺少真实 v8 数据清单，先 download_live_data.py')
    require(manifest.get('complete_universe') is True,'抽样股票池不能生成实盘信号')
    required={'prices.csv','raw_prices.csv','trade_calendar.csv','daily_basic.csv','fundamentals_raw.csv','index_membership.csv','stock_metadata.csv','industry_membership.csv','st_status.csv','benchmark.csv','stock_limits.csv'}
    require(required<=set(manifest['files']),'清单缺少必要数据文件')
    for name,expected in manifest['files'].items():
        require(Path(name).name==name,'非法数据文件路径')
        require(hashlib.sha256((d/name).read_bytes()).hexdigest()==expected,f'{name} 与数据清单不符')
    return manifest,TradingCalendar(pd.read_csv(d/'trade_calendar.csv',dtype={'date':str}))
