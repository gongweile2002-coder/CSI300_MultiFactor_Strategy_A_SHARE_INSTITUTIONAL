"""Auditable daily core signal. Data completion is explicit; no ML live sleeve."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from .live_v8 import require,read_csv,cn_timestamp,board
from .data_contract_v8 import load_verified_dataset
from .research_v4 import build_point_in_time_panel_v4,industry_for_date,st_tickers_for_date

def last_completed_session(calendar,now):
    now=cn_timestamp(now);today=pd.Timestamp(now.date())
    return today if calendar.is_open(today) and now.hour>=18 else calendar.previous(today)

def select_capped_portfolio(panel,gross,cfg):
    """Select up to 30 positive allocations, skipping exhausted sectors.

    Never increase per-name budgets to force full investment. Unallocated
    capital stays in cash; fewer than 15 funded names blocks publication.
    """
    require(len(panel)>=15,'合格股票不足 15 只')
    require(not panel['ticker'].duplicated().any(),'选股输入存在重复证券')
    require(np.isfinite(panel['composite_score']).all(),'选股评分必须有限')
    require(panel['industry_l1'].notna().all() and
            not panel['industry_l1'].isin(['','UNKNOWN']).any(),'选股行业缺失')
    require(np.isfinite(gross) and 0<gross<=cfg.max_gross_weight,'目标总仓位非法')
    ranked=panel.sort_values(['composite_score','ticker'],ascending=[False,True]).reset_index(drop=True)
    budget=min(gross/min(30,len(ranked)),cfg.max_single_weight)
    sector={}; selected=[]; weights=[]
    for idx,row in ranked.iterrows():
        w=min(budget,max(0.,cfg.max_sector_weight-sector.get(row['industry_l1'],0.)),
              max(0.,gross-sum(weights)))
        if w<=1e-12:
            continue
        selected.append(idx); weights.append(w)
        sector[row['industry_l1']]=sector.get(row['industry_l1'],0.)+w
        if len(selected)==30:
            break
    require(len(selected)>=15,'行业/单股约束后实际正权重股票不足 15 只，停止发布目标')
    result=ranked.loc[selected].copy()
    result['target_weight']=weights
    return result

def _generate_signals_unchecked(directory,output,now,cfg):
    manifest,calendar=load_verified_dataset(directory);signal=last_completed_session(calendar,now);d=Path(directory)
    require(manifest.get('as_of')==str(signal.date()),'日线数据未更新到最近完整交易日')
    def load(name,dates):
        x=read_csv(d/name)
        for c in dates:x[c]=pd.to_datetime(x[c],errors='raise')
        return x
    prices=load('prices.csv',['date']);raw=load('raw_prices.csv',['date']);db=load('daily_basic.csv',['date'])
    financial=load('fundamentals_raw.csv',['ann_date','report_date']);members=load('index_membership.csv',['effective_date'])
    meta=load('stock_metadata.csv',[]);industry=load('industry_membership.csv',['in_date','out_date']);st=load('st_status.csv',['trade_date'])
    benchmark=load('benchmark.csv',['date'])
    for x,label,datecol in [(prices,'研究行情','date'),(raw,'原始行情','date'),(db,'估值','date'),(members,'成分','effective_date')]:
        require(not x.duplicated(['ticker',datecol]).any(),f'{label}: 重复主键')
    for x,label in [(prices,'研究行情'),(raw,'原始行情'),(db,'估值')]:require(x['date'].max()==signal,f'{label}: 最新日期不一致/未来数据')
    require(raw['price_basis'].eq('raw').all() and raw['amount_unit'].eq('CNY').all(),'原始行情必须标明 raw/CNY')
    require(not meta['ticker'].duplicated().any(),'重复证券资料')
    snapshot=members[members['effective_date']<=signal]
    require(not snapshot.empty,'没有信号日可用成分')
    effective=snapshot['effective_date'].max();current=set(snapshot.loc[snapshot['effective_date']==effective,'ticker'])
    require((signal-effective).days<=45,'成分快照超过 45 天')
    for frame,label in [(raw,'原始行情'),(db,'估值')]:
        available=set(frame.loc[frame['date']==signal,'ticker'])
        require(len(current & available)>=.95*len(current),f'{label}: 当前成分覆盖不足 95%，先排查漏数或停牌')

    ref=raw[raw['date']==signal][['ticker','close']].rename(columns={'close':'signal_close_raw'})
    ref=ref.merge(industry_for_date(industry,signal),on='ticker',how='left',validate='one_to_one')
    ref=ref.merge(meta[['ticker','name']],on='ticker',how='left',validate='one_to_one')
    adv=raw[raw['date']<=signal].sort_values(['ticker','date']).groupby('ticker').tail(20).groupby('ticker')['amount'].mean()
    ref['adv20_cny']=ref['ticker'].map(adv);ref['signal_date']=str(signal.date())
    ref['is_st']=ref['ticker'].isin(st_tickers_for_date(st,signal))|ref['name'].fillna('').str.upper().str.contains('ST')
    ref['is_delisting']=ref['name'].fillna('').str.contains('退');ref['source']='real'
    panel=build_point_in_time_panel_v4(prices,db,financial,members,metadata=meta,industry_membership=industry,st_status=st,
          signal_dates_override=[signal],momentum_lookback_days=126,factor_weights={'value':.3,'quality':.4,'momentum':.3},
          min_listing_trading_days=180,liquidity_min_quantile=.10,exclude_st=True)
    require(not panel.empty,'缺少足够历史、财报或合格股票')
    panel=panel[(panel['valuation_date']==signal)&((signal-panel['ann_date']).dt.days<=200)].copy()
    panel=panel.merge(ref.drop(columns=['industry_l1','name','signal_date']),on='ticker',how='inner',validate='one_to_one')
    panel=panel[~panel['is_st']&~panel['is_delisting']&panel['ticker'].map(lambda t:board(t) in cfg.allowed_buy_boards)]
    panel=panel[panel['industry_l1'].notna()&~panel['industry_l1'].isin(['','UNKNOWN'])&np.isfinite(panel['composite_score'])]
    require(len(panel)>=15,'合格股票不足 15 只，停止生成实盘目标')
    benchmark=benchmark[benchmark['date']<=signal].sort_values('date')
    require(not benchmark['date'].duplicated().any() and len(benchmark)>=120 and benchmark['date'].max()==signal,'基准历史不足或未更新')
    closes=pd.to_numeric(benchmark['close'],errors='raise').tail(120)
    require(np.isfinite(closes).all() and (closes>0).all(),'基准价格非法')
    trend=closes.iloc[-1]>=closes.mean();gross=min(cfg.max_gross_weight,.8 if trend else .4)
    selected=select_capped_portfolio(panel,gross,cfg)
    weights=selected['target_weight'].tolist()
    selected['signal_date']=str(signal.date());selected['strategy_version']=cfg.version
    columns=['ticker','target_weight','signal_date','signal_close_raw','source','strategy_version','industry_l1','composite_score']
    out=Path(output);out.mkdir(parents=True,exist_ok=True)
    selected[columns].to_csv(out/'targets.csv',index=False);ref.to_csv(out/'execution_reference.csv',index=False)
    panel.to_csv(out/'factor_audit.csv',index=False)
    report={'signal_date':str(signal.date()),'eligible_names':len(panel),'selected_names':len(selected),
      'gross_target':float(sum(weights)),'cash_target':float(1-sum(weights)),
      'allocation_method':'ranked_positive_slots_sector_capped_v9_1','risk_regime':'ABOVE_MA120' if trend else 'BELOW_MA120',
      'strategy':'fixed_core_value30_quality40_momentum30','performance_validated':False,
      'membership_basis':manifest.get('membership_basis'),'financial_vintages':manifest.get('financial_vintages')}
    (out/'signal_report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    return report


def generate_signals(directory,output,now,cfg):
    """v9 publishes a complete, config-bound bundle only after consistent data checks."""
    from .ops_v9 import audit_dataset, atomic_json, generation_lock, publish_signal_manifest, BUNDLE_FILE, sha256
    out=Path(output)
    with generation_lock(out):
        (out/BUNDLE_FILE).unlink(missing_ok=True)
        audit=audit_dataset(directory,now)
        atomic_json(out/'data_audit_v9.json',audit)
        require(audit['status']=='PASS','真实数据一致性检查未通过，查看 data_audit_v9.json')
        report=_generate_signals_unchecked(directory,out,now,cfg)
        load_verified_dataset(directory)
        require(sha256(Path(directory)/'data_manifest_v8.json')==audit['data_manifest_hash'],'计算期间数据版本变化，停止发布信号')
        manifest=publish_signal_manifest(out,directory,cfg,now)
        return {**report,'bundle_hash':manifest['bundle_hash']}
