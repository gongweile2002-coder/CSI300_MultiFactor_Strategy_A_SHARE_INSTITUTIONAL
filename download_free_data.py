"""Free, anonymous CSI300 data for a separately labelled forward Paper variant."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import tempfile

import pandas as pd

from src.baostock_provider import (
    BaoStockDownloader, FREE_CONTRACT, RAW_FIELDS, code_from_ticker,
    derive_paper_limits, merge_industry_snapshots, normalize_calendar,
    normalize_dividends, normalize_financials, normalize_industry,
    normalize_members, normalize_prices, ticker_from_code, to_research_prices,
)
from src.data_contract_v8 import load_verified_dataset
from src.live_v8 import TradingCalendar, require
from src.ops_v9 import atomic_json, audit_dataset, generation_lock, sha256
from src.paper_ledger_v10 import PaperLedger
from src.signals_v8 import last_completed_session

BASE=Path(__file__).resolve().parent
FILES=('prices.csv','raw_prices.csv','trade_calendar.csv','daily_basic.csv','fundamentals_raw.csv',
       'index_membership.csv','stock_metadata.csv','industry_membership.csv','st_status.csv',
       'benchmark.csv','stock_limits.csv','corporate_actions.csv')


def paper_required_tickers(state_dir):
    if state_dir is None or not (Path(state_dir)/'paper.sqlite3').exists():
        return set()
    ledger=PaperLedger(Path(state_dir)/'paper.sqlite3')
    account=ledger.load_account('PAPER-CSI300-LIVE') or {}
    names=set(account.get('positions',{}))
    names|={str(x['ticker']) for x in account.get('share_receivables',{}).values() if x.get('ticker')}
    pending=ledger.load_pending('PAPER-CSI300-LIVE')
    if not pending.empty:names|=set(pending.ticker.astype(str))
    return names


def _read(directory, name, columns=None):
    p=Path(directory)/name
    return pd.read_csv(p,dtype={'ticker':str}) if p.exists() else pd.DataFrame(columns=columns)


def _merge(old,new,keys):
    x=pd.concat([old,new],ignore_index=True)
    date_columns={'date','ann_date','report_date','effective_date','in_date','out_date',
                  'record_date','ex_date','pay_date','div_listdate','imp_ann_date','end_date','base_date'}
    for c in set(x.columns)&date_columns:
        x[c]=pd.to_datetime(x[c],errors='raise',format='mixed')
    return x.drop_duplicates(keys,keep='last').sort_values(keys).reset_index(drop=True)


def download_free_dataset(directory, *, now=None, as_of=None, lookback_years=1,
                          financial_quarters=2, sleep=.05, state_dir=None, api=None):
    require(lookback_years>=1 and financial_quarters>=2, '免费数据至少一年行情、两个已结束财报季度')
    directory=Path(directory).resolve()
    directory.mkdir(parents=True,exist_ok=True)
    now=pd.Timestamp(now) if now is not None else pd.Timestamp.now(tz='Asia/Shanghai')
    if now.tz is None:now=now.tz_localize('Asia/Shanghai')
    else:now=now.tz_convert('Asia/Shanghai')
    with generation_lock(directory):
        old_manifest=None
        if (directory/'data_manifest_v8.json').exists():
            old_manifest,_=load_verified_dataset(directory)
            require(old_manifest['source']=='baostock', '不可把免费数据覆盖到Tushare目录；使用data/free')
        elif any((directory/n).exists() for n in FILES):
            raise ValueError('数据目录含无完整清单的旧文件；请使用新的空数据目录')
        with BaoStockDownloader(api,sleep=sleep) as dl:
            # A holiday run can have asof several days before today. Cover its
            # full lookback too, rather than losing the first price session.
            start=pd.Timestamp(now.date())-pd.DateOffset(years=lookback_years)-pd.Timedelta(days=60)
            cal=normalize_calendar(dl.query('query_trade_dates',start_date=str(start.date()),
                                            end_date=str((pd.Timestamp(now.date())+pd.Timedelta(days=40)).date())))
            calendar=TradingCalendar(cal)
            latest=last_completed_session(calendar,now)
            asof=pd.Timestamp(as_of).normalize() if as_of is not None else latest
            require(asof<=latest and calendar.is_open(asof), '免费数据日期未完成或休市')
            require(old_manifest is None or asof>=pd.Timestamp(old_manifest['as_of']), '禁止用历史下载覆盖较新的数据')
            start=asof-pd.DateOffset(years=lookback_years)
            current=normalize_members(dl.query('query_hs300_stocks',date=str(asof.date())),asof)
            members=_merge(_read(directory,'index_membership.csv',current.columns),current,['effective_date','ticker'])
            names=set(members.ticker)|paper_required_tickers(state_dir)
            tickers=sorted(names)
            basic_frames=[];industry_frames=[]
            for i,ticker in enumerate(tickers,1):
                code=code_from_ticker(ticker)
                basic_frames.append(dl.query('query_stock_basic',code=code))
                industry_frames.append(dl.query('query_stock_industry',code=code,date=str(asof.date())))
                if i%50==0 or i==len(tickers):print(f'免费证券资料: {i}/{len(tickers)}',flush=True)
            basic=pd.concat(basic_frames,ignore_index=True)
            require({'code','code_name','ipoDate','outDate'}<=set(basic), '证券基础资料缺字段')
            basic=basic[basic.code.str.match(r'^(sh|sz)\.\d{6}$')].copy()
            basic['ticker']=basic.code.map(ticker_from_code)
            meta=basic[basic.ticker.isin(names)].rename(columns={'code_name':'name','ipoDate':'list_date','outDate':'delist_date'})
            meta=meta[['ticker','name','list_date','delist_date']].copy()
            require(not meta.ticker.duplicated().any() and names<=set(meta.ticker), '证券资料遗漏当前/历史/持仓证券')
            industry_now=normalize_industry(pd.concat(industry_frames,ignore_index=True),asof)
            industry_now=industry_now[industry_now.ticker.isin(names)].copy()
            industry=merge_industry_snapshots(_read(directory,'industry_membership.csv',industry_now.columns),industry_now)

            old_raw=_read(directory,'raw_prices.csv')
            frames=[]
            for i,ticker in enumerate(tickers,1):
                code=code_from_ticker(ticker)
                previous=old_raw[old_raw.ticker==ticker].copy() if not old_raw.empty else pd.DataFrame()
                expand=old_manifest is not None and lookback_years>old_manifest.get('price_lookback_years',lookback_years)
                since=start if previous.empty or expand else max(start,pd.to_datetime(previous.date).max()-pd.Timedelta(days=7))
                bars=dl.query('query_history_k_data_plus',code=code,fields=RAW_FIELDS,
                              start_date=str(since.date()),end_date=str(asof.date()),frequency='d',adjustflag='3')
                factors=dl.query('query_adjust_factor',code=code,start_date='1990-01-01',end_date=str(asof.date()))
                if previous.empty:
                    if bars.empty:continue
                    raw=normalize_prices(bars,factors)
                else:
                    prev=previous.rename(columns={'pre_close':'preclose','is_st':'isST'}).copy()
                    prev['code']=code
                    columns=RAW_FIELDS.split(',')
                    bars=_merge(prev[columns],bars,['date','code'])
                    raw=normalize_prices(bars,factors)
                frames.append(raw)
                if i%25==0 or i==len(tickers):print(f'免费行情: {i}/{len(tickers)}',flush=True)
            require(bool(frames), '免费行情全部为空')
            raw=pd.concat(frames,ignore_index=True).sort_values(['ticker','date'])
            limits=derive_paper_limits(raw,meta,cal)
            db=raw[['ticker','date','peTTM','pbMRQ','turn']].rename(columns={'peTTM':'pe_ttm','pbMRQ':'pb','turn':'turnover_rate'})
            st=raw.loc[raw.is_st==1,['ticker','date']].rename(columns={'date':'trade_date'})
            st=st.assign(type='vendor_daily_isST')

            # Fetch the financials actually used by this MAIN-only variant.
            supported=set(limits.loc[pd.to_datetime(limits.date)==asof,'ticker'])
            positive=set(db.loc[(pd.to_datetime(db.date)==asof)&(db.pe_ttm>0)&(db.pb>0),'ticker'])
            financial_names=sorted(set(current.ticker)&supported&positive)
            quarters=pd.period_range(end=pd.Period(asof,freq='Q')-1,periods=financial_quarters,freq='Q')
            financial=[]
            for i,ticker in enumerate(financial_names,1):
                for q in quarters:
                    kwargs={'code':code_from_ticker(ticker),'year':q.year,'quarter':q.quarter}
                    profit=dl.query('query_profit_data',**kwargs)
                    growth=dl.query('query_growth_data',**kwargs)
                    balance=dl.query('query_balance_data',**kwargs)
                    f=normalize_financials(profit,growth,balance,asof)
                    if not f.empty:financial.append(f)
                if i%25==0 or i==len(financial_names):print(f'免费财报: {i}/{len(financial_names)}',flush=True)
            require(bool(financial), '免费可用财报全部为空')
            fundamentals=_merge(_read(directory,'fundamentals_raw.csv',financial[0].columns),
                                pd.concat(financial,ignore_index=True),['ticker','ann_date','report_date'])
            actions=[]
            for i,ticker in enumerate(tickers,1):
                for year in [asof.year-1,asof.year]:
                    f=dl.query('query_dividend_data',code=code_from_ticker(ticker),year=year,yearType='operate')
                    a=normalize_dividends(f,asof)
                    if not a.empty:actions.append(a)
                if i%50==0 or i==len(tickers):print(f'免费分红: {i}/{len(tickers)}',flush=True)
            current_actions=pd.concat(actions,ignore_index=True) if actions else normalize_dividends(pd.DataFrame(),asof)
            old_actions=_read(directory,'corporate_actions.csv',current_actions.columns)
            if not old_actions.empty and not current_actions.empty:
                for frame in [old_actions,current_actions]:
                    for col in ['record_date','ex_date']:frame[col]=pd.to_datetime(frame[col],errors='raise')
                joined=old_actions.merge(current_actions,on=['ticker','record_date','ex_date'],suffixes=('_old','_new'))
                require(joined.action_id_old.eq(joined.action_id_new).all(), '已缓存分红事件被供应商修订，停止自动更新并核对账本')
            corporate=_merge(old_actions,current_actions,['action_id'])
            b=dl.query('query_history_k_data_plus',code='sh.000300',fields='date,code,open,high,low,close,volume,amount',
                       start_date=str(start.date()),end_date=str(asof.date()),frequency='d',adjustflag='3')
            require(not b.empty, '沪深300基准为空')
            benchmark=b[['date','close']].copy()
            benchmark.date=pd.to_datetime(benchmark.date,errors='raise')
            benchmark.close=pd.to_numeric(benchmark.close,errors='raise')

        frames={'raw_prices.csv':raw,'prices.csv':to_research_prices(raw),'trade_calendar.csv':cal,
                'daily_basic.csv':db,'fundamentals_raw.csv':fundamentals,'index_membership.csv':members,
                'stock_metadata.csv':meta,'industry_membership.csv':industry,'st_status.csv':st,
                'benchmark.csv':benchmark,'stock_limits.csv':limits,'corporate_actions.csv':corporate}
        manifest={'schema_version':8,'source':'baostock','complete_universe':True,'as_of':str(asof.date()),
                  'created_at':now.isoformat(),'ticker_count':members.ticker.nunique(),'market_ticker_count':len(tickers),
                  'amount_unit':'CNY','volume_unit':'share','st_history_status':'vendor_daily_isST',
                  'refresh_mode':'incremental_bars_refresh_quarterly_financials_and_dividends',
                  'price_lookback_years':lookback_years,
                  'financial_quarters':financial_quarters,'financial_universe':'supported_non_ST_MAIN_positive_PE_PB',
                  'history_available_from':old_manifest.get('history_available_from',old_manifest['as_of']) if old_manifest else str(asof.date()),
                  'index_weight_basis':'membership_marker_only_not_actual_index_weight',
                  'industry_basis':'vendor_weekly_classification_snapshot_not_SW1_exact_changes',
                  'point_in_time_policy':{'financials':'latest joined pubDate strictly before signal_date',
                      'membership':'observed weekly snapshots only; no pre-bootstrap backfill',
                      'execution':'D close fixes shares; next completed session raw open sets fill price'},
                  **FREE_CONTRACT}
        stage=Path(tempfile.mkdtemp(prefix='.free_download_',dir=directory.parent))
        try:
            for name,frame in frames.items():frame.to_csv(stage/name,index=False)
            manifest['files']={n:sha256(stage/n) for n in FILES}
            atomic_json(stage/'data_manifest_v8.json',manifest)
            audit_now=now if as_of is None else pd.Timestamp(str(asof.date())+'T18:30:00',tz='Asia/Shanghai')
            audit=audit_dataset(stage,audit_now)
            require(audit['status']=='PASS', '免费数据审计未通过: '+json.dumps(audit['checks'],ensure_ascii=False))
            # The old bundle remains intact if download/normalization/audit fails.
            # During publication the completion manifest is removed first/last.
            (directory/'data_manifest_v8.json').unlink(missing_ok=True)
            for name in FILES:os.replace(stage/name,directory/name)
            os.replace(stage/'data_manifest_v8.json',directory/'data_manifest_v8.json')
        finally:shutil.rmtree(stage,ignore_errors=True)
        print(f'免费数据审计通过: {asof.date()}；当前沪深300完整300只；可用财务证券{fundamentals.ticker.nunique()}；无需Token',flush=True)
        return manifest


def main(argv=None):
    p=argparse.ArgumentParser(description='BaoStock匿名免费数据；独立免费版Paper口径')
    p.add_argument('--output',default=str(BASE/'data/free'))
    p.add_argument('--as-of',help='冻结历史下载；日常Paper仍要求最近完整交易日')
    p.add_argument('--lookback-years',type=int,default=1)
    p.add_argument('--financial-quarters',type=int,default=2)
    p.add_argument('--sleep',type=float,default=.05)
    p.add_argument('--paper-state-dir')
    args=p.parse_args(argv)
    download_free_dataset(args.output,as_of=args.as_of,lookback_years=args.lookback_years,
                          financial_quarters=args.financial_quarters,sleep=args.sleep,state_dir=args.paper_state_dir)
    return 0


if __name__=='__main__':raise SystemExit(main())
