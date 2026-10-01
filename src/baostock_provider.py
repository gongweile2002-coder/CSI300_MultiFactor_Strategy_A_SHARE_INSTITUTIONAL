"""Anonymous BaoStock adapter for forward Paper; never a broker data feed.

Weekly constituent/industry snapshots are accumulated from the first download.
They are never backfilled over earlier dates. Financials use the latest vendor
quarterly observations and the latest publication date of all joined inputs.
The provider does not archive historical revisions independently.
"""
from __future__ import annotations

from decimal import Decimal, ROUND_HALF_UP
import re
import socket
import time

import numpy as np
import pandas as pd

from .corporate_actions_v10 import normalize_corporate_actions
from .live_v8 import board, require


RAW_FIELDS = 'date,code,open,high,low,close,preclose,volume,amount,turn,tradestatus,peTTM,pbMRQ,isST'
FREE_CONTRACT = {
    'provider_contract': 'baostock_free_v1',
    'paper_only': True,
    'financial_metric_basis': 'roeAvg_YOYNI_liabilityToAsset_percent',
    'size_neutralization': False,
    'limit_basis': 'derived_10pct_seasoned_non_st_main_paper_only',
    'dividend_cash_policy': 'gross_less_flat_20pct_model_withholding',
    'adjustment_basis': 'baostock_price_change_adjustment_not_total_return',
    'membership_basis': 'vendor_weekly_snapshots_observed_from_bootstrap_not_exact_intrawweek_changes',
    'financial_vintages': 'pubDate_vendor_latest_quarterly_history_not_independently_archived',
}


def ticker_from_code(code):
    require(bool(re.fullmatch(r'(sh|sz)\.\d{6}', str(code))), 'BaoStock 股票代码非法: '+str(code))
    market, number = str(code).split('.')
    return number+'.'+market.upper()


def code_from_ticker(ticker):
    require(bool(re.fullmatch(r'\d{6}\.(SH|SZ)', str(ticker))), '股票代码非法: '+str(ticker))
    number, market = str(ticker).split('.')
    return market.lower()+'.'+number


def result_frame(result, label):
    """Inspect both initial and final status, including page-fetch failures."""
    require(str(result.error_code)=='0', f'{label}: {result.error_code} {result.error_msg}')
    rows = []
    while result.next():
        rows.append(result.get_row_data())
    require(str(result.error_code)=='0', f'{label}: 分页/接收失败 {result.error_code} {result.error_msg}')
    return pd.DataFrame(rows, columns=result.fields)


def _dates(series):
    return pd.to_datetime(series.replace('', pd.NA), errors='raise').dt.normalize()


def normalize_calendar(frame):
    x = frame.rename(columns={'calendar_date':'date', 'is_trading_day':'is_open'}).copy()
    require({'date','is_open'}<=set(x) and not x.empty, '免费交易日历为空/缺字段')
    x['date'] = _dates(x['date'])
    x['is_open'] = pd.to_numeric(x['is_open'], errors='raise')
    require(x.date.notna().all() and not x.date.duplicated().any() and x.is_open.isin([0,1]).all(), '交易日历非法')
    return x[['date','is_open']].sort_values('date')


def normalize_members(frame, asof):
    require({'code','updateDate'}<=set(frame) and len(frame)==300, '免费沪深300快照必须完整300只')
    x = frame.rename(columns={'updateDate':'effective_date', 'code_name':'name'}).copy()
    x['effective_date'] = _dates(x['effective_date'])
    require(x.effective_date.nunique()==1 and x.effective_date.notna().all(), '成分更新日期不一致')
    require(x.effective_date.max()<=pd.Timestamp(asof) and
            (pd.Timestamp(asof)-x.effective_date.max()).days<=14, '免费成分返回未来/过旧快照')
    x['ticker'] = x.code.map(ticker_from_code)
    require(not x.ticker.duplicated().any(), '免费成分快照证券重复')
    # This field is only a membership marker, never a claimed CSI300 weight.
    x['weight'] = 1.0
    return x[['effective_date','ticker','weight']]


def normalize_industry(frame, asof):
    x = frame.copy()
    require({'updateDate','code','industry','industryClassification'}<=set(x) and not x.empty, '行业快照为空/缺字段')
    x['in_date'] = _dates(x.updateDate)
    require(x.in_date.notna().all() and x.in_date.max()<=pd.Timestamp(asof), '行业快照缺日期/来自未来')
    x['ticker'] = x.code.map(ticker_from_code)
    x['l1_name'] = x.industry
    x['classification'] = x.industryClassification
    x['out_date'] = pd.NaT
    require(not x.duplicated(['ticker','in_date']).any(), '行业快照重复')
    return x[['ticker','in_date','out_date','l1_name','classification']]


def merge_industry_snapshots(old, new):
    x = pd.concat([old,new], ignore_index=True).drop_duplicates(['ticker','in_date'],keep='last')
    x['in_date'] = pd.to_datetime(x.in_date)
    x = x.sort_values(['ticker','in_date'])
    x['out_date'] = x.groupby('ticker').in_date.shift(-1)-pd.Timedelta(days=1)
    return x.reset_index(drop=True)


def normalize_prices(frame, factors):
    x = frame.copy()
    require({'date','code','open','high','low','close','preclose','volume','amount','turn','tradestatus','peTTM','pbMRQ','isST'}<=set(x), '免费行情缺字段')
    require(not x.empty and x.code.nunique()==1, '单只股票行情为空/证券混用')
    x['ticker'] = x.code.map(ticker_from_code)
    x['date'] = _dates(x.date)
    require(x.date.notna().all() and not x.date.duplicated().any(), '行情日期缺失/重复')
    for c in ['open','high','low','close','preclose','volume','amount','turn','tradestatus','peTTM','pbMRQ','isST']:
        x[c] = pd.to_numeric(x[c].replace('',np.nan),errors='raise')
    require(x.tradestatus.isin([0,1]).all() and x.isST.isin([0,1]).all(), '交易/ST状态缺失或非法')
    # Suspended bars are omitted; never fill a tradable open or volume for them.
    x = x[(x.tradestatus==1)&(x.volume>0)].sort_values('date').copy()
    require(not x.empty, '证券历史中没有有效成交日')
    values=x[['open','high','low','close','preclose','volume','amount']]
    require(np.isfinite(values).all().all() and (values[['open','high','low','close']]>0).all().all()
            and (values[['preclose','volume','amount']]>=0).all().all(), '免费原始行情数值非法')
    require((x.low<=x[['open','close']].min(axis=1)).all() and
            (x.high>=x[['open','close']].max(axis=1)).all(), '免费原始OHLC关系非法')
    if factors.empty:
        # A successful all-history factor query with no events means no adjustment.
        x['adj_factor'] = 1.0
    else:
        require({'code','dividOperateDate','backAdjustFactor'}<=set(factors), '复权因子缺字段')
        f=factors.copy()
        require(set(f.code)==set(frame.code), '复权因子证券混用')
        f['factor_date']=_dates(f.dividOperateDate)
        f['adj_factor']=pd.to_numeric(f.backAdjustFactor,errors='raise')
        require(f.factor_date.notna().all() and not f.factor_date.duplicated().any() and
                np.isfinite(f.adj_factor).all() and (f.adj_factor>0).all(), '复权因子日期/数值非法')
        x=pd.merge_asof(x,f[['factor_date','adj_factor']].sort_values('factor_date'),
                        left_on='date',right_on='factor_date',direction='backward')
        x['adj_factor']=x.adj_factor.fillna(1.0) # before the first reported adjustment
        x=x.drop(columns='factor_date')
    x['price_basis']='raw';x['amount_unit']='CNY';x['volume_unit']='share'
    x=x.rename(columns={'preclose':'pre_close','isST':'is_st'})
    return x.drop(columns=['code']).reset_index(drop=True)


def to_research_prices(raw):
    x=raw.copy()
    for c in ['open','high','low','close']:
        x['raw_'+c]=x[c]
        x[c]=x[c]*x.adj_factor
    x['price_basis']='adjusted_research_only'
    return x


def normalize_financials(profit, growth, balance, asof):
    columns=['ticker','ann_date','report_date','roe','profit_growth','debt_ratio']
    if any(x.empty for x in [profit,growth,balance]):
        return pd.DataFrame(columns=columns)
    parts=[]
    for i,(frame,field,name) in enumerate([(profit,'roeAvg','roe'),(growth,'YOYNI','profit_growth'),(balance,'liabilityToAsset','debt_ratio')]):
        require({'code','pubDate','statDate',field}<=set(frame), '财报缺字段: '+field)
        x=frame[['code','pubDate','statDate',field]].copy()
        require(not x.duplicated(['code','statDate']).any(), '单次季度接口返回重复财报版本')
        x['pubDate']=_dates(x.pubDate);x['statDate']=_dates(x.statDate)
        require(x.pubDate.notna().all() and x.statDate.notna().all() and
                (x.statDate<=x.pubDate).all(), '财务公告/报告期非法')
        x[field]=pd.to_numeric(x[field].mask(x[field].eq(''),np.nan),errors='raise')*100
        parts.append(x.rename(columns={field:name,'pubDate':f'ann_{i}'}))
    x=parts[0].merge(parts[1],on=['code','statDate'],validate='one_to_one').merge(parts[2],on=['code','statDate'],validate='one_to_one')
    # A factor becomes available only after ALL three component publications.
    x['ann_date']=x[['ann_0','ann_1','ann_2']].max(axis=1)
    x['ticker']=x.code.map(ticker_from_code);x['report_date']=x.statDate
    x=x[(x.ann_date<=pd.Timestamp(asof)) & np.isfinite(x[['roe','profit_growth','debt_ratio']]).all(axis=1)]
    return x[columns].sort_values(['ticker','report_date','ann_date']).reset_index(drop=True)


def normalize_dividends(frame, asof):
    if frame.empty:
        return normalize_corporate_actions(None)
    required={'code','dividRegistDate','dividOperateDate','dividPayDate','dividStockMarketDate',
              'dividCashPsBeforeTax','dividStocksPs','dividReserveToStockPs'}
    require(required<=set(frame), '免费分红接口缺字段')
    x=frame.copy()
    for c in ['dividRegistDate','dividOperateDate','dividPayDate','dividStockMarketDate']:
        x[c]=_dates(x[c])
    # Proposals have no enforceable ex date; only completed implementations apply.
    x=x[x.dividOperateDate.notna() & (x.dividOperateDate<=pd.Timestamp(asof))].copy()
    require(x.dividRegistDate.notna().all() and (x.dividRegistDate<=x.dividOperateDate).all(), '已实施分红缺登记日或日期非法')
    x['ticker']=x.code.map(ticker_from_code)
    for c in ['dividCashPsBeforeTax','dividStocksPs','dividReserveToStockPs']:
        x[c]=pd.to_numeric(x[c].mask(x[c].eq(''),0),errors='raise')
        require(np.isfinite(x[c]).all() and (x[c]>=0).all(), '分红比例非法: '+c)
    x['div_proc']='实施';x['record_date']=x.dividRegistDate;x['ex_date']=x.dividOperateDate
    x['pay_date']=x.dividPayDate;x['div_listdate']=x.dividStockMarketDate
    x['stk_bo_rate']=x.dividStocksPs;x['stk_co_rate']=x.dividReserveToStockPs
    x['stk_div']=x.stk_bo_rate+x.stk_co_rate
    x['cash_div_tax']=x.dividCashPsBeforeTax
    # Vendor after-tax strings may contain alternatives (e.g. "0.378或0.42").
    # Use an explicit conservative MODEL, never coerce them to zero/claim actual tax.
    x['cash_div']=x.cash_div_tax*.8
    require(((x.cash_div_tax==0)|x.pay_date.notna()).all(), '现金分红缺派息日')
    require(((x.stk_div==0)|x.div_listdate.notna()).all(), '送转股缺上市日')
    x['ann_date']=pd.NaT
    for c in ['dividPlanDate','dividPlanAnnounceDate','dividAgmPumDate']:
        if c in x:
            x['ann_date']=x.ann_date.fillna(_dates(x[c]))
    require(x.ann_date.notna().all() and (x.ann_date<=x.ex_date).all(), '已实施分红缺可用公告日期')
    x['imp_ann_date']=x.ann_date
    out=normalize_corporate_actions(x)
    out['cash_policy']='gross_less_flat_20pct_model_withholding'
    return out


def derive_paper_limits(raw, metadata, calendar):
    """Normal, seasoned, non-ST MAIN shares only; not exchange-confirmed limits.

    Suspensions, IPOs, other boards, abnormal-day bars and ST/delisting securities
    receive no inferred limit. The Paper engine rejects missing limits.
    """
    meta=metadata.set_index('ticker')
    sessions=pd.DatetimeIndex(sorted(pd.to_datetime(calendar.loc[calendar.is_open==1,'date'])))
    rows=[]
    for r in raw.itertuples():
        if board(r.ticker)!='MAIN' or int(r.is_st)!=0 or r.ticker not in meta.index:
            continue
        name=str(meta.loc[r.ticker,'name'])
        if 'ST' in name.upper() or '退' in name:
            continue
        ipo=pd.to_datetime(meta.loc[r.ticker,'list_date'],errors='coerce')
        if pd.isna(ipo) or (pd.Timestamp(r.date)-ipo).days<365 or float(r.pre_close)<=0:
            continue
        # Calendar only spans the lookback. Calendar age is a conservative check
        # for recent IPOs; old listed shares pass the one-year seasoning rule.
        if ipo>=sessions.min() and ((sessions>=ipo)&(sessions<=r.date)).sum()<180:
            continue
        pre=Decimal(str(r.pre_close))
        up=float((pre*Decimal('1.10')).quantize(Decimal('.01'),rounding=ROUND_HALF_UP))
        down=float((pre*Decimal('.90')).quantize(Decimal('.01'),rounding=ROUND_HALF_UP))
        if r.low<down-.005 or r.high>up+.005:
            continue
        rows.append({'date':r.date,'ticker':r.ticker,'pre_close':r.pre_close,
                     'up_limit':up,'down_limit':down,'limit_source':'derived_paper_model'})
    return pd.DataFrame(rows,columns=['date','ticker','pre_close','up_limit','down_limit','limit_source'])


def guard_held_adjustments(raw, actions, held_tickers, asof):
    """Stop for held-share rights/other adjustment events not explained by dividends."""
    for ticker in held_tickers:
        hist=raw[(raw.ticker==ticker)&(pd.to_datetime(raw.date)<=pd.Timestamp(asof))].sort_values('date').tail(2)
        if len(hist)<2 or pd.Timestamp(hist.iloc[-1].date)!=pd.Timestamp(asof):
            continue
        before, today=hist.iloc[0],hist.iloc[1]
        if np.isclose(before.adj_factor,today.adj_factor,rtol=1e-9,atol=1e-12):
            continue
        event=actions[(actions.ticker==ticker)&(pd.to_datetime(actions.ex_date)==pd.Timestamp(asof))]
        require(len(event)==1, ticker+': 持仓复权变化没有唯一已核实分红事件；停止Paper，排查配股/拆并股')
        r=event.iloc[0]
        expected=(float(before.close)-float(r.cash_div_tax))/(1+float(r.stk_div))
        require(abs(expected-float(today.pre_close))<=.02, ticker+': 持仓除权无法由已知分红解释，停止Paper')


class BaoStockDownloader:
    def __init__(self, api=None, sleep=.05, timeout=20):
        if api is None:
            import baostock
            api=baostock
        self.api=api;self.sleep=float(sleep);self.timeout=float(timeout)
        require(self.sleep>=0 and self.timeout>0, '请求节流/超时参数非法')

    def __enter__(self):
        self.old_timeout=socket.getdefaulttimeout()
        socket.setdefaulttimeout(self.timeout)
        r=self.api.login() # documented anonymous session; no user's credentials
        if str(r.error_code)!='0':
            socket.setdefaulttimeout(self.old_timeout)
            raise RuntimeError('BaoStock匿名连接失败；需要允许TCP 10030的网络/GitHub runner: '+str(r.error_msg))
        return self

    def __exit__(self, *args):
        try:self.api.logout()
        finally:socket.setdefaulttimeout(self.old_timeout)

    def query(self, method, **kwargs):
        result=getattr(self.api,method)(**kwargs)
        out=result_frame(result,method)
        time.sleep(self.sleep)
        return out
