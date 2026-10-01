"""Synthetic adapter/safety tests. No network or performance claims."""
from pathlib import Path
import json
import shutil

import pandas as pd
import pytest

from daily_paper import _refresh, bind_provider
from src.baostock_provider import (
    FREE_CONTRACT, derive_paper_limits, guard_held_adjustments,
    merge_industry_snapshots, normalize_calendar, normalize_dividends,
    normalize_financials, normalize_industry, normalize_members,
    normalize_prices, result_frame, to_research_prices,
)
from src.data_contract_v8 import latest_financial_records
from src.live_v8 import RiskConfig, TradingBlocked, TradingCalendar
from src.ops_v9 import atomic_json, audit_dataset, sha256, verify_signal_bundle
from src.signals_v8 import generate_signals
from src.strategy_lab_v9_4 import load_lab_inputs
from test_ops_v9 import market_fixture, NOW, TRADE_NOW


def bars():
    return pd.DataFrame([
        ['2026-09-28','sh.600000','9','9.2','8.9','9.1','9','100','910','1','1','6','.5','0'],
        ['2026-09-29','sh.600000','0','0','0','0','9.1','0','0','0','0','','','0'],
        ['2026-09-30','sh.600000','9','9.3','8.9','9.2','9.1','200','1840','2','1','6','.5','0'],
    ],columns='date,code,open,high,low,close,preclose,volume,amount,turn,tradestatus,peTTM,pbMRQ,isST'.split(','))


def factors():
    return pd.DataFrame({'code':['sh.600000']*2,'dividOperateDate':['2026-09-30','2026-10-09'],
                         'backAdjustFactor':['2','3']})


def test_raw_units_no_suspension_fill_and_backward_adjustments():
    raw=normalize_prices(bars(),factors())
    assert raw.date.tolist()==[pd.Timestamp('2026-09-28'),pd.Timestamp('2026-09-30')]
    assert raw.adj_factor.tolist()==[1.,2.]
    assert raw.volume.tolist()==[100,200] and raw.amount.tolist()==[910,1840]
    adjusted=to_research_prices(raw)
    assert adjusted.iloc[-1].close==pytest.approx(18.4)
    assert adjusted.iloc[-1].raw_close==pytest.approx(9.2)
    assert raw.price_basis.eq('raw').all() and adjusted.price_basis.eq('adjusted_research_only').all()


@pytest.mark.parametrize('column,value',[('isST',''),('tradestatus','2'),('open','-1'),('high','8')])
def test_invalid_raw_status_and_prices_stop(column,value):
    x=bars();x.loc[0,column]=value
    with pytest.raises(TradingBlocked):normalize_prices(x,factors())


def test_future_duplicate_or_wrong_security_factor_stops():
    for f in [pd.concat([factors(),factors().iloc[[0]]]),factors().assign(code='sz.000001')]:
        with pytest.raises(TradingBlocked):normalize_prices(bars(),f)


def financial(field,value,ann):
    return pd.DataFrame({'code':['sh.600000'],'statDate':['2026-06-30'],
                         'pubDate':[ann],field:[value]})


def test_quality_waits_for_all_publications_and_strict_next_day():
    f=normalize_financials(financial('roeAvg','.08','2026-08-20'),
                          financial('YOYNI','.20','2026-08-22'),
                          financial('liabilityToAsset','.40','2026-08-25'),'2026-09-30')
    assert f.iloc[0].ann_date==pd.Timestamp('2026-08-25')
    assert f.iloc[0].roe==8 and f.iloc[0].profit_growth==20 and f.iloc[0].debt_ratio==40
    assert latest_financial_records(f,'2026-08-25').empty
    assert len(latest_financial_records(f,'2026-08-26'))==1
    assert normalize_financials(financial('roeAvg','.08','2026-08-20'),
                               financial('YOYNI','.20','2026-08-22'),
                               financial('liabilityToAsset','.40','2026-10-10'),'2026-09-30').empty


def test_missing_financials_are_excluded_not_zero_filled():
    f=normalize_financials(financial('roeAvg','','2026-08-20'),
                          financial('YOYNI','.2','2026-08-20'),
                          financial('liabilityToAsset','.4','2026-08-20'),'2026-09-30')
    assert f.empty


def dividend():
    return pd.DataFrame([{'code':'sh.600000','dividPlanDate':'2026-07-10',
        'dividRegistDate':'2026-07-15','dividOperateDate':'2026-07-16','dividPayDate':'2026-07-16',
        'dividStockMarketDate':'','dividCashPsBeforeTax':'.42','dividCashPsAfterTax':'0.378或0.42',
        'dividStocksPs':'0','dividReserveToStockPs':''}])


def test_ambiguous_tax_text_uses_explicit_conservative_cash_model():
    a=normalize_dividends(dividend(),'2026-09-30')
    assert a.iloc[0].cash_div_tax==pytest.approx(.42)
    assert a.iloc[0].cash_div==pytest.approx(.336)
    assert a.iloc[0].cash_policy=='gross_less_flat_20pct_model_withholding'
    assert len(a.action_id.iloc[0])==64


@pytest.mark.parametrize('column,value',[('dividPayDate',''),('dividRegistDate',''),('dividCashPsBeforeTax','unknown')])
def test_incomplete_implemented_cash_action_stops(column,value):
    d=dividend();d.loc[0,column]=value
    with pytest.raises((TradingBlocked,ValueError)):normalize_dividends(d,'2026-09-30')


def test_share_action_requires_listing_date_and_adds_bonus_and_reserves():
    d=dividend();d.loc[0,'dividStocksPs']='.1';d.loc[0,'dividReserveToStockPs']='.2'
    with pytest.raises(TradingBlocked):normalize_dividends(d,'2026-09-30')
    d.loc[0,'dividStockMarketDate']='2026-07-16'
    assert normalize_dividends(d,'2026-09-30').iloc[0].stk_div==pytest.approx(.3)


def test_member_snapshot_is_complete_and_never_backdated():
    m=pd.DataFrame({'code':[f'sh.{600001+i}' for i in range(300)],'updateDate':'2026-09-28'})
    x=normalize_members(m,'2026-09-30')
    assert x.effective_date.eq(pd.Timestamp('2026-09-28')).all()
    with pytest.raises(TradingBlocked):normalize_members(m,'2026-09-25')
    with pytest.raises(TradingBlocked):normalize_members(m.iloc[:299],'2026-09-30')


def test_weekly_industry_intervals_do_not_assign_new_classification_to_past():
    old=normalize_industry(pd.DataFrame({'code':['sh.600000'],'updateDate':['2026-09-21'],
                         'industry':['Old'],'industryClassification':['CSRC']}),'2026-09-30')
    new=normalize_industry(pd.DataFrame({'code':['sh.600000'],'updateDate':['2026-09-28'],
                         'industry':['New'],'industryClassification':['CSRC']}),'2026-09-30')
    x=merge_industry_snapshots(old,new)
    assert x.iloc[0].out_date==pd.Timestamp('2026-09-27') and x.iloc[1].in_date==pd.Timestamp('2026-09-28')


def test_only_supported_seasoned_main_bars_receive_model_limits():
    raw=normalize_prices(bars(),pd.DataFrame())
    raw.loc[0,'pre_close']=9.15
    meta=pd.DataFrame({'ticker':['600000.SH'],'name':['SYNTHETIC'],'list_date':['2000-01-01']})
    cal=pd.DataFrame({'date':pd.bdate_range('2024-01-01','2026-09-30'),'is_open':1})
    limits=derive_paper_limits(raw,meta,cal)
    assert limits.iloc[0].up_limit==10.07 # HALF_UP, never bankers' round
    raw.loc[0,'is_st']=1
    assert len(derive_paper_limits(raw,meta,cal))==1
    raw.loc[1,'high']=50
    assert derive_paper_limits(raw,meta,cal).empty
    meta['list_date']='2026-09-01';raw.loc[0,'is_st']=0
    assert derive_paper_limits(raw,meta,cal).empty


def test_unexplained_held_adjustment_blocks_paper():
    raw=normalize_prices(bars(),factors())
    actions=normalize_dividends(pd.DataFrame(),'2026-09-30')
    with pytest.raises(TradingBlocked,match='复权变化'):
        guard_held_adjustments(raw,actions,{'600000.SH'},'2026-09-30')
    guard_held_adjustments(raw,actions,set(),'2026-09-30')


def test_page_error_cannot_publish_partial_vendor_rows():
    class BrokenPage:
        error_code='0';error_msg='';fields=['code']
        def next(self):
            self.error_code='network_failure';self.error_msg='SYNTHETIC failure'
            return False
    with pytest.raises(TradingBlocked,match='分页'):
        result_frame(BrokenPage(),'SYNTHETIC')


def test_provider_binding_prevents_cross_variant_ledger_reuse(tmp_path):
    bind_provider(tmp_path,'tushare')
    with pytest.raises(ValueError,match='数据源不同'):bind_provider(tmp_path,'baostock')
    fresh=tmp_path/'free';bind_provider(fresh,'baostock');bind_provider(fresh,'baostock')
    legacy=tmp_path/'legacy';legacy.mkdir();(legacy/'paper.sqlite3').touch()
    with pytest.raises(ValueError,match='旧账本'):bind_provider(legacy,'baostock')


def test_free_refresh_invokes_no_tushare_script(monkeypatch,tmp_path):
    calls=[]
    monkeypatch.setattr('daily_paper.subprocess.run',lambda cmd,**kw:calls.append(cmd))
    _refresh(tmp_path/'data',2,tmp_path/'paper')
    assert Path(calls[0][1]).name=='download_free_data.py'
    _refresh(tmp_path/'data',2,tmp_path/'paper','tushare')
    assert Path(calls[1][1]).name=='update_live_data.py'


def test_free_full_signal_is_labelled_and_rejected_by_broker_reader(market_fixture,tmp_path):
    d=tmp_path/'SYNTHETIC_FREE_ONLY';shutil.copytree(market_fixture,d)
    m=json.loads((d/'data_manifest_v8.json').read_text())
    m.update(source='baostock',**FREE_CONTRACT)
    atomic_json(d/'data_manifest_v8.json',m)
    assert audit_dataset(d,NOW)['source']=='baostock'
    out=tmp_path/'signals'
    report=generate_signals(d,out,NOW,RiskConfig())
    assert report['data_source']=='baostock' and report['paper_only'] is True
    assert report['size_neutralization'] is False and len(pd.read_csv(out/'targets.csv'))==30
    bundle=json.loads((out/'signal_manifest_v9.json').read_text())
    assert bundle['kind']=='free_paper_signal_bundle'
    cal=TradingCalendar(pd.read_csv(d/'trade_calendar.csv'))
    with pytest.raises(TradingBlocked):verify_signal_bundle(out/'targets.csv',RiskConfig(),cal,TRADE_NOW)
    with pytest.raises(TradingBlocked,match='原版策略对比'):load_lab_inputs(d)


def test_free_manifest_cannot_claim_unknown_financial_policy(market_fixture,tmp_path):
    d=tmp_path/'SYNTHETIC_BROKEN_FREE';shutil.copytree(market_fixture,d)
    m=json.loads((d/'data_manifest_v8.json').read_text());m.update(source='baostock',**FREE_CONTRACT)
    m['financial_metric_basis']='SYNTHETIC_unknown'
    atomic_json(d/'data_manifest_v8.json',m)
    assert audit_dataset(d,NOW)['status']=='BLOCK'


def test_calendar_does_not_invent_holiday_sessions():
    dates=pd.date_range('2026-09-30','2026-10-09')
    c=normalize_calendar(pd.DataFrame({'calendar_date':dates.strftime('%Y-%m-%d'),
                                      'is_trading_day':[1]+[0]*8+[1]}))
    assert not TradingCalendar(c).is_open('2026-10-01')
    c.loc[0,'is_open']=2
    with pytest.raises(TradingBlocked):normalize_calendar(c)


def test_free_risk_configuration_uses_current_execution_schema():
    cfg=json.loads((Path(__file__).resolve().parents[1]/'config/free.example.json').read_text())
    risk=RiskConfig.from_dict(cfg['risk']).validated()
    assert risk.strategy_id=='CSI300_FREE_QUARTERLY_CORE'
    assert cfg['live_enabled'] is False


class FakeResult:
    error_code='0';error_msg='SYNTHETIC ONLY'
    def __init__(self, frame):
        self.fields=list(frame.columns);self.rows=iter(frame.astype(str).values.tolist())
    def next(self):
        self.row=next(self.rows,None)
        return self.row is not None
    def get_row_data(self):return self.row


class FakeAPI:
    """Exercise the complete anonymous download contract without network access."""
    codes=[f'sh.{600001+i}' for i in range(300)]
    fail=False
    def login(self):return FakeResult(pd.DataFrame())
    def logout(self):return FakeResult(pd.DataFrame())
    def query_trade_dates(self,start_date,end_date):
        dates=pd.date_range(start_date,end_date)
        opened=[int(d.weekday()<5 and not pd.Timestamp('2026-10-01')<=d<=pd.Timestamp('2026-10-08')) for d in dates]
        return FakeResult(pd.DataFrame({'calendar_date':dates.strftime('%Y-%m-%d'),'is_trading_day':opened}))
    def query_hs300_stocks(self,date):
        return FakeResult(pd.DataFrame({'code':self.codes,'updateDate':'2026-09-28','code_name':'SYNTHETIC'}))
    def query_stock_basic(self):
        return FakeResult(pd.DataFrame({'code':self.codes,'code_name':'SYNTHETIC','ipoDate':'2000-01-01','outDate':''}))
    def query_stock_industry(self,date):
        return FakeResult(pd.DataFrame({'code':self.codes,'updateDate':'2026-09-28',
            'industry':[f'SYNTHETIC{i%5}' for i in range(300)],'industryClassification':'SYNTHETIC'}))
    def query_history_k_data_plus(self,code,fields,start_date,end_date,**kwargs):
        dates=pd.bdate_range(end='2026-09-30',periods=190)
        dates=dates[(dates>=pd.Timestamp(start_date))&(dates<=pd.Timestamp(end_date))]
        j=int(code[-6:])%300
        close=pd.Series([(10+j/30)*(1.0003+j%17*.00002)**i for i in range(len(dates))])
        x=pd.DataFrame({'date':dates.strftime('%Y-%m-%d'),'code':code,'open':close*.999,
            'high':close*1.002,'low':close*.998,'close':close,'preclose':close,'volume':1000000,
            'amount':close*1000000,'turn':1,'tradestatus':1,'peTTM':8+j%19,'pbMRQ':1+j%11/10,'isST':0})
        return FakeResult(x[fields.split(',')])
    def query_adjust_factor(self,**kwargs):return FakeResult(pd.DataFrame(columns=['code','dividOperateDate','backAdjustFactor']))
    def _financial(self,field,value,code,year,quarter):
        q=pd.Period(f'{year}Q{quarter}').end_time.normalize()
        return FakeResult(pd.DataFrame({'code':[code],'statDate':[str(q.date())],
            'pubDate':['2026-08-25'],field:[value]}))
    def query_profit_data(self,**kwargs):return self._financial('roeAvg',.05+int(kwargs['code'][-3:])%13*.01,**kwargs)
    def query_growth_data(self,**kwargs):
        if self.fail:raise RuntimeError('SYNTHETIC interrupted vendor request')
        return self._financial('YOYNI',.02+int(kwargs['code'][-3:])%23*.01,**kwargs)
    def query_balance_data(self,**kwargs):return self._financial('liabilityToAsset',.2+int(kwargs['code'][-3:])%37*.01,**kwargs)
    def query_dividend_data(self,**kwargs):return FakeResult(pd.DataFrame())


def test_full_anonymous_download_audit_signal_and_failed_refresh_keep_prior_bundle(tmp_path):
    from download_free_data import download_free_dataset
    directory=tmp_path/'SYNTHETIC_FULL_ADAPTER_ONLY'
    api=FakeAPI()
    now=pd.Timestamp('2026-10-01T20:00:00+08:00')
    m=download_free_dataset(directory,now=now,api=api,sleep=0)
    assert m['source']=='baostock' and m['as_of']=='2026-09-30'
    assert m['ticker_count']==300 and m['history_available_from']=='2026-09-30'
    # Explicitly label injected data so it is never a purported market dataset.
    m['fixture_notice']='SYNTHETIC TEST INPUT, NEVER REAL MARKET DATA'
    atomic_json(directory/'data_manifest_v8.json',m)
    assert audit_dataset(directory,now)['status']=='PASS'
    report=generate_signals(directory,tmp_path/'signals',now,RiskConfig())
    assert report['selected_names']==30 and report['paper_only'] is True
    saved_hash=sha256(directory/'data_manifest_v8.json')
    api.fail=True
    with pytest.raises(RuntimeError,match='interrupted'):
        download_free_dataset(directory,now=now,api=api,sleep=0)
    assert sha256(directory/'data_manifest_v8.json')==saved_hash
    assert audit_dataset(directory,now)['status']=='PASS'
