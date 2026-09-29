"""Safety regressions: synthetic fixtures, never contact a broker."""
import copy,sqlite3
from dataclasses import replace
from types import SimpleNamespace
import numpy as np
import pandas as pd
import pytest
from src.live_v8 import *
from src.order_journal_v8 import OrderJournal,submit_reserved
from src.data_contract_v8 import normalize_tushare_prices,latest_financial_records
from src.accounting_backtest_v8 import run_accounting_backtest

@pytest.fixture
def inp():
    now=pd.Timestamp('2026-09-11T10:00:00+08:00')
    account={'source':'broker','risk_reference_date':'2026-09-10','account_id':'TEST_ONLY','as_of':now.isoformat(),'cash_available':100000.,'cash_total':100000.,'total_asset':100000.,
      'prior_close_nav_adjusted':100000.,'high_water_nav_adjusted':100000.,'open_order_count':0,'positions':[]}
    target=pd.DataFrame([{'ticker':'600000.SH','target_weight':.08,'signal_date':'2026-09-10','signal_close_raw':10.,'source':'real'}])
    quote=pd.DataFrame([{'ticker':'600000.SH','last':10.,'bid':9.99,'ask':10.01,'up_limit':11.,'down_limit':9.,'quote_time':now.isoformat(),
        'price_basis':'raw','status':'TRADING','adv20_cny':1e8,'industry_l1':'BANK','is_st':False,'is_delisting':False}])
    calendar=TradingCalendar(pd.DataFrame({'date':pd.date_range('2026-09-09','2026-09-12'),'is_open':[1,1,1,0]}))
    return account,target,quote,calendar,now

def test_fee_and_exposure(inp):
    p=make_plan(*inp);o=p['orders'][0]
    assert o['qty']==700 and o['estimated_fees']>=5
    assert o['notional']/(100000-o['estimated_fees'])<=.08

@pytest.mark.parametrize('weight',[-.1,np.nan,np.inf,1.01])
def test_bad_weight(inp,weight):
    inp[1].loc[0,'target_weight']=weight
    with pytest.raises(TradingBlocked):make_plan(*inp)

@pytest.mark.parametrize('date',['2026-09-11','2026-09-12','2026-09-09'])
def test_stale_or_future_signal(inp,date):
    inp[1].loc[0,'signal_date']=date
    with pytest.raises(TradingBlocked):make_plan(*inp)

@pytest.mark.parametrize('column,value',[
    ('quote_time','2026-09-11T09:59:29+08:00'),('quote_time','2026-09-11T10:00:01+08:00'),
    ('quote_time','2026-09-11 10:00:00'),('price_basis','qfq'),('last',np.nan),('industry_l1','UNKNOWN'),('status','UNKNOWN')])
def test_invalid_quote_blocks_entire_plan(inp,column,value):
    inp[2].loc[0,column]=value
    with pytest.raises((TradingBlocked,ValueError)):make_plan(*inp)

@pytest.mark.parametrize('column,value',[
    ('status','SUSPENDED'),('up_limit',10.01),('down_limit',0),('is_st',True),('is_delisting',True),('ask',10.2),('bid',10.02)])
def test_untradable_order(inp,column,value):
    inp[2].loc[0,column]=value;p=make_plan(*inp)
    assert not p['orders'] and p['blocked']

@pytest.mark.parametrize('date',['2026-09-11T09:29:59+08:00','2026-09-11T11:30:00+08:00','2026-09-11T12:00:00+08:00','2026-09-11T14:57:00+08:00','2026-09-12T10:00:00+08:00'])
def test_outside_session(inp,date):
    with pytest.raises(TradingBlocked):make_plan(*inp[:4],pd.Timestamp(date))

def test_no_weekday_calendar_fabrication():
    with pytest.raises(TradingBlocked):TradingCalendar(pd.DataFrame({'date':['2026-09-10','2026-09-14'],'is_open':[1,1]}))
    cal=TradingCalendar(pd.DataFrame({'date':pd.date_range('2026-10-01','2026-10-08'),'is_open':[0]*7+[1]}))
    assert not cal.is_open('2026-10-05')

def held(inp,qty=500,sellable=200,ticker='600000.SH'):
    a=inp[0];a['positions']=[{'ticker':ticker,'qty':qty,'sellable_qty':sellable,'last_price':10.}]
    a['cash_total']=a['cash_available']=100000-qty*10

def test_t1(inp):
    held(inp);inp[1].loc[0,'target_weight']=0
    assert make_plan(*inp)['orders'][0]['qty']==200
    inp[0]['positions'][0]['sellable_qty']=0
    assert not make_plan(*inp)['orders']

def test_missing_held_quote(inp):
    held(inp,ticker='000001.SZ')
    with pytest.raises(TradingBlocked):make_plan(*inp)

def test_pending_sales_never_fund_buys(inp):
    held(inp,10000,10000,'000001.SZ');q=inp[2];q.loc[1]=q.loc[0];q.loc[1,'ticker']='000001.SZ'
    p=make_plan(*inp)
    assert p['orders'] and all(o['side']=='SELL' for o in p['orders'])

def test_blocked_sale_still_counts_exposure(inp):
    held(inp,8000,8000,'000001.SZ');q=inp[2];q.loc[1]=q.loc[0];q.loc[1,'ticker']='000001.SZ';q.loc[1,'down_limit']=9.99
    p=make_plan(*inp);assert not p['orders']

def test_drawdown_reduce_only(inp):
    inp[0]['high_water_nav_adjusted']=120000
    p=make_plan(*inp);assert p['risk_reduce_only'] and not p['orders']

def test_kill_and_external_orders(inp):
    with pytest.raises(TradingBlocked):make_plan(*inp,kill_switch=True)
    inp[0]['open_order_count']=1
    with pytest.raises(TradingBlocked):make_plan(*inp)

def test_synthetic_data_cannot_be_live(inp):
    inp[0]['source']='synthetic'
    with pytest.raises(TradingBlocked):make_plan(*inp)
    p=make_plan(*inp,allow_demo=True);assert p['mode']=='DEMO'

def test_duplicate_target(inp):
    inp[1].loc[1]=inp[1].loc[0]
    with pytest.raises(TradingBlocked):make_plan(*inp)

def test_small_account(inp):
    for c in ['cash_total','cash_available','total_asset','prior_close_nav_adjusted','high_water_nav_adjusted']:inp[0][c]=1000
    assert not make_plan(*inp)['orders']

@pytest.mark.parametrize('ticker,raw,side,position,sellable,result',[
    ('688001.SH',399,'BUY',0,0,399),('688001.SH',199,'BUY',0,0,0),('600000.SH',399,'BUY',0,0,300),
    ('600000.SH',51,'SELL',51,51,51),('600000.SH',399,'SELL',1000,399,300),('688001.SH',199,'SELL',199,199,199)])
def test_board_lots(ticker,raw,side,position,sellable,result):
    assert round_quantity(raw,ticker,side,position,sellable)==result

@pytest.mark.parametrize('ticker',['600000','1.SZ','510300.SH','430001.BJ','688001.SZ'])
def test_security_scope(ticker):
    with pytest.raises(TradingBlocked):ticker_code(ticker)

def test_string_false():assert strict_bool('false') is False and strict_bool('0') is False

def test_used_daily_turnover(inp):assert not make_plan(*inp,used_turnover_cny=30000)['orders']

def test_plan_checksum(inp):
    p=make_plan(*inp);p['orders'][0]['qty']+=100
    with pytest.raises(TradingBlocked):verify_plan(p)

def test_same_day_id_independent_of_qty(inp):
    key=make_plan(*inp)['orders'][0]['client_order_id'];inp[1].loc[0,'target_weight']=.05
    assert make_plan(*inp)['orders'][0]['client_order_id']==key
    assert not make_plan(*inp,used_keys={key})['orders']

def test_journal_partial_cancel_and_idempotence(inp,tmp_path):
    p=make_plan(*inp);key=p['orders'][0]['client_order_id'];j=OrderJournal(tmp_path/'journal.db')
    try:
        j.reserve(p);j.update(key,'SUBMITTED','123');assert j.rows()[0]['filled_qty']==0
        with pytest.raises(TradingBlocked):j.context('TEST_ONLY','2026-09-11')
        j.update(key,'PARTIAL','123',100,10);j.update(key,'PARTIAL','123',100,10)
        with pytest.raises(TradingBlocked):j.update(key,'PARTIAL','123',99,10)
        j.update(key,'CANCELLED','123',100,10);keys,turn=j.context('TEST_ONLY','2026-09-11')
        assert key in keys and turn==pytest.approx(1001)
        with pytest.raises(sqlite3.IntegrityError):j.reserve(p)
    finally:j.close()

def test_timeout_survives_restart(inp,tmp_path):
    class Broker:
        calls=0
        def submit_checked(self,o):self.calls+=1;raise TimeoutError('unknown broker acceptance')
    b=Broker();path=tmp_path/'journal.db';p=make_plan(*inp);j=OrderJournal(path)
    assert submit_reserved(p,j,b)[0]['status']=='UNKNOWN';j.close();j=OrderJournal(path)
    try:
        with pytest.raises(TradingBlocked):submit_reserved(p,j,b)
        assert b.calls==1
    finally:j.close()

def test_demo_journal_rejected(inp,tmp_path):
    j=OrderJournal(tmp_path/'journal.db')
    try:
        with pytest.raises(TradingBlocked):j.reserve(make_plan(*inp,allow_demo=True))
    finally:j.close()

def test_price_units_and_missing_adjustment():
    daily=pd.DataFrame([dict(ts_code='600000.SH',trade_date='20260910',open=10,high=11,low=9,close=10,vol=123,amount=456)])
    adj=pd.DataFrame([dict(ts_code='600000.SH',trade_date='20260910',adj_factor=2)])
    row=normalize_tushare_prices(daily,adj).iloc[0]
    assert row['volume']==12300 and row['amount']==456000 and row['close']==10
    with pytest.raises(TradingBlocked):normalize_tushare_prices(daily,adj.iloc[:0])

def test_latest_period_wins_over_older_restatement():
    f=pd.DataFrame([['600000.SH','2026-08-20','2026-06-30',2],['600000.SH','2026-09-01','2025-12-31',999],['600000.SH','2026-09-10','2026-09-30',3]],columns=['ticker','ann_date','report_date','roe'])
    assert latest_financial_records(f,'2026-09-10').iloc[0]['roe']==2

def test_optimizer_never_relaxes_limits(monkeypatch):
    import src.portfolio_v5 as m
    g=pd.DataFrame({'ticker':['A','B'],'composite_score':[1.,0.]});cov=pd.DataFrame(np.eye(2)*.001,index=['A','B'],columns=['A','B'])
    with pytest.raises(ValueError):m.optimize_portfolio(g,cov,max_stock_weight=.08)
    monkeypatch.setattr(m,'minimize',lambda *a,**kw:SimpleNamespace(success=False,message='infeasible'))
    with pytest.raises(RuntimeError):m.optimize_portfolio(g,cov,max_stock_weight=.6)

def simulate(prices,targets,limits=None):
    return run_accounting_backtest(prices,targets,limits,commission_bps=0,slippage_bps=0,minimum_commission=0,transfer_fee_bps=0,use_price_limits=limits is not None)

def test_rebalance_keeps_overnight_return():
    p=pd.DataFrame({'date':pd.date_range('2026-09-08',periods=3),'ticker':'A','open':[10,10,11],'close':[10,10,11]})
    t=pd.DataFrame({'signal_date':['2026-09-08','2026-09-09'],'ticker':'A','target_weight':1.})
    ret,_,_=simulate(p,t);assert ret.iloc[-1]==pytest.approx(.1)

def test_holdings_drift_without_free_rebalance():
    days=pd.date_range('2026-09-07',periods=4)
    p=pd.DataFrame([(d,t,v,v) for t,vs in [('A',[10,10,20,40]),('B',[10,10,10,10])] for d,v in zip(days,vs)],columns=['date','ticker','open','close'])
    t=pd.DataFrame({'signal_date':[days[0],days[0]],'ticker':['A','B'],'target_weight':[.5,.5]})
    ret,_,_=simulate(p,t);assert (1+ret).prod()==pytest.approx(2.5)

def test_adjusted_price_uses_raw_limit_and_blocked_cash():
    days=pd.date_range('2026-09-07',periods=3)
    p=pd.DataFrame([(d,t,v,v,raw,raw) for t,v,raw in [('A',5,10),('B',10,20)] for d in days],columns=['date','ticker','open','close','raw_open','raw_close'])
    limits=pd.DataFrame([(d,t,up,dn) for d in days for t,up,dn in [('A',11,10 if d==days[-1] else 9),('B',22,18)]],columns=['date','ticker','up_limit','down_limit'])
    t=pd.DataFrame({'signal_date':days[:2],'ticker':['A','B'],'target_weight':1.})
    _,h,tr=simulate(p,t,limits)
    assert tr.iloc[-1]['blocked_sell_count']==1 and tr.iloc[-1]['buy_turnover']==0
    assert h.groupby('execution_date')['executed_weight'].sum().max()<=1+1e-8
    with pytest.raises(ValueError):simulate(p.drop(columns=['raw_open','raw_close']),t,limits)

def test_missing_held_bar_does_not_free_cash():
    days=pd.date_range('2026-09-07',periods=3)
    p=pd.DataFrame([(days[0],'A',10,10),(days[1],'A',10,10),(days[2],'B',10,10)],columns=['date','ticker','open','close'])
    t=pd.DataFrame({'signal_date':days[:2],'ticker':['A','B'],'target_weight':1.})
    ret,_,tr=simulate(p,t);assert tr.iloc[-1]['buy_turnover']==0 and ret.attrs['stale_valuation_days']==1

def test_ml_labels_must_have_matured(monkeypatch):
    import src.ml_ranker_elite as m
    x=pd.DataFrame({'signal_date':pd.to_datetime(['2026-01-01','2026-02-01','2026-03-01']),
        'forward_end_trade_date':pd.to_datetime(['2026-02-15','2026-03-15','2026-04-15']),'forward_1m_return':[.1,.2,.3]})
    captured=[]
    def fit(train,features):captured.append(train.copy());return object(),[]
    monkeypatch.setattr(m,'fit_walkforward_ranker',fit);monkeypatch.setattr(m,'predict_rank_score',lambda model,feats,test:pd.Series(0.,index=test.index))
    result=m.strict_walkforward_ml_scores(x,min_train_rows=1)
    assert len(captured)==1 and captured[0]['signal_date'].tolist()==[pd.Timestamp('2026-01-01')]
    assert result['signal_date'].tolist()==[pd.Timestamp('2026-03-01')]
    with pytest.raises(ValueError):m.strict_walkforward_ml_scores(x.drop(columns='forward_end_trade_date'))

@pytest.mark.parametrize('ticker',['600000.SH','000001.SZ','688001.SH','300001.SZ'])
def test_conservative_absolute_share_ceiling(ticker):
    assert round_quantity(2000000,ticker,'BUY')==100000
    assert round_quantity(2000001,ticker,'SELL',2000001,2000001)==100000

def test_broker_acceptance_requires_broker_id(inp,tmp_path):
    p=make_plan(*inp);j=OrderJournal(tmp_path/'journal.db')
    try:
        j.reserve(p)
        with pytest.raises(TradingBlocked):j.update(p['orders'][0]['client_order_id'],'SUBMITTED')
    finally:j.close()
