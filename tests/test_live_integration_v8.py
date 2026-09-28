"""End-to-end fixtures and SDK contracts. All data below is synthetic."""
from dataclasses import replace
from types import SimpleNamespace
import pandas as pd
import pytest
from src.live_v8 import TradingCalendar,RiskConfig,make_plan,TradingBlocked

def test_signal_to_reviewed_orders(tmp_path,monkeypatch):
    from run_ashare_elite_demo import inputs,BASE
    import src.signals_v8 as m
    prices,db,f,mem,meta,industry,st=inputs();signal=prices['date'].max()
    prices['raw_open']=prices['open'];prices['raw_close']=prices['close'];prices['price_basis']='synthetic_unadjusted'
    raw=prices.copy();raw['price_basis']='raw';raw['amount_unit']='CNY';raw['volume_unit']='share'
    # Fill the synthetic valuation snapshot at the selected fixture date.
    latest=db.sort_values('date').groupby('ticker').tail(1).copy();latest['date']=signal
    db=pd.concat([db[db['date']!=signal],latest],ignore_index=True)
    datasets={'prices.csv':prices,'raw_prices.csv':raw,'daily_basic.csv':db,'fundamentals_raw.csv':f,'index_membership.csv':mem,
      'stock_metadata.csv':meta,'industry_membership.csv':industry,'st_status.csv':st,
      'benchmark.csv':pd.read_csv(BASE/'data/demo_benchmark.csv').rename(columns={'benchmark_close':'close'})}
    for name,frame in datasets.items():frame.to_csv(tmp_path/name,index=False)
    days=pd.date_range(signal-pd.Timedelta(days=10),signal+pd.Timedelta(days=5))
    cal=TradingCalendar(pd.DataFrame({'date':days,'is_open':(days.weekday<5).astype(int)}))
    # Replace the authenticated-data contract in the test process only. Do not write a forged real-data manifest.
    monkeypatch.setattr(m,'load_verified_dataset',lambda path:({'as_of':str(signal.date()),'membership_basis':'SYNTHETIC_FIXTURE'},cal))
    cfg=RiskConfig();out=tmp_path/'signal'
    report=m._generate_signals_unchecked(tmp_path,out,pd.Timestamp(str(signal.date())+'T18:00:00+08:00'),cfg)
    assert report['selected_names']>=15 and report['gross_target']<=.8+1e-8 and report['performance_validated'] is False
    targets=pd.read_csv(out/'targets.csv');targets['source']='synthetic'
    ref=pd.read_csv(out/'execution_reference.csv').set_index('ticker');now=pd.Timestamp(str(cal.next(signal).date())+'T10:00:00+08:00')
    quotes=[]
    for t in targets['ticker']:
        r=ref.loc[t];px=round(r['signal_close_raw'],2)
        quotes.append({'ticker':t,'quote_time':now.isoformat(),'price_basis':'raw','last':px,'bid':px-.01,'ask':px+.01,
          'up_limit':round(px*1.1,2),'down_limit':round(px*.9,2),'status':'TRADING','is_st':False,'is_delisting':False,
          'industry_l1':r['industry_l1'],'adv20_cny':r['adv20_cny']})
    a={'source':'synthetic','account_id':'INTEGRATION_TEST_ONLY','as_of':now.isoformat(),'cash_available':500000.,'cash_total':500000.,'total_asset':500000.,
      'prior_close_nav_adjusted':500000.,'high_water_nav_adjusted':500000.,'positions':[],'open_order_count':0}
    p=make_plan(a,targets,pd.DataFrame(quotes),cal,now,cfg,allow_demo=True)
    assert p['orders'] and sum(o['notional'] for o in p['orders'])/500000<=.3+1e-8
    assert sum(o['notional']+o['estimated_fees'] for o in p['orders'])<=500000

def test_real_provider_separates_split_prices(tmp_path):
    from src.tushare_provider import TushareDownloader
    class Pro:
        def daily(self,**kwargs):return pd.DataFrame([dict(ts_code='600000.SH',trade_date=d,open=v,high=v,low=v,close=v,vol=1000,amount=100) for d,v in [('20260909',10),('20260910',5)]])
        def adj_factor(self,**kwargs):return pd.DataFrame([dict(ts_code='600000.SH',trade_date=d,adj_factor=a) for d,a in [('20260909',1),('20260910',2)]])
    dl=object.__new__(TushareDownloader);dl.pro=Pro();dl.output_dir=tmp_path;dl.sleep_seconds=0
    result=dl.fetch_prices_for_tickers(['600000.SH'],'2026-09-09','2026-09-10')
    assert result['close'].tolist()==[10,10] and result['raw_close'].tolist()==[10,5]
    raw=pd.read_csv(tmp_path/'raw_prices.csv');assert raw['close'].tolist()==[10,5] and raw['amount'].eq(100000).all()

def test_cashflow_ttm_and_cny_marketcap():
    from src.fundamental_extension_institutional import add_cash_quality_factors
    cf=pd.DataFrame([
      ['600000.SH','2025-08-20','2025-06-30',20e6,40e6,30e6],
      ['600000.SH','2026-04-20','2025-12-31',40e6,80e6,60e6],
      ['600000.SH','2026-08-20','2026-06-30',30e6,60e6,45e6]],
      columns=['ticker','ann_date','report_date','free_cashflow','n_cashflow_act','net_profit'])
    x=pd.DataFrame({'ticker':['600000.SH'],'total_mv':[100000.]})
    r=add_cash_quality_factors(x,cf,'2026-09-10').iloc[0]
    assert r['free_cashflow']==50e6 and r['fcf_yield_raw']==pytest.approx(.05)
    assert r['cash_quality_raw']==pytest.approx(100e6/75e6)
    assert pd.isna(add_cash_quality_factors(x,None,'2026-09-10').iloc[0]['fcf_yield_raw'])

def test_qmt_snapshot_retains_broker_sellable():
    from src.qmt_v8 import MiniQMT
    q=object.__new__(MiniQMT);now=pd.Timestamp('2026-09-11T10:00:00+08:00')
    q.calendar=TradingCalendar(pd.DataFrame({'date':['2026-09-10','2026-09-11'],'is_open':[1,1]}))
    q.risk_reference={'as_of':'2026-09-10','prior_close_nav_adjusted':100000,'high_water_nav_adjusted':100000};q.cfg=RiskConfig()
    q.account=SimpleNamespace(account_id='STUB_ONLY');q.requested_tickers={'600000.SH'}
    q.reference=pd.DataFrame([{'ticker':'600000.SH','signal_date':'2026-09-10','industry_l1':'BANK','adv20_cny':1e8,'is_st':False,'is_delisting':False}])
    q.constant=SimpleNamespace(ORDER_SUCCEEDED=56,ORDER_CANCELED=54,ORDER_PART_CANCEL=53,ORDER_JUNK=57)
    q.trader=SimpleNamespace(query_stock_asset=lambda a:SimpleNamespace(cash=96000,frozen_cash=1000,total_asset=100000),
      query_stock_positions=lambda a:[SimpleNamespace(stock_code='600000.SH',volume=300,can_use_volume=100,market_value=3000)],query_stock_orders=lambda a,b:[])
    q.xtdata=SimpleNamespace(get_full_tick=lambda names:{'600000.SH':dict(time=now.value//1000000,lastPrice=10,bidPrice=[9.99],askPrice=[10.01],stockStatus=13)},
      get_instrument_detail=lambda t:dict(InstrumentName='测试证券',UpStopPrice=11,DownStopPrice=9))
    a,quotes=q.snapshot(now)
    assert a['positions'][0]['sellable_qty']==100 and a['cash_total']==97000 and a['cash_available']==96000
    assert quotes.iloc[0]['price_basis']=='raw'

def test_manifest_file_tampering_blocks(tmp_path):
    import hashlib,json
    from src.data_contract_v8 import load_verified_dataset
    names=['prices.csv','raw_prices.csv','trade_calendar.csv','daily_basic.csv','fundamentals_raw.csv','index_membership.csv','stock_metadata.csv','industry_membership.csv','st_status.csv','benchmark.csv','stock_limits.csv']
    for n in names:(tmp_path/n).write_text('fixture\n')
    manifest={'schema_version':8,'source':'tushare','complete_universe':True,'files':{n:hashlib.sha256((tmp_path/n).read_bytes()).hexdigest() for n in names}}
    (tmp_path/'data_manifest_v8.json').write_text(json.dumps(manifest));(tmp_path/'prices.csv').write_text('changed\n')
    with pytest.raises(TradingBlocked,match='不符'):load_verified_dataset(tmp_path)

def test_qmt_pre_submit_recheck_and_acceptance_only(tmp_path,monkeypatch):
    import src.qmt_v8 as module
    from src.live_v8 import cn_timestamp
    from src.order_journal_v8 import OrderJournal,submit_reserved
    now=cn_timestamp('2026-09-11T10:00:00+08:00');cal=TradingCalendar(pd.DataFrame({'date':['2026-09-10','2026-09-11'],'is_open':[1,1]}))
    account={'source':'broker','risk_reference_date':'2026-09-10','account_id':'STUB_ONLY','as_of':now.isoformat(),'cash_available':100000.,'cash_total':100000.,'total_asset':100000.,
      'prior_close_nav_adjusted':100000.,'high_water_nav_adjusted':100000.,'positions':[],'open_order_count':0}
    targets=pd.DataFrame([{'ticker':'600000.SH','target_weight':.08,'signal_date':'2026-09-10','signal_close_raw':10.,'source':'real'}])
    quotes=pd.DataFrame([{'ticker':'600000.SH','quote_time':now.isoformat(),'last':10.,'bid':9.99,'ask':10.01,'up_limit':11.,'down_limit':9.,'adv20_cny':1e8,
       'industry_l1':'BANK','is_st':False,'is_delisting':False,'status':'TRADING','price_basis':'raw'}])
    p=make_plan(account,targets,quotes,cal,now);q=object.__new__(module.MiniQMT);q.cfg=RiskConfig();q.calendar=cal
    q.account=SimpleNamespace(account_id='STUB_ONLY');q.reference=targets.assign(industry_l1='BANK');q.own_ids=set();q.accepted_buy_value={}
    q.snapshot=lambda now:(account,quotes);q.orders=lambda:[];q.terminal_statuses=lambda:set()
    q.constant=SimpleNamespace(STOCK_BUY=23,STOCK_SELL=24,FIX_PRICE=11);calls=[]
    q.trader=SimpleNamespace(order_stock=lambda *args:calls.append(args) or 17)
    monkeypatch.setattr(module,'pd',SimpleNamespace(Timestamp=SimpleNamespace(now=lambda tz:now)))
    q.enable_submission=False
    with pytest.raises(TradingBlocked):q.submit_checked(p['orders'][0])
    q.enable_submission=True;monkeypatch.delenv('QMT_ENABLE_LIVE',raising=False)
    with pytest.raises(TradingBlocked):q.submit_checked(p['orders'][0])
    monkeypatch.setenv('QMT_ENABLE_LIVE','YES_I_UNDERSTAND_REAL_ORDERS');quotes.loc[0,'last']=10.5
    with pytest.raises(TradingBlocked):q.submit_checked(p['orders'][0])
    assert not calls
    quotes.loc[0,'last']=10.;j=OrderJournal(tmp_path/'journal.db')
    try:
        result=submit_reserved(p,j,q);assert len(calls)==1
        assert result[0]['status']=='SUBMITTED' and result[0]['filled_qty']==0 and result[0]['broker_order_id']=='17'
    finally:j.close()

def test_etf_all_below_trend_can_be_cash():
    from src.etf_rotation_institutional import etf_rotation_scores
    days=pd.bdate_range('2025-01-01',periods=150)
    data=pd.DataFrame({'date':days,'ticker':'SYNTHETIC_ETF','close':[200-i for i in range(150)],'amount':1e8})
    assert etf_rotation_scores(data,days[-1]).empty
