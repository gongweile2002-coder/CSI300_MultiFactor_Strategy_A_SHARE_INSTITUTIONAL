"""v9 operational regressions. Vendor-shaped inputs below are synthetic fixtures only.
No fixture is distributed as real market data and no test contacts Tushare/QMT.
"""
from dataclasses import asdict,replace
import copy,hashlib,json,shutil,sqlite3
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from src.live_v8 import RiskConfig,TradingCalendar,TradingBlocked,make_plan
from src.ops_v9 import (audit_dataset,sha256,atomic_json,publish_signal_manifest,verify_signal_bundle,
    generation_lock,archive_run,verify_archive,BUNDLE_FILE,BUNDLE_FILES)

@pytest.fixture(scope='module')
def market_fixture(tmp_path_factory):
    d=tmp_path_factory.mktemp('SYNTHETIC_MARKET_ONLY');dates=pd.bdate_range(end='2026-09-11',periods=185)
    names=[f'{600001+i:06d}.SH' for i in range(300)];rows=[]
    for j,t in enumerate(names):
        for k,day in enumerate(dates):
            price=(10+j/30)*(1.0003+(j%17)*.00002)**k
            rows.append((str(day.date()),t,price*.999,price*1.002,price*.998,price,1.,1000000,price*1000000,'raw','CNY','share'))
    raw=pd.DataFrame(rows,columns=['date','ticker','open','high','low','close','adj_factor','volume','amount','price_basis','amount_unit','volume_unit'])
    research=raw.copy()
    for c in ['open','high','low','close']:research['raw_'+c]=research[c]
    research['price_basis']='adjusted_research_only'
    db=pd.DataFrame([('2026-09-11',t,8+j%19,1+(j%11)/10,1.,100000+j*1300,80000+j*1000) for j,t in enumerate(names)],
        columns=['date','ticker','pe_ttm','pb','turnover_rate','total_mv','circ_mv'])
    f=pd.DataFrame([(t,'2026-04-30','2026-03-31',5+j%13,5+j%13,2+j%23,20+j%37) for j,t in enumerate(names)],
        columns=['ticker','ann_date','report_date','roe','roe_dt','profit_growth','debt_ratio'])
    mem=pd.DataFrame({'ticker':names,'effective_date':'2026-09-01','index_code':'399300.SZ','weight':100/300})
    meta=pd.DataFrame({'ticker':names,'name':['Synthetic'+str(i) for i in range(300)],'list_date':'2000-01-01','market':'Main','exchange':'SSE'})
    ind=pd.DataFrame({'ticker':names,'l1_name':['Sector'+str(i%5) for i in range(300)],'in_date':'2000-01-01','out_date':pd.NaT})
    bench=pd.DataFrame({'date':dates.strftime('%Y-%m-%d'),'close':np.linspace(1000,1200,len(dates)),'index_code':'399300.SZ'})
    limits=raw[['date','ticker']].copy();limits['up_limit']=raw['close']*1.1;limits['down_limit']=raw['close']*.9
    caldates=pd.date_range(dates[0],'2026-09-16');cal=pd.DataFrame({'date':caldates.strftime('%Y-%m-%d'),'is_open':(caldates.weekday<5).astype(int)})
    frames={'raw_prices.csv':raw,'prices.csv':research,'daily_basic.csv':db,'fundamentals_raw.csv':f,'index_membership.csv':mem,
        'stock_metadata.csv':meta,'industry_membership.csv':ind,'st_status.csv':pd.DataFrame(columns=['ticker','trade_date']),
        'benchmark.csv':bench,'stock_limits.csv':limits,'trade_calendar.csv':cal}
    for name,x in frames.items():x.to_csv(d/name,index=False)
    manifest={'schema_version':8,'source':'tushare','complete_universe':True,'as_of':'2026-09-11','ticker_count':300,
        'fixture_notice':'SYNTHETIC TEST INPUT, NEVER REAL MARKET DATA',
        'files':{name:sha256(d/name) for name in frames},'membership_basis':'SYNTHETIC_TEST','financial_vintages':'SYNTHETIC_TEST'}
    atomic_json(d/'data_manifest_v8.json',manifest)
    return d

NOW=pd.Timestamp('2026-09-12T18:00:00+08:00')
TRADE_NOW=pd.Timestamp('2026-09-14T10:00:00+08:00')

def calendar(d):return TradingCalendar(pd.read_csv(d/'trade_calendar.csv'))
def refresh_manifest(d,name):
    p=d/'data_manifest_v8.json';m=json.loads(p.read_text());m['files'][name]=sha256(d/name);atomic_json(p,m)

def test_complete_data_audit(market_fixture):
    result=audit_dataset(market_fixture,NOW)
    assert result['status']=='PASS' and result['performance_validated'] is False

@pytest.mark.parametrize('mutation,expected',[
    ('research_price','raw_adjusted_alignment'),('volume_unit','raw_adjusted_alignment'),
    ('duplicate','dates_and_primary_keys'),('coverage','universe_coverage'),('future_market_date','dates_and_primary_keys')])
def test_semantic_errors_even_with_updated_hash(market_fixture,tmp_path,mutation,expected):
    d=tmp_path/'data';shutil.copytree(market_fixture,d)
    name='prices.csv' if mutation=='research_price' else 'daily_basic.csv' if mutation=='coverage' else 'raw_prices.csv'
    x=pd.read_csv(d/name)
    if mutation=='research_price':x.loc[0,'close']*=2
    if mutation=='volume_unit':x.loc[0,'volume_unit']='hand'
    if mutation=='duplicate':x=pd.concat([x,x.iloc[[0]]],ignore_index=True)
    if mutation=='coverage':x=x.iloc[:280]
    if mutation=='future_market_date':x.loc[0,'date']='2026-09-14'
    x.to_csv(d/name,index=False);refresh_manifest(d,name)
    report=audit_dataset(d,NOW)
    assert report['status']=='BLOCK' and any(r['check']==expected and r['status']=='BLOCK' for r in report['checks'])

@pytest.fixture(scope='module')
def signal_fixture(market_fixture,tmp_path_factory):
    from src.signals_v8 import generate_signals
    out=tmp_path_factory.mktemp('SYNTHETIC_SIGNAL_ONLY');cfg=RiskConfig()
    report=generate_signals(market_fixture,out,NOW,cfg)
    return out,report

def test_published_signal_lineage_and_real_core(signal_fixture,market_fixture):
    out,report=signal_fixture
    m,targets,reference=verify_signal_bundle(out/'targets.csv',RiskConfig(),calendar(market_fixture),TRADE_NOW,return_inputs=True)
    assert len(targets)==30 and len(reference)==300 and targets['target_weight'].sum()<=.8+1e-8
    assert m['bundle_hash']==report['bundle_hash'] and m['performance_validated'] is False
    assert m['data_manifest_hash']==sha256(market_fixture/'data_manifest_v8.json')

@pytest.mark.parametrize('name',list(BUNDLE_FILES))
def test_changed_signal_file_rejected(signal_fixture,market_fixture,tmp_path,name):
    out,_=signal_fixture;shutil.copytree(out,tmp_path/'signal');p=tmp_path/'signal'/name;p.write_bytes(p.read_bytes()+b'\n')
    with pytest.raises(TradingBlocked):verify_signal_bundle(tmp_path/'signal/targets.csv',RiskConfig(),calendar(market_fixture),TRADE_NOW)

def test_config_change_and_future_execution_rejected(signal_fixture,market_fixture):
    out,_=signal_fixture;cal=calendar(market_fixture)
    with pytest.raises(TradingBlocked):verify_signal_bundle(out/'targets.csv',replace(RiskConfig(),max_single_weight=.07),cal,TRADE_NOW)
    with pytest.raises(TradingBlocked):verify_signal_bundle(out/'targets.csv',RiskConfig(),cal,pd.Timestamp('2026-09-15T10:00:00+08:00'))

def test_generation_lock_blocks_reader(signal_fixture,market_fixture):
    out,_=signal_fixture
    with generation_lock(out):
        with pytest.raises(TradingBlocked):verify_signal_bundle(out/'targets.csv',RiskConfig(),calendar(market_fixture),TRADE_NOW)
        with pytest.raises(TradingBlocked):
            with generation_lock(out):pass

def test_failed_generation_invalidates_prior_bundle(signal_fixture,market_fixture,tmp_path,monkeypatch):
    import src.signals_v8 as m
    out,_=signal_fixture;shutil.copytree(out,tmp_path/'signal')
    monkeypatch.setattr(m,'_generate_signals_unchecked',lambda *a,**kw:(_ for _ in ()).throw(RuntimeError('interrupted generation')))
    with pytest.raises(RuntimeError):m.generate_signals(market_fixture,tmp_path/'signal',NOW,RiskConfig())
    assert not (tmp_path/'signal'/BUNDLE_FILE).exists()
    with pytest.raises(FileNotFoundError):verify_signal_bundle(tmp_path/'signal/targets.csv',RiskConfig(),calendar(market_fixture),TRADE_NOW)

def test_archive_keeps_wal_commits_and_detects_tampering(tmp_path):
    db=tmp_path/'live.db';writer=sqlite3.connect(db);writer.execute('PRAGMA journal_mode=WAL');writer.execute('PRAGMA wal_autocheckpoint=0')
    writer.execute('CREATE TABLE evidence(value TEXT)');writer.execute("INSERT INTO evidence VALUES('committed_in_wal')");writer.commit()
    note=tmp_path/'note.json';note.write_text('{"fixture":true}')
    try:
        archived=archive_run(tmp_path/'archives','demo',{'note.json':note},db)
        snap=sqlite3.connect(archived/'orders.sqlite3')
        try:assert snap.execute('SELECT value FROM evidence').fetchone()[0]=='committed_in_wal'
        finally:snap.close()
        assert verify_archive(archived)['kind']=='demo'
        (archived/'note.json').write_text('changed')
        with pytest.raises(TradingBlocked):verify_archive(archived)
    finally:writer.close()

def test_archive_failure_does_not_publish_partial_directory(tmp_path):
    root=tmp_path/'archives'
    with pytest.raises(TradingBlocked):archive_run(root,'plan',{'missing.json':tmp_path/'missing'})
    assert not list(root.iterdir())

def test_archive_binds_signal_and_configuration(signal_fixture,market_fixture,tmp_path):
    out,_=signal_fixture;cfgfile=tmp_path/'config.json';atomic_json(cfgfile,{'risk':asdict(RiskConfig())})
    files={name:out/name for name in (*BUNDLE_FILES,BUNDLE_FILE)};files['settings.json']=cfgfile
    archived=archive_run(tmp_path/'archives','prepare',files)
    assert verify_archive(archived)['files']['targets.csv']==sha256(out/'targets.csv')
    atomic_json(cfgfile,{'risk':asdict(replace(RiskConfig(),max_single_weight=.06))})
    with pytest.raises(TradingBlocked):archive_run(tmp_path/'archives','prepare',files)

def test_real_plan_requires_current_risk_reference_date():
    cal=TradingCalendar(pd.DataFrame({'date':['2026-09-10','2026-09-11'],'is_open':[1,1]}));now=pd.Timestamp('2026-09-11T10:00:00+08:00')
    a={'source':'broker','account_id':'TEST','as_of':now.isoformat(),'positions':[],'cash_available':100000.,'cash_total':100000.,'total_asset':100000.,
       'prior_close_nav_adjusted':100000.,'high_water_nav_adjusted':100000.,'open_order_count':0,'risk_reference_date':'2026-09-09'}
    t=pd.DataFrame([{'ticker':'600000.SH','target_weight':.05,'signal_date':'2026-09-10','signal_close_raw':10.,'source':'real'}])
    q=pd.DataFrame()
    with pytest.raises(TradingBlocked,match='风险基准日期'):make_plan(a,t,q,cal,now)

def test_doctor_does_not_expose_token(tmp_path,monkeypatch):
    from src.doctor_v9 import environment_report
    monkeypatch.delenv('TUSHARE_TOKEN',raising=False);secret='TEST_SECRET_NEVER_REAL'
    (tmp_path/'.env').write_text('TUSHARE_TOKEN='+secret)
    r=environment_report(tmp_path,{},tmp_path,NOW,{'status':'BLOCK','checks':[]})
    assert r['tushare_token_present'] and secret not in json.dumps(r)

def test_prepare_plan_register_and_reconcile_cli(market_fixture,tmp_path,monkeypatch):
    import live
    from types import SimpleNamespace
    cfgfile=tmp_path/'settings.json';atomic_json(cfgfile,{'risk':asdict(RiskConfig()),'live_enabled':False,'broker_rules_confirmed':True})
    out=tmp_path/'signal';journal=tmp_path/'orders.db';archives=tmp_path/'archives'
    common=['--config',str(cfgfile),'--journal',str(journal),'--archive-root',str(archives)]
    monkeypatch.setattr(live,'pd',SimpleNamespace(Timestamp=SimpleNamespace(now=lambda tz:NOW)))
    assert live.main(common+['prepare','--data',str(market_fixture),'--output',str(out)])==0
    refs=pd.read_csv(out/'execution_reference.csv');quotes=[]
    for r in refs.to_dict('records'):
        px=round(r['signal_close_raw'],2)
        quotes.append({'ticker':r['ticker'],'quote_time':TRADE_NOW.isoformat(),'last':px,'bid':px-.01,'ask':px+.01,'up_limit':round(px*1.1,2),'down_limit':round(px*.9,2),
          'adv20_cny':r['adv20_cny'],'industry_l1':r['industry_l1'],'is_st':False,'is_delisting':False,'status':'TRADING','price_basis':'raw'})
    acct={'source':'broker','account_id':'SYNTHETIC_CLI_TEST_ONLY','as_of':TRADE_NOW.isoformat(),'risk_reference_date':'2026-09-11',
        'cash_available':500000.,'cash_total':500000.,'total_asset':500000.,'prior_close_nav_adjusted':500000.,'high_water_nav_adjusted':500000.,'open_order_count':0,'positions':[]}
    atomic_json(tmp_path/'account.json',acct);pd.DataFrame(quotes).to_csv(tmp_path/'quotes.csv',index=False)
    monkeypatch.setattr(live,'pd',SimpleNamespace(Timestamp=SimpleNamespace(now=lambda tz:TRADE_NOW)))
    inputs=['--account',str(tmp_path/'account.json'),'--quotes',str(tmp_path/'quotes.csv'),'--targets',str(out/'targets.csv'),'--calendar',str(market_fixture/'trade_calendar.csv')]
    assert live.main(common+['plan']+inputs+['--output',str(tmp_path/'plan')])==0
    plan=json.loads((tmp_path/'plan/plan.json').read_text());assert plan['orders'] and plan.get('signal_bundle_hash')
    assert live.main(common+['manual-register']+inputs+['--plan',str(tmp_path/'plan/plan.json'),'--confirm-plan',plan['plan_hash']])==0
    # Registered only. No broker call occurs; synthetic rejection reports clear the test state.
    reports=pd.DataFrame([{'client_order_id':o['client_order_id'],'broker_order_id':'','status':'REJECTED','filled_qty':0,'avg_fill_price':0} for o in plan['orders']])
    reports.to_csv(tmp_path/'reports.csv',index=False)
    monkeypatch.setattr(live,'pd',pd)
    assert live.main(common+['reconcile','--reports',str(tmp_path/'reports.csv')])==0
    directories=list(archives.iterdir());assert len(directories)==4
    for folder in directories:assert verify_archive(folder)['schema_version']==9
