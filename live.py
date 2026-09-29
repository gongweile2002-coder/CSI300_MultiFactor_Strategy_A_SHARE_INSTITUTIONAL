"""v8 reviewed planning, persistent journal and optional MiniQMT execution."""
import argparse,json,sqlite3,sys,subprocess,importlib.util
from dataclasses import asdict
from pathlib import Path
import pandas as pd
from src.live_v8 import RiskConfig,TradingBlocked,TradingCalendar,read_csv,make_plan,save_plan,verify_plan,require,digest
from src.order_journal_v8 import OrderJournal,submit_reserved
from src.signals_v8 import generate_signals,last_completed_session
from src.data_contract_v8 import load_verified_dataset
from src.ops_v9 import audit_dataset,atomic_json,verify_signal_bundle,archive_run,verify_archive,BUNDLE_FILE,BUNDLE_FILES
BASE=Path(__file__).resolve().parent

def read_json(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def write_json(path,obj):Path(path).write_text(json.dumps(obj,ensure_ascii=False,indent=2),encoding='utf-8')

def demo(output,cfg):
    now=pd.Timestamp('2026-09-11T10:00:00+08:00');d=Path(output);d.mkdir(parents=True,exist_ok=True)
    cal=pd.DataFrame({'date':pd.date_range('2026-09-09','2026-09-12'),'is_open':[1,1,1,0]})
    account={'source':'synthetic','account_id':'DEMO_ONLY','as_of':now.isoformat(),'cash_available':97000.,'cash_total':97000.,
      'total_asset':100000.,'prior_close_nav_adjusted':100000.,'high_water_nav_adjusted':100000.,'open_order_count':0,
      'positions':[{'ticker':'000001.SZ','qty':300,'sellable_qty':100,'last_price':10.}]}
    names=['000001.SZ','600000.SH','688001.SH','600036.SH']
    targets=pd.DataFrame({'ticker':names,'target_weight':[0,.08,.03,.03],'signal_date':'2026-09-10','signal_close_raw':10.,'source':'synthetic'})
    quotes=pd.DataFrame([{'ticker':t,'quote_time':now.isoformat(),'last':10.,'bid':9.99,'ask':10.01,'up_limit':11.,'down_limit':9.,
      'adv20_cny':1e8,'industry_l1':'TECH' if t.startswith('688') else 'BANK','is_st':t=='600036.SH','is_delisting':False,'price_basis':'raw','status':'TRADING'} for t in names])
    plan=make_plan(account,targets,quotes,TradingCalendar(cal),now,cfg,allow_demo=True);save_plan(plan,d)
    write_json(d/'account.json',account);targets.to_csv(d/'targets.csv',index=False);quotes.to_csv(d/'quotes.csv',index=False);cal.to_csv(d/'calendar.csv',index=False)
    print(f'DEMO：{len(plan["orders"])} 笔模拟计划，{len(plan["blocked"])} 项拦截。合成数据，不是今日推荐，不会发送委托。\n{d}')

def build_parser():
    p=argparse.ArgumentParser(description='A股 v9.2：环境诊断 -> 数据核对 -> 信号溯源 -> 限时计划 -> 持久订单状态 -> 券商对账')
    p.add_argument('--config',default=str(BASE/'config/live.example.json'));p.add_argument('--journal',default=str(BASE/'ops/live_v8.sqlite3'))
    p.add_argument('--archive-root',default=str(BASE/'ops/archive'))
    sub=p.add_subparsers(dest='command',required=True)
    q=sub.add_parser('demo');q.add_argument('--output',default=str(BASE/'outputs/v8_demo'))
    q=sub.add_parser('signal');q.add_argument('--data',default=str(BASE/'data/live'));q.add_argument('--output',default=str(BASE/'outputs/v8_signal'))
    q=sub.add_parser('check');q.add_argument('--data',default=str(BASE/'data/live'))
    for command in ['audit','doctor','prepare']:
        q=sub.add_parser(command);q.add_argument('--data',default=str(BASE/'data/live'))
        q.add_argument('--output',default=str(BASE/('outputs/v9_'+command if command!='prepare' else 'outputs/v8_signal')))
        if command=='prepare':q.add_argument('--download',action='store_true',help='显式联网更新真实数据，再核查并生成信号；不下单')
    q=sub.add_parser('archive-verify');q.add_argument('--directory',required=True)
    for command in ['plan','manual-register']:
        q=sub.add_parser(command)
        for f in ['account','targets','quotes','calendar']:q.add_argument('--'+f,required=True)
        if command=='plan':q.add_argument('--output',default=str(BASE/'outputs/v8_plan'))
        else:q.add_argument('--plan',required=True);q.add_argument('--confirm-plan',required=True)
    q=sub.add_parser('reconcile');q.add_argument('--reports',required=True)
    q=sub.add_parser('ledger');q.add_argument('--account-id')
    q=sub.add_parser('expire');q.add_argument('--account-id')
    for command in ['qmt-snapshot','qmt-submit','qmt-reconcile']:
        q=sub.add_parser(command)
        for f in ['reference','risk-reference','calendar']:q.add_argument('--'+f,required=True)
        if command!='qmt-reconcile':q.add_argument('--targets',required=True)
        if command=='qmt-snapshot':q.add_argument('--output',default=str(BASE/'outputs/v8_broker'))
        if command=='qmt-submit':q.add_argument('--plan',required=True);q.add_argument('--confirm-plan',required=True)
    return p

def main(argv=None):
    args=build_parser().parse_args(argv);journal=None;broker=None
    try:
        settings=read_json(args.config);cfg=RiskConfig.from_dict(settings['risk']);now=pd.Timestamp.now(tz='Asia/Shanghai')
        kill=Path(settings.get('kill_switch_file','ops/KILL_SWITCH'));kill=kill if kill.is_absolute() else BASE/kill
        cmd=args.command
        if cmd=='demo':demo(args.output,cfg);return 0
        if cmd=='signal':print(json.dumps(generate_signals(args.data,args.output,now,cfg),ensure_ascii=False,indent=2));return 0
        if cmd=='archive-verify':
            result=verify_archive(args.directory);print(f"归档完整性通过：{result['run_id']}，{len(result['files'])} 个文件。不是业绩认证。");return 0
        if cmd in {'check','audit','doctor'}:
            audit=audit_dataset(args.data,now)
            if cmd=='doctor':
                from src.doctor_v9 import environment_report,write_doctor_report
                report=environment_report(BASE,settings,args.data,now,audit)
                write_doctor_report(report,args.output);print(json.dumps(report,ensure_ascii=False,indent=2))
                return 0 if report['signal_preparation_ready'] else 2
            if cmd=='audit':
                out=Path(args.output);out.mkdir(parents=True,exist_ok=True);atomic_json(out/'data_audit_v9.json',audit)
            audit['broker_rules_confirmed']=settings.get('broker_rules_confirmed') is True
            audit['live_enabled']=settings.get('live_enabled') is True
            print(json.dumps(audit,ensure_ascii=False,indent=2));return 0 if audit['status']=='PASS' else 2
        if cmd=='prepare':
            if args.download:
                result=subprocess.run([sys.executable,str(BASE/'download_live_data.py'),'--output',str(Path(args.data).resolve())],cwd=BASE)
                require(result.returncode==0,'下载没有完成，未生成新信号')
            report=generate_signals(args.data,args.output,now,cfg)
            out=Path(args.output);data=Path(args.data)
            files={name:out/name for name in (*BUNDLE_FILES,BUNDLE_FILE,'data_audit_v9.json')}
            files['settings.json']=Path(args.config);files['data_manifest_v8.json']=data/'data_manifest_v8.json'
            data_manifest=read_json(data/'data_manifest_v8.json')
            files.update({'data_'+name:data/name for name in data_manifest['files']})
            archived=archive_run(args.archive_root,'prepare',files,metadata={'bundle_hash':report['bundle_hash']})
            print(json.dumps({**report,'archive':str(archived),'orders_sent':0},ensure_ascii=False,indent=2));return 0
        journal=OrderJournal(args.journal)
        if cmd=='ledger':
            print(json.dumps({'orders':journal.rows(args.account_id),'reservations':journal.reservations(args.account_id)},ensure_ascii=False,indent=2));return 0
        if cmd=='expire':
            count=journal.expire_approved(now,args.account_id);print(f'已将 {count} 笔未提交且超过计划有效期的 APPROVED 委托标记为 EXPIRED；未联系券商。');return 0
        if cmd=='reconcile':
            reports=pd.read_csv(args.reports,dtype={'client_order_id':str,'broker_order_id':str})
            require({'client_order_id','broker_order_id','status','filled_qty','avg_fill_price'}<=set(reports),'回报列不完整')
            for r in reports.to_dict('records'):
                bid=None if pd.isna(r['broker_order_id']) else str(r['broker_order_id'])
                journal.update(r['client_order_id'],r['status'],bid,r['filled_qty'],r['avg_fill_price'])
            archived=archive_run(args.archive_root,'reconcile',{'reports.csv':args.reports},args.journal)
            print(f'累计券商回报已对账；刷新资金、持仓和行情。归档：{archived}');return 0
        cal=TradingCalendar(read_csv(args.calendar))
        bundle=None;reference=None
        if cmd!='qmt-reconcile':
            bundle,targets,reference=verify_signal_bundle(args.targets,cfg,cal,now,
                getattr(args,'reference',None),return_inputs=True)
        if cmd.startswith('qmt-'):
            from src.qmt_v8 import MiniQMT
            if cmd=='qmt-submit':
                require(settings.get('live_enabled') is True,'live_enabled 尚未开启')
                require(settings.get('broker_rules_confirmed') is True,'须先向券商确认程序化交易报告和账户权限')
            targets=targets if cmd!='qmt-reconcile' else None
            broker=MiniQMT(reference if reference is not None else read_csv(args.reference),read_json(args.risk_reference),cal,cfg,
                enable_submission=cmd=='qmt-submit',requested_tickers=targets['ticker'].tolist() if targets is not None else None)
            if cmd=='qmt-reconcile':
                rows=broker.reconcile(journal)
                archived=archive_run(args.archive_root,'reconcile',{'settings.json':args.config},args.journal)
                print(json.dumps({'orders':rows,'archive':str(archived)},ensure_ascii=False,indent=2));return 0
            account,quotes=broker.snapshot()
            if cmd=='qmt-snapshot':
                out=Path(args.output);out.mkdir(parents=True,exist_ok=True);write_json(out/'account.json',account);quotes.to_csv(out/'quotes.csv',index=False)
                parent=Path(args.targets).parent
                files={name:parent/name for name in (*BUNDLE_FILES,BUNDLE_FILE)}
                files.update({'account.json':out/'account.json','quotes.csv':out/'quotes.csv','settings.json':args.config,'risk_reference.json':args.risk_reference,'calendar.csv':args.calendar})
                archived=archive_run(args.archive_root,'snapshot',files,args.journal,{'bundle_hash':bundle['bundle_hash']})
                print(f'只读券商快照已生成：{out}；归档：{archived}');return 0
        else:account=read_json(args.account);quotes=read_csv(args.quotes)
        now=pd.Timestamp.now(tz='Asia/Shanghai');keys,turnover=journal.context(account['account_id'],str(now.date()))
        fresh=make_plan(account,targets,quotes,cal,now,cfg,used_keys=keys,used_turnover_cny=turnover,kill_switch=kill.exists())
        fresh['signal_bundle_hash']=bundle['bundle_hash']
        fresh['plan_hash']=digest({k:v for k,v in fresh.items() if k!='plan_hash'})
        def evidence_files():
            parent=Path(args.targets).parent
            files={name:parent/name for name in (*BUNDLE_FILES,BUNDLE_FILE)}
            files.update({'settings.json':args.config,'calendar.csv':args.calendar})
            if hasattr(args,'account'):files['account.json']=args.account;files['quotes.csv']=args.quotes
            if hasattr(args,'risk_reference'):files['risk_reference.json']=args.risk_reference
            return files
        if cmd=='plan':
            save_plan(fresh,args.output);files=evidence_files()
            files.update({name:Path(args.output)/name for name in ['plan.json','orders.csv','blocked.csv']})
            archived=archive_run(args.archive_root,'plan',files,args.journal,{'plan_hash':fresh['plan_hash'],'bundle_hash':bundle['bundle_hash']})
            print(f'计划 {fresh["plan_hash"]}\n{len(fresh["orders"])} 笔待复核；{len(fresh["blocked"])} 项拦截。\n归档：{archived}');return 0
        original=read_json(args.plan);verify_plan(original,now,require_fresh=True)
        require(original.get('signal_bundle_hash')==bundle['bundle_hash'],'复核计划与当前信号包不是同一版本，重新生成计划')
        require(original['mode']=='MANUAL_REVIEW','不能执行演示或未知模式计划')
        require(args.confirm_plan==original['plan_hash'],'确认值必须是复核过的完整 plan_hash')
        require(original['account_id']==account['account_id'] and original['config_hash']==digest(asdict(cfg)),'账户/策略参数与原计划不符')
        require(original['orders']==fresh['orders'] and original['trade_date']==fresh['trade_date'],'最新资金/行情/风控导致计划变化，请重新生成并复核')
        if cmd=='manual-register':
            require(settings.get('broker_rules_confirmed') is True,'须先向券商确认程序化交易报告和账户权限')
            verify_plan(original,pd.Timestamp.now(tz='Asia/Shanghai'),require_fresh=True)
            journal.reserve(original)
            files=evidence_files();files['plan.json']=args.plan
            archived=archive_run(args.archive_root,'manual_register',files,args.journal,{'plan_hash':original['plan_hash']},objects={'revalidated_plan.json':fresh})
            print(f'已登记 APPROVED 待人工委托；登记不代表已提交或成交。按原计划操作并录入累计回报。归档：{archived}');return 0
        verify_plan(original,pd.Timestamp.now(tz='Asia/Shanghai'),require_fresh=True)
        rows=submit_reserved(original,journal,broker,kill.exists)
        files=evidence_files();files['plan.json']=args.plan
        archived=archive_run(args.archive_root,'qmt_submit',files,args.journal,{'plan_hash':original['plan_hash']},
            objects={'revalidated_plan.json':fresh,'broker_account_before_batch.json':account,'broker_quotes_before_batch.json':quotes.to_dict('records')})
        print(json.dumps({'orders':rows,'archive':str(archived)},ensure_ascii=False,indent=2))
        return 2 if any(r['status'] in {'UNKNOWN','REJECTED','BLOCKED'} for r in rows if r['plan_hash']==original['plan_hash']) else 0
    except (TradingBlocked,FileNotFoundError,KeyError,ImportError,ValueError,TypeError,OSError,sqlite3.Error) as exc:
        print(f'BLOCK：{exc}。若此前已登记/提交委托，先查询 ledger 和券商回报；不要因后续错误重新发送。');return 2
    finally:
        if broker is not None:broker.close()
        if journal is not None:journal.close()

if __name__=='__main__':raise SystemExit(main())
