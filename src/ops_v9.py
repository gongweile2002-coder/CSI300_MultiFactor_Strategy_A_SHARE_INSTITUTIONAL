"""Operational data checks, signal lineage and consistent local archives.
Hashes detect accidental changes; they do not authenticate vendor data or attest returns.
"""
from __future__ import annotations
import hashlib
import io
import json
import os
import shutil
import sqlite3
import tempfile
import uuid
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path
import numpy as np
import pandas as pd
from .live_v8 import TradingBlocked, require, cn_timestamp, digest, read_csv
from .data_contract_v8 import load_verified_dataset

BUNDLE_FILE = 'signal_manifest_v9.json'
BUNDLE_FILES = ('targets.csv', 'execution_reference.csv', 'factor_audit.csv', 'signal_report.json')

def sha256(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()

def atomic_json(path, value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    fd,name=tempfile.mkstemp(prefix='.'+path.name+'.',dir=path.parent)
    try:
        with os.fdopen(fd,'w',encoding='utf-8') as stream:
            json.dump(value,stream,ensure_ascii=False,indent=2,allow_nan=False,default=str)
            stream.write('\n');stream.flush();os.fsync(stream.fileno())
        os.replace(name,path)
    finally:
        if os.path.exists(name):os.unlink(name)

@contextmanager
def generation_lock(directory):
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
    lock=directory/'.signal_generation.lock'
    try:fd=os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY,0o600)
    except FileExistsError:raise TradingBlocked('信号目录正在生成或保留了中断锁；确认没有运行进程后使用新输出目录') from None
    try:
        with os.fdopen(fd,'w') as stream:stream.write(str(os.getpid()))
        yield
    finally:lock.unlink(missing_ok=True)

def completed_date(calendar,now):
    now=cn_timestamp(now);today=pd.Timestamp(now.date())
    return today if calendar.is_open(today) and now.hour>=18 else calendar.previous(today)

def audit_dataset(directory,now):
    """Inspect all required data together; checks are stronger than a checksum alone."""
    directory=Path(directory);now=cn_timestamp(now);checks=[]
    def check(name,fn):
        try:
            details=fn()
            checks.append({'check':name,'status':'PASS','detail':str(details or '通过')})
        except (ValueError,KeyError,TypeError,OSError,AssertionError) as exc:
            checks.append({'check':name,'status':'BLOCK','detail':str(exc)})
    report={'schema_version':9,'checked_at':now.isoformat(),'source':'tushare',
            'status':'BLOCK','performance_validated':False,'checks':checks}
    try:manifest,calendar=load_verified_dataset(directory)
    except (ValueError,KeyError,TypeError,OSError) as exc:
        checks.append({'check':'manifest_and_hashes','status':'BLOCK','detail':str(exc)});return report
    checks.append({'check':'manifest_and_hashes','status':'PASS','detail':'必要文件和 SHA256 一致；不构成第三方真实性证明'})
    report['source']=manifest['source']
    report['data_manifest_hash']=sha256(directory/'data_manifest_v8.json');report['as_of']=manifest.get('as_of')
    asof=pd.Timestamp(manifest['as_of'])
    check('latest_completed_session',lambda:require(asof==completed_date(calendar,now),'数据未更新到最近完整交易日'))
    frames={}
    for name in manifest['files']:
        if name.endswith('.csv'):
            try:frames[name]=read_csv(directory/name)
            except (ValueError,OSError) as exc:checks.append({'check':'read:'+name,'status':'BLOCK','detail':str(exc)})
    required=['prices.csv','raw_prices.csv','daily_basic.csv','fundamentals_raw.csv','index_membership.csv','stock_metadata.csv','industry_membership.csv','st_status.csv','benchmark.csv','stock_limits.csv']
    if not set(required)<=set(frames):return report
    raw=frames['raw_prices.csv'];research=frames['prices.csv'];db=frames['daily_basic.csv'];members=frames['index_membership.csv']
    def dated(frame,column):
        x=pd.to_datetime(frame[column],errors='raise')
        require(x.notna().all() and x.dt.tz is None and x.eq(x.dt.normalize()).all(),column+': 日期必须有效且不含时刻')
        return x
    def market_dates():
        for name in ['prices.csv','raw_prices.csv','daily_basic.csv','benchmark.csv','stock_limits.csv']:
            x=frames[name];dates=dated(x,'date');keys=['date','ticker'] if 'ticker' in x else ['date']
            require(not x.empty and not x.duplicated(keys).any(),name+': 空表或重复主键')
            require(dates.max()==asof,name+': 最新日期不等于清单日期')
            require(all(calendar.is_open(d) for d in dates.unique()),name+': 出现休市日行情')
    check('dates_and_primary_keys',market_dates)
    def units_and_prices():
        require(raw['price_basis'].eq('raw').all() and raw['amount_unit'].eq('CNY').all() and raw['volume_unit'].eq('share').all(),'原始价/成交量/成交额单位错误')
        values=raw[['open','high','low','close','adj_factor','volume','amount']].apply(pd.to_numeric,errors='raise')
        require(np.isfinite(values.to_numpy()).all(),'行情存在非有限值')
        require((values[['open','high','low','close','adj_factor']]>0).all().all() and (values[['volume','amount']]>=0).all().all(),'价格/复权因子/成交量额非法')
        require((values['low']<=values[['open','close']].min(axis=1)).all() and (values['high']>=values[['open','close']].max(axis=1)).all(),'OHLC 关系错误')
        require(research['price_basis'].eq('adjusted_research_only').all(),'研究价格口径未明确')
        joined=raw.merge(research,on=['date','ticker'],suffixes=('_r','_a'),how='outer',indicator=True,validate='one_to_one')
        require(joined['_merge'].eq('both').all(),'原始/研究行情主键不一致')
        for c in ['open','high','low','close']:
            require(np.allclose(joined[c+'_r']*joined['adj_factor_r'],joined[c+'_a'],rtol=1e-9,atol=1e-8),c+': 研究价不等于原始价乘复权因子')
            require(np.allclose(joined[c+'_r'],joined['raw_'+c],rtol=1e-9,atol=1e-8),c+': raw 副本不一致')
        for c in ['volume','amount','adj_factor']:
            require(np.allclose(joined[c+'_r'],joined[c+'_a'],rtol=1e-9,atol=1e-8),c+': 两份行情口径不一致')
    check('raw_adjusted_alignment',units_and_prices)
    def universe():
        dates=dated(members,'effective_date');require(not members.duplicated(['effective_date','ticker']).any(),'重复成分主键')
        require(dates.max()<=asof,'未来成分快照')
        latest=dates.max();names=set(members.loc[dates==latest,'ticker'])
        require(len(names)==300 and 0<=(asof-latest).days<=45,'最新成分不是300只或超过45天')
        require(len(set(members['ticker']))==int(manifest['ticker_count']),'历史成分数量与清单不一致')
        for x,label in [(raw,'原始行情'),(db,'估值')]:
            current=set(x.loc[pd.to_datetime(x['date'])==asof,'ticker']);missing=sorted(names-current)
            require(len(names & current)>=285,label+': 当前成分覆盖不足95%')
            if missing:checks.append({'check':label+'_missing_members','status':'WARN','detail':f'{len(missing)} 只缺失；核实停牌或漏数：'+','.join(missing[:10])})
        return f'最新成分300只；历史并集{manifest["ticker_count"]}只'
    check('universe_coverage',universe)
    def reference_data():
        meta=frames['stock_metadata.csv'];require(not meta.empty and not meta['ticker'].duplicated().any(),'证券资料空或重复')
        require(set(members['ticker'])<=set(meta['ticker']),'证券资料遗漏历史成分')
        industry=frames['industry_membership.csv'];require(not industry.empty,'历史行业为空')
        starts=pd.to_datetime(industry['in_date'],errors='raise');ends=pd.to_datetime(industry['out_date'],errors='raise')
        require((starts.notna()).all() and ((ends.isna())|(ends>=starts)).all(),'行业生效区间非法')
        st=frames['st_status.csv'];require({'ticker','trade_date'}<=set(st),'ST 字段缺失')
        if not st.empty:require(dated(st,'trade_date').max()<=asof,'ST 历史含未来日期')
        f=frames['fundamentals_raw.csv'];ann=dated(f,'ann_date');period=dated(f,'report_date')
        require(not f.empty and (period<=ann).all(),'财务报告期/公告日期关系错误')
        require(not f.duplicated(['ticker','ann_date','report_date']).any(),'重复财务版本主键')
        benchmark=frames['benchmark.csv'];require(len(benchmark)>=120,'基准历史不足120期')
        closes=pd.to_numeric(benchmark['close'],errors='raise');require(np.isfinite(closes).all() and (closes>0).all(),'基准价格非法')
    check('financial_and_reference_contracts',reference_data)
    if manifest['source']=='baostock':
        def free_contract():
            require(manifest.get('financial_metric_basis')=='roeAvg_YOYNI_liabilityToAsset_percent', '免费财务指标口径未知')
            require(manifest.get('size_neutralization') is False, '免费版不能冒充每日总市值中性化')
            require(manifest.get('limit_basis')=='derived_10pct_seasoned_non_st_main_paper_only', '免费涨跌停口径未知')
            require(manifest.get('dividend_cash_policy')=='gross_less_flat_20pct_model_withholding', '免费分红税模型未知')
            return 'BaoStock 免费 Paper 口径；派息税/特殊交易情形有模型限制，不能用于券商下单'
        check('free_provider_contract',free_contract)
    report['status']='BLOCK' if any(x['status']=='BLOCK' for x in checks) else 'PASS'
    return report

def publish_signal_manifest(directory,data_directory,cfg,now):
    directory=Path(directory);data_directory=Path(data_directory)
    report=json.loads((directory/'signal_report.json').read_text(encoding='utf-8'))
    data=json.loads((data_directory/'data_manifest_v8.json').read_text(encoding='utf-8'))
    kind='free_paper_signal_bundle' if data['source']=='baostock' else 'real_signal_bundle'
    manifest={'schema_version':9,'kind':kind,'signal_date':report['signal_date'],
      'created_at':cn_timestamp(now).isoformat(),'config_hash':digest(asdict(cfg)),
      'data_manifest_hash':sha256(data_directory/'data_manifest_v8.json'),
      'files':{name:sha256(directory/name) for name in BUNDLE_FILES},'performance_validated':False}
    manifest['bundle_hash']=digest(manifest);atomic_json(directory/BUNDLE_FILE,manifest)
    return manifest

def verify_signal_bundle(targets_path,cfg,calendar,now,reference_path=None,*,return_inputs=False):
    targets_path=Path(targets_path);directory=targets_path.parent
    require(not (directory/'.signal_generation.lock').exists(),'信号仍在生成，稍后重试')
    require(targets_path.name=='targets.csv','使用完整信号目录中的 targets.csv，不能重命名混用')
    manifest=json.loads((directory/BUNDLE_FILE).read_text(encoding='utf-8'))
    require(manifest.get('schema_version')==9 and manifest.get('kind')=='real_signal_bundle','需要重新生成 v9 信号包')
    require(manifest.get('bundle_hash')==digest({k:v for k,v in manifest.items() if k!='bundle_hash'}),'信号清单内容被修改')
    require(manifest['config_hash']==digest(asdict(cfg)),'当前风控配置与生成信号时不同，请重新生成信号')
    require(set(manifest['files'])==set(BUNDLE_FILES),'信号文件清单不完整')
    payloads={name:(directory/name).read_bytes() for name in BUNDLE_FILES}
    for name,content in payloads.items():
        require(hashlib.sha256(content).hexdigest()==manifest['files'][name],name+': 信号文件被修改或混用了旧版本')
    now=cn_timestamp(now);created=cn_timestamp(manifest['created_at'])
    require(created<=now,'信号生成时间来自未来')
    require(pd.Timestamp(manifest['signal_date'])==calendar.previous(now.date()),'信号不是本次交易日前一交易日')
    if reference_path is not None:
        require(sha256(reference_path)==manifest['files']['execution_reference.csv'],'QMT参考资料与信号包不是同一版本')
    targets=pd.read_csv(io.BytesIO(payloads['targets.csv']),dtype={'ticker':str})
    require(targets['signal_date'].astype(str).eq(manifest['signal_date']).all() and targets['source'].eq('real').all(),'目标日期/来源与清单不符')
    require(targets['strategy_version'].eq(cfg.version).all(),'目标策略版本不符')
    if return_inputs:
        reference=pd.read_csv(io.BytesIO(payloads['execution_reference.csv']),dtype={'ticker':str})
        return manifest,targets,reference
    return manifest

def archive_run(output_root,kind,files,journal_path=None,metadata=None,objects=None):
    """Commit a distinct directory only after all file copies and SQLite backup succeed."""
    require(kind in {'prepare','plan','manual_register','qmt_submit','reconcile','snapshot','demo'},'归档类型非法')
    root=Path(output_root);root.mkdir(parents=True,exist_ok=True)
    stamp=pd.Timestamp.now(tz='Asia/Shanghai');run_id=stamp.strftime('%Y%m%dT%H%M%S')+'_'+kind+'_'+uuid.uuid4().hex[:10]
    temp=Path(tempfile.mkdtemp(prefix='.staging_',dir=root));final=root/run_id
    try:
        entries={}
        for logical,path in files.items():
            require(Path(logical).name==logical and logical not in {'archive_manifest.json','orders.sqlite3'},'归档名称非法')
            path=Path(path);require(path.is_file(),'归档输入不存在：'+str(path));shutil.copyfile(path,temp/logical)
            entries[logical]=sha256(temp/logical)
        for logical,value in (objects or {}).items():
            require(Path(logical).name==logical and logical not in entries and logical not in {'archive_manifest.json','orders.sqlite3'},'归档对象名称非法')
            atomic_json(temp/logical,value);entries[logical]=sha256(temp/logical)
        if journal_path is not None and Path(journal_path).exists():
            source=sqlite3.connect(Path(journal_path).resolve().as_uri()+'?mode=ro',uri=True,timeout=10)
            target=sqlite3.connect(temp/'orders.sqlite3')
            try:
                source.backup(target);target.commit()
                require(target.execute('PRAGMA integrity_check').fetchone()[0]=='ok','SQLite备份完整性检查失败')
            finally:target.close();source.close()
            entries['orders.sqlite3']=sha256(temp/'orders.sqlite3')
        _check_archive_lineage(temp,entries)
        manifest={'schema_version':9,'run_id':run_id,'kind':kind,'created_at':stamp.isoformat(),
          'metadata':metadata or {},'files':entries,'note':'本地操作证据；不是外部认证，也不是投资业绩证明'}
        manifest['archive_hash']=digest(manifest);atomic_json(temp/'archive_manifest.json',manifest)
        os.replace(temp,final);return final
    except Exception:
        shutil.rmtree(temp,ignore_errors=True);raise

def verify_archive(directory):
    directory=Path(directory);manifest=json.loads((directory/'archive_manifest.json').read_text(encoding='utf-8'))
    require(manifest.get('schema_version')==9,'归档版本错误')
    require(manifest['archive_hash']==digest({k:v for k,v in manifest.items() if k!='archive_hash'}),'归档清单不符')
    for name,expected in manifest['files'].items():
        require(Path(name).name==name and name not in {'.','..'},'归档路径非法')
        require(sha256(directory/name)==expected,'归档内容变动：'+name)
    _check_archive_lineage(directory,manifest['files'])
    return manifest


def _check_archive_lineage(directory,entries):
    """Detect mixed-generation copies before committing or verifying an archive."""
    directory=Path(directory)
    if BUNDLE_FILE not in entries:return
    bundle=json.loads((directory/BUNDLE_FILE).read_text(encoding='utf-8'))
    require(bundle['bundle_hash']==digest({k:v for k,v in bundle.items() if k!='bundle_hash'}),'归档中的信号清单被修改')
    for name in BUNDLE_FILES:require(entries.get(name)==bundle['files'].get(name),'归档中的信号文件版本混用：'+name)
    if 'plan.json' in entries:
        from .live_v8 import verify_plan
        plan=json.loads((directory/'plan.json').read_text(encoding='utf-8'));verify_plan(plan)
        require(plan.get('signal_bundle_hash')==bundle['bundle_hash'],'归档计划不属于此信号版本')
    if 'settings.json' in entries:
        from .live_v8 import RiskConfig
        settings=json.loads((directory/'settings.json').read_text(encoding='utf-8-sig'))
        require(digest(asdict(RiskConfig.from_dict(settings['risk'])))==bundle['config_hash'],'归档配置与信号版本不符')
    if 'data_manifest_v8.json' in entries:
        require(entries['data_manifest_v8.json']==bundle['data_manifest_hash'],'归档的数据清单与信号来源不符')
        data=json.loads((directory/'data_manifest_v8.json').read_text(encoding='utf-8'))
        for name,h in data['files'].items():require(entries.get('data_'+name)==h,'归档的数据内容与清单不符：'+name)
