"""Download the full historical CSI300 union; publish a manifest only on success."""
import argparse,hashlib,json,os
from pathlib import Path
import pandas as pd
from src.corporate_actions_v10 import normalize_corporate_actions
from src.tushare_provider_v4 import TushareDownloaderV4
from src.tushare_provider import _yyyymmdd
from src.live_v8 import TradingCalendar,require
from src.signals_v8 import last_completed_session

BASE=Path(__file__).resolve().parent

_DIVIDEND_FIELDS = (
    "ts_code,end_date,ann_date,div_proc,stk_div,stk_bo_rate,stk_co_rate,"
    "cash_div,cash_div_tax,record_date,ex_date,pay_date,div_listdate,"
    "imp_ann_date,base_date,base_share"
)

def fetch_corporate_actions(dl, tickers, start, end, output_dir):
    frames = []
    for ticker in tickers:
        df = dl.pro.dividend(ts_code=ticker, fields=_DIVIDEND_FIELDS)
        dl._pause()
        if df is None or df.empty:
            continue
        if len(df) >= 2000:
            raise RuntimeError(f"{ticker}: dividend 达到单次 2000 行上限，不能确认完整性")
        frames.append(df)
    raw = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    actions = normalize_corporate_actions(raw)
    if not actions.empty:
        start = pd.Timestamp(start).normalize()
        end = pd.Timestamp(end).normalize()
        actions = actions[
            (actions["ex_date"] >= start) & (actions["ex_date"] <= end)
        ].copy()
    actions.to_csv(Path(output_dir) / "corporate_actions.csv", index=False)
    return actions


def load_env():
    p=BASE/'.env'
    if p.exists():
        for line in p.read_text(encoding='utf-8').splitlines():
            if line.strip() and not line.lstrip().startswith('#') and '=' in line:
                key,value=line.split('=',1);os.environ.setdefault(key.strip(),value.strip().strip('"').strip("'"))

def main():
    p=argparse.ArgumentParser(description='完整沪深300实盘信号数据；不支持抽样替代全股票池')
    p.add_argument('--output',default=str(BASE/'data/live'));p.add_argument('--as-of')
    p.add_argument('--lookback-years',type=int,default=3);p.add_argument('--sleep',type=float,default=.15)
    args=p.parse_args();require(args.lookback_years>=2,'至少需要两年历史');load_env()
    d=Path(args.output);d.mkdir(parents=True,exist_ok=True)
    (d/'data_manifest_v8.json').unlink(missing_ok=True)
    dl=TushareDownloaderV4(os.getenv('TUSHARE_TOKEN',''),d,args.sleep)
    now=pd.Timestamp.now(tz='Asia/Shanghai');start=pd.Timestamp(now.date())-pd.DateOffset(years=args.lookback_years)
    cal=dl.pro.trade_cal(exchange='SSE',start_date=_yyyymmdd(start),end_date=_yyyymmdd(pd.Timestamp(now.date())+pd.Timedelta(days=40)))
    dl._pause();require(cal is not None and not cal.empty,'交易日历下载失败')
    cal=cal.rename(columns={'cal_date':'date'});cal['date']=pd.to_datetime(cal['date'],format='%Y%m%d')
    cal=cal[['date','is_open']].sort_values('date');calendar=TradingCalendar(cal);latest=last_completed_session(calendar,now)
    asof=pd.Timestamp(args.as_of) if args.as_of else latest
    require(asof<=latest and calendar.is_open(asof),'日期尚未完成日线更新或为休市日')
    cal.to_csv(d/'trade_calendar.csv',index=False)
    mem=dl.fetch_index_membership('399300.SZ',start,asof);date=mem['effective_date'].max()
    require(mem.loc[mem['effective_date']==date,'ticker'].nunique()==300,'最新沪深300快照不完整')
    require(0<=(asof-date).days<=45,'指数成分快照过旧')
    tickers=sorted(mem['ticker'].unique());dl.fetch_stock_metadata()
    dl.fetch_prices_for_tickers(tickers,start,asof);dl.fetch_daily_basic_for_tickers(tickers,start,asof)
    dl.fetch_fundamentals_for_tickers(tickers,start-pd.Timedelta(days=800),asof)
    dl.fetch_stock_limits_for_tickers(tickers,start,asof)
    fetch_corporate_actions(dl,tickers,start,asof,d)
    dl.fetch_sw_industry_membership(tickers)
    # Historical ST data can require a higher Tushare permission tier. For the
    # live signal, stock_basic names still provide a conservative current-name
    # fallback; record the degraded provenance explicitly instead of pretending
    # the history is complete.
    st_history_status = "downloaded"
    try:
        dl.fetch_st_status_for_tickers(tickers,start,asof)
    except Exception as exc:
        pd.DataFrame(
            columns=["ticker","name","trade_date","type","type_name"]
        ).to_csv(d/'st_status.csv',index=False)
        st_history_status = f"fallback_stock_basic_name:{type(exc).__name__}"
    dl.fetch_benchmark('399300.SZ',start,asof)
    files=['prices.csv','raw_prices.csv','trade_calendar.csv','daily_basic.csv','fundamentals_raw.csv','index_membership.csv',
           'stock_metadata.csv','industry_membership.csv','st_status.csv','benchmark.csv','stock_limits.csv','corporate_actions.csv']
    manifest={'schema_version':8,'source':'tushare','complete_universe':True,'as_of':str(asof.date()),
        'created_at':now.isoformat(),'ticker_count':len(tickers),'amount_unit':'CNY','volume_unit':'share',
        'membership_basis':'latest_available_index_weight_snapshot_not_exact_intramonth_changes',
        'financial_vintages':'ann_date_point_in_time_vendor_history_not_independently_archived',
        'point_in_time_policy':{
            'financials':'ann_date strictly before signal_date',
            'index_membership':'latest effective_date not after signal_date',
            'industry':'in_date/out_date active on signal_date',
            'corporate_actions':'implemented distributions applied on ex_date using the ledger record-date snapshot',
            'execution':'post-close signal; paper fills only on a later completed session raw open'
        },
        'refresh_mode':'full_bootstrap',
        'industry_refreshed_at':str(asof.date()),
        'st_history_status':st_history_status,
        'files':{name:hashlib.sha256((d/name).read_bytes()).hexdigest() for name in files}}
    (d/'data_manifest_v8.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    print(f'数据下载完成：{len(tickers)} 只历史成分股，截止 {asof.date()}。请运行 live.py signal。')

if __name__=='__main__':main()
