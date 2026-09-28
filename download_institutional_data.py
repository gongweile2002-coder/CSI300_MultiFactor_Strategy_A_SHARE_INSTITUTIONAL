import argparse, json, os
from pathlib import Path
import pandas as pd
from src.tushare_provider_institutional import TushareDownloaderInstitutional
from src.tushare_provider import _yyyymmdd

BASE=Path(__file__).resolve().parent

def load_dotenv(path):
    if not path.exists(): return
    for line in path.read_text(encoding='utf-8').splitlines():
        line=line.strip()
        if not line or line.startswith('#') or '=' not in line: continue
        k,v=line.split('=',1); os.environ.setdefault(k.strip(),v.strip())

def main():
    p=argparse.ArgumentParser(description='Download CSI300/500/1000 institutional research data')
    p.add_argument('--max-stocks-per-universe',type=int,default=50,help='Safety cap; use 0 for full universe')
    p.add_argument('--sleep',type=float,default=0.06)
    p.add_argument('--etf-watchlist',default=str(BASE/'config'/'etf_watchlist.csv'))
    args=p.parse_args()
    cfg=json.loads((BASE/'config.example.json').read_text(encoding='utf-8'))
    load_dotenv(BASE/'.env')
    out=BASE/'data'/'real_multi'; out.mkdir(parents=True,exist_ok=True)
    dl=TushareDownloaderInstitutional(token=os.getenv('TUSHARE_TOKEN',''),output_dir=out,sleep_seconds=args.sleep)

    universes=cfg['institutional']['universes']
    membership=dl.fetch_multi_index_membership(universes,cfg['start_date'],cfg['end_date'])

    selected=[]
    cap=int(args.max_stocks_per_universe)
    for u,g in membership.groupby('universe'):
        names=sorted(g['ticker'].dropna().astype(str).unique().tolist())
        if cap>0: names=names[:cap]
        selected.extend(names)
    tickers=sorted(set(selected))
    pd.DataFrame({'ticker':tickers}).to_csv(out/'selected_tickers.csv',index=False)

    start_ext=pd.Timestamp(cfg['start_date'])-pd.Timedelta(days=300)
    fin_start=pd.Timestamp(cfg['start_date'])-pd.Timedelta(days=900)
    dl.fetch_stock_metadata()
    dl.fetch_prices_for_tickers(tickers,start_ext,cfg['end_date'])
    dl.fetch_daily_basic_for_tickers(tickers,start_ext,cfg['end_date'])
    dl.fetch_fundamentals_for_tickers(tickers,fin_start,cfg['end_date'])
    dl.fetch_cashflow_for_tickers(tickers,fin_start,cfg['end_date'])
    dl.fetch_forecast_for_tickers(tickers,fin_start,cfg['end_date'])
    dl.fetch_express_for_tickers(tickers,fin_start,cfg['end_date'])
    dl.fetch_stock_limits_for_tickers(tickers,start_ext,cfg['end_date'])
    dl.fetch_sw_industry_membership(tickers)
    try:
        dl.fetch_st_status_for_tickers(tickers,start_ext,cfg['end_date'])
    except Exception:
        pd.DataFrame(columns=['ticker','name','trade_date','type','type_name']).to_csv(out/'st_status.csv',index=False)

    # Benchmarks for all three universes, stored together.
    frames=[]
    for u,code in universes.items():
        df=dl.pro.index_daily(ts_code=code,start_date=_yyyymmdd(start_ext),end_date=_yyyymmdd(cfg['end_date']))
        dl._pause()
        if df is not None and not df.empty:
            df=df.rename(columns={'trade_date':'date','ts_code':'index_code'}); df['universe']=u; frames.append(df)
    bench=pd.concat(frames,ignore_index=True) if frames else pd.DataFrame()
    if not bench.empty: bench['date']=pd.to_datetime(bench['date'],errors='coerce')
    bench.to_csv(out/'multi_benchmarks.csv',index=False)

    wl=Path(args.etf_watchlist)
    if wl.exists():
        try: dl.fetch_etf_daily_watchlist(wl,start_ext,cfg['end_date'])
        except Exception as e: print('ETF watchlist skipped:',type(e).__name__)

    print(f'Downloaded institutional data for {len(tickers)} unique stocks into {out}')
    if cap>0: print(f'Safety cap active: max {cap} stocks per universe. Use --max-stocks-per-universe 0 for full run.')

if __name__=='__main__': main()
