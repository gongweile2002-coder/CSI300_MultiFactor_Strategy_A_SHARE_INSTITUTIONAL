from pathlib import Path
import json
import pandas as pd
import numpy as np

from src.research_v4 import build_point_in_time_panel_v4
from src.ashare_factors import augment_ashare_factors
from src.strategy_zoo_elite import build_strategy_sleeves, combine_sleeves
from src.institutional_engine import institutional_enhance_cross_section
from src.ashare_regime import compute_market_regime, regime_weight_map
from src.multi_universe_institutional import select_multi_universe_candidates
from src.portfolio_v5 import estimate_covariance, optimize_portfolio, run_optimized_backtest_v5, portfolio_diagnostics
from src.research_v2 import performance_summary

BASE=Path(__file__).resolve().parent

def read_optional(path,date_cols=None):
    if not path.exists(): return pd.DataFrame()
    x=pd.read_csv(path)
    for c in (date_cols or []):
        if c in x.columns: x[c]=pd.to_datetime(x[c],errors='coerce')
    return x

def main():
    cfg=json.loads((BASE/'config.example.json').read_text(encoding='utf-8'))
    d=BASE/'data'/'real_multi'; out=BASE/'outputs'/'institutional_real'; out.mkdir(parents=True,exist_ok=True)
    prices=pd.read_csv(d/'prices.csv',parse_dates=['date'])
    db=pd.read_csv(d/'daily_basic.csv',parse_dates=['date'])
    fundamentals=pd.read_csv(d/'fundamentals_raw.csv',parse_dates=['ann_date','report_date'])
    membership=pd.read_csv(d/'multi_index_membership.csv',parse_dates=['effective_date'])
    meta=read_optional(d/'stock_metadata.csv',['list_date','delist_date'])
    industry=read_optional(d/'industry_membership.csv',['in_date','out_date'])
    st=read_optional(d/'st_status.csv',['trade_date'])
    limits=read_optional(d/'stock_limits.csv',['date'])
    cashflow=read_optional(d/'cashflow_raw.csv',['ann_date','f_ann_date','pit_ann_date','report_date'])
    forecast=read_optional(d/'forecast_raw.csv',['ann_date','report_date'])
    express=read_optional(d/'express_raw.csv',['ann_date','report_date'])
    bench=pd.read_csv(d/'multi_benchmarks.csv',parse_dates=['date'])
    bench300=bench[bench['universe']=='CSI300'][['date','close']].sort_values('date')

    panels=[]
    for universe,gmem in membership.groupby('universe'):
        p=build_point_in_time_panel_v4(
            prices=prices,daily_basic=db,fundamentals=fundamentals,membership=gmem,
            metadata=meta,industry_membership=industry,st_status=st,
            start_date=cfg['start_date'],end_date=cfg['end_date'],
            momentum_lookback_days=cfg['momentum_lookback_days'],
            winsor_lower=cfg['winsor_lower'],winsor_upper=cfg['winsor_upper'],
            factor_weights=cfg['factor_weights'],min_turnover_rate=cfg['min_turnover_rate'],
            min_market_cap_cny_10k=cfg['min_market_cap_cny_10k'],exclude_st=cfg['exclude_st'],
            min_listing_trading_days=cfg['min_listing_trading_days'],
            liquidity_lookback_days=cfg['liquidity_lookback_days'],liquidity_min_quantile=cfg['liquidity_min_quantile'],
            neutralize_market_cap=cfg['neutralize_market_cap'],neutralize_industry=cfg['neutralize_industry'])
        if not p.empty:
            p['universe']=universe; panels.append(p)
    if not panels: raise RuntimeError('No institutional panels produced')
    panel=pd.concat(panels,ignore_index=True)

    rich_rows=[]; regime_rows=[]; candidate_rows=[]
    base_sleeves=cfg['ashare_elite']['sleeves']
    for sig,g in panel.groupby('signal_date'):
        try:
            reg=compute_market_regime(prices,bench300,pd.Timestamp(sig),universe_tickers=g['ticker'].astype(str).tolist(),
                breadth_ma_days=cfg['ashare_pro']['breadth_ma_days'],
                turnover_short_days=cfg['ashare_pro']['turnover_short_days'],turnover_long_days=cfg['ashare_pro']['turnover_long_days'])
        except Exception as exc:
            raise RuntimeError(f'{sig}: 市场状态计算失败') from exc
        regime_rows.append(reg)
        ax=augment_ashare_factors(g,prices,db,industry,pd.Timestamp(sig),cfg)
        ax=build_strategy_sleeves(ax)
        ax['ml_score']=0.0  # real ML sleeve should be added only after strict historical OOS training
        ax=combine_sleeves(ax,base_sleeves)
        ax=institutional_enhance_cross_section(ax,cashflow,forecast,express,pd.Timestamp(sig),cfg)
        ax['institutional_score']=(0.70*pd.to_numeric(ax['elite_score'],errors='coerce').fillna(0.0)
            +0.10*pd.to_numeric(ax['fcf_yield_score'],errors='coerce').fillna(0.0)
            +0.10*pd.to_numeric(ax['cash_quality_score'],errors='coerce').fillna(0.0)
            +0.10*pd.to_numeric(ax['earnings_event_score'],errors='coerce').fillna(0.0))
        ax['regime']=reg['regime']; rich_rows.append(ax)
        cand=select_multi_universe_candidates(ax,reg['regime'],cfg,score_col='institutional_score',total_names=max(int(cfg['max_names']*2),60))
        if not cand.empty: candidate_rows.append(cand)

    rich=pd.concat(rich_rows,ignore_index=True); rich.to_csv(out/'institutional_factor_panel.csv',index=False)
    regimes=pd.DataFrame(regime_rows); regimes.to_csv(out/'regime_history.csv',index=False)
    candidates=pd.concat(candidate_rows,ignore_index=True); candidates.to_csv(out/'candidate_history.csv',index=False)

    # Benchmark-aware portfolio optimization using candidates, then scale gross exposure by regime.
    targets=[]; risks=[]; prev=pd.Series(dtype=float)
    for sig,g in candidates.groupby('signal_date'):
        names=g.sort_values('institutional_score',ascending=False).head(int(cfg['max_names']))['ticker'].astype(str).tolist()
        cov,cov_end=estimate_covariance(prices,names,pd.Timestamp(sig),lookback_days=cfg['risk_lookback_days'],shrinkage=cfg['covariance_shrinkage'])
        if cov.empty or len(cov)<2: raise RuntimeError(f'{sig}: 协方差不足')
        z=g.copy(); z['composite_score']=z['institutional_score']
        try:
            t,r=optimize_portfolio(z,cov,prev_weights=prev,max_names=cfg['max_names'],max_stock_weight=cfg['max_stock_weight'],
                min_stock_weight=cfg['min_stock_weight'],sector_active_limit=cfg['sector_active_limit'],
                tracking_error_limit_annual=cfg['tracking_error_limit_annual'],alpha_strength=cfg['alpha_strength'],
                risk_aversion=cfg['risk_aversion'],turnover_penalty=cfg['turnover_penalty'])
        except Exception as exc:
            raise RuntimeError(f'{sig}: 组合优化失败') from exc
        regime=str(g['regime'].iloc[0]) if 'regime' in g.columns else 'RANGE'
        gross,_=regime_weight_map(regime,cfg)
        t['target_weight_full_invested']=t['target_weight']; t['target_weight']*=gross; t['cash_weight']=1-t['target_weight'].sum()
        t['signal_date']=pd.Timestamp(sig); t['regime']=regime; t['covariance_end_date']=cov_end
        targets.append(t); prev=t.set_index('ticker')['target_weight']
        risks.append({'signal_date':sig,'regime':regime,'gross_exposure':gross,'ex_ante_vol_annual':r.get('ex_ante_vol_annual',np.nan),'ex_ante_tracking_error_annual':r.get('ex_ante_tracking_error_annual',np.nan)})
    if not targets: raise RuntimeError('No optimized targets')
    targets=pd.concat(targets,ignore_index=True); targets.to_csv(out/'optimized_targets_institutional.csv',index=False)
    pd.DataFrame(risks).to_csv(out/'risk_history.csv',index=False)
    portfolio_diagnostics(targets).to_csv(out/'portfolio_diagnostics.csv',index=False)

    ret,holds,trades=run_optimized_backtest_v5(prices,targets,limits,commission_bps=cfg['commission_bps'],slippage_bps=cfg['slippage_bps'],use_price_limits=cfg['use_price_limits'])
    holds.to_csv(out/'executed_holdings.csv',index=False); trades.to_csv(out/'trade_costs.csv',index=False)
    bret=bench300.set_index('date')['close'].pct_change().fillna(0).reindex(ret.index).fillna(0)
    active=ret.index[ret.ne(0)]
    if len(active): ret=ret.loc[active.min():]; bret=bret.loc[active.min():]
    pd.DataFrame([performance_summary(ret,bret)]).to_csv(out/'performance_summary.csv',index=False)
    pd.DataFrame({'date':ret.index,'strategy_return':ret.values,'benchmark_return':bret.values,'strategy_nav':(1+ret).cumprod().values,'benchmark_nav':(1+bret).cumprod().values}).to_csv(out/'nav_timeseries.csv',index=False)
    print('Institutional real pipeline completed:',out)

if __name__=='__main__': main()
