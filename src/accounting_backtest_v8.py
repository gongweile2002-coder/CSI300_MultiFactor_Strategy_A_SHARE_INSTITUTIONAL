"""Cash-accounted TOTAL-RETURN research simulation, not a broker tax-lot simulator.
Holds adjusted economic units; raw prices determine limit checks and approximate
lot sizing. Adjusted prices assume reinvested gross distributions. Dividend tax,
payment dates, rights elections and delisting recovery are not modelled.
"""
import numpy as np
import pandas as pd
from .research_v4 import stamp_duty_rate

def run_accounting_backtest(prices,targets,stock_limits=None,commission_bps=3.,slippage_bps=2.,use_price_limits=True,
        initial_cash=500000.,minimum_commission=5.,transfer_fee_bps=.1,max_adv_participation=.01):
    p=prices.copy();p['date']=pd.to_datetime(p['date'])
    if p.duplicated(['date','ticker']).any():raise ValueError('重复行情主键')
    if initial_cash<=0:raise ValueError('初始资金须为正')
    for c in ['open','close']:
        if c not in p or ((p[c]<=0)&p[c].notna()).any():raise ValueError(f'价格非法: {c}')
    if {'raw_open','raw_close'}<=set(p):pass
    elif 'price_basis' in p and p['price_basis'].eq('raw').all():
        if 'adj_factor' not in p:raise ValueError('原始行情缺少复权因子')
        p['raw_open']=p['open'];p['raw_close']=p['close'];p['open']*=p['adj_factor'];p['close']*=p['adj_factor']
    else:
        if stock_limits is not None and not stock_limits.empty:raise ValueError('行情复权口径未标注，不能与原始涨跌停比较')
        p['raw_open']=p['open'];p['raw_close']=p['close']
    dates=pd.DatetimeIndex(sorted(p['date'].unique()));bars={d:g.set_index('ticker') for d,g in p.groupby('date')}
    limits={}
    if stock_limits is not None and not stock_limits.empty:
        x=stock_limits.copy();x['date']=pd.to_datetime(x['date'])
        if x.duplicated(['date','ticker']).any():raise ValueError('重复涨跌停主键')
        limits={(r.date,r.ticker):(r.up_limit,r.down_limit) for r in x.itertuples()}
    t=targets.copy();t['signal_date']=pd.to_datetime(t['signal_date'])
    if t.duplicated(['signal_date','ticker']).any() or not np.isfinite(t['target_weight']).all() or (t['target_weight']<0).any():raise ValueError('目标权重非法或重复')
    if (t.groupby('signal_date')['target_weight'].sum()>1+1e-8).any():raise ValueError('不支持杠杆')
    schedule={}
    for sig,g in t.groupby('signal_date'):
        i=dates.searchsorted(sig,side='right')
        if i<len(dates):
            if dates[i] in schedule:raise ValueError('多组信号对应一个执行日')
            schedule[dates[i]]=(sig,g.set_index('ticker')['target_weight'].to_dict())
    adv={}
    if 'amount' in p:
        p=p.sort_values(['ticker','date']);p['adv']=p.groupby('ticker')['amount'].transform(lambda s:s.shift().rolling(20,min_periods=1).mean())
        adv={(r.date,r.ticker):r.adv for r in p.itertuples()}
    cash=float(initial_cash);units={};marks={};last_nav=cash;returns=[];holdings=[];trades=[];stale_days=0
    for day in dates:
        b=bars[day]
        for n in units:
            if n in b.index and np.isfinite(b.loc[n,'open']):marks[n]=float(b.loc[n,'open'])
        opening_nav=cash+sum(u*marks[n] for n,u in units.items())
        if day in schedule:
            sig,desired=schedule[day];names=sorted(set(units)|set(desired));old={n:units.get(n,0)*marks.get(n,0) for n in names}
            sell=[n for n in names if old[n]>desired.get(n,0)*opening_nav+1e-7]
            buy=[n for n in names if old[n]<desired.get(n,0)*opening_nav-1e-7]
            total={k:0. for k in ['buys','sells','commission','stamp','slip','transfer','blocked_buys','blocked_sells']}
            for n in sell+buy:
                side='SELL' if n in sell else 'BUY';blocked='blocked_buys' if side=='BUY' else 'blocked_sells'
                if n not in b.index or not np.isfinite(b.loc[n,'open']) or not np.isfinite(b.loc[n,'raw_open']):total[blocked]+=1;continue
                row=b.loc[n];op=float(row['open']);raw=float(row['raw_open'])
                if raw<=0 or op<=0 or ('volume' in row and (not np.isfinite(row['volume']) or row['volume']<=0)):total[blocked]+=1;continue
                up,dn=limits.get((day,n),(np.nan,np.nan))
                if use_price_limits and (not np.isfinite(up) or not np.isfinite(dn) or (side=='BUY' and raw>=up-.005) or (side=='SELL' and raw<=dn+.005)):total[blocked]+=1;continue
                change=abs(desired.get(n,0)*opening_nav-old[n]);a=adv.get((day,n),np.nan)
                if np.isfinite(a):change=min(change,max_adv_participation*a)
                step=1 if str(n).startswith('688') else 100;minimum=200 if step==1 else 100
                qty=int(change/raw)//step*step
                if qty<minimum and not(side=='SELL' and desired.get(n,0)==0):total[blocked]+=1;continue
                amount=min(qty*raw,old[n]) if side=='SELL' else qty*raw
                if side=='SELL' and desired.get(n,0)==0 and change>=old[n]-1e-6:amount=old[n]
                def cost(v):
                    return (max(minimum_commission,v*commission_bps/10000) if v else 0,
                        v*stamp_duty_rate(day) if side=='SELL' else 0,v*slippage_bps/10000,v*transfer_fee_bps/10000)
                while side=='BUY' and amount>0 and amount+sum(cost(amount))>cash+1e-9:
                    qty-=step;amount=qty*raw if qty>=minimum else 0
                if amount<=0:total[blocked]+=1;continue
                comm,stamp,slip,transfer=cost(amount);fee=comm+stamp+slip+transfer
                if side=='SELL' and cash+amount-fee<0:total[blocked]+=1;continue
                if side=='SELL':units[n]=max(0,units.get(n,0)-amount/op);cash+=amount-fee;total['sells']+=amount
                else:units[n]=units.get(n,0)+amount/op;cash-=amount+fee;total['buys']+=amount
                marks[n]=op
                for k,v in [('commission',comm),('stamp',stamp),('slip',slip),('transfer',transfer)]:total[k]+=v
            units={n:u for n,u in units.items() if u>1e-12}
            if cash < -1e-6:raise AssertionError('negative cash')
            nav=cash+sum(u*marks[n] for n,u in units.items())
            for n,u in units.items():holdings.append({'signal_date':sig,'execution_date':day,'ticker':n,'executed_weight':u*marks[n]/nav,'cash_weight':cash/nav})
            trades.append({'signal_date':sig,'execution_date':day,'gross_turnover':(total['buys']+total['sells'])/opening_nav,
                'buy_turnover':total['buys']/opening_nav,'sell_turnover':total['sells']/opening_nav,
                'commission_cost':total['commission']/opening_nav,'stamp_duty_cost':total['stamp']/opening_nav,
                'slippage_cost':total['slip']/opening_nav,'transfer_cost':total['transfer']/opening_nav,
                'total_cost':sum(total[k] for k in ['commission','stamp','slip','transfer'])/opening_nav,
                'blocked_buy_count':int(total['blocked_buys']),'blocked_sell_count':int(total['blocked_sells']),'cash_weight':cash/nav})
        stale=False
        for n in units:
            if n in b.index and np.isfinite(b.loc[n,'close']):marks[n]=float(b.loc[n,'close'])
            else:stale=True
        stale_days+=int(stale);closing_nav=cash+sum(u*marks[n] for n,u in units.items())
        returns.append(closing_nav/last_nav-1);last_nav=closing_nav
    result=pd.Series(returns,index=dates,name='strategy_return')
    result.attrs.update(model='cash_accounted_adjusted_units_gross_distributions',stale_valuation_days=stale_days,
        not_modelled='dividend_tax/payment_dates, rights_elections, order_book, delisting_recovery')
    return result,pd.DataFrame(holdings),pd.DataFrame(trades)
