"""Optional MiniQMT SDK adapter. Requires the user's broker Windows environment.
Official contracts: https://dict.thinktrader.net/nativeApi/xttrader.html and xtdata.html.
Offline tests use SDK fixtures; no claim of a certified live connection.
"""
import os,time
from pathlib import Path
import pandas as pd
from .live_v8 import require,cn_timestamp,strict_bool,validate_account,quote_map,board,round_quantity,fee_estimate

class MiniQMT:
    def __init__(self,reference,risk_reference,calendar,config,*,enable_submission=False,requested_tickers=None):
        from xtquant import xtdata,xtconstant
        from xtquant.xttrader import XtQuantTrader
        from xtquant.xttype import StockAccount
        path=os.getenv('QMT_USERDATA_PATH','');account_id=os.getenv('QMT_ACCOUNT_ID','')
        require(bool(path) and Path(path).is_dir(),'请在券商 Windows 电脑设置 QMT_USERDATA_PATH');require(bool(account_id),'缺少 QMT_ACCOUNT_ID')
        self.xtdata=xtdata;self.constant=xtconstant;self.account=StockAccount(account_id,'STOCK')
        self.trader=XtQuantTrader(path,int(time.time()*1000)%2147483647);self.trader.start()
        require(self.trader.connect()==0,'MiniQMT 连接失败');require(self.trader.subscribe(self.account)==0,'账户订阅失败')
        self.reference=reference;self.risk_reference=risk_reference;self.calendar=calendar;self.cfg=config
        self.enable_submission=enable_submission;self.requested_tickers=set(requested_tickers or [])
        self.own_ids=set();self.accepted_buy_value={}
    def close(self):self.trader.stop()
    def orders(self):
        orders=self.trader.query_stock_orders(self.account,False);require(orders is not None,'委托查询失败');return orders
    def terminal_statuses(self):
        c=self.constant;return {c.ORDER_SUCCEEDED,c.ORDER_CANCELED,c.ORDER_PART_CANCEL,c.ORDER_JUNK}
    def snapshot(self,now=None):
        now=cn_timestamp(now or pd.Timestamp.now(tz='Asia/Shanghai'));self.calendar.validate_session(now)
        require(str(self.risk_reference.get('as_of'))==str(self.calendar.previous(now.date()).date()),'风险参考净值须更新到前一交易日并调整出入金')
        a=self.trader.query_stock_asset(self.account);pos=self.trader.query_stock_positions(self.account)
        require(a is not None and pos is not None,'券商资金/持仓查询失败')
        active=[o for o in self.orders() if o.order_status not in self.terminal_statuses()]
        require(bool(self.requested_tickers),'需要本次目标股票清单 --targets')
        names=self.requested_tickers|{p.stock_code for p in pos if p.volume>0};ticks=self.xtdata.get_full_tick(sorted(names))
        require(ticks is not None,'实时行情查询失败');refs=self.reference.set_index('ticker');rows=[]
        for t in sorted(names):
            require(t in refs.index and t in ticks,f'{t}: 缺少行情、行业或 ADV 资料');ref=refs.loc[t];tick=ticks[t]
            require(str(ref['signal_date'])==str(self.calendar.previous(now.date()).date()),'参考资料过期')
            detail=self.xtdata.get_instrument_detail(t);require(detail is not None,'证券资料缺失')
            status=int(tick.get('stockStatus',0));require(status in {13,17,20},f'{t}: 交易状态未知/非连续竞价 {status}')
            stamp=pd.Timestamp(int(tick['time']),unit='ms',tz='UTC').tz_convert('Asia/Shanghai');name=str(detail.get('InstrumentName',''))
            require(bool(name.strip()),'证券名称缺失，无法复核风险标记')
            rows.append({'ticker':t,'quote_time':stamp.isoformat(),'last':tick['lastPrice'],
              'bid':(tick.get('bidPrice') or [0])[0],'ask':(tick.get('askPrice') or [0])[0],
              'up_limit':detail.get('UpStopPrice',0),'down_limit':detail.get('DownStopPrice',0),
              'status':'TRADING' if status==13 else 'SUSPENDED','price_basis':'raw','adv20_cny':ref['adv20_cny'],
              'industry_l1':ref['industry_l1'],'is_st':strict_bool(ref['is_st']) or 'ST' in name.upper(),
              'is_delisting':strict_bool(ref['is_delisting']) or '退' in name})
        account={'source':'broker','account_id':self.account.account_id,'as_of':now.isoformat(),
                 'cash_available':float(a.cash),'cash_total':float(a.cash)+float(a.frozen_cash),'total_asset':float(a.total_asset),
                 'open_order_count':len(active),'risk_reference_date':str(self.risk_reference['as_of']),'prior_close_nav_adjusted':self.risk_reference['prior_close_nav_adjusted'],
                 'high_water_nav_adjusted':self.risk_reference['high_water_nav_adjusted'],
                 'positions':[{'ticker':p.stock_code,'qty':int(p.volume),'sellable_qty':int(p.can_use_volume),'last_price':float(p.market_value)/p.volume} for p in pos if p.volume>0]}
        return account,pd.DataFrame(rows)
    def submit_checked(self,o):
        require(self.enable_submission,'QMT 默认只读')
        require(os.getenv('QMT_ENABLE_LIVE')=='YES_I_UNDERSTAND_REAL_ORDERS','尚未在本机明确开启真实委托')
        now=pd.Timestamp.now(tz='Asia/Shanghai');self.calendar.validate_session(now)
        require(str(now.date())==o['trade_date'],'委托交易日不符');a,quotes=self.snapshot(now)
        active=[x for x in self.orders() if x.order_status not in self.terminal_statuses()]
        require(all(str(x.order_id) in self.own_ids for x in active),'账户出现本批次之外的活动委托')
        a['open_order_count']=0;positions,reduce_only=validate_account(a,now,self.cfg)
        q=quote_map(quotes,{o['ticker']},now,self.cfg)[o['ticker']];qty=o['qty'];px=o['limit_price'];side=o['side']
        require(q['status']=='TRADING' and 0<q['down_limit']<=px<=q['up_limit'],'停牌/价格超限')
        pos=positions.get(o['ticker'],{'qty':0,'sellable_qty':0})
        require(round_quantity(qty,o['ticker'],side,pos['qty'],pos['sellable_qty'])==qty,'股数/可卖数量变化')
        require(0<q['bid']<=q['ask'] and (q['ask']-q['bid'])/q['last']*10000<=self.cfg.max_spread_bps,'最新盘口异常或价差超限')
        if side=='BUY':
            ref=self.reference.set_index('ticker').loc[o['ticker']]
            require(abs(q['last']/float(ref['signal_close_raw'])-1)<=self.cfg.max_signal_gap_pct,'最新行情偏离信号过大/除权需复核')
            require(not reduce_only,'触发日亏损或回撤限制');require(board(o['ticker']) in self.cfg.allowed_buy_boards,'板块权限不足')
            require(not q['is_st'] and not q['is_delisting'],'ST/退市风险不买入')
            require(0<q['ask']<q['up_limit']-.005 and px<=q['ask']*1.02+1e-9,'涨停/无卖盘/价格笼子超限')
            fee=fee_estimate(side,qty,px,self.cfg);nav=a['total_asset']-fee
            require(qty*px+fee<=a['cash_available']-self.cfg.cash_buffer_pct*a['total_asset'],'可用资金不足')
            pending=list(self.accepted_buy_value.values())
            # Conservative double counting of already-filled own buys is intentional.
            gross=sum(p['qty']*p['last_price'] for p in positions.values())+sum(v for v,s in pending)
            require(gross+qty*px<=self.cfg.max_gross_weight*nav,'实际/未成交买入总仓位超限')
            require(pos['qty']*q['last']+qty*px+self.accepted_buy_value.get(o['ticker'],(0,''))[0]<=self.cfg.max_single_weight*nav,'单股仓位超限')
            refs=self.reference.set_index('ticker')
            sector=sum(p['qty']*p['last_price'] for t,p in positions.items() if refs.loc[t,'industry_l1']==q['industry_l1'])+sum(v for v,s in pending if s==q['industry_l1'])
            require(sector+qty*px<=self.cfg.max_sector_weight*nav,'行业仓位超限')
        else:require(0<q['bid'] and q['bid']>q['down_limit']+.005 and px>=q['bid']*.98-1e-9,'跌停/无买盘/价格笼子超限')
        require(qty*px<=self.cfg.max_order_nav*a['total_asset']+1e-8,'单笔金额超限')
        require(qty*px<=self.cfg.max_adv_participation*q['adv20_cny']+1e-8,'流动性容量超限')
        result=self.trader.order_stock(self.account,o['ticker'],self.constant.STOCK_BUY if side=='BUY' else self.constant.STOCK_SELL,
                   qty,self.constant.FIX_PRICE,px,self.cfg.strategy_id,o['client_order_id'])
        if result and int(result)>0:
            self.own_ids.add(str(result))
            if side=='BUY':self.accepted_buy_value[o['ticker']]=(qty*px,q['industry_l1'])
        return result
    def reconcile(self,journal):
        c=self.constant;mapping={c.ORDER_SUCCEEDED:'FILLED',c.ORDER_CANCELED:'CANCELLED',c.ORDER_PART_CANCEL:'CANCELLED',c.ORDER_JUNK:'REJECTED'}
        known={r['client_order_id']:r for r in journal.rows(self.account.account_id)}
        for o in self.orders():
            key=str(o.order_remark)
            if key not in known:continue
            require(str(o.stock_code)==known[key]['ticker'] and int(o.order_volume)==known[key]['qty'],'券商回报与原委托不一致')
            expected=c.STOCK_BUY if known[key]['side']=='BUY' else c.STOCK_SELL
            require(o.order_type==expected,'券商回报买卖方向不符')
            journal.update(key,mapping.get(o.order_status,'PARTIAL' if o.traded_volume else 'SUBMITTED'),str(o.order_id),int(o.traded_volume),float(o.traded_price))
        return journal.rows(self.account.account_id)
