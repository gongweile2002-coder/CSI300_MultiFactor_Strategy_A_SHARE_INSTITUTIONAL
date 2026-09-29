"""A-share cash-account planning. CNY, shares, raw quotes, Shanghai timestamps.
No presumed fills, no credit for pending sales. Shared by manual and QMT paths.
"""
from __future__ import annotations
import hashlib, json, math, re
from dataclasses import dataclass, asdict
from datetime import time
from decimal import Decimal, ROUND_CEILING, ROUND_FLOOR
from pathlib import Path
import pandas as pd

CURRENT_VERSION='A_SHARE_V9_2_2026_09_28'

class TradingBlocked(ValueError): pass

def require(ok,message):
    if not ok: raise TradingBlocked(message)

def finite(value,label,minimum=0):
    try: x=float(value)
    except (ValueError,TypeError): raise TradingBlocked(f'{label}: 必须为数字') from None
    require(math.isfinite(x) and x>=minimum,f'{label}: 非法数值')
    return x

def integer(value,label):
    x=finite(value,label);require(x.is_integer(),f'{label}: 必须为整数');return int(x)

def strict_bool(v):
    if str(v).strip().lower() in {'true','1'}: return True
    if str(v).strip().lower() in {'false','0'}: return False
    raise TradingBlocked(f'非法布尔值: {v!r}')

def ticker_code(value):
    s=str(value).strip().upper()
    require(bool(re.fullmatch(r'\d{6}\.(SH|SZ)',s)),f'代码须为 000001.SZ / 600000.SH: {s}')
    code,ex=s.split('.')
    require((ex=='SH' and code.startswith(('600','601','603','605','688'))) or
            (ex=='SZ' and code.startswith(('000','001','002','003','300','301'))),
            f'v8 只支持沪深现金账户 A 股股票，不支持 ETF/北交所/融资融券: {s}')
    return s

def board(t):
    ticker_code(t)
    return 'STAR' if t.startswith('688') else 'CHINEXT' if t.startswith(('300','301')) else 'MAIN'

def cn_timestamp(value):
    t=pd.Timestamp(value)
    require(not pd.isna(t) and t.tzinfo is not None,'时间须带时区，例如 2026-09-11T10:00:00+08:00')
    return t.tz_convert('Asia/Shanghai')

def digest(obj):
    return hashlib.sha256(json.dumps(obj,sort_keys=True,ensure_ascii=False,allow_nan=False,separators=(',',':'),default=str).encode()).hexdigest()

def read_csv(path):
    return pd.read_csv(path,dtype={'ticker':str,'ts_code':str,'date':str,'cal_date':str})

class TradingCalendar:
    def __init__(self,frame):
        x=frame.rename(columns={'cal_date':'date'}).copy()
        require({'date','is_open'}<=set(x),'交易日历需要 date/is_open')
        x['date']=pd.to_datetime(x['date'],errors='raise').dt.normalize()
        require(not x.empty and not x['date'].duplicated().any(),'日历为空/重复')
        require(x['date'].dt.tz is None,'日历使用不含时区的日期')
        require(len(x)==len(pd.date_range(x['date'].min(),x['date'].max())),'日历必须包含休市日，不可用工作日代替交易日')
        self.days=dict(zip(x['date'],x['is_open'].map(strict_bool)))
    def is_open(self,date):
        d=pd.Timestamp(str(date)[:10]);require(d in self.days,f'日历不覆盖 {d.date()}');return self.days[d]
    def previous(self,date):
        d=pd.Timestamp(str(date)[:10]);self.is_open(d)
        v=[x for x,o in self.days.items() if o and x<d];require(bool(v),'缺少前一交易日');return max(v)
    def next(self,date):
        d=pd.Timestamp(str(date)[:10]);self.is_open(d)
        v=[x for x,o in self.days.items() if o and x>d];require(bool(v),'缺少下一交易日');return min(v)
    def validate_session(self,now):
        now=cn_timestamp(now);require(self.is_open(now.date()),'当天休市');t=now.time()
        require(time(9,30)<=t<time(11,30) or time(13)<=t<time(14,57),
                '本入口只支持连续竞价，不支持集合竞价、午休或盘后交易')

@dataclass(frozen=True)
class RiskConfig:
    version:str=CURRENT_VERSION
    strategy_id:str='CSI300_CORE_V8'
    max_single_weight:float=.08
    max_sector_weight:float=.30
    max_gross_weight:float=.80
    cash_buffer_pct:float=.05
    max_turnover_per_day:float=.30
    max_order_nav:float=.08
    max_adv_participation:float=.01
    max_spread_bps:float=40
    max_signal_gap_pct:float=.03
    max_daily_loss:float=.025
    max_drawdown:float=.10
    max_quote_age_seconds:int=30
    max_account_age_seconds:int=120
    min_order_cny:float=2000
    rebalance_band_pct:float=.01
    plan_valid_minutes:int=10
    max_orders_per_batch:int=60
    commission_bps:float=3
    minimum_commission:float=5
    transfer_fee_bps:float=.1
    sell_stamp_bps:float=5
    allowed_buy_boards:tuple=('MAIN',)
    @classmethod
    def from_dict(cls,obj):
        require(set(obj)<=set(cls.__dataclass_fields__),'存在未知风控配置项');return cls(**obj).validated()
    def validated(self):
        for k,v in asdict(self).items():
            if k not in {'version','strategy_id','allowed_buy_boards'}: finite(v,k)
        for k in ['max_single_weight','max_sector_weight','max_gross_weight','max_turnover_per_day','max_order_nav','max_adv_participation','max_daily_loss','max_drawdown']:
            require(0<getattr(self,k)<=1,f'{k} 必须位于 (0,1]')
        require(0<=self.cash_buffer_pct<1 and self.max_gross_weight<=1-self.cash_buffer_pct+1e-9,'现金缓冲/总仓位约束非法')
        require(bool(self.allowed_buy_boards) and set(self.allowed_buy_boards)<={'MAIN','STAR','CHINEXT'},'板块权限非法')
        require(float(self.plan_valid_minutes).is_integer() and int(self.plan_valid_minutes)>0,'plan_valid_minutes 必须为正整数')
        require(float(self.max_orders_per_batch).is_integer() and int(self.max_orders_per_batch)>0,'max_orders_per_batch 必须为正整数')
        require(self.version==CURRENT_VERSION,f'配置版本 {self.version!r} 与代码版本 {CURRENT_VERSION!r} 不一致，请升级配置并重新生成信号')
        require(bool(self.strategy_id),'策略ID 不可为空')
        return self

def fee_estimate(side,qty,price,cfg):
    n=int(qty)*float(price)
    if n==0:return 0.
    fee=max(cfg.minimum_commission,n*cfg.commission_bps/10000)+n*(cfg.transfer_fee_bps+(cfg.sell_stamp_bps if side=='SELL' else 0))/10000
    return math.ceil(fee*100-1e-8)/100

def round_price(price,side):
    return float(Decimal(str(price)).quantize(Decimal('.01'),rounding=ROUND_CEILING if side=='BUY' else ROUND_FLOOR))

def round_quantity(raw,ticker,side='BUY',current_qty=0,sellable_qty=0):
    # Conservative shared operational ceiling; prevents oversized SDK orders on any supported board.
    n=min(100000,max(0,int(math.floor(raw+1e-9))))
    if side=='SELL':
        n=min(n,int(current_qty),int(sellable_qty))
        if n==int(current_qty):return n
    if board(ticker)=='STAR':return n if n>=200 else 0
    return n//100*100

def validate_account(a,now,cfg,allow_demo=False):
    require(a.get('source') in ({'broker','synthetic'} if allow_demo else {'broker'}),'需要真实券商快照，不能使用演示账户')
    require(bool(a.get('account_id')),'缺少账户编号')
    stamp=cn_timestamp(a['as_of']);age=(now-stamp).total_seconds()
    require(stamp.date()==now.date() and 0<=age<=cfg.max_account_age_seconds,'账户快照过期或来自未来')
    cash=finite(a['cash_available'],'可用资金');total=finite(a['cash_total'],'总现金')
    require(cash<=total+.01,'可用资金大于总现金')
    nav=finite(a['total_asset'],'总资产',.01);prev=finite(a['prior_close_nav_adjusted'],'调整后昨收净值',.01)
    high=finite(a['high_water_nav_adjusted'],'调整后高水位',.01);require(high>=prev-.01,'高水位低于昨收净值')
    require(a.get('open_order_count')==0,'存在未完成委托，先查询/撤单/对账')
    positions={}
    for r in a['positions']:
        t=ticker_code(r['ticker']);require(t not in positions,'重复持仓')
        qty=integer(r['qty'],'持仓股数');sellable=integer(r['sellable_qty'],'可卖股数')
        require(sellable<=qty,'可卖股数超过持仓');px=finite(r['last_price'],'持仓估值价',.01)
        positions[t]={**r,'qty':qty,'sellable_qty':sellable,'last_price':px}
    calculated=total+sum(p['qty']*p['last_price'] for p in positions.values())
    require(abs(calculated-nav)<=max(1,nav*.002),'账户总资产与持仓、现金不符')
    return positions,nav/prev-1<=-cfg.max_daily_loss or nav/max(high,nav)-1<=-cfg.max_drawdown

def quote_map(frame,names,now,cfg):
    require(not frame['ticker'].duplicated().any(),'重复行情代码');result={}
    for r in frame.to_dict('records'):
        t=ticker_code(r['ticker'])
        if t not in names:continue
        require(r.get('price_basis')=='raw','交易只接受未复权行情')
        stamp=cn_timestamp(r['quote_time'])
        require(stamp.date()==now.date() and 0<=(now-stamp).total_seconds()<=cfg.max_quote_age_seconds,f'{t}: 行情过期或来自未来')
        for k in ['last','bid','ask','up_limit','down_limit','adv20_cny']:r[k]=finite(r[k],f'{t}/{k}')
        require(r['last']>0,'缺少估值价格');require(r['status'] in {'TRADING','SUSPENDED'},'交易状态未知')
        require(str(r.get('industry_l1','')).strip() not in {'','nan','UNKNOWN'},'缺少行业分类')
        r['is_st']=strict_bool(r['is_st']);r['is_delisting']=strict_bool(r['is_delisting']);result[t]=r
    require(names<=set(result),f'缺少行情或持仓估值: {sorted(names-set(result))}');return result

def make_plan(account,targets,quotes,calendar,now,cfg=None,*,allow_demo=False,used_keys=(),used_turnover_cny=0.,kill_switch=False):
    cfg=(cfg or RiskConfig()).validated();now=cn_timestamp(now)
    require(not kill_switch,'总开关已停止交易');calendar.validate_session(now)
    positions,reduce_only=validate_account(account,now,cfg,allow_demo)
    needed={'ticker','target_weight','signal_date','signal_close_raw','source'}
    require(needed<=set(targets),f'目标缺少 {needed-set(targets)}')
    require(not targets.empty and not targets['ticker'].duplicated().any(),'目标为空或重复；全现金目标保留权重为0的行')
    previous=calendar.previous(now.date());target={}
    if not allow_demo:
        require(account.get('risk_reference_date')==str(previous.date()),'账户风险基准日期不是前一交易日，请更新昨收净资产与高水位')
    for r in targets.to_dict('records'):
        t=ticker_code(r['ticker'])
        require(r['source'] in ({'real','synthetic'} if allow_demo else {'real'}),'演示信号不能用于真实委托')
        require(pd.Timestamp(r['signal_date']).normalize()==previous,'信号须来自前一交易日，不接受同日/未来/过期信号')
        r['target_weight']=finite(r['target_weight'],'目标权重');require(r['target_weight']<=cfg.max_single_weight+1e-9,'单股目标超限')
        r['signal_close_raw']=finite(r['signal_close_raw'],'信号日原始收盘价',.01);target[t]=r
    require(sum(r['target_weight'] for r in target.values())<=cfg.max_gross_weight+1e-9,'目标总仓位超限')
    names=set(target)|set(positions);qm=quote_map(quotes,names,now,cfg)
    nav=float(account['cash_total'])+sum(p['qty']*qm[t]['last'] for t,p in positions.items())
    require(abs(nav-account['total_asset'])<=max(1,nav*.005),'行情估值与券商总资产偏差过大')
    reduce_only=reduce_only or nav/account['prior_close_nav_adjusted']-1<=-cfg.max_daily_loss or nav/max(nav,account['high_water_nav_adjusted'])-1<=-cfg.max_drawdown
    gross=sum(p['qty']*qm[t]['last'] for t,p in positions.items());sectors={};target_sectors={}
    for t,r in target.items():
        s=qm[t]['industry_l1'];target_sectors[s]=target_sectors.get(s,0)+r['target_weight']
    require(all(w<=cfg.max_sector_weight+1e-9 for w in target_sectors.values()),'目标行业权重超限')
    for t,p in positions.items():
        s=qm[t]['industry_l1'];sectors[s]=sectors.get(s,0)+p['qty']*qm[t]['last']
    # Original exposures are retained for all BUY checks: sales may fail to fill.
    cash_room=max(0,account['cash_available']-cfg.cash_buffer_pct*nav)
    turn_room=max(0,cfg.max_turnover_per_day*nav-finite(used_turnover_cny,'已使用日换手金额'))
    initial_nav=nav;orders=[];blocked=[]
    def priority(t):
        desired=0 if reduce_only else int(initial_nav*target.get(t,{'target_weight':0})['target_weight']/qm[t]['last'])
        return desired>=positions.get(t,{'qty':0})['qty'],t
    for t in sorted(names,key=priority):
        q=qm[t];pos=positions.get(t,{'qty':0,'sellable_qty':0});r=target.get(t,{'target_weight':0})
        desired=0 if reduce_only else int(nav*r['target_weight']/q['last']);diff=desired-pos['qty']
        if not diff:continue
        side='BUY' if diff>0 else 'SELL';reason=None
        if not reduce_only and r['target_weight']>0 and abs(diff)*q['last']<cfg.rebalance_band_pct*initial_nav:reason='within_rebalance_band'
        key=hashlib.sha256(f"{account['account_id']}|{cfg.strategy_id}|{now.date()}|{t}|{side}".encode()).hexdigest()[:32]
        if key in used_keys:reason='already_submitted_today'
        elif q['status']!='TRADING':reason='suspended'
        elif not 0<q['down_limit']<q['up_limit']:reason='missing_price_limits_or_unlimited_listing'
        elif not 0<q['bid']<=q['ask']:reason='missing_or_crossed_order_book'
        elif (q['ask']-q['bid'])/q['last']*10000>cfg.max_spread_bps:reason='spread_too_wide'
        elif side=='BUY' and (q['is_st'] or q['is_delisting']):reason='st_or_delisting'
        elif side=='BUY' and board(t) not in cfg.allowed_buy_boards:reason='board_permission_missing'
        elif side=='BUY' and q['ask']>=q['up_limit']-.005:reason='limit_up'
        elif side=='SELL' and q['bid']<=q['down_limit']+.005:reason='limit_down'
        elif side=='BUY' and abs(q['last']/r['signal_close_raw']-1)>cfg.max_signal_gap_pct:reason='signal_gap_or_corporate_action_requires_review'
        px=round_price(q['ask'] if side=='BUY' else q['bid'],side)
        if not reason and not q['down_limit']<=px<=q['up_limit']:reason='price_outside_limits'
        cap=min(abs(diff),cfg.max_order_nav*nav/max(px,.01),cfg.max_adv_participation*q['adv20_cny']/max(px,.01),turn_room/max(px,.01))
        if side=='BUY':
            cap=min(cap,max(0,cfg.max_single_weight*nav-pos['qty']*q['last'])/max(px,.01),
                    max(0,cfg.max_gross_weight*nav-gross)/max(px,.01),
                    max(0,cfg.max_sector_weight*nav-sectors.get(q['industry_l1'],0))/max(px,.01),cash_room/max(px,.01))
        qty=round_quantity(cap,t,side,pos['qty'],pos['sellable_qty'])
        while side=='BUY' and qty>0:
            fee=fee_estimate(side,qty,px,cfg)
            if qty*px+fee<=cash_room+1e-8 and gross+qty*px<=cfg.max_gross_weight*(nav-fee)+1e-8 and sectors.get(q['industry_l1'],0)+qty*px<=cfg.max_sector_weight*(nav-fee)+1e-8 and pos['qty']*q['last']+qty*px<=cfg.max_single_weight*(nav-fee)+1e-8:break
            qty=round_quantity(qty-(1 if board(t)=='STAR' else 100),t)
        if not reason and qty<=0:reason='t1_cash_turnover_lot_or_exposure_limit'
        if not reason and side=='BUY' and qty*px<cfg.min_order_cny:reason='below_minimum_economic_order_size'
        if reason:
            blocked.append({'ticker':t,'side':side,'desired_delta':diff,'reason':reason});continue
        fee=fee_estimate(side,qty,px,cfg)
        orders.append({'client_order_id':key,'ticker':t,'side':side,'qty':qty,'limit_price':px,'notional':round(qty*px,2),
                       'estimated_fees':fee,'signal_date':str(previous.date()),'trade_date':str(now.date()),'industry_l1':q['industry_l1'],'reason':'risk_reduce' if reduce_only else 'rebalance'})
        turn_room-=qty*px
        if side=='BUY':
            cash_room-=qty*px+fee;gross+=qty*px;nav-=fee;s=q['industry_l1'];sectors[s]=sectors.get(s,0)+qty*px
    orders.sort(key=lambda o:(o['side']!='SELL',o['ticker']))
    require(len(orders)<=int(cfg.max_orders_per_batch),f'单批委托数量 {len(orders)} 超过上限 {int(cfg.max_orders_per_batch)}，拆分策略或人工复核')
    expires_at=(now+pd.Timedelta(minutes=int(cfg.plan_valid_minutes))).isoformat()
    result={'schema_version':9,'mode':'DEMO' if allow_demo else 'MANUAL_REVIEW','account_id':account['account_id'],
            'strategy_id':cfg.strategy_id,'created_at':now.isoformat(),'expires_at':expires_at,'trade_date':str(now.date()),'signal_date':str(previous.date()),
            'config_hash':digest(asdict(cfg)),'account_hash':digest(account),'orders':orders,'blocked':blocked,'risk_reduce_only':bool(reduce_only),
            'note':'委托计划不是成交回报；买入未使用待成交卖单资金。'}
    result['plan_hash']=digest(result);return result

def verify_plan(plan,now=None,require_fresh=False):
    require(digest({k:v for k,v in plan.items() if k!='plan_hash'})==plan.get('plan_hash'),'计划内容发生修改，重新生成')
    if require_fresh:
        require('expires_at' in plan,'旧计划没有失效时间，请重新生成')
        current=cn_timestamp(now or pd.Timestamp.now(tz='Asia/Shanghai'))
        expires=cn_timestamp(plan['expires_at'])
        require(current<=expires,'计划已过复核有效期，请刷新账户、行情后重新生成')
        require(str(current.date())==str(plan.get('trade_date')),'计划不是当前交易日，请重新生成')

def save_plan(plan,directory):
    d=Path(directory);d.mkdir(parents=True,exist_ok=True)
    (d/'plan.json').write_text(json.dumps(plan,ensure_ascii=False,indent=2),encoding='utf-8')
    cols=['client_order_id','ticker','side','qty','limit_price','notional','estimated_fees','signal_date','trade_date','industry_l1','reason']
    pd.DataFrame(plan['orders'],columns=cols).to_csv(d/'orders.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(plan['blocked'],columns=['ticker','side','desired_delta','reason']).to_csv(d/'blocked.csv',index=False,encoding='utf-8-sig')
