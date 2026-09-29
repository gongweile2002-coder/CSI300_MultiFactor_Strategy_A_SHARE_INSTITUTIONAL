> 历史研究资料：这里的旧命令、路线图和输出不代表 v8 已验证功能或实盘业绩。当前操作以 README.md 和 docs/实盘操作指南.md 为准。


# CSI300 Multi-Factor Strategy — A-SHARE PRO

这是在 ULTIMATE 版本之上，专门针对 **A股交易机制、风格轮动和当前高成交/高轮动环境** 加的一层自适应策略引擎。

## 核心变化

原版本的固定三因子：

```text
Value 1/3 + Quality 1/3 + Momentum 1/3
```

升级为：

```text
市场状态识别
↓
动态风险仓位
↓
动态因子权重
↓
A股特有追涨/拥挤过滤
↓
行业轮动
↓
组合优化
↓
A股交易时段与T+1执行规划
```

---

## 一、A股 Regime Engine

每天/每个调仓日计算：

- 沪深300 20日收益
- 60日收益
- MA20 / MA60 / MA120
- 20日年化波动
- 60日最大回撤
- 股票池站上20日均线比例（Breadth）
- 市场20日成交额 / 60日成交额
- 截面5日收益离散度

分成：

```text
TREND_UP
HIGH_ROTATION
RANGE
RISK_OFF
PANIC
```

---

## 二、动态仓位

默认：

```text
TREND_UP      100%
HIGH_ROTATION  85%
RANGE          75%
RISK_OFF       50%
PANIC          30%
```

现金不是“没选到股票”，而是显式风险预算。

---

## 三、A股增强因子

### 1. Momentum Skip-1M

不是简单使用：

```text
Close(t) / Close(t-126)
```

而是：

```text
Close(t-20) / Close(t-126) - 1
```

即约6个月动量，但跳过最近约1个月。

目的：

- 降低极短期过热
- 避免把刚刚冲高的情绪直接当成中期趋势

### 2. Short-term Reversal

```text
- 5D Return
```

只在 RANGE / HIGH_ROTATION 等环境给较高权重。

### 3. Low Volatility

使用20日年化波动率的反向信号：

```text
Lower Vol → Higher Score
```

在 RISK_OFF / PANIC 中权重明显提高。

### 4. Dividend Yield

读取 Tushare `dv_ttm`。

尤其在风险下降阶段，给高股息/现金回报更高权重。

### 5. Sector Rotation

按历史申万一级行业：

```text
60% × 20D行业相对强度
+
40% × 60D行业相对强度
```

让策略适应A股明显的行业轮动。

### 6. Crowding Penalty

使用：

```text
Turnover Z-score
Volume Ratio
```

极端换手、极端量比会被扣分。

---

## 四、避免A股“追高陷阱”

新开仓默认跳过：

```text
收盘涨停
一字涨停
```

并对：

```text
Volume Ratio > 3
且
5D Return > 8%
```

的股票标记为 `new_buy_block=True`。

目的不是证明它一定会跌，而是避免系统用日频模型去追已经严重情绪化的短线价格。

---

## 五、动态因子权重

### TREND_UP

```text
Momentum Skip    35%
Quality          20%
Sector Rotation  15%
Value            10%
Low Vol          10%
Dividend          5%
Crowding          5%
```

### HIGH_ROTATION

```text
Quality          20%
Momentum Skip    20%
Short Reversal   15%
Sector Rotation  15%
Value            10%
Low Vol          10%
Dividend          5%
Crowding          5%
```

### RANGE

```text
Quality          25%
Value            20%
Short Reversal   20%
Dividend         10%
Low Vol          10%
Momentum Skip     5%
Sector Rotation   5%
Crowding          5%
```

### RISK_OFF

```text
Quality          30%
Low Vol          20%
Dividend         15%
Value            15%
Momentum Skip     5%
Short Reversal    5%
Sector Rotation   5%
Crowding          5%
```

### PANIC

```text
Quality          30%
Low Vol          25%
Dividend         20%
Value            10%
Crowding         10%
Short Reversal    5%
```

同时只有约30%总仓位。

---

## 六、为什么适合当前A股而不是死守固定因子

固定多因子的问题：

```text
科技趋势行情
和
高股息行情
和
快速轮动震荡行情
```

不应该用完全一样的权重。

A-SHARE PRO 通过：

```text
趋势
波动
回撤
市场宽度
成交活跃度
截面分化
```

识别环境，再切换因子和风险敞口。

---

## 七、2026 A股交易机制适配

配置里：

```text
post_close_fixed_price_from = 2026-07-06
```

对2026年7月6日后的当前A股，执行规划默认支持：

```text
next_day_post_close_fixed
```

而不是盲目假设只能 next-open。

原因：

- 当前沪深交易所规则已将盘后固定价格交易扩展到全部A股；
- 盘后固定价格交易使用当日收盘价；
- 对日频/月频策略，可以降低开盘跳空对执行的影响。

但默认使用 **next-day** post-close，而不是 same-day post-close。

因为如果你依赖 Tushare 等盘后更新的每日指标，数据并不保证在15:05前全部准备好。

只有使用经过验证的实时行情/实时基础数据栈，才能考虑：

```text
same_day_post_close_fixed
```

---

## 八、丰富 daily_basic

A-SHARE PRO 下载字段增加：

```text
turnover_rate
turnover_rate_f
volume_ratio
pe
pe_ttm
pb
ps_ttm
dv_ratio
dv_ttm
total_mv
circ_mv
limit_status
```

---

## 九、自由现金流接口

新增可选：

```text
cashflow_raw.csv
```

读取：

```text
ann_date
f_ann_date
report_date
n_cashflow_act
c_pay_acq_const_fiolta
free_cashflow
```

用于下一阶段 FCF Yield / Cash Quality。

该数据仍必须按公告日 Point-in-Time 使用。

---

## 十、真实数据

```bash
python download_ashare_pro_data.py --max-stocks 30
```

确认权限后：

```bash
python download_ashare_pro_data.py
```

---

## 十一、Demo

```bash
python run_ashare_pro_demo.py
```

输出：

```text
outputs/ashare_pro_demo/
├── adaptive_targets.csv
├── scored_cross_section.csv
├── regime_summary.csv
└── execution_plan.csv
```

---

## 十二、实盘推荐结构

不建议把账户全部变成“短线题材追涨”。

推荐结构：

```text
Core 70–85%
  ├─ Quality
  ├─ Value
  ├─ Dividend / Cash Quality
  ├─ Low Vol
  └─ Medium-term Momentum

Tactical 15–30%
  ├─ Sector Rotation
  ├─ Momentum Skip-1M
  ├─ Short-term Reversal
  └─ Crowding Filter

Cash
  └─ Regime Engine 动态决定
```

这比“追涨停板”更适合日频量化实盘。

---

## 十三、当前仍建议增加的数据

如果你以后有更强数据权限，可以继续加入：

- FCF Yield
- Operating Cash Flow / Net Profit
- Earnings Revision
- Analyst Consensus Revision
- 融资余额变化
- 龙虎榜/异常交易只作为拥挤风险，而不是直接追涨
- 股东减持/解禁风险
- 指数调入调出事件
- 业绩预告/快报 Surprise

但必须全部按 **当时可得信息** 做 Point-in-Time。

---

## 十四、实盘底线

A股 T+1 意味着：

> 今天买错，今天不能靠“止损单”卖掉解决。

所以真正重要的是：

```text
买入前仓位控制
+
不要追一字板/极端放量
+
现金仓位
+
组合风险
+
第二天可执行的退出计划
```

策略的目的不是保证赚钱，而是让亏损、拥挤、成交和错误暴露尽量可控。
