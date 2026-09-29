> 历史研究资料：这里的旧命令、路线图和输出不代表 v8 已验证功能或实盘业绩。当前操作以 README.md 和 docs/实盘操作指南.md 为准。

# CSI300 / CSI500 / CSI1000 — A-SHARE INSTITUTIONAL

这是在 A-SHARE ELITE 上继续推进的一层，重点不再是“继续堆技术指标”，而是补齐更接近机构化 A 股实盘研究的四块：**多股票池、现金流质量、业绩事件、ETF 轮动与策略生命周期**。

## 1. 多股票池

支持历史 CSI300 / CSI500 / CSI1000，并按 Regime 动态分配候选名额：

- TREND_UP：55% / 30% / 15%
- HIGH_ROTATION：45% / 35% / 20%
- RANGE：65% / 25% / 10%
- RISK_OFF：80% / 20% / 0%
- PANIC：90% / 10% / 0%

小盘暴露不再永久存在，而是随风险状态变化。

## 2. FCF Yield + Cash Quality

新增：

```text
FCF Yield = Free Cash Flow / Market Cap
Cash Quality = Operating Cash Flow / Net Profit
```

都使用公告日可得数据做 Point-in-Time 匹配，并做行业/市值中性化后再进入 Core Sleeve。

## 3. Earnings Event

使用两类公告：

- forecast：预增/预减/扭亏/首亏、净利润区间、同报告期修正
- express：YoY Net Profit、YoY Sales、Diluted ROE

它不是“分析师一致预期 Surprise”的冒充版；没有一致预期数据时，只把它称为 **Earnings Event / Guidance Revision Proxy**。

事件信号默认 120 天衰减。

## 4. ETF Rotation Sleeve

不硬编码任何“推荐 ETF”。ETF 观察池由用户自己维护：

```text
config/etf_watchlist.example.csv
```

评分使用：

```text
35% 20D Momentum
30% 60D Momentum
20% 120D Momentum
15% Low Volatility
+ 60D MA Trend Filter
+ 20D Liquidity
```

Top-K 使用 inverse-volatility 权重。

## 5. 策略生命周期

每个 Sleeve 进入：

```text
ACTIVE / WATCH / REDUCE / PAUSE
```

主要依据 6 个月与 12 个月 OOS Rank IC。策略失效后降低未来权重，而不是删除历史结果或偷偷重新拟合。

ML Sleeve 继续设置硬上限，Core Sleeve 保留最低治理权重。

## 6. 真实数据接口新增

`TushareDownloaderInstitutional` 增加：

```text
multi_index_membership.csv
forecast_raw.csv
express_raw.csv
etf_daily.csv
```

原 A-SHARE PRO 已有 `cashflow_raw.csv`。

历史指数成分继续使用 `index_weight`，避免 survivorship bias。

## 7. Demo

```bash
python run_institutional_demo.py
```

输出：

```text
outputs/institutional_demo/
├── latest_institutional_ranking.csv
├── multi_universe_candidates.csv
├── strategy_lifecycle.csv
├── dynamic_sleeve_weights.csv
├── etf_rotation.csv
└── universe_slots_*.csv
```

## 8. 当前结构

```text
A股 Portfolio
│
├── Stock Sleeve
│   ├── CSI300
│   ├── CSI500
│   └── CSI1000
│
├── Alpha
│   ├── Quality / Value
│   ├── FCF Yield / Cash Quality
│   ├── Momentum Skip-1M / Reversal
│   ├── Sector Rotation
│   ├── Earnings Event
│   └── ML Rank
│
├── ETF Sleeve
├── Cash
└── Regime Engine
```

## 9. 下一层才值得继续的方向

- 股票 / ETF / Cash 的组合级动态风险预算
- 可靠一致预期数据后的真正 Earnings Revision
- 解禁 / 减持 / 股东人数作为风险惩罚
- Barra-like A 股风险模型
- 固定版本连续 Forward Paper Test（不再每天调参）

做到这里后，“停止改模型、开始验证未来表现”本身就是重要的研究纪律。

## 10. 真实运行

先下载三层股票池数据。默认每个指数只取50只用于权限/流程验证：

```bash
python download_institutional_data.py --max-stocks-per-universe 50
```

全量运行：

```bash
python download_institutional_data.py --max-stocks-per-universe 0
```

然后：

```bash
python run_institutional_real.py
```

输出：

```text
outputs/institutional_real/
├── institutional_factor_panel.csv
├── regime_history.csv
├── candidate_history.csv
├── optimized_targets_institutional.csv
├── risk_history.csv
├── portfolio_diagnostics.csv
├── executed_holdings.csv
├── trade_costs.csv
├── performance_summary.csv
└── nav_timeseries.csv
```

真实 ML Sleeve 在该 runner 中默认权重为0，只有完成严格历史 Walk-forward 训练并确认 OOS 有效后才应该打开，避免把 demo ML 直接带入真钱流程。
