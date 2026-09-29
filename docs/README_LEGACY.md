> 旧版研究说明归档。semi_live 和 production_check 的使用方式已经替换；v8 实盘操作以根目录 README 和实盘操作指南为准。

# CSI300 Multi-Factor Strategy — A-SHARE PRO / ULTIMATE

> 新增 A股自适应 Regime、行业轮动、Momentum Skip-1M、短期反转、低波、股息率、拥挤过滤与2026盘后固定价格执行规划。详见 `README_A_SHARE_PRO.md`。


# CSI300 Multi-Factor Strategy — ULTIMATE
准生产级个人量化研究、组合构建与交易运营框架

> **默认仅支持 Research / Paper Trading / Manual Approval。**
> 项目故意不包含可直接提交真实券商订单的实现，Live Broker Submission 默认永久关闭，除非你另行开发、审查并明确启用自己的合规券商适配器。

---

## 现在已经覆盖的完整链路

```text
Historical CSI300 Constituents
↓
Point-in-Time Fundamentals
↓
ST / Listing Age / Liquidity Filters
↓
Value / Quality / Momentum
↓
Winsorization + Standardization
↓
Industry + Size Neutralization
↓
Composite Alpha
↓
Shrinkage Covariance Risk Model
↓
Benchmark-aware Portfolio Optimization
↓
Single-name / Sector / TE / Turnover Constraints
↓
Latest Target Weights
↓
Board-lot Sizing
↓
T+1 / Cash / Position / Price-limit Checks
↓
Capacity / ADV Participation Checks
↓
Restricted List / Compliance Checks
↓
Manual Approval Gate
↓
Paper Execution
↓
FIFO Cost / Realized & Unrealized P&L
↓
Reconciliation
↓
Stress / VaR / CVaR
↓
Audit Log / Run Manifest / SQLite Ledger
↓
Dashboard / HTML Research Report
↓
CI / Docker / Production Readiness Check
```

---

# ULTIMATE 新增能力

## 1. Semi-Live Read-Only Broker Import

可以把真实券商导出的持仓 CSV 导入，但只读取，不下单。

格式：

```text
ticker,qty,sellable_qty,avg_cost
000001.SZ,1200,1200,10.25
600000.SH,800,800,9.84
```

导入：

```bash
python semi_live.py import-broker \
  --positions my_broker_positions.csv \
  --cash 125000 \
  --date 2026-09-04
```

它会形成策略内部账户镜像：

```text
ops/semi_live_account.json
```

---

## 2. Order Proposal

```bash
python semi_live.py propose \
  --targets outputs/latest_v5/latest_optimized_portfolio_2026-09-04.csv \
  --prices data/real/prices.csv \
  --date 2026-09-04
```

流程：

```text
目标组合
↓
价格新鲜度检查
↓
Restricted List
↓
单股权重
↓
行业权重
↓
Gross Exposure
↓
目标股数
↓
T+1
↓
现金
↓
ADV / Capacity
↓
Order Notional Limit
↓
Idempotency Fingerprint
↓
proposed_orders.csv
```

---

## 3. Restricted List

文件：

```text
ops/restricted_list.csv
```

格式：

```text
ticker,reason,effective_date
000001.SZ,internal restriction,2026-09-04
```

只要股票出现在 Restricted List，组合风险检查直接 BLOCK。

---

## 4. Capacity / ADV Participation

不再只问：

> “我想买多少？”

还会问：

> “这个量相对于股票平时成交额大不大？”

默认：

```json
"max_adv_participation_pct": 0.05
```

即单个订单名义金额默认不得超过过去20日平均成交额约 5%。

此外：

```json
"max_order_notional_pct_nav": 0.10
```

默认单笔订单不得超过账户 NAV 的 10%。

---

## 5. Manual Approval Gate

订单不会自动进入执行。

必须人工审批：

```bash
python semi_live.py approve \
  --orders ops/proposed_orders.csv \
  --approved-by "YourName"
```

只有：

```text
Risk Checks = PASS
Capacity = PASS
Approved = True
```

才能进入下一步 Paper Execution。

---

## 6. Kill Switch

开启：

```bash
python semi_live.py kill-switch on --reason "manual risk stop"
```

查看：

```bash
python semi_live.py kill-switch status
```

关闭：

```bash
python semi_live.py kill-switch off
```

只要 Kill Switch 为 ON：

```text
Propose     BLOCK
Approve     BLOCK
Execute     BLOCK
```

---

## 7. Live Broker Submission 默认禁用

配置：

```json
"live_broker_submission_enabled": false
```

`LiveBrokerDisabled` 类任何 submit_orders 调用都会直接抛错。

ULTIMATE 版本只提供：

```text
Paper Execution
Manual Order Export
Read-only Broker Snapshot
```

不会自动碰真实资金。

---

## 8. Stress Testing

运行：

```bash
python semi_live.py risk-report \
  --targets outputs/latest_v5/latest_optimized_portfolio_2026-09-04.csv \
  --prices data/real/prices.csv \
  --date 2026-09-04
```

输出：

```text
ops/risk_summary.csv
ops/stress_report.csv
```

包括：

### Market Shock
```text
-3%
-5%
-10%
```

### Sector Shock
默认对前几大行业分别做：

```text
-15%
```

### Historical VaR / CVaR
基于历史组合日收益。

### Bootstrap VaR / CVaR
默认：

```text
2000 scenarios
```

---

## 9. Audit Trail

每次运行生成：

```text
Run Manifest
```

记录：

```text
执行时间
Python版本
操作系统
Git Commit
Config SHA256
输入数据 SHA256
输入文件大小
运行模式
As-of Date
```

因此可以回答：

> “这次组合到底是用哪一版代码、哪一个配置和哪一份数据产生的？”

---

## 10. Idempotency

订单生成：

```text
order_fingerprint
```

由：

```text
trade_date
ticker
side
qty
reference_price
```

生成稳定 Hash。

重复订单会被识别并阻止，降低同一批订单被重复执行的风险。

---

## 11. SQLite Ledger

新增：

```text
ops/ledger.sqlite3
```

记录：

```text
Runs
Orders
Approvals
Fills
Reconciliations
```

比单纯 CSV/JSON 更适合作为长期 Paper / Semi-live 账本。

---

## 12. Production Readiness Check

运行：

```bash
python production_check.py
```

检查：

```text
Live submission disabled
Manual approval required
Kill switch configured
T+1 enabled
Max weight sane
Turnover limit sane
ADV participation sane
Cash buffer
Critical modules exist
```

---

## 13. Docker

构建：

```bash
docker build -t csi300-quant .
```

运行 Dashboard：

```bash
docker run -p 8501:8501 csi300-quant
```

---

## 14. GitHub Actions CI

已加入：

```text
.github/workflows/ci.yml
```

每次 Push / PR 自动：

```text
pip install
pytest
production_check.py
```

---

## 15. Makefile

```bash
make test
make demo
make ultimate-demo
make check
make dashboard
```

---

# 最推荐的真实使用方式

## Phase A — Research
```text
真实历史数据
↓
run_real_v6.py
```

## Phase B — Latest Portfolio
```text
run_latest_optimized.py
```

## Phase C — Broker Read-only Snapshot
```text
semi_live.py import-broker
```

## Phase D — Proposal
```text
semi_live.py propose
```

## Phase E — Human Review
```text
检查：
目标股票
目标权重
订单股数
T+1
行业
风险
容量
Stress
```

## Phase F — Manual Approval
```text
semi_live.py approve
```

## Phase G — Paper / Manual Real Execution
```text
Paper:
semi_live.py paper-execute

Real:
人工在券商客户端按批准订单执行
```

## Phase H — Reconciliation
将真实券商新持仓再次导入，核对策略目标和实际账户。

---

# 目前成熟度

```text
Research Engine                     ✅
Point-in-Time                       ✅
Historical Constituents             ✅
Factor Neutralization               ✅
Risk Model                          ✅
Portfolio Optimizer                 ✅
Walk-forward                        ✅
Realistic Backtest                  ✅
Latest Target Portfolio             ✅
Board-lot Sizing                    ✅
T+1                                 ✅
Cash Management                     ✅
Restricted List                     ✅
ADV Capacity                        ✅
Stress / VaR / CVaR                 ✅
Order Fingerprint / Idempotency     ✅
Manual Approval Gate                ✅
Kill Switch                         ✅
Paper Execution                     ✅
P&L / NAV / Reconciliation          ✅
Read-only Broker Import             ✅
Audit Manifest                      ✅
SQLite Ledger                       ✅
Dashboard                           ✅
Research Report                     ✅
Docker                              ✅
CI                                  ✅

Automatic Live Broker Orders        intentionally disabled
```

---

# 简历最终版

**CSI 300 Multi-Factor Quant Research & Portfolio Trading Framework | Python**

Built an end-to-end point-in-time CSI 300 quantitative investment framework covering alpha research, risk modeling, benchmark-aware portfolio optimization, execution-aware backtesting and paper/semi-live portfolio operations. Constructed Value, Quality and Momentum signals with industry/size neutralization, a shrinkage covariance risk model, and constrained portfolio optimization with single-name, sector, tracking-error and turnover controls. Implemented A-share board-lot sizing, T+1 sellability, price-limit handling, transaction costs, ADV-based capacity checks, restricted lists, stress testing, historical/bootstrap VaR-CVaR, manual approval gates, a kill switch, order idempotency, persistent SQLite audit ledger, read-only broker snapshot import, reconciliation, Streamlit dashboard, automated HTML research reports, Docker packaging and CI tests.

---

# 重要边界

“更完整的系统”不等于“保证赚钱”。

模型仍然会面临：

- 因子失效
- 市场结构变化
- 数据修订
- 极端流动性
- 滑点高于预期
- 容量限制
- 回测与真实成交差异
- 参数不稳定
- 监管和券商规则变化

ULTIMATE 版本的设计目标不是把风险藏起来，而是让研究、风险、执行和审计尽可能透明、可重复和可控制。


## A-SHARE ELITE 新增

参考你提供的量化学习路线图后，本版新增了真正适合实盘研究的多策略层：

- Core Multi-Factor Sleeve
- Sector Rotation Sleeve
- Mean Reversion Sleeve
- Strict Walk-forward ML Rank Sleeve
- OOS Rank IC comparison
- Multiple-testing haircut
- Probabilistic Sharpe Ratio
- Strategy allowlist / denylist

运行：

```bash
python run_ashare_elite_demo.py
```

详见 `ROADMAP_INTEGRATION.md`。


## Institutional Layer

详见 `README_INSTITUTIONAL.md`。
