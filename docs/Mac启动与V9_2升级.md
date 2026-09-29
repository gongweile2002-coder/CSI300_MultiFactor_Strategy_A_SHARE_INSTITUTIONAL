# Mac 启动与 v9.2 升级

## 这次升级解决什么

v9.2 不改变价值 30% / 质量 40% / 动量 30% 的核心因子权重，也不声称提高收益。它主要修复“从计划到真实委托”之间的操作风险：计划可能放太久才提交、复核完成与券商受理混为一谈、程序重启后看不到本地冻结资源、状态可以被错误回退，以及 Kill Switch 命中后被记成普通券商拒单。

新版本号为 `A_SHARE_V9_2_2026_09_28`。`strategy_id` 继续保持 `CSI300_CORE_V8`，这样同一台账中的日内防重复逻辑不会因为升级被绕开。

如果你已经复制过 `config/live.local.json`，请把其中 `risk.version` 更新为 `A_SHARE_V9_2_2026_09_28`。v9.2 会拒绝旧版本号，避免新代码继续消费旧版配置/信号。更新配置后必须重新生成信号和计划。

## 已经完成的 v9.2 改动

1. `plan.json` 增加 `expires_at`，默认生成后 10 分钟失效；`manual-register` 与 `qmt-submit` 在消费计划前强制检查。
2. 订单台账新增持久状态：`APPROVED`、`SUBMITTING`、`UNKNOWN`、`SUBMITTED`、`PARTIAL`、`FILLED`、`CANCELLED`、`REJECTED`、`EXPIRED`、`BLOCKED`。
3. API 调用前先落盘为 `SUBMITTING`。若网络/SDK异常导致“可能已受理但本地不知道”，记录为 `UNKNOWN`，必须先对账，禁止盲目重发。
4. 台账保存 `reserved_cash`、`reserved_shares`、`estimated_fees`、`expires_at`。部分成交后本地冻结量递减，终态释放。
5. 状态迁移不可回退；终态只接受完全一致的幂等回报。
6. Kill Switch 在券商调用前触发时记录为 `BLOCKED`，不会发送订单。
7. 默认单批最多 60 笔委托；超过上限整批阻止，不自动截断。
8. 旧 `orders_v8` SQLite 表启动时原地增加字段，不重建、不清空。

## 你现在的 Mac 环境

你已经把本项目切到 Python 3.12 的 `.venv312`，并在 v9.1 上跑过 136 项测试和 `python live.py demo`。升级 v9.2 后继续沿用同一 Python 3.12 环境即可，不需要再安装一套 Python。

在新版项目目录：

```bash
source .venv312/bin/activate
python --version
python -m pip install -r requirements.txt
python -m pytest -q
python live.py demo
```

预期 v9.2 软件测试为 **147 passed**，Demo 仍为 **2 笔模拟计划 + 2 项拦截**，只是 `plan.json` 的 schema 升为 9，并多出 `expires_at`。

## 升级旧订单台账

如果你已经有：

```text
ops/live_v8.sqlite3
```

**不要删除，不要新建空文件替换。** v9.2 第一次打开 `OrderJournal` 时会自动增加 4 个字段：

```text
estimated_fees
reserved_cash
reserved_shares
expires_at
```

旧订单记录保留。升级前如果存在 `SUBMITTED / PARTIAL / UNKNOWN / SUBMITTING`，应先完成券商对账，再生成新的计划。

可查看：

```bash
python live.py ledger
```

v9.2 的输出包含 `orders` 和 `reservations`，其中 `reservations` 只是本地安全台账，不替代券商真实可用资金、冻结资金和可卖数量。

## v9.2 的计划有效期

配置在 `config/live.example.json`：

```json
"plan_valid_minutes": 10,
"max_orders_per_batch": 60
```

真实操作时不要把旧 `plan.json` 留到很久以后继续点确认。计划过期后重新获取资金、持仓、可卖数量和行情，再生成新计划。

## 真实下单仍然默认关闭

v9.2 没有替你打开任何实盘开关。配置仍要求：

```json
"live_enabled": false,
"broker_rules_confirmed": false
```

MiniQMT 真实提交还需要本机显式设置 `QMT_ENABLE_LIVE=YES_I_UNDERSTAND_REAL_ORDERS`。没有完成你的券商 Windows / MiniQMT 实机联调前，不应把软件测试通过理解成“可无人值守实盘”。

## 当前验证边界

本版完成的是软件安全层验证。测试使用合成数据、临时 SQLite 和 SDK 替身，不连接你的券商、不发送真实订单，也没有产生新的真实历史收益、胜率、Sharpe 或最大回撤结论。真实部署仍需要完整真实数据、样本外验证、前向模拟、券商字段/权限核对和小规模人工监控试运行。
