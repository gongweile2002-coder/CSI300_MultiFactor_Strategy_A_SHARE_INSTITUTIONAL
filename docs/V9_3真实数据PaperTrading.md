# v9.3 真实数据 Paper Trading

本层只做 **真实市场数据 + 纸面交易**。不会调用 MiniQMT，也不会把任何委托发送到券商。

## 数据与时序

1. 收盘后通过 Tushare 下载完整沪深300研究数据。
2. 财务数据按公告日做 point-in-time 过滤；信号日看不到当日或未来公告。
3. 当日收盘后生成候选组合，并用当日 raw close 固定下一开放交易日的 share intents。
4. 下一开放交易日只使用该日真实未复权开盘价决定 Paper fill price，不再用开盘价重新计算股数。
5. 同一交易日真实未复权收盘价用于 NAV 标记；如果预期执行日被漏跑，程序会停止而不是把旧 intent 自动平移到更晚的 open。

因此，D 日收盘信号不会获得 D 日开盘成交价。首次运行只建立候选组合，不成交；第二个完成交易日开始才会产生纸面成交。

## 本地运行

```bash
export TUSHARE_TOKEN="你的 token"
python -m pip install -r requirements-live.txt
python daily_paper.py run --refresh
python daily_paper.py status
```

状态写入 `paper/live/`：

- `paper.sqlite3`：**权威事务账本**，保存账户状态、pending intents、orders、fills、NAV、持仓与 run 记录；
- `paper_account.json` / `state.json`：由 SQLite 导出的兼容快照；
- `pending_intents.csv`：下一开放交易日的固定股数 intent 导出；
- `candidate_history.csv` / `order_history.csv` / `fill_history.csv` / `nav_history.csv` / `position_history.csv`：审计导出；
- `runs/YYYY-MM-DD/`：单日导出快照。

同一 signal_date + signal bundle 重复执行会返回 `IDEMPOTENT`；同一日期若 signal bundle hash 不同则报冲突，禁止静默覆盖。SQLite 使用事务提交，故障注入测试覆盖了 commit 前崩溃后重跑不会重复入账。

## GitHub Actions

当前仓库是公开仓库，因此 `.github/workflows/daily-paper.yml` **不做定时 Paper Trading，也不把 Paper 状态提交回 Git**。为解决 `workflow_dispatch` 仅在默认分支可直接触发的限制，当前 owned feature 分支上的相关代码 push 会自动触发一次临时验收；合并到默认分支后也可以手动触发。验收只在 runner 临时目录里完成 Tushare 数据刷新、候选生成、SQLite Paper run 和 broker-disabled 检查，任务结束后状态即丢弃。

在仓库 **Settings → Secrets and variables → Actions** 新增：

`TUSHARE_TOKEN`

Secret 只注入确实需要访问 Tushare 的 step；workflow 的 `GITHUB_TOKEN` 只有 `contents: read`。不要把 token 写入仓库文件，也不要把真实 pending intents、持仓或 fill history 提交到公开仓库。`paper/live/` 已加入仓库 `.gitignore`，降低本地状态误提交风险。

要做连续 forward shadow，请在本机持续保留 `paper/live/paper.sqlite3`，或者把 ledger 放入私有、持久、可备份的状态存储；不要依赖 GitHub Actions cache 充当权威账本。

## 边界

- 目前使用 Tushare 日线 raw open/close，不是 tick 级实时行情。
- 纸面成交包含佣金、滑点、印花税和 T+1 规则；缺失当日涨跌停数据时 fail-closed，滑点后的价格被限制在合法涨跌停区间内，但仍无法复现真实盘口排队、部分成交、停牌复牌瞬间流动性等全部微观结构。
- 结果是前向 Paper Trading 记录，不是实盘业绩。
- 真实下单仍由 `config/live.example.json` 中的 `live_enabled=false` 默认关闭。
