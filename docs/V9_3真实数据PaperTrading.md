# v9.3 真实数据 Paper Trading

本层只做 **真实市场数据 + 纸面交易**。不会调用 MiniQMT，也不会把任何委托发送到券商。

## 数据与时序

1. 收盘后通过 Tushare 下载完整沪深300研究数据。
2. 财务数据按公告日做 point-in-time 过滤；信号日看不到当日或未来公告。
3. 当日收盘后生成候选组合，保存为 pending targets。
4. 下一完成交易日使用该交易日真实未复权开盘价进行纸面成交。
5. 同一交易日真实未复权收盘价用于 NAV 标记。

因此，D 日收盘信号不会获得 D 日开盘成交价。首次运行只建立候选组合，不成交；第二个完成交易日开始才会产生纸面成交。

## 本地运行

```bash
export TUSHARE_TOKEN="你的 token"
python -m pip install -r requirements-live.txt
python daily_paper.py run --refresh
python daily_paper.py status
```

状态写入 `paper/live/`：

- `paper_account.json`：纸面现金和分批持仓；
- `pending_targets.csv`：下一交易日待执行目标；
- `candidate_history.csv`：每日候选组合；
- `order_history.csv`：纸面委托记录；
- `fill_history.csv`：纸面成交记录；
- `nav_history.csv`：每日收盘 NAV；
- `position_history.csv`：每日收盘持仓；
- `runs/YYYY-MM-DD/`：单日审计快照。

同一 signal_date 重复执行会返回 `IDEMPOTENT`，避免重复入账。

## GitHub Actions

`.github/workflows/daily-paper.yml` 默认在周一至周五上海时间 19:30 运行，也可手动触发。

在仓库 **Settings → Secrets and variables → Actions** 新增：

`TUSHARE_TOKEN`

不要把 token 写入仓库文件。真实市场原始数据位于 `data/live/` 并继续被 `.gitignore` 排除；Actions 用缓存复用数据，纸面账本则保存在 `paper/live/`。

## 边界

- 目前使用 Tushare 日线 raw open/close，不是 tick 级实时行情。
- 纸面成交包含佣金、滑点、印花税和 T+1 规则，但无法复现真实盘口排队、部分成交、停牌复牌瞬间流动性等全部微观结构。
- 结果是前向 Paper Trading 记录，不是实盘业绩。
- 真实下单仍由 `config/live.example.json` 中的 `live_enabled=false` 默认关闭。
