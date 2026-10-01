# v9.5 免费数据与 Paper Trading

默认数据源改为 **BaoStock 匿名接口**。不需要注册账户、积分、Tushare Token 或 GitHub Secret。

这是明确标注的免费策略变体：保留价值30%、质量40%、126交易日动量30%、基准MA120仓位规则与单股/行业上限。数据口径与Tushare原版有差异，不能声称两个版本的收益可以直接比较。

## 本地运行

在仓库目录执行：

```bash
python -m pip install -r requirements-free.txt
python daily_paper.py run --refresh
python daily_paper.py status
```

收盘后重复执行第二条命令即可刷新。网络需要支持 BaoStock 的 TCP 10030 连接；如果本地网络不支持，可先查看 GitHub 的真实数据验收。

首次默认拉取一年行情，覆盖126交易日动量、20日流动性与120日基准均线；已有记录继续保留，行情随后按重叠区间增量更新。需要更长行情时可添加`--lookback-years 2`。证券/行业资料按所需证券读取，避免全市场分页；连接意外中断时最多完整重试该接口两次，不使用未收完的分页数据。

- `data/free/`：真实行情、财报、成分/行业快照、日历、分红和带哈希的数据清单。
- `outputs/free_signal/targets.csv`：当日候选组合。
- `outputs/free_signal/factor_audit.csv`：因子与公告日审计。
- `outputs/free_signal/signal_report.json`：数据源、口径、筛选数量、仓位和模型限制。
- `paper/free/paper.sqlite3`：连续运行的权威 Paper 账本。
- `paper/free/nav_history.csv`、`fill_history.csv`、`candidate_history.csv`：账本导出；无成交时成交文件可能尚未建立。

首次运行只建立下一开放交易日的固定股数意图，不产生当日成交。下一**完成交易日**运行时，才用该日未复权开盘价模拟上一交易日订单。漏跑预期执行日会停止。节假日重跑同一个完成交易日不会产生新成交。

免费版使用独立的数据、信号与账本目录；不能把旧Tushare账本直接当作免费版账本继续用。这些真实数据和账本目录已被Git忽略。

## GitHub 查看验收

打开仓库 **Actions → Free A-share data validation → 最新运行 → Summary**，可以看到真实数据日期、完整股票池、合格证券数、候选数、目标仓位和验收结果。

合并到默认分支后，可以点击 **Run workflow** 手动运行。原 **Manual A-share paper validation** 工作流也改用免费数据源。

GitHub工作流只读、无Secret，使用临时数据与临时Paper状态，不发布候选/持仓/成交，也不保存跨日账本。每次新工作流都是初次验收，不能把它当成连续Paper收益记录。连续账本需用上述本地命令保持同一个`paper/free`目录。

## 免费版数据口径

| 内容 | 免费版处理 |
|---|---|
| 行情 | BaoStock不复权日线；量为股、额为元。停牌和零成交条目不填充成可交易行情 |
| 研究价格 | 原始价乘BaoStock后复权因子。该供应商使用涨跌幅复权算法，不能把它标为分红再投资总收益 |
| 财务质量 | `roeAvg`、`YOYNI`、`liabilityToAsset`转为百分比；不是原版扣非ROE字段 |
| 公告日期 | 三个财务接口的公告日期取最大值；严格早于信号日才可使用。缺字段的证券排除，不用0代替 |
| 财务历史 | 默认拉最近两个已结束季度，之后保留已下载记录；供应商历史版本没有独立存证 |
| 股票池 | 当前快照必须完整300只；逐次累积已观察到的周度快照，不把当前名单回填历史 |
| 行业 | 使用供应商返回的分类名称及更新时间，记录周度快照区间；不冒充精确历史申万一级行业 |
| 中性化 | 行业中性化；免费日线缺少同口径每日总市值，因此关闭市值中性化 |
| 涨跌停 | 仅对上市足够久、非ST、非退市的普通主板股，依据供应商当日前收价推算10%边界并四舍五入到分。异常日、其他板块、数据缺失时不给推算边界，Paper拒单。这不是交易所确认的实盘涨跌停数据 |
| 分红现金 | 读取税前每股派息并在模型中预扣固定20%；不把供应商含多种持有期税率的文本解析成0，不声称已核算真实持有期税款 |
| 送转股 | 保留登记、除权、派息和新增股份上市日期；缺失必要日期时停止 |
| 特殊除权 | 持仓复权变化无法由已知分红/送转解释时停止，需核对配股等事件 |

免费信号清单标记为`free_paper_signal_bundle`。券商计划入口会拒绝该类型；`live_enabled=false`仍保持关闭。

原`strategy_lab.py`的真实四策略对比仍要求Tushare原版完整历史数据，不能将免费版当前快照冒充历史沪深300股票池。合成演示仍可运行：

```bash
python -m pip install -r requirements.txt
python strategy_lab.py --demo --output outputs/free_strategy_demo
```

## 保留的Tushare入口

如果以后开通Tushare权限，可显式运行独立原版：

```bash
python -m pip install -r requirements-live.txt
python daily_paper.py run --provider tushare --refresh
python daily_paper.py status --state-dir paper/live
```

以上命令会使用`data/live`、`outputs/v8_signal`、`paper/live`和原配置，不与免费版混用。旧Secret可以保留；免费工作流不会读取它。

## 官方参考

- [BaoStock Python API文档](https://www.baostock.com/mainContent?file=pythonAPI.md)
- [BaoStock复权因子说明](https://www.baostock.com/helpdocs/pdf/BaoStock复权因子简介.pdf)
- [上交所交易规则2026年修订](https://www.sse.com.cn/lawandrules/sselawsrules2025/fund/trading/c/c_20260424_10817739.shtml)
- [深交所普通主板交易规则说明](https://investor.szse.cn/knowledge/t20230308_599141.html)

软件检查和真实数据验收只证明该次数据/候选/初始Paper流程可运行，不证明策略盈利，也不证明后续所有交易日或特殊事件都已验收。
