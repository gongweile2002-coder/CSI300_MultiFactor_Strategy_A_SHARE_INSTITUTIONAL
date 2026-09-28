# Mac 启动与 v9.1 升级

## 此次修复

原版先取评分前30名，再按行业截断权重，行业额度用尽后零权重仍占据名额，且“至少15只”的检查发生在实际分配前。
新版按评分及证券代码稳定排序，跳过行业额度耗尽的候选，向后补足最多30只正权重股票。实际分配不足15只则阻止发布。单股预算不因候选不足而提高，余款保留现金。行业和总仓位上限不变。信号报告增加现金目标和分配方法标识。

这是分配逻辑修订，会改变持仓，也可能增加实际仓位和换手，并不证明收益改善。价值30%、质量40%、动量30%的参数未验证最优；未新增收益结论。默认真实下单仍关闭。

版本标识为 A_SHARE_V9_1_2026_09_26；原订单策略ID保持不变以维持防重复机制。升级后重新生成信号和计划；不要把旧信号与新配置混用。历史券商规则资料日期不因本次代码修复而更新。

## 打开正确文件夹

1. 把新版ZIP解压到独立文件夹，建议命名 CSI300_v9_1，避免与旧目录混淆。
2. PyCharm：文件 → 打开 → 选择含 live.py、VERSION.txt、requirements-live.txt 的目录。
3. 终端运行 `cat VERSION.txt` 确认版本。旧目录若无这些文件，不应继续运行旧README的semi_live命令。
4. 已经登记或发送过委托的用户，先完成对账，并保留原订单台账及归档。不要清空或用新建空台账代替原台账。

## 统一 Python 环境

本项目测试环境是 Linux / Python 3.12.14，未在你的Mac实机验证。建议使用Python 3.12系列。截图中终端为3.13而IDE为3.14，二者指向不同解释器。
先运行 `python3.12 --version`。若找不到命令，先安装Python 3.12，再继续。Python官方macOS页面：https://www.python.org/downloads/macos/

在新版项目的终端逐行执行；任一步报错先处理，不继续下一行：

```bash
python3.12 -m venv .venv312
source .venv312/bin/activate
python -m pip install -r requirements-live.txt
python -c "import sys; print(sys.executable); print(sys.version)"
python live.py demo
python live.py doctor
```

PyCharm设置 → Python → 解释器 → 添加本地解释器 → 使用现有环境，选择本项目 `.venv312/bin/python`。菜单名称因版本略有不同。官方说明：https://www.jetbrains.com/help/pycharm/creating-virtual-environment.html

之后执行命令也可以直接用 `.venv312/bin/python live.py demo`，明确指定同一环境。

## 预期结果

- demo：合成行情，2笔模拟计划、2项拦截，不是今日选股，不会下单。
- doctor：没有数据或Token时BLOCK属于诊断结果，不意味着程序安装失败。查看 outputs/v9_doctor/doctor.md。
- 实盘前仍需真实完整数据、样本外验证、模拟跟踪和券商联调。Mac可用于数据与计划；MiniQMT依赖券商支持的Windows环境。

## GitHub

本次连接未列出用户可访问仓库，未推送代码。提供量化项目仓库URL并授予该仓库访问权限后，才能创建修订分支并提交可审查的变更。不要分享Token、密码或券商账号凭据。
