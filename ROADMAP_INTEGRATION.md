> 历史研究资料：这里的旧命令、路线图和输出不代表 v8 已验证功能或实盘业绩。当前操作以 README.md 和 docs/实盘操作指南.md 为准。


# 从你发来的量化学习路线图中，真正值得吸收到实盘系统的部分

我没有机械照搬“所有策略”。

## 已吸收

- 小市值/风格思想 → 不做纯小市值，改为未来可扩展的中小盘 Quality + Liquidity 版本
- 反转策略 → `mean_reversion sleeve`
- 动量策略 → `Momentum Skip-1M`
- 多因子 → 已经是主框架
- 行业轮动 → `sector_rotation sleeve`
- 机器学习 → 严格 Walk-forward 的 `ML rank sleeve`
- 择时 → A股 Regime Engine
- 基金/指数轮动思想 → 以后可直接扩成 ETF sleeve

## 没有直接照搬

- 纯低价股
- 纯小市值
- 单独 MACD / RSI
- 追涨停
- 训练集随机切分的机器学习

原因：这些东西非常容易在历史样本里看起来漂亮，但在真实A股里受到壳价值退潮、涨跌停/T+1、拥挤、交易成本和数据泄漏影响。

## 这版新增的“策略组合”思想

不再追求一个神奇策略，而是把不同 Alpha 来源分成：

1. Core Multi-Factor
2. Sector Rotation
3. Mean Reversion
4. ML Rank

最终 `Elite Score` 是 sleeve ensemble。

机器学习最高权重被限制，不允许黑盒模型控制整个账户。

## 为什么比“再加10个指标”更强

真正的问题不是指标数量，而是：

- Alpha 是否独立
- 是否通过样本外验证
- 是否有数据泄漏
- 是否在A股交易规则下可成交
- 是否经得起多重策略试验的过拟合
- 是否与其他策略相关性过高

所以加入：
- OOS Rank IC
- ML Strict Walk-forward
- Multiple-testing haircut
- Probabilistic Sharpe Ratio

这比简单再加 MACD/KDJ/RSI 更接近正式量化研究。
