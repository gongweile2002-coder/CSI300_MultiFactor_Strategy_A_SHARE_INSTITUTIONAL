"""Compare fixed A-share research variants; never submit broker orders."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.live_v8 import RiskConfig, require
from src.strategy_lab_v9_4 import LabConfig, compare_strategies, demo_inputs, load_lab_inputs, save_lab_results


def main(argv=None):
    parser = argparse.ArgumentParser(description="A股固定策略对比（独立研究，不生成实盘信号）")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--demo", action="store_true", help="显式合成数据，仅检查流程")
    source.add_argument("--data", help="已审计的 data/live 或冻结真实数据目录")
    parser.add_argument("--output", required=True, help="新的空输出目录")
    parser.add_argument("--frequency", choices=["daily", "weekly", "monthly"], default="weekly")
    parser.add_argument("--start")
    parser.add_argument("--end")
    parser.add_argument("--initial-cash", type=float, default=500000.)
    parser.add_argument("--rank-buffer", type=int, default=10)
    parser.add_argument("--max-replacements", type=int, default=3)
    parser.add_argument("--min-hold-sessions", type=int, default=5)
    parser.add_argument("--risk-config", default=str(Path(__file__).resolve().parent/"config/live.example.json"))
    args = parser.parse_args(argv)
    output = Path(args.output).resolve()
    require(not output.exists() or not any(output.iterdir()), "输出目录非空，请使用新的研究运行目录")
    protected = [Path(__file__).resolve().parent/"paper/live", Path(__file__).resolve().parent/"outputs/v8_signal"]
    if args.data:
        protected.append(Path(args.data).resolve())
    require(all(output != d.resolve() and d.resolve() not in output.parents for d in protected),
            "研究输出不得写入数据、Paper 状态或实盘信号目录")
    cfg = LabConfig(frequency=args.frequency, initial_cash=args.initial_cash, rank_buffer=args.rank_buffer,
                    max_replacements=args.max_replacements, min_hold_sessions=args.min_hold_sessions).validated()
    settings = json.loads(Path(args.risk_config).read_text(encoding="utf-8-sig"))
    risk = RiskConfig.from_dict(settings["risk"])
    if args.demo:
        inputs = demo_inputs()
        provenance = {"source": "synthetic", "seed": 94, "notice": "SYNTHETIC DEMO ONLY; NOT MARKET PERFORMANCE"}
    else:
        inputs, provenance = load_lab_inputs(args.data)
    result = compare_strategies(inputs, cfg, risk, args.start, args.end)
    save_lab_results(result, output, cfg, risk, provenance)
    print("研究完成；source="+provenance["source"]+"；performance_validated=false；未调用券商。")
    print(result["comparison"][["strategy", "scenario", "period", "total_return", "max_drawdown",
                                 "mean_rebalance_turnover"]].to_string(index=False))
    print("结果目录: "+str(output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
