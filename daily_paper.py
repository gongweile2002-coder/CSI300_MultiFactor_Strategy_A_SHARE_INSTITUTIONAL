from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

from src.daily_paper_v9 import run_paper_day
from src.live_v8 import RiskConfig
from src.signals_v8 import generate_signals

BASE = Path(__file__).resolve().parent


def _json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def _refresh(data_dir: Path, lookback_years: int, state_dir: Path) -> None:
    cmd = [
        sys.executable,
        str(BASE / "update_live_data.py"),
        "--output",
        str(data_dir),
        "--lookback-years",
        str(lookback_years),
        "--paper-state-dir",
        str(state_dir),
    ]
    subprocess.run(cmd, cwd=BASE, check=True)


def main(argv=None):
    p = argparse.ArgumentParser(
        description="A股真实日线数据 -> PIT候选组合 -> 次交易日开盘纸面成交 -> 收盘NAV"
    )
    sub = p.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run")
    run.add_argument("--data", default=str(BASE / "data/live"))
    run.add_argument("--signals", default=str(BASE / "outputs/v8_signal"))
    run.add_argument("--state-dir", default=str(BASE / "paper/live"))
    run.add_argument("--paper-config", default=str(BASE / "config.example.json"))
    run.add_argument("--risk-config", default=str(BASE / "config/live.example.json"))
    run.add_argument("--initial-cash", type=float, default=500000.0)
    run.add_argument("--lookback-years", type=int, default=3)
    run.add_argument("--refresh", action="store_true")

    status = sub.add_parser("status")
    status.add_argument("--state-dir", default=str(BASE / "paper/live"))

    args = p.parse_args(argv)

    if args.command == "status":
        d = Path(args.state_dir)
        state_path = d / "state.json"
        if not state_path.exists():
            print(json.dumps({
                "status": "NOT_INITIALIZED",
                "state_dir": str(d),
                "real_broker_submission": False,
            }, ensure_ascii=False, indent=2))
            return 0
        state = _json(state_path)
        print(json.dumps(state, ensure_ascii=False, indent=2))
        nav = d / "nav_history.csv"
        if nav.exists():
            print(pd.read_csv(nav).tail(10).to_string(index=False))
        return 0

    data_dir = Path(args.data)
    signal_dir = Path(args.signals)
    state_dir = Path(args.state_dir)
    if args.refresh:
        _refresh(data_dir, args.lookback_years, state_dir)

    risk_settings = _json(args.risk_config)
    cfg = RiskConfig.from_dict(risk_settings["risk"])
    now = pd.Timestamp.now(tz="Asia/Shanghai")
    report = generate_signals(data_dir, signal_dir, now, cfg)

    targets = pd.read_csv(signal_dir / "targets.csv", dtype={"ticker": str})
    raw_prices = pd.read_csv(data_dir / "raw_prices.csv", dtype={"ticker": str})
    limits = pd.read_csv(data_dir / "stock_limits.csv", dtype={"ticker": str})
    trade_calendar = pd.read_csv(data_dir / "trade_calendar.csv")
    paper_cfg = _json(args.paper_config)

    result = run_paper_day(
        state_dir,
        targets,
        raw_prices,
        limits,
        report,
        paper_cfg,
        trade_calendar,
        initial_cash=args.initial_cash,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print("PAPER ONLY：未调用任何券商下单接口。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
