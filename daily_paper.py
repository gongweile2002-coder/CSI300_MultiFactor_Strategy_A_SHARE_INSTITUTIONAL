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
from src.data_contract_v8 import load_verified_dataset
from src.ops_v9 import atomic_json
from src.baostock_provider import guard_held_adjustments
from src.paper_ledger_v10 import PaperLedger

BASE = Path(__file__).resolve().parent


def _json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def _refresh(data_dir: Path, lookback_years: int, state_dir: Path, provider: str = "baostock") -> None:
    cmd = [
        sys.executable,
        str(BASE / ("download_free_data.py" if provider == "baostock" else "update_live_data.py")),
        "--output",
        str(data_dir),
        "--lookback-years",
        str(lookback_years),
        "--paper-state-dir",
        str(state_dir),
    ]
    subprocess.run(cmd, cwd=BASE, check=True)


def bind_provider(state_dir, provider):
    """Keep the new variant in a separate ledger, including explicit path overrides."""
    state_dir = Path(state_dir)
    binding = state_dir / "provider_binding.json"
    if binding.exists():
        if _json(binding).get("provider") != provider:
            raise ValueError("Paper账本数据源不同；请使用独立的paper/free或paper/live目录")
    elif provider == "baostock" and (state_dir / "paper.sqlite3").exists():
        raise ValueError("旧账本没有免费数据源绑定；请使用新的paper/free目录")
    else:
        atomic_json(binding, {"provider": provider, "broker_submission": False})


def main(argv=None):
    p = argparse.ArgumentParser(
        description="A股真实日线数据 -> PIT候选组合 -> 次交易日开盘纸面成交 -> 收盘NAV"
    )
    sub = p.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run")
    run.add_argument("--provider", choices=["baostock", "tushare"], default="baostock")
    run.add_argument("--data")
    run.add_argument("--signals")
    run.add_argument("--state-dir")
    run.add_argument("--paper-config", default=str(BASE / "config.example.json"))
    run.add_argument("--risk-config")
    run.add_argument("--initial-cash", type=float, default=500000.0)
    run.add_argument("--lookback-years", type=int)
    run.add_argument("--refresh", action="store_true")

    status = sub.add_parser("status")
    status.add_argument("--state-dir", default=str(BASE / "paper/free"))

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

    free = args.provider == "baostock"
    data_dir = Path(args.data or BASE / ("data/free" if free else "data/live"))
    signal_dir = Path(args.signals or BASE / ("outputs/free_signal" if free else "outputs/v8_signal"))
    state_dir = Path(args.state_dir or BASE / ("paper/free" if free else "paper/live"))
    bind_provider(state_dir, args.provider)
    if args.refresh:
        lookback=args.lookback_years if args.lookback_years is not None else (1 if free else 3)
        _refresh(data_dir, lookback, state_dir, args.provider)

    manifest, _ = load_verified_dataset(data_dir)
    if manifest["source"] != args.provider:
        raise ValueError("--provider与数据清单不一致；请使用所选数据源的独立数据目录")
    risk_settings = _json(args.risk_config or BASE / ("config/free.example.json" if free else "config/live.example.json"))
    cfg = RiskConfig.from_dict(risk_settings["risk"])
    now = pd.Timestamp.now(tz="Asia/Shanghai")
    report = generate_signals(data_dir, signal_dir, now, cfg)

    targets = pd.read_csv(signal_dir / "targets.csv", dtype={"ticker": str})
    raw_prices = pd.read_csv(data_dir / "raw_prices.csv", dtype={"ticker": str})
    limits = pd.read_csv(data_dir / "stock_limits.csv", dtype={"ticker": str})
    trade_calendar = pd.read_csv(data_dir / "trade_calendar.csv")
    corporate_actions_path = data_dir / "corporate_actions.csv"
    if not corporate_actions_path.exists():
        raise FileNotFoundError(
            "缺少 corporate_actions.csv；请先运行 daily_paper.py run --refresh"
        )
    corporate_actions = pd.read_csv(
        corporate_actions_path,
        dtype={"ticker": str, "ts_code": str},
    )
    if free and (state_dir / "paper.sqlite3").exists():
        ledger = PaperLedger(state_dir / "paper.sqlite3")
        account = ledger.load_account("PAPER-CSI300-LIVE") or {}
        held = set(account.get("positions", {}))
        held |= {str(x["ticker"]) for x in account.get("share_receivables", {}).values() if x.get("ticker")}
        guard_held_adjustments(raw_prices, corporate_actions, held, report["signal_date"])
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
        corporate_actions=corporate_actions,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"PAPER ONLY：data_source={args.provider}；未调用任何券商下单接口。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
