from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd

from src.data_contract_v8 import load_verified_dataset, normalize_tushare_prices
from src.live_v8 import TradingCalendar, require
from src.signals_v8 import last_completed_session
from src.tushare_provider import _yyyymmdd
from src.tushare_provider_v4 import TushareDownloaderV4

BASE = Path(__file__).resolve().parent


def load_env() -> None:
    path = BASE / ".env"
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip() and not line.lstrip().startswith("#") and "=" in line:
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def merge_frame(old: pd.DataFrame, new: pd.DataFrame, keys: list[str], sort: list[str]) -> pd.DataFrame:
    if old is None or old.empty:
        out = new.copy()
    elif new is None or new.empty:
        out = old.copy()
    else:
        out = pd.concat([old, new], ignore_index=True)
    if out.empty:
        return out
    return out.drop_duplicates(keys, keep="last").sort_values(sort).reset_index(drop=True)


def raw_to_adjusted(raw: pd.DataFrame) -> pd.DataFrame:
    out = raw.copy()
    for col in ["open", "high", "low", "close"]:
        out["raw_" + col] = out[col]
        out[col] = pd.to_numeric(out[col], errors="raise") * pd.to_numeric(out["adj_factor"], errors="raise")
    out["price_basis"] = "adjusted_research_only"
    return out


def _read(path: Path, dates: list[str] | None = None) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    x = pd.read_csv(path, dtype={"ticker": str, "ts_code": str})
    for col in dates or []:
        if col in x.columns:
            x[col] = pd.to_datetime(x[col], errors="coerce")
    return x


def _financial_rows(dl: TushareDownloaderV4, tickers: list[str], start, end) -> pd.DataFrame:
    frames = []
    fields = "ts_code,ann_date,end_date,roe,roe_dt,netprofit_yoy,debt_to_assets"
    for ticker in tickers:
        df = dl.pro.fina_indicator(
            ts_code=ticker,
            start_date=_yyyymmdd(start),
            end_date=_yyyymmdd(end),
            fields=fields,
        )
        dl._pause()
        if df is None or df.empty:
            continue
        frames.append(df.rename(columns={
            "ts_code": "ticker",
            "end_date": "report_date",
            "netprofit_yoy": "profit_growth",
            "debt_to_assets": "debt_ratio",
        }))
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    out["ann_date"] = pd.to_datetime(out["ann_date"], errors="coerce")
    out["report_date"] = pd.to_datetime(out["report_date"], errors="coerce")
    return out.dropna(subset=["ann_date"])


def _ticker_backfill(dl: TushareDownloaderV4, ticker: str, start, end):
    daily = dl.pro.daily(ts_code=ticker, start_date=_yyyymmdd(start), end_date=_yyyymmdd(end))
    dl._pause()
    adj = dl.pro.adj_factor(ts_code=ticker, start_date=_yyyymmdd(start), end_date=_yyyymmdd(end))
    dl._pause()
    raw = pd.DataFrame()
    if daily is not None and not daily.empty and adj is not None and not adj.empty:
        raw = normalize_tushare_prices(daily, adj)

    db = dl.pro.daily_basic(
        ts_code=ticker,
        start_date=_yyyymmdd(start),
        end_date=_yyyymmdd(end),
        fields="ts_code,trade_date,turnover_rate,pe,pe_ttm,pb,total_share,float_share,free_share,total_mv,circ_mv",
    )
    dl._pause()
    if db is None:
        db = pd.DataFrame()
    if not db.empty:
        db = db.rename(columns={"ts_code": "ticker", "trade_date": "date"})
        db["date"] = pd.to_datetime(db["date"], errors="coerce")

    lim = dl.pro.stk_limit(ts_code=ticker, start_date=_yyyymmdd(start), end_date=_yyyymmdd(end))
    dl._pause()
    if lim is None:
        lim = pd.DataFrame()
    if not lim.empty:
        lim = lim.rename(columns={"ts_code": "ticker", "trade_date": "date"})
        lim["date"] = pd.to_datetime(lim["date"], errors="coerce")
        lim = lim[[c for c in ["date", "ticker", "pre_close", "up_limit", "down_limit"] if c in lim]]

    return raw, db, lim


def _bulk_daily_rows(dl: TushareDownloaderV4, sessions: list[pd.Timestamp], current: set[str]):
    raw_frames, db_frames, limit_frames = [], [], []
    db_fields = "ts_code,trade_date,turnover_rate,pe,pe_ttm,pb,total_share,float_share,free_share,total_mv,circ_mv"
    for session in sessions:
        day = _yyyymmdd(session)
        daily = dl.pro.daily(trade_date=day)
        dl._pause()
        adj = dl.pro.adj_factor(trade_date=day)
        dl._pause()
        if daily is None or daily.empty:
            raise RuntimeError(f"{session.date()}: daily 未返回数据")
        daily = daily[daily["ts_code"].isin(current)].copy()
        adj = adj[adj["ts_code"].isin(current)].copy() if adj is not None else pd.DataFrame()
        if not daily.empty:
            if adj.empty:
                raise RuntimeError(f"{session.date()}: adj_factor 未返回当前成分数据")
            raw_frames.append(normalize_tushare_prices(daily, adj))

        db = dl.pro.daily_basic(trade_date=day, fields=db_fields)
        dl._pause()
        if db is not None and not db.empty:
            db = db[db["ts_code"].isin(current)].rename(columns={"ts_code": "ticker", "trade_date": "date"})
            db["date"] = pd.to_datetime(db["date"], errors="coerce")
            db_frames.append(db)

        lim = dl.pro.stk_limit(trade_date=day)
        dl._pause()
        if lim is not None and not lim.empty:
            lim = lim[lim["ts_code"].isin(current)].rename(columns={"ts_code": "ticker", "trade_date": "date"})
            lim["date"] = pd.to_datetime(lim["date"], errors="coerce")
            limit_frames.append(lim[[c for c in ["date", "ticker", "pre_close", "up_limit", "down_limit"] if c in lim]])

    def cat(frames):
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    return cat(raw_frames), cat(db_frames), cat(limit_frames)


def _manifest(output: Path, previous: dict, asof: pd.Timestamp, current_count: int, industry_refreshed_at: str, st_status: str):
    files = [
        "prices.csv", "raw_prices.csv", "trade_calendar.csv", "daily_basic.csv",
        "fundamentals_raw.csv", "index_membership.csv", "stock_metadata.csv",
        "industry_membership.csv", "st_status.csv", "benchmark.csv", "stock_limits.csv",
    ]
    now = pd.Timestamp.now(tz="Asia/Shanghai")
    return {
        "schema_version": 8,
        "source": "tushare",
        "complete_universe": True,
        "as_of": str(asof.date()),
        "created_at": now.isoformat(),
        "ticker_count": int(current_count),
        "amount_unit": "CNY",
        "volume_unit": "share",
        "membership_basis": "latest_available_index_weight_snapshot_not_exact_intramonth_changes",
        "financial_vintages": "ann_date_point_in_time_vendor_history_not_independently_archived",
        "point_in_time_policy": {
            "financials": "ann_date strictly before signal_date",
            "index_membership": "latest effective_date not after signal_date",
            "industry": "in_date/out_date active on signal_date",
            "execution": "signal generated after close; paper fill uses next completed session raw open",
        },
        "refresh_mode": "incremental",
        "previous_as_of": previous.get("as_of"),
        "industry_refreshed_at": industry_refreshed_at,
        "st_history_status": st_status,
        "files": {name: hashlib.sha256((output / name).read_bytes()).hexdigest() for name in files},
    }


def main(argv=None):
    p = argparse.ArgumentParser(description="Incrementally refresh the verified Tushare live-data bundle")
    p.add_argument("--output", default=str(BASE / "data/live"))
    p.add_argument("--lookback-years", type=int, default=3)
    p.add_argument("--sleep", type=float, default=0.25)
    args = p.parse_args(argv)
    require(args.lookback_years >= 2, "至少需要两年历史")
    load_env()

    output = Path(args.output)
    manifest_path = output / "data_manifest_v8.json"
    if not manifest_path.exists():
        cmd = [
            sys.executable, str(BASE / "download_live_data.py"),
            "--output", str(output),
            "--lookback-years", str(args.lookback_years),
            "--sleep", str(args.sleep),
        ]
        subprocess.run(cmd, cwd=BASE, check=True)
        print("首次运行已完成完整数据 bootstrap；后续运行使用增量刷新。")
        return 0

    previous, _ = load_verified_dataset(output)
    last = pd.Timestamp(previous["as_of"]).normalize()
    dl = TushareDownloaderV4(os.getenv("TUSHARE_TOKEN", ""), output, args.sleep)

    now = pd.Timestamp.now(tz="Asia/Shanghai")
    cal_old = _read(output / "trade_calendar.csv", ["date"])
    cal_start = cal_old["date"].min() if not cal_old.empty else pd.Timestamp(now.date()) - pd.DateOffset(years=args.lookback_years)
    cal_api = dl.pro.trade_cal(
        exchange="SSE",
        start_date=_yyyymmdd(cal_start),
        end_date=_yyyymmdd(pd.Timestamp(now.date()) + pd.Timedelta(days=40)),
    )
    dl._pause()
    require(cal_api is not None and not cal_api.empty, "交易日历刷新失败")
    cal = cal_api.rename(columns={"cal_date": "date"})
    cal["date"] = pd.to_datetime(cal["date"], format="%Y%m%d")
    cal = cal[["date", "is_open"]].sort_values("date")
    calendar = TradingCalendar(cal)
    latest = last_completed_session(calendar, now)
    if latest <= last:
        print(f"数据已经是最近完整交易日 {last.date()}，无需刷新。")
        return 0

    sessions = [
        d for d in sorted(calendar.days)
        if calendar.days[d] and last < d <= latest
    ]
    require(bool(sessions), "没有需要增量刷新的完成交易日")
    cal.to_csv(output / "trade_calendar.csv", index=False)

    members_old = _read(output / "index_membership.csv", ["effective_date"])
    mem_api = dl.pro.index_weight(
        index_code="399300.SZ",
        start_date=_yyyymmdd(last - pd.Timedelta(days=60)),
        end_date=_yyyymmdd(latest),
    )
    dl._pause()
    if mem_api is not None and not mem_api.empty:
        mem_new = mem_api.rename(columns={"con_code": "ticker", "trade_date": "effective_date"})
        mem_new["effective_date"] = pd.to_datetime(mem_new["effective_date"], errors="raise")
        mem_new = mem_new[["index_code", "ticker", "effective_date", "weight"]]
    else:
        mem_new = pd.DataFrame(columns=members_old.columns)
    members = merge_frame(
        members_old, mem_new,
        ["effective_date", "ticker"],
        ["effective_date", "ticker"],
    )
    members.to_csv(output / "index_membership.csv", index=False)
    eligible = members[members["effective_date"] <= latest]
    effective = eligible["effective_date"].max()
    current = set(eligible.loc[eligible["effective_date"] == effective, "ticker"].astype(str))
    require(len(current) == 300, "最新沪深300成分快照不是 300 只")
    require(0 <= (latest - effective).days <= 45, "最新沪深300成分快照过旧")

    raw_old = _read(output / "raw_prices.csv", ["date"])
    existing_tickers = set(raw_old["ticker"].astype(str)) if not raw_old.empty else set()
    new_tickers = sorted(current - existing_tickers)

    raw_new, db_new, limit_new = _bulk_daily_rows(dl, sessions, current)

    backfill_start = max(
        pd.Timestamp(latest) - pd.Timedelta(days=500),
        pd.Timestamp(latest) - pd.DateOffset(years=args.lookback_years),
    )
    for ticker in new_tickers:
        r, d, l = _ticker_backfill(dl, ticker, backfill_start, latest)
        if not r.empty:
            raw_new = pd.concat([raw_new, r], ignore_index=True)
        if not d.empty:
            db_new = pd.concat([db_new, d], ignore_index=True)
        if not l.empty:
            limit_new = pd.concat([limit_new, l], ignore_index=True)

    raw = merge_frame(raw_old, raw_new, ["date", "ticker"], ["date", "ticker"])
    raw.to_csv(output / "raw_prices.csv", index=False)
    raw_to_adjusted(raw).to_csv(output / "prices.csv", index=False)

    db_old = _read(output / "daily_basic.csv", ["date"])
    db = merge_frame(db_old, db_new, ["date", "ticker"], ["date", "ticker"])
    db.to_csv(output / "daily_basic.csv", index=False)

    limits_old = _read(output / "stock_limits.csv", ["date"])
    limits = merge_frame(limits_old, limit_new, ["date", "ticker"], ["date", "ticker"])
    limits.to_csv(output / "stock_limits.csv", index=False)

    financial_old = _read(output / "fundamentals_raw.csv", ["ann_date", "report_date"])
    financial_new = _financial_rows(dl, sorted(current), last + pd.Timedelta(days=1), latest)
    for ticker in new_tickers:
        extra = _financial_rows(dl, [ticker], latest - pd.Timedelta(days=900), latest)
        if not extra.empty:
            financial_new = pd.concat([financial_new, extra], ignore_index=True)
    financial = merge_frame(
        financial_old, financial_new,
        ["ticker", "ann_date", "report_date"],
        ["ticker", "ann_date", "report_date"],
    )
    financial.to_csv(output / "fundamentals_raw.csv", index=False)

    dl.fetch_stock_metadata()

    old_industry = _read(output / "industry_membership.csv", ["in_date", "out_date"])
    previous_industry = previous.get("industry_refreshed_at")
    month_changed = (
        previous_industry is None
        or pd.Timestamp(previous_industry).to_period("M") != latest.to_period("M")
    )
    if new_tickers or month_changed or old_industry.empty:
        dl.fetch_sw_industry_membership(sorted(current))
        industry_refreshed_at = str(latest.date())
    else:
        industry_refreshed_at = str(pd.Timestamp(previous_industry).date())

    st_status = previous.get("st_history_status", "unknown")
    try:
        st_api = dl.pro.stock_st(trade_date=_yyyymmdd(latest))
        dl._pause()
        if st_api is not None:
            st_new = st_api.rename(columns={"ts_code": "ticker"})
            if not st_new.empty:
                if "trade_date" in st_new:
                    st_new["trade_date"] = pd.to_datetime(st_new["trade_date"], errors="coerce")
                st_old = _read(output / "st_status.csv", ["trade_date"])
                st = merge_frame(
                    st_old,
                    st_new[[c for c in ["ticker", "name", "trade_date", "type", "type_name"] if c in st_new]],
                    ["trade_date", "ticker"],
                    ["trade_date", "ticker"],
                )
                st.to_csv(output / "st_status.csv", index=False)
            st_status = "incremental_bulk"
    except Exception as exc:
        st_status = f"fallback_stock_basic_name:{type(exc).__name__}"

    benchmark_old = _read(output / "benchmark.csv", ["date"])
    bench_api = dl.pro.index_daily(
        ts_code="399300.SZ",
        start_date=_yyyymmdd(last + pd.Timedelta(days=1)),
        end_date=_yyyymmdd(latest),
    )
    dl._pause()
    if bench_api is None:
        bench_api = pd.DataFrame()
    if not bench_api.empty:
        benchmark_new = bench_api.rename(columns={"trade_date": "date", "ts_code": "index_code"})
        benchmark_new["date"] = pd.to_datetime(benchmark_new["date"], errors="coerce")
    else:
        benchmark_new = pd.DataFrame(columns=benchmark_old.columns)
    benchmark = merge_frame(benchmark_old, benchmark_new, ["date"], ["date"])
    benchmark.to_csv(output / "benchmark.csv", index=False)

    manifest = _manifest(
        output, previous, latest, len(current), industry_refreshed_at, st_status
    )
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    load_verified_dataset(output)
    print(
        f"增量刷新完成：{last.date()} -> {latest.date()}，"
        f"{len(sessions)} 个交易日，当前成分 {len(current)}，新成分 {len(new_tickers)}。"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
