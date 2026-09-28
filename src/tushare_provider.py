
from __future__ import annotations

import os
import time
from dataclasses import dataclass
from pathlib import Path
import pandas as pd

try:
    import tushare as ts
except ImportError:
    ts = None


def _yyyymmdd(x) -> str:
    return pd.Timestamp(x).strftime("%Y%m%d")


def _month_ranges(start_date, end_date):
    start = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date)
    months = pd.period_range(start=start.to_period("M"), end=end.to_period("M"), freq="M")
    for m in months:
        s = max(start, m.start_time)
        e = min(end, m.end_time)
        yield s, e


@dataclass
class TushareDownloader:
    token: str
    output_dir: Path
    sleep_seconds: float = 0.06

    def __post_init__(self):
        if ts is None:
            raise ImportError("未安装 tushare。请先运行: pip install -r requirements.txt")
        if not self.token or self.token == "put_your_token_here":
            raise ValueError("缺少 TUSHARE_TOKEN。请复制 .env.example 为 .env 后填入自己的 token。")
        self.output_dir = Path(self.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        ts.set_token(self.token)
        self.pro = ts.pro_api(self.token)

    def _pause(self):
        if self.sleep_seconds:
            time.sleep(self.sleep_seconds)

    def fetch_index_membership(self, index_code, start_date, end_date) -> pd.DataFrame:
        frames = []
        for s, e in _month_ranges(start_date, end_date):
            df = self.pro.index_weight(
                index_code=index_code,
                start_date=_yyyymmdd(s),
                end_date=_yyyymmdd(e)
            )
            self._pause()
            if df is not None and not df.empty:
                frames.append(df)
        if not frames:
            raise RuntimeError("未取得指数成分数据。请检查 Tushare 权限、指数代码和日期范围。")
        out = pd.concat(frames, ignore_index=True).drop_duplicates()
        out = out.rename(columns={
            "con_code": "ticker",
            "trade_date": "effective_date"
        })
        out["effective_date"] = pd.to_datetime(out["effective_date"])
        out = out[["index_code", "ticker", "effective_date", "weight"]]
        out = out.sort_values(["effective_date", "ticker"]).reset_index(drop=True)
        out.to_csv(self.output_dir / "index_membership.csv", index=False)
        return out

    def fetch_stock_metadata(self) -> pd.DataFrame:
        frames = []
        # Include listed + delisted so historical constituents are not silently lost.
        for status in ["L", "D", "P"]:
            df = self.pro.stock_basic(
                exchange="",
                list_status=status,
                fields="ts_code,symbol,name,area,industry,market,exchange,list_status,list_date,delist_date"
            )
            self._pause()
            if df is not None and not df.empty:
                frames.append(df)
        if not frames:
            return pd.DataFrame()
        out = pd.concat(frames, ignore_index=True).drop_duplicates("ts_code")
        out = out.rename(columns={"ts_code": "ticker"})
        for c in ["list_date", "delist_date"]:
            if c in out.columns:
                out[c] = pd.to_datetime(out[c], errors="coerce")
        out.to_csv(self.output_dir / "stock_metadata.csv", index=False)
        return out

    def fetch_prices_for_tickers(self, tickers, start_date, end_date) -> pd.DataFrame:
        from .data_contract_v8 import normalize_tushare_prices
        frames = []
        for ticker in tickers:
            daily = self.pro.daily(ts_code=ticker, start_date=_yyyymmdd(start_date), end_date=_yyyymmdd(end_date))
            self._pause()
            adj = self.pro.adj_factor(ts_code=ticker, start_date=_yyyymmdd(start_date), end_date=_yyyymmdd(end_date))
            self._pause()
            if daily is None or daily.empty or adj is None or adj.empty:
                raise RuntimeError(f'{ticker}: 缺少原始行情或复权因子，下载中止')
            frames.append(normalize_tushare_prices(daily, adj))
        if not frames:
            raise RuntimeError('未取得股票行情')
        raw = pd.concat(frames, ignore_index=True).sort_values(['date','ticker'])
        raw.to_csv(self.output_dir/'raw_prices.csv', index=False)
        out = raw.copy()
        for col in ['open','high','low','close']:
            out['raw_'+col] = out[col]
            out[col] = out[col] * out['adj_factor']
        out['price_basis'] = 'adjusted_research_only'
        out.to_csv(self.output_dir/'prices.csv', index=False)
        return out

    def fetch_daily_basic_for_tickers(self, tickers, start_date, end_date) -> pd.DataFrame:
        frames = []
        fields = (
            "ts_code,trade_date,turnover_rate,pe,pe_ttm,pb,"
            "total_share,float_share,free_share,total_mv,circ_mv"
        )
        for ticker in tickers:
            df = self.pro.daily_basic(
                ts_code=ticker,
                start_date=_yyyymmdd(start_date),
                end_date=_yyyymmdd(end_date),
                fields=fields
            )
            self._pause()
            if df is None or df.empty:
                continue
            df = df.rename(columns={"trade_date":"date","ts_code":"ticker"})
            frames.append(df)
        if not frames:
            raise RuntimeError("未取得 daily_basic 数据。")
        out = pd.concat(frames, ignore_index=True)
        out["date"] = pd.to_datetime(out["date"])
        out = out.sort_values(["date","ticker"]).drop_duplicates(["date","ticker"])
        out.to_csv(self.output_dir / "daily_basic.csv", index=False)
        return out

    def fetch_fundamentals_for_tickers(self, tickers, start_date, end_date) -> pd.DataFrame:
        frames = []
        fields = "ts_code,ann_date,end_date,roe,roe_dt,netprofit_yoy,debt_to_assets"
        for ticker in tickers:
            df = self.pro.fina_indicator(
                ts_code=ticker,
                start_date=_yyyymmdd(start_date),
                end_date=_yyyymmdd(end_date),
                fields=fields
            )
            self._pause()
            if df is None or df.empty:
                continue
            df = df.rename(columns={
                "ts_code":"ticker",
                "end_date":"report_date",
                "netprofit_yoy":"profit_growth",
                "debt_to_assets":"debt_ratio"
            })
            frames.append(df)
        if not frames:
            raise RuntimeError("未取得财务指标数据。")
        out = pd.concat(frames, ignore_index=True)
        out["ann_date"] = pd.to_datetime(out["ann_date"], errors="coerce")
        out["report_date"] = pd.to_datetime(out["report_date"], errors="coerce")
        # Some companies may publish duplicate/revised records. Keep the latest record per announcement/report pair.
        out = out.dropna(subset=["ann_date"]).sort_values(
            ["ticker","ann_date","report_date"]
        ).drop_duplicates(["ticker","ann_date","report_date"], keep="last")
        out.to_csv(self.output_dir / "fundamentals_raw.csv", index=False)
        return out

    def fetch_benchmark(self, index_code, start_date, end_date) -> pd.DataFrame:
        df = self.pro.index_daily(
            ts_code=index_code,
            start_date=_yyyymmdd(start_date),
            end_date=_yyyymmdd(end_date)
        )
        self._pause()
        if df is None or df.empty:
            raise RuntimeError("未取得指数基准行情。")
        out = df.rename(columns={"trade_date":"date","ts_code":"index_code"})
        out["date"] = pd.to_datetime(out["date"])
        out = out.sort_values("date")
        out.to_csv(self.output_dir / "benchmark.csv", index=False)
        return out

    def download_all(self, index_code, start_date, end_date, max_stocks=None):
        membership = self.fetch_index_membership(index_code, start_date, end_date)
        metadata = self.fetch_stock_metadata()

        tickers = sorted(membership["ticker"].dropna().unique().tolist())
        if max_stocks:
            tickers = tickers[:int(max_stocks)]

        # Extra lookback for 6m momentum and trailing liquidity calculations.
        extended_start = pd.Timestamp(start_date) - pd.Timedelta(days=230)

        self.fetch_prices_for_tickers(tickers, extended_start, end_date)
        self.fetch_daily_basic_for_tickers(tickers, extended_start, end_date)
        self.fetch_fundamentals_for_tickers(tickers, extended_start - pd.Timedelta(days=400), end_date)
        self.fetch_benchmark(index_code, extended_start, end_date)

        return {
            "tickers": len(tickers),
            "index_code": index_code,
            "start_date": str(pd.Timestamp(start_date).date()),
            "end_date": str(pd.Timestamp(end_date).date()),
            "output_dir": str(self.output_dir)
        }
