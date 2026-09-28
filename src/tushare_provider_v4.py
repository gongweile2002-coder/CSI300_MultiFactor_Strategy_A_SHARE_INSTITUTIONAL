
from __future__ import annotations

from pathlib import Path
import time
import pandas as pd

from .tushare_provider import TushareDownloader, _yyyymmdd


class TushareDownloaderV4(TushareDownloader):

    def fetch_stock_limits_for_tickers(self, tickers, start_date, end_date) -> pd.DataFrame:
        frames = []
        for ticker in tickers:
            df = self.pro.stk_limit(
                ts_code=ticker,
                start_date=_yyyymmdd(start_date),
                end_date=_yyyymmdd(end_date)
            )
            self._pause()
            if df is None or df.empty:
                continue
            df = df.rename(columns={"trade_date":"date","ts_code":"ticker"})
            keep = [c for c in ["date","ticker","pre_close","up_limit","down_limit"] if c in df.columns]
            frames.append(df[keep])

        out = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(
            columns=["date","ticker","pre_close","up_limit","down_limit"]
        )
        if not out.empty:
            out["date"] = pd.to_datetime(out["date"], errors="coerce")
            out = out.sort_values(["date","ticker"]).drop_duplicates(["date","ticker"])
        out.to_csv(self.output_dir/"stock_limits.csv", index=False)
        return out

    def fetch_st_status_for_tickers(self, tickers, start_date, end_date) -> pd.DataFrame:
        """
        stock_st currently requires higher Tushare permission than the core 2000-point interfaces.
        If permission is unavailable, caller can catch the exception and continue without historical ST filtering.
        """
        frames = []
        for ticker in tickers:
            cursor = pd.Timestamp(start_date)
            end = pd.Timestamp(end_date)
            while cursor <= end:
                stop = min(end, cursor + pd.Timedelta(days=365))
                df = self.pro.stock_st(ts_code=ticker, start_date=_yyyymmdd(cursor), end_date=_yyyymmdd(stop))
                self._pause()
                if df is None:
                    raise RuntimeError(f'{ticker}: ST 接口未返回有效响应')
                if len(df) >= 1000:
                    raise RuntimeError('ST 接口达到返回上限，不能确认数据完整性')
                if not df.empty:
                    df = df.rename(columns={'ts_code':'ticker'})
                    keep = [c for c in ['ticker','name','trade_date','type','type_name'] if c in df]
                    frames.append(df[keep])
                cursor = stop + pd.Timedelta(days=1)

        out = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(
            columns=["ticker","name","trade_date","type","type_name"]
        )
        if not out.empty:
            out["trade_date"] = pd.to_datetime(out["trade_date"], errors="coerce")
            out = out.sort_values(["trade_date","ticker"]).drop_duplicates(["trade_date","ticker"])
        out.to_csv(self.output_dir/"st_status.csv", index=False)
        return out

    def fetch_sw_industry_membership(self, tickers) -> pd.DataFrame:
        frames = []
        for ticker in tickers:
            per_ticker = []
            # Pull current and historical memberships.
            for is_new in ["Y","N"]:
                df = self.pro.index_member_all(ts_code=ticker, is_new=is_new)
                self._pause()
                if df is not None and not df.empty:
                    per_ticker.append(df)
            if per_ticker:
                frames.append(pd.concat(per_ticker, ignore_index=True))

        out = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(
            columns=["l1_code","l1_name","l2_code","l2_name","l3_code","l3_name",
                     "ticker","name","in_date","out_date","is_new"]
        )
        if not out.empty:
            out = out.rename(columns={"ts_code":"ticker"})
            for c in ["in_date","out_date"]:
                if c in out.columns:
                    out[c] = pd.to_datetime(out[c], errors="coerce")
            out = out.drop_duplicates()
        out.to_csv(self.output_dir/"industry_membership.csv", index=False)
        return out

    def download_all_v4(self, index_code, start_date, end_date, max_stocks=None):
        summary = self.download_all(index_code, start_date, end_date, max_stocks=max_stocks)

        membership = pd.read_csv(self.output_dir/"index_membership.csv")
        tickers = sorted(membership["ticker"].dropna().unique().tolist())
        if max_stocks:
            tickers = tickers[:int(max_stocks)]

        extended_start = pd.Timestamp(start_date) - pd.Timedelta(days=230)

        # Price limits and industry are core v4 data.
        self.fetch_stock_limits_for_tickers(tickers, extended_start, end_date)
        self.fetch_sw_industry_membership(tickers)

        # ST history may require a higher permission tier. Keep pipeline usable if unavailable.
        try:
            self.fetch_st_status_for_tickers(tickers, extended_start, end_date)
            summary["st_history"] = "downloaded"
        except Exception as e:
            pd.DataFrame(
                columns=["ticker","name","trade_date","type","type_name"]
            ).to_csv(self.output_dir/"st_status.csv", index=False)
            summary["st_history"] = f"unavailable: {type(e).__name__}"

        summary["v4_advanced_data"] = True
        return summary
