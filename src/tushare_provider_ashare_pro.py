
from __future__ import annotations

import pandas as pd
from .tushare_provider_v4 import TushareDownloaderV4, _yyyymmdd


class TushareDownloaderASharePro(TushareDownloaderV4):
    """
    Adds A-share-specific fields:
      daily_basic: turnover_rate_f, volume_ratio, dv_ttm, limit_status
      cashflow: free_cashflow and operating cash flow
    """

    def fetch_daily_basic_for_tickers(self, tickers, start_date, end_date) -> pd.DataFrame:
        frames = []
        fields = (
            "ts_code,trade_date,turnover_rate,turnover_rate_f,volume_ratio,"
            "pe,pe_ttm,pb,ps_ttm,dv_ratio,dv_ttm,total_share,float_share,"
            "free_share,total_mv,circ_mv,limit_status"
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
            raise RuntimeError("未取得 daily_basic A-share Pro 数据。")

        out = pd.concat(frames, ignore_index=True)
        out["date"] = pd.to_datetime(out["date"], errors="coerce")
        out = out.sort_values(["date","ticker"]).drop_duplicates(["date","ticker"])
        out.to_csv(self.output_dir/"daily_basic.csv", index=False)
        return out

    def fetch_cashflow_for_tickers(self, tickers, start_date, end_date) -> pd.DataFrame:
        frames = []
        fields = (
            "ts_code,ann_date,f_ann_date,end_date,comp_type,report_type,"
            "n_cashflow_act,c_pay_acq_const_fiolta,free_cashflow,net_profit,update_flag"
        )
        for ticker in tickers:
            df = self.pro.cashflow(
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
                "end_date":"report_date"
            })
            frames.append(df)

        out = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        if not out.empty:
            out["ann_date"] = pd.to_datetime(out["ann_date"], errors="coerce")
            out["f_ann_date"] = pd.to_datetime(out["f_ann_date"], errors="coerce")
            out["report_date"] = pd.to_datetime(out["report_date"], errors="coerce")
            # Prefer actual announcement date when present.
            out["pit_ann_date"] = out["f_ann_date"].fillna(out["ann_date"])
            out = out.sort_values(["ticker","pit_ann_date","report_date"]).drop_duplicates(
                ["ticker","pit_ann_date","report_date"], keep="last"
            )
        out.to_csv(self.output_dir/"cashflow_raw.csv", index=False)
        return out

    def download_all_ashare_pro(self, index_code, start_date, end_date, max_stocks=None):
        summary = self.download_all_v4(index_code, start_date, end_date, max_stocks=max_stocks)
        membership = pd.read_csv(self.output_dir/"index_membership.csv")
        tickers = sorted(membership["ticker"].dropna().unique().tolist())
        if max_stocks:
            tickers = tickers[:int(max_stocks)]

        # Re-fetch richer daily basic fields.
        extended_start = pd.Timestamp(start_date) - pd.Timedelta(days=300)
        self.fetch_daily_basic_for_tickers(tickers, extended_start, end_date)

        try:
            self.fetch_cashflow_for_tickers(
                tickers,
                pd.Timestamp(start_date) - pd.Timedelta(days=800),
                end_date
            )
            summary["cashflow"] = "downloaded"
        except Exception as e:
            pd.DataFrame().to_csv(self.output_dir/"cashflow_raw.csv", index=False)
            summary["cashflow"] = f"unavailable:{type(e).__name__}"

        summary["ashare_pro"] = True
        return summary
