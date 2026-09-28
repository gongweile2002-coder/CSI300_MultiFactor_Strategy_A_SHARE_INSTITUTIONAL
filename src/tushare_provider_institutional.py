from __future__ import annotations
import pandas as pd
from .tushare_provider_ashare_pro import TushareDownloaderASharePro, _yyyymmdd

class TushareDownloaderInstitutional(TushareDownloaderASharePro):
    def fetch_multi_index_membership(self, universes: dict, start_date, end_date):
        frames=[]
        for name,code in universes.items():
            df=self.fetch_index_membership(code,start_date,end_date); df["universe"]=name; frames.append(df)
        out=pd.concat(frames,ignore_index=True); out.to_csv(self.output_dir/"multi_index_membership.csv",index=False); return out

    def fetch_forecast_for_tickers(self,tickers,start_date,end_date):
        frames=[]; fields="ts_code,ann_date,end_date,type,p_change_min,p_change_max,net_profit_min,net_profit_max,last_parent_net,first_ann_date"
        for t in tickers:
            df=self.pro.forecast(ts_code=t,start_date=_yyyymmdd(start_date),end_date=_yyyymmdd(end_date),fields=fields); self._pause()
            if df is None or df.empty: continue
            frames.append(df.rename(columns={"ts_code":"ticker","end_date":"report_date"}))
        out=pd.concat(frames,ignore_index=True) if frames else pd.DataFrame()
        if not out.empty:
            out["ann_date"]=pd.to_datetime(out["ann_date"],errors="coerce"); out["report_date"]=pd.to_datetime(out["report_date"],errors="coerce")
        out.to_csv(self.output_dir/"forecast_raw.csv",index=False); return out

    def fetch_express_for_tickers(self,tickers,start_date,end_date):
        frames=[]; fields="ts_cod