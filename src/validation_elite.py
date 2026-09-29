
from __future__ import annotations
import math
import numpy as np
import pandas as pd


def annualized_sharpe(r: pd.Series):
    r = pd.to_numeric(r, errors="coerce").dropna()
    if len(r) < 2 or r.std(ddof=0) == 0:
        return np.nan
    return float(r.mean()/r.std(ddof=0)*np.sqrt(252))


def probabilistic_sharpe_ratio(r: pd.Series, benchmark_sr=0.0):
    """
    Approximate PSR following Bailey & Lopez de Prado intuition.
    Useful as a guard against small-sample Sharpe overconfidence.
    """
    x = pd.to_numeric(r, errors="coerce").dropna()
    n = len(x)
    if n < 30:
        return np.nan
    sr = annualized_sharpe(x)
    if not np.isfinite(sr):
        return np.nan
    # Convert annualized SR to daily scale for the approximation.
    sr_d = sr/np.sqrt(252)
    b_d = float(benchmark_sr)/np.sqrt(252)
    skew = float(x.skew())
    kurt = float(x.kurt()+3)
    denom = math.sqrt(max((1 - skew*sr_d + ((kurt-1)/4)*(sr_d**2))/(n-1), 1e-12))
    z = (sr_d-b_d)/denom
    # normal CDF without scipy dependency
    return 0.5*(1+math.erf(z/math.sqrt(2)))


def multiple_testing_haircut(sharpes: dict):
    """
    Simple transparent haircut: the more strategy variants you test, the higher
    the required Sharpe hurdle. This is deliberately conservative and easy to audit.
    """
    n = max(len(sharpes), 1)
    hurdle = math.sqrt(max(2*math.log(n), 0))
    rows = []
    for name,sr in sharpes.items():
        rows.append({
            "strategy":name,
            "raw_sharpe":sr,
            "testing_count":n,
            "haircut_hurdle":hurdle,
            "haircut_score":(float(sr)-hurdle if np.isfinite(sr) else np.nan)
        })
    return pd.DataFrame(rows)


def oos_rank_ic_summary(df, score_cols, target_col="forward_1m_return"):
    rows = []
    for c in score_cols:
        vals = []
        for d,g in df.groupby("signal_date"):
            t = g[[c,target_col]].dropna()
            if len(t) >= 10:
                vals.append(t[c].corr(t[target_col], method="spearman"))
        arr = pd.Series(vals).dropna()
        rows.append({
            "strategy":c,
            "mean_rank_ic":arr.mean() if len(arr) else np.nan,
            "ic_std":arr.std(ddof=0) if len(arr) else np.nan,
            "icir":arr.mean()/arr.std(ddof=0) if len(arr)>1 and arr.std(ddof=0)>0 else np.nan,
            "months":len(arr)
        })
    return pd.DataFrame(rows)
