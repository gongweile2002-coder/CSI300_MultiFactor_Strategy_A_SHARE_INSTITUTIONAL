
from __future__ import annotations

from pathlib import Path
import base64
import io
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def _safe_read_csv(path):
    path = Path(path)
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def _fmt_pct(x, digits=2):
    try:
        if pd.isna(x):
            return "—"
        return f"{100*float(x):.{digits}f}%"
    except Exception:
        return "—"


def _fmt_num(x, digits=2):
    try:
        if pd.isna(x):
            return "—"
        return f"{float(x):.{digits}f}"
    except Exception:
        return "—"


def _fig_to_base64(fig):
    bio = io.BytesIO()
    fig.savefig(bio, format="png", dpi=160, bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(bio.getvalue()).decode("ascii")


def _img_html(b64, alt):
    return f'<img alt="{alt}" src="data:image/png;base64,{b64}" style="max-width:100%;height:auto;">'


def load_run_folder(folder):
    folder = Path(folder)

    files = {
        "performance": ["performance_summary_v6.csv", "performance_summary_v5.csv", "performance_summary_v4.csv"],
        "targets": ["optimized_targets_v6.csv", "optimized_targets_v5.csv"],
        "risk": ["ex_ante_risk_v6.csv", "ex_ante_risk_v5.csv"],
        "diagnostics": ["portfolio_diagnostics_v6.csv", "portfolio_diagnostics_v5.csv"],
        "trades": ["trade_costs_v6.csv", "trade_costs_v5.csv"],
        "holdings": ["executed_holdings_v6.csv", "executed_holdings_v5.csv"],
        "factor_ic": ["factor_ic_timeseries_v6.csv", "factor_ic_timeseries.csv"],
        "ic_summary": ["ic_summary_v6.csv", "ic_summary.csv"],
        "quintile": ["quintile_returns_v6.csv", "quintile_returns.csv"],
        "walk_forward": ["walk_forward_folds_v6.csv", "walk_forward_folds_v5.csv"],
        "walk_forward_ic": ["walk_forward_test_ic_v6.csv", "walk_forward_test_ic_v5.csv"],
        "nav": ["nav_timeseries_v6.csv", "nav_timeseries.csv"],
    }

    out = {}
    for key, names in files.items():
        df = pd.DataFrame()
        for n in names:
            p = folder/n
            if p.exists():
                df = pd.read_csv(p)
                break
        out[key] = df
    return out


def build_charts(data):
    charts = {}

    nav = data.get("nav", pd.DataFrame())
    if not nav.empty and {"date","strategy_nav"}.issubset(nav.columns):
        x = nav.copy()
        x["date"] = pd.to_datetime(x["date"])
        fig, ax = plt.subplots(figsize=(10, 4.8))
        ax.plot(x["date"], x["strategy_nav"], label="Strategy")
        if "benchmark_nav" in x.columns:
            ax.plot(x["date"], x["benchmark_nav"], label="Benchmark")
        ax.set_title("Net Asset Value")
        ax.set_xlabel("Date")
        ax.set_ylabel("NAV")
        ax.legend()
        fig.tight_layout()
        charts["nav"] = _fig_to_base64(fig)

        dd = x["strategy_nav"] / x["strategy_nav"].cummax() - 1
        fig, ax = plt.subplots(figsize=(10, 3.8))
        ax.plot(x["date"], dd)
        ax.set_title("Strategy Drawdown")
        ax.set_xlabel("Date")
        ax.set_ylabel("Drawdown")
        fig.tight_layout()
        charts["drawdown"] = _fig_to_base64(fig)

    risk = data.get("risk", pd.DataFrame())
    if not risk.empty and {"signal_date","ex_ante_tracking_error_annual"}.issubset(risk.columns):
        x = risk.copy()
        x["signal_date"] = pd.to_datetime(x["signal_date"])
        fig, ax = plt.subplots(figsize=(10, 3.8))
        ax.plot(x["signal_date"], x["ex_ante_tracking_error_annual"])
        ax.set_title("Ex-ante Tracking Error")
        ax.set_xlabel("Date")
        ax.set_ylabel("Annualized TE")
        fig.tight_layout()
        charts["te"] = _fig_to_base64(fig)

    wf = data.get("walk_forward_ic", pd.DataFrame())
    if not wf.empty and {"signal_date","rank_ic"}.issubset(wf.columns):
        x = wf.copy()
        x["signal_date"] = pd.to_datetime(x["signal_date"])
        fig, ax = plt.subplots(figsize=(10, 3.8))
        ax.plot(x["signal_date"], x["rank_ic"], marker="o")
        ax.axhline(0, linewidth=1)
        ax.set_title("Walk-forward Out-of-Sample Rank IC")
        ax.set_xlabel("Date")
        ax.set_ylabel("Rank IC")
        fig.tight_layout()
        charts["wf_ic"] = _fig_to_base64(fig)

    ic_summary = data.get("ic_summary", pd.DataFrame())
    if not ic_summary.empty and {"factor","mean"}.issubset(ic_summary.columns):
        x = ic_summary.copy()
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.bar(x["factor"].astype(str), x["mean"].astype(float))
        ax.set_title("Mean Rank IC by Factor")
        ax.set_ylabel("Mean Rank IC")
        ax.tick_params(axis="x", rotation=25)
        fig.tight_layout()
        charts["ic"] = _fig_to_base64(fig)

    diag = data.get("diagnostics", pd.DataFrame())
    if not diag.empty and {"signal_date","effective_n"}.issubset(diag.columns):
        x = diag.copy()
        x["signal_date"] = pd.to_datetime(x["signal_date"])
        fig, ax = plt.subplots(figsize=(10, 3.8))
        ax.plot(x["signal_date"], x["effective_n"])
        ax.set_title("Effective Number of Holdings")
        ax.set_xlabel("Date")
        ax.set_ylabel("Effective N")
        fig.tight_layout()
        charts["effective_n"] = _fig_to_base64(fig)

    return charts


def generate_html_report(folder, output_html, title="CSI300 Multi-Factor Quant Research Report"):
    folder = Path(folder)
    output_html = Path(output_html)
    output_html.parent.mkdir(parents=True, exist_ok=True)

    data = load_run_folder(folder)
    charts = build_charts(data)

    perf = data["performance"]
    perf_row = perf.iloc[0].to_dict() if not perf.empty else {}

    # Latest target portfolio
    targets = data["targets"].copy()
    latest_targets = pd.DataFrame()
    if not targets.empty and "signal_date" in targets.columns:
        targets["signal_date"] = pd.to_datetime(targets["signal_date"])
        latest_date = targets["signal_date"].max()
        latest_targets = targets[targets["signal_date"] == latest_date].copy()
        latest_targets = latest_targets.sort_values("target_weight", ascending=False)

    risk = data["risk"].copy()
    latest_risk = {}
    if not risk.empty:
        if "signal_date" in risk.columns:
            risk["signal_date"] = pd.to_datetime(risk["signal_date"])
            latest_risk = risk.sort_values("signal_date").iloc[-1].to_dict()
        else:
            latest_risk = risk.iloc[-1].to_dict()

    trades = data["trades"]
    total_cost = float(trades["total_cost"].sum()) if not trades.empty and "total_cost" in trades.columns else np.nan
    avg_turnover = float(trades["gross_turnover"].mean()) if not trades.empty and "gross_turnover" in trades.columns else np.nan

    ic_summary = data["ic_summary"]
    wf = data["walk_forward"]
    wf_ic = data["walk_forward_ic"]

    def table_html(df, cols=None, n=15, pct_cols=None):
        if df is None or df.empty:
            return "<p class='muted'>No data available.</p>"
        x = df.copy()
        if cols:
            cols = [c for c in cols if c in x.columns]
            x = x[cols]
        x = x.head(n)
        pct_cols = pct_cols or []
        for c in pct_cols:
            if c in x.columns:
                x[c] = x[c].map(lambda v: _fmt_pct(v))
        return x.to_html(index=False, border=0, classes="dataframe")

    metrics = [
        ("Total Return", _fmt_pct(perf_row.get("total_return"))),
        ("Annualized Return", _fmt_pct(perf_row.get("annualized_return"))),
        ("Annualized Volatility", _fmt_pct(perf_row.get("annualized_volatility"))),
        ("Sharpe", _fmt_num(perf_row.get("sharpe"))),
        ("Max Drawdown", _fmt_pct(perf_row.get("max_drawdown"))),
        ("Information Ratio", _fmt_num(perf_row.get("information_ratio"))),
        ("Beta", _fmt_num(perf_row.get("beta"))),
        ("Annualized Alpha", _fmt_pct(perf_row.get("alpha_annualized"))),
    ]
    metric_cards = "".join(
        f"<div class='metric'><div class='metric-name'>{k}</div><div class='metric-value'>{v}</div></div>"
        for k, v in metrics
    )

    latest_cols = [
        "ticker","industry_l1","target_weight","benchmark_weight","active_weight",
        "composite_score","optimizer_status"
    ]

    html = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{title}</title>
<style>
body {{
  font-family: -apple-system,BlinkMacSystemFont,"Segoe UI",Arial,sans-serif;
  margin: 0;
  background: #f5f6f8;
  color: #111827;
}}
.container {{ max-width: 1200px; margin: 0 auto; padding: 28px; }}
.hero {{ background: white; padding: 28px; border-radius: 16px; margin-bottom: 18px; }}
.hero h1 {{ margin: 0 0 8px 0; font-size: 30px; }}
.muted {{ color: #6b7280; }}
.grid {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(170px,1fr)); gap:12px; margin:16px 0 22px; }}
.metric {{ background:white; padding:16px; border-radius:12px; border:1px solid #e5e7eb; }}
.metric-name {{ color:#6b7280; font-size:13px; }}
.metric-value {{ font-size:24px; font-weight:700; margin-top:6px; }}
.card {{ background:white; padding:22px; border-radius:16px; margin-bottom:18px; }}
.card h2 {{ margin-top:0; }}
.card h3 {{ margin-bottom:8px; }}
img {{ display:block; margin:0 auto; }}
.dataframe {{ width:100%; border-collapse:collapse; font-size:13px; }}
.dataframe th,.dataframe td {{ padding:8px 10px; border-bottom:1px solid #e5e7eb; text-align:right; }}
.dataframe th:first-child,.dataframe td:first-child {{ text-align:left; }}
.tag {{ display:inline-block; padding:5px 9px; border-radius:999px; background:#eef2f7; margin-right:6px; font-size:12px; }}
.two {{ display:grid; grid-template-columns:1fr 1fr; gap:18px; }}
@media(max-width:800px) {{ .two {{ grid-template-columns:1fr; }} }}
code,pre {{ background:#f3f4f6; border-radius:8px; }}
pre {{ padding:14px; overflow:auto; }}
</style>
</head>
<body>
<div class="container">

<section class="hero">
  <h1>{title}</h1>
  <p class="muted">Point-in-Time Multi-Factor Research • Risk Model • Portfolio Optimization • Realistic Execution • Walk-Forward Validation</p>
  <span class="tag">Value</span>
  <span class="tag">Quality</span>
  <span class="tag">Momentum</span>
  <span class="tag">CSI 300</span>
  <span class="tag">Python</span>
</section>

<section class="card">
  <h2>Executive Summary</h2>
  <p>This report summarizes the end-to-end research pipeline from point-in-time factor construction through benchmark-aware portfolio optimization and execution-aware backtesting.</p>
  <div class="grid">{metric_cards}</div>
</section>

<section class="card">
  <h2>Performance</h2>
  { _img_html(charts["nav"], "NAV") if "nav" in charts else "<p class='muted'>NAV time series unavailable.</p>" }
  { _img_html(charts["drawdown"], "Drawdown") if "drawdown" in charts else "" }
</section>

<section class="two">
  <div class="card">
    <h2>Risk Model</h2>
    <p><b>Latest ex-ante vol:</b> {_fmt_pct(latest_risk.get("ex_ante_vol_annual"))}</p>
    <p><b>Latest ex-ante tracking error:</b> {_fmt_pct(latest_risk.get("ex_ante_tracking_error_annual"))}</p>
    <p><b>Average turnover:</b> {_fmt_pct(avg_turnover)}</p>
    <p><b>Cumulative modeled trading cost:</b> {_fmt_pct(total_cost)}</p>
    { _img_html(charts["te"], "Tracking Error") if "te" in charts else "" }
  </div>
  <div class="card">
    <h2>Portfolio Concentration</h2>
    { _img_html(charts["effective_n"], "Effective N") if "effective_n" in charts else "<p class='muted'>Portfolio diagnostics unavailable.</p>" }
  </div>
</section>

<section class="card">
  <h2>Latest Optimized Portfolio</h2>
  {table_html(latest_targets, latest_cols, n=20, pct_cols=["target_weight","benchmark_weight","active_weight"])}
</section>

<section class="two">
  <div class="card">
    <h2>Factor IC</h2>
    { _img_html(charts["ic"], "Factor IC") if "ic" in charts else "" }
    {table_html(ic_summary, ["factor","mean","std","count","icir"], n=10)}
  </div>
  <div class="card">
    <h2>Walk-Forward OOS Validation</h2>
    { _img_html(charts["wf_ic"], "Walk-forward IC") if "wf_ic" in charts else "" }
    {table_html(wf, ["fold","train_start","train_end","test_start","test_end","value_weight","quality_weight","momentum_weight","train_mean_rank_ic","test_mean_rank_ic"], n=10)}
  </div>
</section>

<section class="card">
  <h2>Methodology</h2>
  <pre>Historical CSI300 Constituents
→ Point-in-Time Fundamentals
→ ST / Listing-age / Liquidity Filters
→ Value / Quality / Momentum
→ Winsorization + Z-score
→ Industry + Size Neutralization
→ Composite Alpha Score
→ Shrinkage Covariance Risk Model
→ Benchmark-aware Portfolio Optimization
→ Single-name / Sector / Tracking-error Constraints
→ Turnover Penalty
→ Next-open Execution
→ Suspension / Limit-up / Limit-down Constraints
→ Commission / Slippage / Historical Stamp Duty
→ Backtest + Walk-forward OOS Validation</pre>
</section>

<section class="card">
  <h2>Research Caveats</h2>
  <p>This is a research framework, not investment advice. Demo outputs are synthetic and must not be presented as live or historical real-market performance. Real results depend on data entitlements, point-in-time coverage, transaction-cost assumptions, market microstructure, and implementation details.</p>
</section>

</div>
</body>
</html>
"""
    output_html.write_text(html, encoding="utf-8")
    return output_html
