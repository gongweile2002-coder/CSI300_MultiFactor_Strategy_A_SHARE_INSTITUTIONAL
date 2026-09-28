
from pathlib import Path
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

from src.reporting_v6 import load_run_folder

BASE = Path(__file__).resolve().parent

st.set_page_config(
    page_title="CSI300 Multi-Factor Quant Dashboard",
    layout="wide"
)

st.title("CSI300 Multi-Factor Quant Dashboard")
st.info("此面板显示历史研究/合成演示输出。v8 实盘入口为 live.py；这里的收益图不是新版实盘或样本外业绩证明。")
st.caption("Point-in-Time • Multi-Factor • Risk Model • Portfolio Optimization • Walk-Forward")

default_demo = BASE/"outputs"/"demo_v6"
default_real = BASE/"outputs"/"real_v6"

with st.sidebar:
    st.header("Data Source")
    source = st.radio(
        "Run folder",
        ["Demo v6", "Real v6", "Custom folder"],
        index=0
    )
    if source == "Demo v6":
        folder = default_demo
    elif source == "Real v6":
        folder = default_real
    else:
        custom = st.text_input("Folder path", str(default_demo))
        folder = Path(custom)

    st.caption(f"Reading: {folder}")

data = load_run_folder(folder)

perf = data["performance"]
nav = data["nav"]
targets = data["targets"]
risk = data["risk"]
diag = data["diagnostics"]
trades = data["trades"]
ic_summary = data["ic_summary"]
ic_ts = data["factor_ic"]
wf = data["walk_forward"]
wf_ic = data["walk_forward_ic"]

if perf.empty:
    st.warning("No performance summary found in this folder. Run the corresponding pipeline first.")
    st.stop()

p = perf.iloc[0].to_dict()

def pct(x):
    if pd.isna(x):
        return "—"
    return f"{100*float(x):.2f}%"

def num(x):
    if pd.isna(x):
        return "—"
    return f"{float(x):.2f}"

c1,c2,c3,c4 = st.columns(4)
c1.metric("Annualized Return", pct(p.get("annualized_return")))
c2.metric("Sharpe", num(p.get("sharpe")))
c3.metric("Max Drawdown", pct(p.get("max_drawdown")))
c4.metric("Information Ratio", num(p.get("information_ratio")))

c5,c6,c7,c8 = st.columns(4)
c5.metric("Annualized Volatility", pct(p.get("annualized_volatility")))
c6.metric("Alpha", pct(p.get("alpha_annualized")))
c7.metric("Beta", num(p.get("beta")))
c8.metric("Tracking Error", pct(p.get("tracking_error")))

tabs = st.tabs([
    "Performance",
    "Factors",
    "Portfolio",
    "Risk",
    "Walk-Forward",
    "Trades",
    "Latest Holdings",
    "Paper Trading",
    "Risk & Ops"
])

with tabs[0]:
    if not nav.empty and {"date","strategy_nav"}.issubset(nav.columns):
        x = nav.copy()
        x["date"] = pd.to_datetime(x["date"])
        fig, ax = plt.subplots(figsize=(11,4.5))
        ax.plot(x["date"], x["strategy_nav"], label="Strategy")
        if "benchmark_nav" in x.columns:
            ax.plot(x["date"], x["benchmark_nav"], label="Benchmark")
        ax.set_title("Net Asset Value")
        ax.set_xlabel("Date")
        ax.set_ylabel("NAV")
        ax.legend()
        st.pyplot(fig)

        dd = x["strategy_nav"]/x["strategy_nav"].cummax()-1
        fig, ax = plt.subplots(figsize=(11,3.6))
        ax.plot(x["date"], dd)
        ax.set_title("Drawdown")
        ax.set_xlabel("Date")
        ax.set_ylabel("Drawdown")
        st.pyplot(fig)
    else:
        st.info("NAV file not found.")

with tabs[1]:
    if not ic_summary.empty:
        st.subheader("IC Summary")
        st.dataframe(ic_summary, use_container_width=True)

        if {"factor","mean"}.issubset(ic_summary.columns):
            fig, ax = plt.subplots(figsize=(8,4))
            ax.bar(ic_summary["factor"].astype(str), ic_summary["mean"].astype(float))
            ax.set_title("Mean Rank IC by Factor")
            ax.set_ylabel("Mean Rank IC")
            ax.tick_params(axis="x", rotation=25)
            st.pyplot(fig)

    if not ic_ts.empty and {"signal_date","factor","rank_ic"}.issubset(ic_ts.columns):
        x = ic_ts.copy()
        x["signal_date"] = pd.to_datetime(x["signal_date"])
        for f in sorted(x["factor"].unique()):
            g = x[x["factor"] == f]
            fig, ax = plt.subplots(figsize=(10,3.2))
            ax.plot(g["signal_date"], g["rank_ic"])
            ax.axhline(0, linewidth=1)
            ax.set_title(f"{f} Rank IC")
            ax.set_xlabel("Date")
            ax.set_ylabel("Rank IC")
            st.pyplot(fig)
    else:
        st.info("Factor IC time series not found.")

with tabs[2]:
    if not diag.empty:
        st.subheader("Portfolio Diagnostics")
        st.dataframe(diag.tail(24), use_container_width=True)

        if {"signal_date","effective_n"}.issubset(diag.columns):
            x = diag.copy()
            x["signal_date"] = pd.to_datetime(x["signal_date"])
            fig, ax = plt.subplots(figsize=(10,3.6))
            ax.plot(x["signal_date"], x["effective_n"])
            ax.set_title("Effective Number of Holdings")
            ax.set_xlabel("Date")
            ax.set_ylabel("Effective N")
            st.pyplot(fig)

        if {"signal_date","max_weight"}.issubset(diag.columns):
            x = diag.copy()
            x["signal_date"] = pd.to_datetime(x["signal_date"])
            fig, ax = plt.subplots(figsize=(10,3.6))
            ax.plot(x["signal_date"], x["max_weight"])
            ax.set_title("Maximum Single-Name Weight")
            ax.set_xlabel("Date")
            ax.set_ylabel("Weight")
            st.pyplot(fig)
    else:
        st.info("Portfolio diagnostics not found.")

with tabs[3]:
    if not risk.empty:
        st.dataframe(risk.tail(24), use_container_width=True)
        if {"signal_date","ex_ante_vol_annual","ex_ante_tracking_error_annual"}.issubset(risk.columns):
            x = risk.copy()
            x["signal_date"] = pd.to_datetime(x["signal_date"])
            fig, ax = plt.subplots(figsize=(10,3.8))
            ax.plot(x["signal_date"], x["ex_ante_vol_annual"], label="Ex-ante Vol")
            ax.plot(x["signal_date"], x["ex_ante_tracking_error_annual"], label="Ex-ante TE")
            ax.set_title("Ex-ante Risk")
            ax.set_xlabel("Date")
            ax.set_ylabel("Annualized")
            ax.legend()
            st.pyplot(fig)
    else:
        st.info("Risk output not found.")

with tabs[4]:
    if not wf.empty:
        st.subheader("Walk-Forward Folds")
        st.dataframe(wf, use_container_width=True)
    if not wf_ic.empty and {"signal_date","rank_ic"}.issubset(wf_ic.columns):
        x = wf_ic.copy()
        x["signal_date"] = pd.to_datetime(x["signal_date"])
        fig, ax = plt.subplots(figsize=(10,3.8))
        ax.plot(x["signal_date"], x["rank_ic"], marker="o")
        ax.axhline(0, linewidth=1)
        ax.set_title("Out-of-Sample Rank IC")
        ax.set_xlabel("Date")
        ax.set_ylabel("Rank IC")
        st.pyplot(fig)
    if wf.empty and wf_ic.empty:
        st.info("Walk-forward output not found.")

with tabs[5]:
    if not trades.empty:
        st.dataframe(trades.tail(36), use_container_width=True)
        if {"execution_date","gross_turnover"}.issubset(trades.columns):
            x = trades.copy()
            x["execution_date"] = pd.to_datetime(x["execution_date"])
            fig, ax = plt.subplots(figsize=(10,3.6))
            ax.plot(x["execution_date"], x["gross_turnover"])
            ax.set_title("Gross Turnover")
            ax.set_xlabel("Date")
            ax.set_ylabel("Turnover")
            st.pyplot(fig)
        if "total_cost" in trades.columns:
            st.metric("Cumulative Modeled Trading Cost", pct(trades["total_cost"].sum()))
    else:
        st.info("Trade-cost output not found.")

with tabs[6]:
    if not targets.empty and "signal_date" in targets.columns:
        x = targets.copy()
        x["signal_date"] = pd.to_datetime(x["signal_date"])
        latest = x["signal_date"].max()
        g = x[x["signal_date"] == latest].sort_values("target_weight", ascending=False)
        st.subheader(f"Latest Optimized Target — {latest.date()}")
        st.dataframe(g, use_container_width=True)

        if {"ticker","target_weight"}.issubset(g.columns):
            top = g.head(15)
            fig, ax = plt.subplots(figsize=(10,5))
            ax.barh(top["ticker"].astype(str), top["target_weight"].astype(float))
            ax.set_title("Top Target Weights")
            ax.set_xlabel("Target Weight")
            ax.invert_yaxis()
            st.pyplot(fig)
    else:
        st.info("Optimized target file not found.")


with tabs[7]:
    paper_dir = BASE/"paper"
    summary_path = paper_dir/"account_summary.csv"
    pos_path = paper_dir/"positions.csv"
    orders_path = paper_dir/"orders.csv"
    fills_path = paper_dir/"fills.csv"

    st.subheader("Paper Trading")
    st.caption("Simulation only — no broker connection and no live order submission.")

    if summary_path.exists():
        s = pd.read_csv(summary_path)
        st.dataframe(s, use_container_width=True)
    else:
        st.info("No paper account summary yet.")

    if pos_path.exists():
        st.subheader("Positions")
        st.dataframe(pd.read_csv(pos_path), use_container_width=True)

    if orders_path.exists():
        st.subheader("Orders")
        st.dataframe(pd.read_csv(orders_path), use_container_width=True)

    if fills_path.exists():
        st.subheader("Fills")
        st.dataframe(pd.read_csv(fills_path), use_container_width=True)


with tabs[8]:
    ops_dir = BASE/"ops"
    st.subheader("Risk & Operations")
    st.caption("Operational controls for paper/manual-approval workflows. Live broker submission remains disabled.")

    kill_file = BASE/"ops"/"KILL_SWITCH"
    st.metric("Kill Switch", "ON" if kill_file.exists() else "OFF")

    for title, filename in [
        ("Target Risk Checks", "target_risk_checks.csv"),
        ("Capacity Checks", "capacity_checks.csv"),
        ("Risk Summary", "risk_summary.csv"),
        ("Stress Report", "stress_report.csv"),
        ("Reconciliation", "reconciliation_summary.csv"),
    ]:
        p = ops_dir/filename
        if p.exists():
            st.subheader(title)
            st.dataframe(pd.read_csv(p), use_container_width=True)
