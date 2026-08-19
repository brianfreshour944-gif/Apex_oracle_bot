"""Modern Streamlit Live Monitoring Dashboard for Apex Oracle Bot."""

import asyncio
import httpx
import streamlit as st
import polars as pl
import pandas as pd

from src.config import settings

st.set_page_config(
    page_title="Apex Oracle Bot Dashboard",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("🤖 Apex Oracle Bot — Live Dashboard")
st.markdown("Real-time Quantitative Crypto Trading & Regime Monitoring Interface")

# --- Sidebar Controls & Connection Info ---
st.sidebar.header("⚙️ System Status")

status_url = f"http://localhost:{settings.STATUS_PORT}/api/v1/health"

@st.cache_data(ttl=5)
def fetch_health_status(url: str):
    try:
        resp = httpx.get(url, timeout=3.0)
        if resp.status_code == 200:
            return resp.json()
    except Exception:
        return None
    return None

health_data = fetch_health_status(status_url)

if health_data:
    st.sidebar.success(f"Bot API: {health_data.get('status', 'online').upper()}")
    st.sidebar.write(f"**Version**: {health_data.get('version', '2.0.0')}")
    st.sidebar.write(f"**Database Connected**: {health_data.get('database', {}).get('connected', False)}")
else:
    st.sidebar.warning("Bot API: Offline / Unreachable")

st.sidebar.markdown("---")
st.sidebar.header("🎯 Monitored Assets")
st.sidebar.write(", ".join(settings.SYMBOLS))

# --- Metrics Overview Section ---
col1, col2, col3, col4 = st.columns(4)

col1.metric("Base Account Equity", f"${settings.ACCOUNT_BASE:,.2f}")
col2.metric("Max Portfolio Cap", f"${settings.ACCOUNT_BASE * getattr(settings, 'MAX_PORTFOLIO_PCT', 0.5):,.2f}")
col3.metric("Max Drawdown Stop", f"{settings.MAX_DRAWDOWN_STOP:.1f}%")
col4.metric("Daily Loss Limit", f"{settings.DAILY_LOSS_LIMIT:.1f}%")

st.markdown("---")

# --- Market Regime & Strategy Section ---
st.subheader("📊 Market Regime & Risk Configuration")

r_col1, r_col2, r_col3 = st.columns(3)
r_col1.metric("Hurst Trend Threshold", f">{settings.HURST_TREND_UP:.2f}")
r_col2.metric("Hurst Mean-Revert Ceiling", f"<{settings.HURST_MEAN_REVERT:.2f}")
r_col3.metric("High Volatility Stand-Aside ATR", f">{settings.HIGH_VOLATILITY_PCT:.1f}%")

st.markdown("---")

# --- Direct Backtest Quick Runner ---
st.subheader("⚡ Quick Strategy Backtest")
with st.expander("Run Historical Backtest Simulation"):
    b_col1, b_col2, b_col3 = st.columns(3)
    sym_select = b_col1.selectbox("Symbol", settings.SYMBOLS, index=0)
    bars_input = b_col2.number_input("Bars Count", min_value=100, max_value=2000, value=400)
    regime_select = b_col3.selectbox("Simulated Regime", ["trending", "mean_reverting", "volatile"])

    if st.button("Run Simulation"):
        from src.backtest import run_backtest
        with st.spinner("Simulating strategy execution over synthetic historical bars..."):
            res = asyncio.run(run_backtest(symbol=sym_select, n_bars=bars_input, regime=regime_select))

        res_col1, res_col2, res_col3, res_col4 = st.columns(4)
        res_col1.metric("Final Equity", f"${res.end_equity:,.2f}", delta=f"{res.total_return_pct:+.2f}%")
        res_col2.metric("Win Rate", f"{res.win_rate:.1f}%")
        res_col3.metric("Max Drawdown", f"{res.max_drawdown_pct:.2f}%")
        res_col4.metric("Total Trades", f"{res.n_trades}")

        st.line_chart(res.equity_curve)
