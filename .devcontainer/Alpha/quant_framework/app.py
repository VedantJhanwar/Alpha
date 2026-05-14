"""
Streamlit entry point for the Quant Trading & Risk Analytics Framework.
=====================================================================

Streamlit-based web app that lets users customize every parameter:
- Ticker symbol (any Yahoo Finance ticker)
- Date range
- MA short/long windows

Results render live in the browser with interactive Plotly charts and 
institutional-grade metrics.

Usage:
    streamlit run app.py

Author: Quant Framework
"""

import logging
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# Module imports
from data.pipeline import fetch_market_data, fetch_live_info
from strategy.ma_crossover import compute_ma_signals
from strategy.rsi_signal import compute_rsi, compute_rsi_signals, apply_confluence_filter
from backtesting.engine import run_backtest
from risk.analytics import (
    benchmark_comparison,
    calmar_ratio,
    compute_annualized_return,
    compute_daily_returns,
    maximum_drawdown,
    sharpe_ratio,
)

# ---------------------------------------------------------------------------
# App setup & Color palette
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)-30s | %(levelname)-5s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

COLORS = {
    "cyan": "#22d3ee",
    "purple": "#a78bfa",
    "green": "#10b981",
    "red": "#ef4444",
    "amber": "#f59e0b",
    "muted": "#64748b",
}


def build_figures(df: pd.DataFrame, strategy_cum: pd.Series, benchmark_cum: pd.Series, portfolio_values: pd.Series, metrics: dict) -> dict:
    """Build all 4 Plotly charts directly as go.Figure objects for Streamlit."""
    theme = dict(
        template="plotly_dark",
        paper_bgcolor="#0a0e1a",
        plot_bgcolor="#0f1525",
        font=dict(family="Inter, system-ui", color="#e2e8f0"),
        margin=dict(l=50, r=30, t=40, b=40),
    )
    charts = {}
    
    # --- Chart 1: Price + MA Crossover ---
    fig1 = go.Figure()
    dates = df["Date"].astype(str).tolist()

    fig1.add_trace(go.Candlestick(
        x=dates, open=df["Open"], high=df["High"],
        low=df["Low"], close=df["Close"], name="OHLC",
        increasing_line_color=COLORS["green"],
        decreasing_line_color=COLORS["red"],
        showlegend=False,
    ))

    if "SMA_Short" in df.columns:
        fig1.add_trace(go.Scatter(
            x=dates, y=df["SMA_Short"], name="SMA Short",
            line=dict(color=COLORS["cyan"], width=1.5),
        ))
    if "SMA_Long" in df.columns:
        fig1.add_trace(go.Scatter(
            x=dates, y=df["SMA_Long"], name="SMA Long",
            line=dict(color=COLORS["purple"], width=1.5),
        ))

    # BUY / SELL markers
    if "Confirmed_Signal" in df.columns:
        buys = df[df["Confirmed_Signal"] == 1]
        if len(buys) > 0:
            fig1.add_trace(go.Scatter(
                x=buys["Date"].astype(str).tolist(),
                y=(buys["Low"] * 0.98).tolist(),
                mode="markers", name="BUY",
                marker=dict(symbol="triangle-up", size=14,
                            color=COLORS["green"],
                            line=dict(width=1, color="white")),
            ))
        sells = df[df["Confirmed_Signal"] == -1]
        if len(sells) > 0:
            fig1.add_trace(go.Scatter(
                x=sells["Date"].astype(str).tolist(),
                y=(sells["High"] * 1.02).tolist(),
                mode="markers", name="SELL",
                marker=dict(symbol="triangle-down", size=14,
                            color=COLORS["red"],
                            line=dict(width=1, color="white")),
            ))

    fig1.update_layout(xaxis_rangeslider_visible=False, height=420, title="Price Action & MA Crossover Signals", **theme)
    charts["price"] = fig1

    # --- Chart 2: RSI Oscillator ---
    fig2 = go.Figure()
    if "RSI" in df.columns:
        fig2.add_trace(go.Scatter(
            x=dates, y=df["RSI"], name="RSI (14)",
            line=dict(color=COLORS["cyan"], width=1.5),
        ))
        fig2.add_hline(y=30, line_dash="dash", line_color=COLORS["green"], line_width=1)
        fig2.add_hline(y=70, line_dash="dash", line_color=COLORS["red"], line_width=1)
        fig2.add_hrect(y0=0, y1=30, fillcolor=COLORS["green"], opacity=0.08, line_width=0)
        fig2.add_hrect(y0=70, y1=100, fillcolor=COLORS["red"], opacity=0.08, line_width=0)

    fig2.update_layout(yaxis_range=[0, 100], height=300, title="RSI Oscillator", **theme)
    charts["rsi"] = fig2

    # --- Chart 3: Cumulative Returns ---
    fig3 = go.Figure()
    if not strategy_cum.empty:
        fig3.add_trace(go.Scatter(
            x=[str(d)[:10] for d in strategy_cum.index],
            y=strategy_cum.values.tolist(),
            name="Strategy", line=dict(color=COLORS["cyan"], width=2),
        ))
    if not benchmark_cum.empty:
        fig3.add_trace(go.Scatter(
            x=[str(d)[:10] for d in benchmark_cum.index],
            y=benchmark_cum.values.tolist(),
            name="Buy & Hold", line=dict(color=COLORS["amber"], width=2, dash="dot"),
        ))
    fig3.update_layout(height=350, title="Cumulative Returns: Strategy vs Benchmark", **theme)
    charts["returns"] = fig3

    # --- Chart 4: Drawdown ---
    fig4 = go.Figure()
    if not portfolio_values.empty:
        rolling_peak = portfolio_values.cummax()
        drawdown = (portfolio_values - rolling_peak) / rolling_peak * 100
        fig4.add_trace(go.Scatter(
            x=[str(d)[:10] for d in drawdown.index],
            y=drawdown.values.tolist(),
            name="Drawdown", fill="tozeroy",
            fillcolor="rgba(239, 68, 68, 0.3)",
            line=dict(color=COLORS["red"], width=1),
        ))

        max_dd_val = metrics.get("max_drawdown", 0)
        max_dd_date = metrics.get("max_dd_date", None)
        if max_dd_date and max_dd_date != "N/A":
            fig4.add_hline(y=max_dd_val, line_dash="dash",
                           line_color=COLORS["red"], line_width=1)
            fig4.add_trace(go.Scatter(
                x=[max_dd_date], y=[max_dd_val],
                mode="markers+text", name="Max DD",
                marker=dict(size=10, color=COLORS["red"]),
                text=[f"Max DD: {max_dd_val:.1f}%"],
                textposition="bottom center",
                textfont=dict(color=COLORS["red"], size=11),
            ))

    fig4.update_layout(yaxis_title="Drawdown (%)", height=350, title="Drawdown Timeline", **theme)
    charts["drawdown"] = fig4

    return charts


def main():
    st.set_page_config(page_title="Alpha - Quant Analytics", page_icon="📈", layout="wide")
    
    st.title("Alpha - Institutional-Grade Quantitative Analytics")
    st.markdown("Test strategies on any stock, index, or crypto with real market data, realistic friction costs, and institutional risk metrics.")

    # Sidebar inputs
    st.sidebar.header("Market Selection")
    
    # Initialize session state for ticker if not exists
    if "ticker_input" not in st.session_state:
        st.session_state.ticker_input = "^NSEI"
        
    def set_ticker(ticker_symbol):
        st.session_state.ticker_input = ticker_symbol

    # The primary way to enter a ticker
    ticker = st.sidebar.text_input("Search Company or Ticker (Hit Enter)", key="ticker_input")
    
    # Live Search Suggestions
    if ticker:
        from data.pipeline import search_ticker
        search_results = search_ticker(ticker)
        
        # Only show search options if the user hasn't typed an exact matching symbol
        exact_match = any(res["symbol"].upper() == ticker.upper() for res in search_results)
        
        if search_results and not exact_match:
            st.sidebar.caption(f"Search Results for '{ticker}':")
            for res in search_results:
                # Format: AAPL - Apple Inc (NASDAQ)
                label = f"{res['symbol']} — {res['name']} ({res['exchange']})"
                st.sidebar.button(label, key=f"btn_{res['symbol']}", on_click=set_ticker, args=(res["symbol"],), use_container_width=True)
    
    # Quick select options "downside"
    st.sidebar.markdown("---")
    st.sidebar.caption("Or choose a popular preset:")
    
    # Create a compact grid of buttons for presets
    col1, col2, col3 = st.sidebar.columns(3)
    col1.button("NIFTY", use_container_width=True, on_click=set_ticker, args=("^NSEI",))
    col2.button("S&P 500", use_container_width=True, on_click=set_ticker, args=("^GSPC",))
    col3.button("BTC", use_container_width=True, on_click=set_ticker, args=("BTC-USD",))
        
    col4, col5, col6 = st.sidebar.columns(3)
    col4.button("AAPL", use_container_width=True, on_click=set_ticker, args=("AAPL",))
    col5.button("RELIANCE", use_container_width=True, on_click=set_ticker, args=("RELIANCE.NS",))
    col6.button("GOLD", use_container_width=True, on_click=set_ticker, args=("GC=F",))

    st.sidebar.header("Date Range")
    default_end = datetime.today()
    default_start = datetime.today() - timedelta(days=7 * 365)
    
    start_date = st.sidebar.date_input("Start Date", value=default_start)
    end_date = st.sidebar.date_input("End Date", value=default_end)
    
    st.sidebar.header("Strategy Parameters")
    short_window = st.sidebar.number_input("Short MA Window", min_value=5, max_value=100, value=20)
    long_window = st.sidebar.number_input("Long MA Window", min_value=10, max_value=300, value=50)

    if st.sidebar.button("Run Analysis", type="primary"):
        if short_window >= long_window:
            st.sidebar.error("Short MA window must be less than Long MA window.")
            return

        with st.spinner("Fetching data and running backtest..."):
            try:
                # 0. Fetch Live Info
                live_info = fetch_live_info(ticker)
                
                # 1. Fetch data
                df = fetch_market_data(ticker=ticker, start_date=start_date.strftime("%Y-%m-%d"), end_date=end_date.strftime("%Y-%m-%d"))

                # 2. MA signals
                df = compute_ma_signals(df, short_window=short_window, long_window=long_window)

                # 3. RSI signals + confluence
                df = compute_rsi(df)
                df = compute_rsi_signals(df)
                df = apply_confluence_filter(df)

                # 4. Backtest
                bt_results = run_backtest(df, ticker=ticker)

                # 5. Risk analytics
                portfolio_values = bt_results["portfolio_values"]
                daily_returns = compute_daily_returns(portfolio_values)

                is_indian = ticker.endswith((".NS", ".BO")) or ticker.startswith("^NS")
                risk_free = 0.065 if is_indian else 0.043

                sr = sharpe_ratio(daily_returns, risk_free_rate=risk_free)
                max_dd, max_dd_date = maximum_drawdown(portfolio_values)
                ann_return = compute_annualized_return(portfolio_values)
                cr = calmar_ratio(ann_return, max_dd)

                strategy_cum, benchmark_cum = benchmark_comparison(
                    daily_returns, ticker=ticker,
                    start_date=start_date.strftime("%Y-%m-%d"), end_date=end_date.strftime("%Y-%m-%d")
                )

                strategy_return_pct = (strategy_cum.iloc[-1] - 1) * 100 if len(strategy_cum) > 0 else 0
                benchmark_return_pct = (benchmark_cum.iloc[-1] - 1) * 100 if len(benchmark_cum) > 0 else 0
                alpha = strategy_return_pct - benchmark_return_pct

                metrics = {
                    "sharpe_ratio": round(float(sr), 4),
                    "max_drawdown": round(float(max_dd * 100), 2),
                    "max_dd_date": max_dd_date or "N/A",
                    "calmar_ratio": round(float(cr), 4) if cr != float("inf") else "Inf",
                    "total_trades": bt_results["total_trades"],
                    "win_rate": round(float(bt_results["win_rate"]), 2),
                    "strategy_return": round(float(strategy_return_pct), 2),
                    "benchmark_return": round(float(benchmark_return_pct), 2),
                    "alpha": round(float(alpha), 2),
                    "initial_cash": bt_results["initial_cash"],
                    "final_value": round(float(bt_results["final_value"]), 2),
                }

                st.success(f"Analysis Complete! Processed {len(df)} data points and executed {metrics['total_trades']} trades.")

                # Ticker display
                st.markdown("---")
                display_name = live_info.get("name", ticker) if live_info else ticker
                st.header(f"📊 Analysis Results: `{display_name}`")
                
                if live_info:
                    st.caption("🔴 Live Screener Data")
                    l1, l2, l3, l4 = st.columns(4)
                    currency = live_info.get("currency", "")
                    l1.metric("Current Price", f"{live_info.get('price', 'N/A')} {currency}")
                    l2.metric("Market Cap", f"{live_info.get('market_cap', 'N/A')} {currency}")
                    l3.metric("P/E Ratio", str(live_info.get('pe_ratio', 'N/A')))
                    l4.metric("52W High / Low", f"{live_info.get('high_52w', 'N/A')} / {live_info.get('low_52w', 'N/A')}")
                    st.divider()
                
                # Metrics display
                st.subheader("Backtest Performance Metrics")
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Sharpe Ratio", f"{metrics['sharpe_ratio']:.2f}")
                col2.metric("Max Drawdown", f"{metrics['max_drawdown']:.2f}%")
                col3.metric("Win Rate", f"{metrics['win_rate']:.1f}%")
                col4.metric("Alpha", f"{metrics['alpha']:+.2f}%")

                col5, col6, col7, col8 = st.columns(4)
                col5.metric("Strategy Return", f"{metrics['strategy_return']:+.2f}%")
                col6.metric("Benchmark Return", f"{metrics['benchmark_return']:+.2f}%")
                col7.metric("Calmar Ratio", str(metrics['calmar_ratio']))
                col8.metric("Final Portfolio", f"{metrics['final_value']:,.0f}")

                # Build charts
                charts = build_figures(df, strategy_cum, benchmark_cum, portfolio_values, metrics)
                
                # Render charts
                st.plotly_chart(charts["price"], use_container_width=True)
                
                col_chart1, col_chart2 = st.columns(2)
                with col_chart1:
                    st.plotly_chart(charts["rsi"], use_container_width=True)
                with col_chart2:
                    st.plotly_chart(charts["returns"], use_container_width=True)
                
                st.plotly_chart(charts["drawdown"], use_container_width=True)

                # Trade Log
                trade_log = bt_results.get("trade_log", [])
                if trade_log:
                    st.subheader("Trade Log")
                    st.dataframe(pd.DataFrame(trade_log), use_container_width=True)

            except Exception as e:
                logger.error("Analysis failed: %s", str(e), exc_info=True)
                st.error(f"Analysis failed: {str(e)}")

if __name__ == "__main__":
    main()
