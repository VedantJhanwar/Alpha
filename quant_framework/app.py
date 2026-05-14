"""
Interactive Quant Framework Web Application
=============================================

Flask-based web app that lets users customize every parameter:
- Ticker symbol (any Yahoo Finance ticker)
- Date range
- MA short/long windows
- RSI period and thresholds

Results render live in the browser with interactive Plotly charts.

Usage:
    python app.py
    Then open http://localhost:5000 in your browser

Author: Quant Framework
"""

import json
import logging
import traceback
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import plotly
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from flask import Flask, render_template, request, jsonify

# Module imports
from data.pipeline import fetch_market_data
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
# App setup
# ---------------------------------------------------------------------------
app = Flask(__name__, template_folder="templates")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)-30s | %(levelname)-5s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Color palette
# ---------------------------------------------------------------------------
COLORS = {
    "cyan": "#22d3ee",
    "purple": "#a78bfa",
    "green": "#10b981",
    "red": "#ef4444",
    "amber": "#f59e0b",
    "muted": "#64748b",
}


@app.route("/")
def index():
    """Serve the interactive dashboard page."""
    default_end = datetime.today().strftime("%Y-%m-%d")
    default_start = (datetime.today() - timedelta(days=7 * 365)).strftime("%Y-%m-%d")
    return render_template(
        "index.html",
        default_start=default_start,
        default_end=default_end,
    )


@app.route("/analyze", methods=["POST"])
def analyze():
    """
    Run the full analysis pipeline with user-specified parameters.
    Returns JSON with chart data and metrics.
    """
    try:
        # Parse form data
        data = request.get_json()
        ticker = data.get("ticker", "^NSEI")
        start_date = data.get("start_date", "")
        end_date = data.get("end_date", "")
        short_window = int(data.get("short_window", 20))
        long_window = int(data.get("long_window", 50))

        logger.info(
            "Analysis request: ticker=%s, period=%s to %s, MA=%d/%d",
            ticker, start_date, end_date, short_window, long_window,
        )

        # ---- Pipeline ----
        # 1. Fetch data
        df = fetch_market_data(ticker=ticker, start_date=start_date, end_date=end_date)

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
            start_date=start_date, end_date=end_date,
        )

        strategy_return_pct = (
            (strategy_cum.iloc[-1] - 1) * 100 if len(strategy_cum) > 0 else 0
        )
        benchmark_return_pct = (
            (benchmark_cum.iloc[-1] - 1) * 100 if len(benchmark_cum) > 0 else 0
        )
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

        # 6. Build charts
        charts = _build_charts(
            df, strategy_cum, benchmark_cum, portfolio_values, metrics
        )

        # 7. Trade log
        trade_log = bt_results.get("trade_log", [])

        return jsonify({
            "success": True,
            "metrics": metrics,
            "charts": charts,
            "trade_log": trade_log,
            "data_points": len(df),
        })

    except Exception as e:
        logger.error("Analysis failed: %s", str(e), exc_info=True)
        return jsonify({
            "success": False,
            "error": str(e),
            "traceback": traceback.format_exc(),
        }), 400


def _build_charts(
    df: pd.DataFrame,
    strategy_cum: pd.Series,
    benchmark_cum: pd.Series,
    portfolio_values: pd.Series,
    metrics: dict,
) -> dict:
    """Build all 4 Plotly charts and return as JSON."""

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

    fig1.update_layout(xaxis_rangeslider_visible=False, height=420, **theme)
    charts["price"] = json.loads(fig1.to_json())

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

    fig2.update_layout(yaxis_range=[0, 100], height=300, **theme)
    charts["rsi"] = json.loads(fig2.to_json())

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
    fig3.update_layout(height=350, **theme)
    charts["returns"] = json.loads(fig3.to_json())

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

    fig4.update_layout(yaxis_title="Drawdown (%)", height=350, **theme)
    charts["drawdown"] = json.loads(fig4.to_json())

    return charts


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("  QUANT FRAMEWORK - Interactive Dashboard")
    print("  Open http://localhost:5000 in your browser")
    print("=" * 60 + "\n")
    app.run(debug=True, port=5000)
