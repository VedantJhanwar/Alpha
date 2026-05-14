"""
Dark-Themed Risk Dashboard — Plotly Visualization
====================================================

This module generates an institutional-quality risk dashboard as a standalone
HTML file. The dark theme is chosen deliberately — it reduces eye strain during
extended analysis sessions and is the industry standard for trading desks
(Bloomberg Terminal, Refinitiv Eikon, and most prop-trading platforms use
dark backgrounds).

The dashboard presents four critical views:
    1. Price action with MA crossover signals
    2. RSI oscillator with overbought/oversold zones
    3. Cumulative returns vs benchmark (alpha visualization)
    4. Drawdown timeline (tail risk visualization)

Plus a summary metrics table for at-a-glance performance assessment.

Author: Quant Framework
"""

import logging
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Global dark theme configuration
# ---------------------------------------------------------------------------
DARK_THEME = dict(
    template="plotly_dark",
    paper_bgcolor="#0a0e1a",      # Near-black background
    plot_bgcolor="#0f1525",        # Slightly lighter plot area
    font=dict(
        family="Inter, system-ui",
        color="#e2e8f0",           # Soft white text
    ),
    margin=dict(l=40, r=40, t=50, b=40),
)

# Color palette — carefully chosen for contrast on dark backgrounds
COLORS = {
    "cyan": "#22d3ee",            # Short SMA, strategy returns
    "purple": "#a78bfa",          # Long SMA
    "green": "#10b981",           # BUY signals, oversold zone
    "red": "#ef4444",             # SELL signals, overbought zone, drawdown
    "amber": "#f59e0b",           # Benchmark returns
    "white": "#e2e8f0",           # Text
    "muted": "#64748b",           # Grid lines, secondary text
    "bg_card": "#1e293b",         # Card backgrounds for metrics table
}

# Default output path
DEFAULT_OUTPUT_FILE = "risk_dashboard.html"


def generate_dashboard(
    df: pd.DataFrame,
    strategy_cum_returns: pd.Series,
    benchmark_cum_returns: pd.Series,
    portfolio_values: pd.Series,
    metrics: dict[str, Any],
    output_path: Optional[str] = None,
) -> str:
    """
    Generate a comprehensive dark-themed risk dashboard as a standalone HTML file.

    The dashboard contains 4 interactive Plotly charts arranged in a 2×2 grid,
    plus a styled metrics summary table below. The HTML file is fully self-contained
    (Plotly.js is embedded inline) so it can be shared via email or Slack without
    any server infrastructure.

    Parameters
    ----------
    df : pd.DataFrame
        Full signal DataFrame with columns: Date, Open, High, Low, Close,
        SMA_Short, SMA_Long, RSI, Confirmed_Signal.
    strategy_cum_returns : pd.Series
        Cumulative strategy returns (starting from 1.0).
    benchmark_cum_returns : pd.Series
        Cumulative benchmark returns (starting from 1.0).
    portfolio_values : pd.Series
        Daily portfolio values for drawdown calculation.
    metrics : dict
        Dictionary containing: sharpe_ratio, max_drawdown, max_dd_date,
        calmar_ratio, total_trades, win_rate, strategy_return,
        benchmark_return, alpha.
    output_path : str, optional
        File path for the HTML output. Defaults to "risk_dashboard.html".

    Returns
    -------
    str
        Absolute path to the generated HTML file.
    """
    if output_path is None:
        output_path = DEFAULT_OUTPUT_FILE

    logger.info("Generating dark-themed risk dashboard...")

    # Create 2×2 subplot grid with descriptive titles
    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=(
            "📈 Price Action & MA Crossover Signals",
            "📊 RSI Oscillator",
            "💰 Cumulative Returns: Strategy vs Benchmark",
            "📉 Drawdown Timeline",
        ),
        vertical_spacing=0.12,
        horizontal_spacing=0.08,
    )

    # ===================================================================
    # CHART 1 — Price + MA Crossover (top-left)
    # ===================================================================
    _add_price_chart(fig, df, row=1, col=1)

    # ===================================================================
    # CHART 2 — RSI Oscillator (top-right)
    # ===================================================================
    _add_rsi_chart(fig, df, row=1, col=2)

    # ===================================================================
    # CHART 3 — Cumulative Returns Comparison (bottom-left)
    # ===================================================================
    _add_returns_chart(
        fig, strategy_cum_returns, benchmark_cum_returns, metrics, row=2, col=1
    )

    # ===================================================================
    # CHART 4 — Drawdown Timeline (bottom-right)
    # ===================================================================
    _add_drawdown_chart(fig, portfolio_values, metrics, row=2, col=2)

    # ===================================================================
    # Apply global dark theme
    # ===================================================================
    fig.update_layout(
        height=1000,
        showlegend=True,
        legend=dict(
            bgcolor="rgba(15, 21, 37, 0.8)",
            bordercolor=COLORS["muted"],
            borderwidth=1,
            font=dict(size=11),
        ),
        **DARK_THEME,
    )

    # Style all axes
    fig.update_xaxes(gridcolor="rgba(100, 116, 139, 0.2)", zeroline=False)
    fig.update_yaxes(gridcolor="rgba(100, 116, 139, 0.2)", zeroline=False)

    # ===================================================================
    # Build full HTML with metrics table
    # ===================================================================
    html_content = _build_full_html(fig, metrics)

    # Save to file
    output_file = Path(output_path).resolve()
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(html_content)

    logger.info("Dashboard saved to: %s", output_file)
    return str(output_file)


def _add_price_chart(
    fig: go.Figure,
    df: pd.DataFrame,
    row: int,
    col: int,
) -> None:
    """
    Add candlestick chart with MA overlays and buy/sell signal markers.

    We use candlesticks instead of line charts because they reveal intraday
    price behavior (open-high-low-close) that is invisible in line charts.
    This is critical for understanding whether signals fire near support/
    resistance levels.
    """
    dates = df["Date"]

    # Candlestick chart for OHLC
    fig.add_trace(
        go.Candlestick(
            x=dates,
            open=df["Open"],
            high=df["High"],
            low=df["Low"],
            close=df["Close"],
            name="OHLC",
            increasing_line_color=COLORS["green"],
            decreasing_line_color=COLORS["red"],
            showlegend=False,
        ),
        row=row,
        col=col,
    )

    # Short SMA overlay (cyan)
    if "SMA_Short" in df.columns:
        fig.add_trace(
            go.Scatter(
                x=dates,
                y=df["SMA_Short"],
                name="SMA Short (20)",
                line=dict(color=COLORS["cyan"], width=1.5),
                opacity=0.9,
            ),
            row=row,
            col=col,
        )

    # Long SMA overlay (purple)
    if "SMA_Long" in df.columns:
        fig.add_trace(
            go.Scatter(
                x=dates,
                y=df["SMA_Long"],
                name="SMA Long (50)",
                line=dict(color=COLORS["purple"], width=1.5),
                opacity=0.9,
            ),
            row=row,
            col=col,
        )

    # BUY signals — green upward triangles
    if "Confirmed_Signal" in df.columns:
        buy_signals = df[df["Confirmed_Signal"] == 1]
        if len(buy_signals) > 0:
            fig.add_trace(
                go.Scatter(
                    x=buy_signals["Date"],
                    y=buy_signals["Low"] * 0.98,  # Slightly below the low
                    mode="markers",
                    name="BUY Signal",
                    marker=dict(
                        symbol="triangle-up",
                        size=12,
                        color=COLORS["green"],
                        line=dict(width=1, color="white"),
                    ),
                ),
                row=row,
                col=col,
            )

        # SELL signals — red downward triangles
        sell_signals = df[df["Confirmed_Signal"] == -1]
        if len(sell_signals) > 0:
            fig.add_trace(
                go.Scatter(
                    x=sell_signals["Date"],
                    y=sell_signals["High"] * 1.02,  # Slightly above the high
                    mode="markers",
                    name="SELL Signal",
                    marker=dict(
                        symbol="triangle-down",
                        size=12,
                        color=COLORS["red"],
                        line=dict(width=1, color="white"),
                    ),
                ),
                row=row,
                col=col,
            )

    # Disable rangeslider (clutters the chart)
    fig.update_xaxes(rangeslider_visible=False, row=row, col=col)


def _add_rsi_chart(
    fig: go.Figure,
    df: pd.DataFrame,
    row: int,
    col: int,
) -> None:
    """
    Add RSI oscillator chart with overbought/oversold zones.

    The shaded zones provide instant visual feedback on market sentiment:
    - Green zone (RSI < 30): Market is oversold → mean-reversion opportunity
    - Red zone (RSI > 70): Market is overbought → reversal risk
    """
    dates = df["Date"]

    if "RSI" not in df.columns:
        logger.warning("RSI column missing — skipping RSI chart")
        return

    # RSI line
    fig.add_trace(
        go.Scatter(
            x=dates,
            y=df["RSI"],
            name="RSI (14)",
            line=dict(color=COLORS["cyan"], width=1.5),
        ),
        row=row,
        col=col,
    )

    # Oversold threshold line (RSI = 30)
    fig.add_hline(
        y=30,
        line_dash="dash",
        line_color=COLORS["green"],
        line_width=1,
        annotation_text="Oversold (30)",
        annotation_font_color=COLORS["green"],
        row=row,
        col=col,
    )

    # Overbought threshold line (RSI = 70)
    fig.add_hline(
        y=70,
        line_dash="dash",
        line_color=COLORS["red"],
        line_width=1,
        annotation_text="Overbought (70)",
        annotation_font_color=COLORS["red"],
        row=row,
        col=col,
    )

    # Shade oversold zone (RSI < 30) — translucent green
    fig.add_hrect(
        y0=0,
        y1=30,
        fillcolor=COLORS["green"],
        opacity=0.1,
        line_width=0,
        row=row,
        col=col,
    )

    # Shade overbought zone (RSI > 70) — translucent red
    fig.add_hrect(
        y0=70,
        y1=100,
        fillcolor=COLORS["red"],
        opacity=0.1,
        line_width=0,
        row=row,
        col=col,
    )

    # Set y-axis range for RSI (always 0-100)
    fig.update_yaxes(range=[0, 100], row=row, col=col)


def _add_returns_chart(
    fig: go.Figure,
    strategy_cum: pd.Series,
    benchmark_cum: pd.Series,
    metrics: dict[str, Any],
    row: int,
    col: int,
) -> None:
    """
    Add cumulative returns comparison chart: Strategy vs Buy-and-Hold.

    This is the ultimate test of active management — does the strategy
    outperform simple buy-and-hold after accounting for all costs?
    """
    # Strategy cumulative returns (cyan)
    if not strategy_cum.empty:
        fig.add_trace(
            go.Scatter(
                x=strategy_cum.index,
                y=strategy_cum.values,
                name="Strategy Returns",
                line=dict(color=COLORS["cyan"], width=2),
            ),
            row=row,
            col=col,
        )

    # Benchmark cumulative returns (amber)
    if not benchmark_cum.empty:
        fig.add_trace(
            go.Scatter(
                x=benchmark_cum.index,
                y=benchmark_cum.values,
                name="Buy & Hold",
                line=dict(color=COLORS["amber"], width=2, dash="dot"),
            ),
            row=row,
            col=col,
        )

    # Add alpha annotation
    alpha = metrics.get("alpha", 0)
    alpha_sign = "+" if alpha >= 0 else ""
    alpha_color = COLORS["green"] if alpha >= 0 else COLORS["red"]

    # Annotation: alpha value and slippage note
    fig.add_annotation(
        text=(
            f"<b>Alpha: {alpha_sign}{alpha:.2f}%</b><br>"
            f"<span style='font-size:10px; color:{COLORS['muted']}'>Slippage + Commissions included</span>"
        ),
        xref="x3 domain",
        yref="y3 domain",
        x=0.05,
        y=0.95,
        xanchor="left",
        yanchor="top",
        showarrow=False,
        font=dict(size=13, color=alpha_color),
        bgcolor="rgba(15, 21, 37, 0.8)",
        bordercolor=alpha_color,
        borderwidth=1,
        borderpad=6,
    )


def _add_drawdown_chart(
    fig: go.Figure,
    portfolio_values: pd.Series,
    metrics: dict[str, Any],
    row: int,
    col: int,
) -> None:
    """
    Add drawdown timeline as a filled area chart.

    Drawdown visualization is critical for risk management because it shows
    not just the magnitude but the DURATION of losses. A 15% drawdown lasting
    3 months is very different from one lasting 18 months.
    """
    if portfolio_values.empty:
        logger.warning("Empty portfolio values — skipping drawdown chart")
        return

    # Compute rolling drawdown
    rolling_peak = portfolio_values.cummax()
    drawdown = (portfolio_values - rolling_peak) / rolling_peak * 100  # As percentage

    # Drawdown area (filled red below zero)
    fig.add_trace(
        go.Scatter(
            x=drawdown.index,
            y=drawdown.values,
            name="Drawdown",
            fill="tozeroy",
            fillcolor="rgba(239, 68, 68, 0.3)",  # Translucent red
            line=dict(color=COLORS["red"], width=1),
        ),
        row=row,
        col=col,
    )

    # Max Drawdown marker
    max_dd = metrics.get("max_drawdown", 0)
    max_dd_date = metrics.get("max_dd_date", None)

    if max_dd_date is not None:
        # Dashed line at max drawdown level
        fig.add_hline(
            y=max_dd * 100,
            line_dash="dash",
            line_color=COLORS["red"],
            line_width=1,
            row=row,
            col=col,
        )

        # Red dot at max drawdown point
        try:
            fig.add_trace(
                go.Scatter(
                    x=[pd.Timestamp(max_dd_date)],
                    y=[max_dd * 100],
                    mode="markers+text",
                    name="Max Drawdown",
                    marker=dict(size=10, color=COLORS["red"], symbol="circle"),
                    text=[f"Max DD: {max_dd * 100:.1f}%"],
                    textposition="bottom center",
                    textfont=dict(color=COLORS["red"], size=11),
                ),
                row=row,
                col=col,
            )
        except Exception:
            pass  # Date parsing may fail; chart still renders

    fig.update_yaxes(title_text="Drawdown (%)", row=row, col=col)


def _build_full_html(
    fig: go.Figure,
    metrics: dict[str, Any],
) -> str:
    """
    Assemble the complete HTML document with charts and metrics table.

    The HTML is fully self-contained — Plotly.js is embedded inline so the
    file works offline and can be shared as an email attachment.
    """
    # Convert Plotly figure to HTML div (not full page — we'll wrap it)
    chart_html = fig.to_html(
        full_html=False,
        include_plotlyjs="cdn",
        config={"displayModeBar": True, "responsive": True},
    )

    # Build metrics table
    metrics_rows = _build_metrics_table_rows(metrics)

    html = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Quant Framework — Risk Analytics Dashboard</title>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}

        body {{
            background-color: #0a0e1a;
            color: #e2e8f0;
            font-family: 'Inter', system-ui, -apple-system, sans-serif;
            line-height: 1.6;
        }}

        .dashboard-container {{
            max-width: 1400px;
            margin: 0 auto;
            padding: 24px;
        }}

        .dashboard-header {{
            text-align: center;
            padding: 32px 0;
            border-bottom: 1px solid rgba(100, 116, 139, 0.3);
            margin-bottom: 24px;
        }}

        .dashboard-header h1 {{
            font-size: 28px;
            font-weight: 700;
            background: linear-gradient(135deg, #22d3ee, #a78bfa);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            margin-bottom: 8px;
        }}

        .dashboard-header p {{
            color: #64748b;
            font-size: 14px;
            font-weight: 400;
        }}

        .chart-container {{
            background: #0f1525;
            border-radius: 12px;
            border: 1px solid rgba(100, 116, 139, 0.2);
            padding: 16px;
            margin-bottom: 32px;
        }}

        .metrics-section {{
            margin-top: 40px;
        }}

        .metrics-section h2 {{
            font-size: 20px;
            font-weight: 600;
            color: #e2e8f0;
            margin-bottom: 20px;
            padding-left: 12px;
            border-left: 3px solid #22d3ee;
        }}

        .metrics-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
            gap: 16px;
        }}

        .metric-card {{
            background: #1e293b;
            border-radius: 10px;
            padding: 20px 24px;
            border: 1px solid rgba(100, 116, 139, 0.2);
            transition: transform 0.2s ease, box-shadow 0.2s ease;
        }}

        .metric-card:hover {{
            transform: translateY(-2px);
            box-shadow: 0 8px 25px rgba(0, 0, 0, 0.3);
        }}

        .metric-label {{
            font-size: 12px;
            text-transform: uppercase;
            letter-spacing: 1px;
            color: #64748b;
            margin-bottom: 6px;
            font-weight: 500;
        }}

        .metric-value {{
            font-size: 24px;
            font-weight: 700;
        }}

        .metric-value.positive {{
            color: #10b981;
        }}

        .metric-value.negative {{
            color: #ef4444;
        }}

        .metric-value.neutral {{
            color: #22d3ee;
        }}

        .footer {{
            text-align: center;
            padding: 32px 0;
            margin-top: 40px;
            border-top: 1px solid rgba(100, 116, 139, 0.2);
            color: #64748b;
            font-size: 12px;
        }}

        /* Make Plotly chart responsive */
        .js-plotly-plot {{
            width: 100% !important;
        }}
    </style>
</head>
<body>
    <div class="dashboard-container">
        <div class="dashboard-header">
            <h1>⚡ Quantitative Risk Analytics Dashboard</h1>
            <p>Institutional-Grade Performance Analysis • Slippage & Commissions Included</p>
        </div>

        <div class="chart-container">
            {chart_html}
        </div>

        <div class="metrics-section">
            <h2>Performance Metrics Summary</h2>
            <div class="metrics-grid">
                {metrics_rows}
            </div>
        </div>

        <div class="footer">
            <p>Generated by Quant Framework • Data sourced from Yahoo Finance</p>
            <p style="margin-top: 4px;">⚠️ Past performance does not guarantee future results. This is a research tool, not investment advice.</p>
        </div>
    </div>
</body>
</html>
"""
    return html


def _build_metrics_table_rows(metrics: dict[str, Any]) -> str:
    """Build HTML cards for each metric."""

    def _card(label: str, value: str, css_class: str = "neutral") -> str:
        return f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value {css_class}">{value}</div>
        </div>
        """

    cards = []

    # Sharpe Ratio
    sharpe = metrics.get("sharpe_ratio", 0)
    sharpe_class = "positive" if sharpe > 1 else ("negative" if sharpe < 0 else "neutral")
    cards.append(_card("Sharpe Ratio", f"{sharpe:.2f}", sharpe_class))

    # Maximum Drawdown
    max_dd = metrics.get("max_drawdown", 0) * 100
    dd_class = "negative" if max_dd < -10 else "neutral"
    cards.append(_card("Maximum Drawdown", f"{max_dd:.2f}%", dd_class))

    # Calmar Ratio
    calmar = metrics.get("calmar_ratio", 0)
    calmar_class = "positive" if calmar > 1 else "neutral"
    cards.append(_card("Calmar Ratio", f"{calmar:.2f}", calmar_class))

    # Total Trades
    trades = metrics.get("total_trades", 0)
    cards.append(_card("Total Trades", str(trades), "neutral"))

    # Win Rate
    win_rate = metrics.get("win_rate", 0)
    wr_class = "positive" if win_rate > 50 else ("negative" if win_rate < 40 else "neutral")
    cards.append(_card("Win Rate", f"{win_rate:.2f}%", wr_class))

    # Strategy Return
    strat_ret = metrics.get("strategy_return", 0)
    sr_class = "positive" if strat_ret > 0 else "negative"
    sign = "+" if strat_ret >= 0 else ""
    cards.append(_card("Strategy Return", f"{sign}{strat_ret:.2f}%", sr_class))

    # Benchmark Return
    bench_ret = metrics.get("benchmark_return", 0)
    br_class = "positive" if bench_ret > 0 else "negative"
    sign = "+" if bench_ret >= 0 else ""
    cards.append(_card("Benchmark Return", f"{sign}{bench_ret:.2f}%", br_class))

    # Alpha
    alpha = metrics.get("alpha", 0)
    a_class = "positive" if alpha > 0 else "negative"
    sign = "+" if alpha >= 0 else ""
    cards.append(_card("Alpha (Strategy − Benchmark)", f"{sign}{alpha:.2f}%", a_class))

    return "\n".join(cards)
