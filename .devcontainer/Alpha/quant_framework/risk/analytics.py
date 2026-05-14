"""
Risk Analytics Module
=======================

This module computes institutional-grade risk metrics that quantify the
quality of a trading strategy beyond simple return percentages.

In professional fund management, raw returns are meaningless without
risk-adjustment. A 20% return means nothing if it came with 50% drawdown
risk — a passive index fund would have been a better choice.

Metrics implemented:
    1. **Sharpe Ratio** — Risk-adjusted return per unit of volatility
    2. **Maximum Drawdown** — Worst peak-to-trough decline (tail risk)
    3. **Calmar Ratio** — Return per unit of drawdown risk
    4. **Benchmark Comparison** — Alpha generation vs passive investing

Author: Quant Framework
"""

import logging
from datetime import datetime
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
TRADING_DAYS_PER_YEAR: int = 252        # Standard assumption for annualization
INDIA_RISK_FREE_RATE: float = 0.065     # 10-year Indian G-Sec yield (~6.5%)
US_RISK_FREE_RATE: float = 0.043        # 10-year US Treasury yield (~4.3%)


def sharpe_ratio(
    returns: pd.Series,
    risk_free_rate: float = INDIA_RISK_FREE_RATE,
) -> float:
    """
    Compute the annualized Sharpe Ratio of a return series.

    The Sharpe Ratio measures excess return per unit of total risk (volatility).
    It answers the question: "Is the strategy's return worth the volatility
    the investor must endure?"

    Formula:
        Sharpe = (Mean Daily Return - Daily Risk-Free Rate) / Std Dev of Daily Returns × √252

    Interpretation:
        - Sharpe > 1.0 is acceptable for most institutional mandates
        - Sharpe > 2.0 is institutional-grade (top-quartile hedge funds)
        - Sharpe > 3.0 is exceptional and warrants scrutiny for overfitting
        - Sharpe < 0 means the strategy underperforms risk-free instruments

    Parameters
    ----------
    returns : pd.Series
        Daily returns series (not cumulative).
    risk_free_rate : float
        Annualized risk-free rate (default: 6.5% for Indian G-Sec).

    Returns
    -------
    float
        Annualized Sharpe Ratio.
    """
    if returns.empty or returns.std() == 0:
        logger.warning("Cannot compute Sharpe: empty or zero-variance returns")
        return 0.0

    # Convert annual risk-free rate to daily
    daily_rf = risk_free_rate / TRADING_DAYS_PER_YEAR

    # Excess return = strategy return - risk-free return
    excess_returns = returns - daily_rf

    # Annualize: multiply by √252 to convert daily to annual scale
    sharpe = (excess_returns.mean() / excess_returns.std()) * np.sqrt(TRADING_DAYS_PER_YEAR)

    # Sharpe > 1.0 is acceptable; > 2.0 is institutional-grade
    logger.info("Sharpe Ratio: %.4f (risk-free rate: %.2f%%)", sharpe, risk_free_rate * 100)

    return round(float(sharpe), 4)


def maximum_drawdown(
    portfolio_values: pd.Series,
) -> Tuple[float, Optional[str]]:
    """
    Compute the Maximum Drawdown — the largest peak-to-trough decline
    in portfolio value over the entire period.

    Maximum Drawdown is the single most important tail-risk metric because
    it tells you: "What is the worst loss an investor would have experienced
    if they invested at the worst possible time?"

    A strategy with 50% max drawdown means an investor could have lost half
    their capital before recovering — psychologically devastating and a
    career-ending event for most fund managers.

    Formula:
        Drawdown(t) = (Peak(t) - Value(t)) / Peak(t)
        Max Drawdown = max(Drawdown(t)) for all t

    Parameters
    ----------
    portfolio_values : pd.Series
        Time series of portfolio values (not returns).

    Returns
    -------
    tuple[float, str | None]
        (max_drawdown_percentage, date_of_max_drawdown)
        Drawdown is returned as a negative percentage (e.g., -0.15 = -15%).
    """
    if portfolio_values.empty:
        logger.warning("Cannot compute Max Drawdown: empty portfolio series")
        return 0.0, None

    # Rolling peak: the highest portfolio value seen up to each point
    rolling_peak = portfolio_values.cummax()

    # Drawdown at each timestep: how far below the peak
    drawdown = (portfolio_values - rolling_peak) / rolling_peak

    # Maximum drawdown: the deepest trough
    max_dd = drawdown.min()

    # Date of maximum drawdown
    max_dd_idx = drawdown.idxmin()
    max_dd_date = str(max_dd_idx)[:10] if max_dd_idx is not None else None

    logger.info(
        "Maximum Drawdown: %.2f%% on %s",
        max_dd * 100,
        max_dd_date,
    )

    return round(float(max_dd), 4), max_dd_date


def benchmark_comparison(
    strategy_returns: pd.Series,
    ticker: str,
    start_date: str,
    end_date: str,
) -> Tuple[pd.Series, pd.Series]:
    """
    Compare strategy cumulative returns against a Buy-and-Hold benchmark.

    Alpha = Strategy Return - Benchmark Return.
    If Alpha is negative, a passive index fund beats our strategy.
    This is the most honest test of any active management approach —
    after all costs and complexity, did the strategy actually ADD value
    beyond what a simple buy-and-hold ETF would have delivered?

    Parameters
    ----------
    strategy_returns : pd.Series
        Daily strategy returns.
    ticker : str
        Benchmark ticker symbol.
    start_date : str
        Start date for benchmark data.
    end_date : str
        End date for benchmark data.

    Returns
    -------
    tuple[pd.Series, pd.Series]
        (strategy_cumulative_returns, benchmark_cumulative_returns)
        Both series are cumulative returns starting from 1.0.
    """
    logger.info(
        "Downloading benchmark data for %s (%s to %s)", ticker, start_date, end_date
    )

    try:
        bench_df = yf.download(ticker, start=start_date, end=end_date, progress=False)

        if isinstance(bench_df.columns, pd.MultiIndex):
            bench_df.columns = bench_df.columns.get_level_values(0)

        if bench_df.empty:
            logger.warning("Benchmark data empty — returning strategy-only comparison")
            strategy_cum = (1 + strategy_returns).cumprod()
            return strategy_cum, pd.Series(dtype=float)

        # Compute benchmark daily returns
        bench_returns = bench_df["Close"].pct_change().dropna()

        # Align dates: use intersection of strategy and benchmark dates
        common_dates = strategy_returns.index.intersection(bench_returns.index)

        if len(common_dates) == 0:
            # Dates might not align perfectly; use all available
            strategy_cum = (1 + strategy_returns).cumprod()
            benchmark_cum = (1 + bench_returns).cumprod()
        else:
            strategy_cum = (1 + strategy_returns.loc[common_dates]).cumprod()
            benchmark_cum = (1 + bench_returns.loc[common_dates]).cumprod()

        # Alpha = Strategy Return - Benchmark Return
        strategy_total = strategy_cum.iloc[-1] - 1 if len(strategy_cum) > 0 else 0
        benchmark_total = benchmark_cum.iloc[-1] - 1 if len(benchmark_cum) > 0 else 0
        alpha = strategy_total - benchmark_total

        # Alpha = Strategy Return - Benchmark Return.
        # If Alpha is negative, a passive index fund beats our strategy.
        logger.info(
            "Strategy Return: %.2f%%, Benchmark Return: %.2f%%, Alpha: %.2f%%",
            strategy_total * 100,
            benchmark_total * 100,
            alpha * 100,
        )

        return strategy_cum, benchmark_cum

    except Exception as e:
        logger.error("Benchmark comparison failed: %s", str(e))
        strategy_cum = (1 + strategy_returns).cumprod()
        return strategy_cum, pd.Series(dtype=float)


def calmar_ratio(
    annualized_return: float,
    max_drawdown: float,
) -> float:
    """
    Compute the Calmar Ratio — a metric favored by hedge funds to evaluate
    fund manager skill.

    Formula:
        Calmar = Annualized Return / |Max Drawdown|

    Interpretation:
        - Calmar > 1.0 means the strategy earns more per unit of drawdown risk —
          a metric hedge funds use to evaluate fund managers.
        - Calmar > 3.0 is exceptional (top-decile performance).
        - Calmar < 0.5 suggests the strategy takes excessive drawdown risk
          relative to its returns.

    Parameters
    ----------
    annualized_return : float
        Annualized return of the strategy (e.g., 0.15 for 15%).
    max_drawdown : float
        Maximum drawdown as a negative decimal (e.g., -0.20 for -20%).

    Returns
    -------
    float
        Calmar Ratio.
    """
    if abs(max_drawdown) < 1e-10:
        logger.warning("Max drawdown is near zero — Calmar ratio undefined")
        return float("inf")

    # Calmar = Annualized Return / |Max Drawdown|
    ratio = annualized_return / abs(max_drawdown)

    # Calmar > 1.0 means the strategy earns more per unit of
    # drawdown risk — a metric hedge funds use to evaluate fund managers.
    logger.info(
        "Calmar Ratio: %.4f (Return: %.2f%%, Max DD: %.2f%%)",
        ratio,
        annualized_return * 100,
        max_drawdown * 100,
    )

    return round(float(ratio), 4)


def compute_daily_returns(portfolio_values: pd.Series) -> pd.Series:
    """
    Compute daily percentage returns from a portfolio value series.

    Parameters
    ----------
    portfolio_values : pd.Series
        Time series of daily portfolio values.

    Returns
    -------
    pd.Series
        Daily percentage returns.
    """
    returns = portfolio_values.pct_change().dropna()
    # Replace infinite values (from division by zero) with 0
    returns = returns.replace([np.inf, -np.inf], 0.0)
    return returns


def compute_annualized_return(
    portfolio_values: pd.Series,
) -> float:
    """
    Compute the annualized return from a portfolio value series using
    the compound annual growth rate (CAGR) formula.

    Formula:
        CAGR = (Final Value / Initial Value)^(252 / N) - 1

    Parameters
    ----------
    portfolio_values : pd.Series
        Time series of daily portfolio values.

    Returns
    -------
    float
        Annualized return as a decimal.
    """
    if len(portfolio_values) < 2:
        return 0.0

    initial = portfolio_values.iloc[0]
    final = portfolio_values.iloc[-1]
    n_days = len(portfolio_values)

    if initial <= 0:
        return 0.0

    # CAGR formula: (Final/Initial)^(252/N) - 1
    cagr = (final / initial) ** (TRADING_DAYS_PER_YEAR / n_days) - 1

    return round(float(cagr), 4)
