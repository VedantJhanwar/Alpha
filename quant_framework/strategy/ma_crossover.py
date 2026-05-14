"""
Moving Average Crossover Strategy
====================================

This module implements a dual Moving Average Crossover strategy — one of the
oldest and most widely studied systematic trading signals in quantitative finance.

**Market inefficiency captured:**
    Price momentum and trend-following exploit the empirically observed tendency
    of institutional investors to react slowly to new information. When a stock
    begins trending upward, large funds (mutual funds, pension funds, insurance
    companies) adjust their positions gradually due to:
        1. Internal committee approvals and mandate constraints
        2. Liquidity requirements — large orders must be spread across days
        3. Behavioral anchoring — analysts anchor to old price targets

    The Short SMA captures recent momentum (fast signal), while the Long SMA
    represents the slower structural trend. A Golden Cross (short > long) signals
    that recent momentum has overcome the historical trend, suggesting the start
    of a new uptrend. The reverse (Death Cross) signals the opposite.

    Academic reference: Jegadeesh & Titman (1993) — "Returns to Buying Winners
    and Selling Losers" documents this momentum anomaly.

Author: Quant Framework
"""

import logging
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants — no magic numbers
# ---------------------------------------------------------------------------
DEFAULT_SHORT_WINDOW: int = 20   # 20-day SMA — captures ~1 month of momentum
DEFAULT_LONG_WINDOW: int = 50    # 50-day SMA — captures ~2.5 months of trend


def compute_ma_signals(
    df: pd.DataFrame,
    short_window: int = DEFAULT_SHORT_WINDOW,
    long_window: int = DEFAULT_LONG_WINDOW,
) -> pd.DataFrame:
    """
    Compute dual Moving Average Crossover signals on a cleaned OHLCV DataFrame.

    The strategy generates discrete signals at crossover points:
        - **Golden Cross (+1 BUY):** Short SMA crosses above Long SMA,
          indicating that recent price momentum has turned bullish.
        - **Death Cross (-1 SELL):** Short SMA crosses below Long SMA,
          indicating that momentum has turned bearish.
        - **No crossover (0 HOLD):** Maintain current position.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned DataFrame with at least a 'Close' column.
    short_window : int
        Period for the short-term (fast) Simple Moving Average.
    long_window : int
        Period for the long-term (slow) Simple Moving Average.

    Returns
    -------
    pd.DataFrame
        Original DataFrame augmented with columns:
        - 'SMA_Short': Short-term moving average values
        - 'SMA_Long': Long-term moving average values
        - 'MA_Signal': +1 (BUY), -1 (SELL), or 0 (HOLD)

    Raises
    ------
    ValueError
        If 'Close' column is missing or short_window >= long_window.
    """

    if "Close" not in df.columns:
        raise ValueError("DataFrame must contain a 'Close' column")

    if short_window >= long_window:
        raise ValueError(
            f"short_window ({short_window}) must be less than "
            f"long_window ({long_window})"
        )

    logger.info(
        "Computing MA Crossover signals: short=%d, long=%d",
        short_window,
        long_window,
    )

    result = df.copy()

    # -------------------------------------------------------------------
    # Step 1: Compute Simple Moving Averages
    # -------------------------------------------------------------------
    # SMA = arithmetic mean of the last N closing prices.
    # We use .rolling().mean() which handles edge cases (NaN at start) cleanly.
    result["SMA_Short"] = result["Close"].rolling(window=short_window).mean()
    result["SMA_Long"] = result["Close"].rolling(window=long_window).mean()

    # -------------------------------------------------------------------
    # Step 2: Determine position based on SMA relationship
    # -------------------------------------------------------------------
    # position = 1 when Short SMA > Long SMA (bullish), 0 otherwise
    # We need this intermediate column to detect crossover EVENTS
    # (transitions from 0→1 or 1→0), not just the current state.
    result["_position"] = 0
    result.loc[result["SMA_Short"] > result["SMA_Long"], "_position"] = 1

    # -------------------------------------------------------------------
    # Step 3: Detect crossover events via position changes
    # -------------------------------------------------------------------
    # diff() of position:
    #   +1 = just crossed above (Golden Cross)  → BUY
    #   -1 = just crossed below (Death Cross)   → SELL
    #    0 = no crossover today                  → HOLD
    result["MA_Signal"] = result["_position"].diff()

    # Fill NaN in signal column (first row has no previous value to diff)
    result["MA_Signal"] = result["MA_Signal"].fillna(0).astype(int)

    # Clean up temporary column
    result.drop(columns=["_position"], inplace=True)

    # -------------------------------------------------------------------
    # Logging summary
    # -------------------------------------------------------------------
    buy_count = (result["MA_Signal"] == 1).sum()
    sell_count = (result["MA_Signal"] == -1).sum()
    logger.info(
        "MA Crossover signals generated: %d BUY, %d SELL signals over %d days",
        buy_count,
        sell_count,
        len(result),
    )

    return result
