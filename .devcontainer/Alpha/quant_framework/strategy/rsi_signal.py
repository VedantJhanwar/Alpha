"""
RSI (Relative Strength Index) Signal Layer
============================================

This module implements the RSI indicator using J. Welles Wilder's original
smoothing method and combines it with the MA Crossover signals to create a
confluence-based trading filter.

**Why RSI reduces false signals:**
    Moving Average Crossovers are inherently lagging indicators — they detect
    trends AFTER they've begun, which means they can trigger false signals in
    choppy, range-bound (sideways) markets. RSI is a momentum oscillator that
    measures the speed and magnitude of recent price changes, providing an
    independent assessment of whether a security is overbought or oversold.

    By requiring BOTH indicators to agree (confluence), we filter out scenarios
    where:
    - A Golden Cross fires during a temporary bounce in a downtrend (RSI > 70
      would reject this as overbought)
    - A Death Cross fires during a temporary pullback in an uptrend (RSI < 30
      would flag this as oversold, rejecting the sell)

**Why confluence filtering improves signal-to-noise ratio:**
    Each indicator captures a different dimension of price behavior:
    - MA Crossover captures TREND (directional bias over weeks/months)
    - RSI captures MOMENTUM (speed of recent price changes over days)

    Requiring agreement between two independent dimensions dramatically reduces
    the probability of false positives. If each indicator has a 30% false signal
    rate independently, the combined false signal rate drops to ~9% (0.3 × 0.3),
    assuming reasonable independence between the two signal sources.

    This is analogous to the concept of "orthogonal signals" in information theory
    — combining uncorrelated information sources improves prediction accuracy
    more than doubling down on a single source.

Author: Quant Framework
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants — all thresholds are named and documented
# ---------------------------------------------------------------------------
RSI_PERIOD: int = 14          # Wilder's original 14-period lookback
RSI_OVERSOLD: float = 30.0   # Below 30 → oversold → potential BUY
RSI_OVERBOUGHT: float = 70.0  # Above 70 → overbought → potential SELL


def compute_rsi(
    df: pd.DataFrame,
    period: int = RSI_PERIOD,
) -> pd.DataFrame:
    """
    Compute the Relative Strength Index using Wilder's smoothing method.

    Wilder's smoothing (also called exponential moving average with alpha=1/period)
    differs from a standard EMA because it uses a smoothing factor of 1/N rather
    than 2/(N+1). This gives more weight to older observations, making the RSI
    less reactive to single-day spikes — which is desirable for reducing noise.

    The manual implementation avoids ta-lib dependency, ensuring:
    1. Full transparency in the calculation (no black-box library)
    2. Reproducibility across platforms without compiled C dependencies
    3. Easier debugging when RSI values seem anomalous

    Formula:
        RSI = 100 - (100 / (1 + RS))
        RS  = Average Gain / Average Loss  (over `period` days)

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with a 'Close' column.
    period : int
        Lookback period for RSI calculation (default: 14).

    Returns
    -------
    pd.DataFrame
        DataFrame augmented with 'RSI' column.
    """

    if "Close" not in df.columns:
        raise ValueError("DataFrame must contain a 'Close' column")

    logger.info("Computing RSI with period=%d", period)

    result = df.copy()

    # -------------------------------------------------------------------
    # Step 1: Compute daily price changes (deltas)
    # -------------------------------------------------------------------
    delta = result["Close"].diff()

    # -------------------------------------------------------------------
    # Step 2: Separate gains and losses
    # -------------------------------------------------------------------
    # Gains = positive deltas, losses = absolute value of negative deltas
    # We clip to isolate each: gains have losses set to 0 and vice versa
    gains = delta.clip(lower=0)           # Keep only positive changes
    losses = (-delta).clip(lower=0)       # Absolute value of negative changes

    # -------------------------------------------------------------------
    # Step 3: Compute initial average gain/loss using simple mean
    # -------------------------------------------------------------------
    # The first `period` values use a simple average as the seed
    # for the Wilder smoothing recursion
    avg_gain = gains.copy()
    avg_loss = losses.copy()

    # First average: simple mean of the first `period` gains/losses
    avg_gain.iloc[:period] = np.nan
    avg_loss.iloc[:period] = np.nan

    # Seed value: simple mean of first `period` observations
    first_avg_gain = gains.iloc[1:period + 1].mean()
    first_avg_loss = losses.iloc[1:period + 1].mean()

    avg_gain.iloc[period] = first_avg_gain
    avg_loss.iloc[period] = first_avg_loss

    # -------------------------------------------------------------------
    # Step 4: Apply Wilder's smoothing recursion
    # -------------------------------------------------------------------
    # Wilder's smoothing: avg = (prev_avg * (period - 1) + current) / period
    # This is equivalent to EMA with alpha = 1/period
    for i in range(period + 1, len(avg_gain)):
        avg_gain.iloc[i] = (
            avg_gain.iloc[i - 1] * (period - 1) + gains.iloc[i]
        ) / period
        avg_loss.iloc[i] = (
            avg_loss.iloc[i - 1] * (period - 1) + losses.iloc[i]
        ) / period

    # -------------------------------------------------------------------
    # Step 5: Compute RS and RSI
    # -------------------------------------------------------------------
    # RS = Relative Strength = Average Gain / Average Loss
    rs = avg_gain / avg_loss

    # RSI = 100 - (100 / (1 + RS))
    # When avg_loss = 0 (all gains), RS → ∞, RSI → 100
    # When avg_gain = 0 (all losses), RS = 0, RSI = 0
    result["RSI"] = 100 - (100 / (1 + rs))

    logger.info(
        "RSI computed: min=%.1f, max=%.1f, mean=%.1f",
        result["RSI"].min(),
        result["RSI"].max(),
        result["RSI"].mean(),
    )

    return result


def compute_rsi_signals(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Generate RSI-based trading signals using overbought/oversold thresholds.

    Signal logic:
        - RSI < 30  → OVERSOLD  → BUY signal  (+1)
        - RSI > 70  → OVERBOUGHT → SELL signal (-1)
        - 30 ≤ RSI ≤ 70 → NEUTRAL → HOLD       (0)

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with an 'RSI' column (output of compute_rsi).

    Returns
    -------
    pd.DataFrame
        DataFrame augmented with 'RSI_Signal' column.
    """

    if "RSI" not in df.columns:
        raise ValueError("DataFrame must contain an 'RSI' column — run compute_rsi first")

    result = df.copy()

    # Vectorized signal assignment
    result["RSI_Signal"] = 0  # Default: HOLD (neutral)
    result.loc[result["RSI"] < RSI_OVERSOLD, "RSI_Signal"] = 1     # Oversold → BUY
    result.loc[result["RSI"] > RSI_OVERBOUGHT, "RSI_Signal"] = -1  # Overbought → SELL

    buy_count = (result["RSI_Signal"] == 1).sum()
    sell_count = (result["RSI_Signal"] == -1).sum()
    logger.info(
        "RSI signals: %d BUY (oversold), %d SELL (overbought)", buy_count, sell_count
    )

    return result


def apply_confluence_filter(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Apply confluence filtering: a trade is confirmed ONLY when both the
    MA Crossover and RSI signals agree on direction.

    Practical confluence logic:
        The MA Crossover is the primary directional trigger (trend signal).
        RSI acts as a confirmation filter to block trades where momentum
        contradicts the trend:

        - Confirmed BUY  (+1):  MA_Signal == +1  AND  RSI < 70
          (Golden Cross fires AND market is NOT overbought — room to run)
        - Confirmed SELL (-1):  MA_Signal == -1  AND  RSI > 30
          (Death Cross fires AND market is NOT oversold — room to fall)
        - Otherwise       (0):  Signals disagree → no action

    Why this works better than requiring both extremes on the same day:
        MA crossovers are rare events (~14 per 7 years). RSI extremes
        (<30 or >70) occur on different days. Requiring BOTH on the exact
        same day produces near-zero trades. Instead, we use RSI as a
        directional filter: it blocks a BUY if the market is already
        overbought (unlikely to continue rising) and blocks a SELL if the
        market is already oversold (unlikely to continue falling).

        This mirrors how institutional desks actually deploy multi-indicator
        systems: one indicator triggers, another confirms.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with 'MA_Signal', 'RSI_Signal', and 'RSI' columns.

    Returns
    -------
    pd.DataFrame
        DataFrame augmented with 'Confirmed_Signal' column.
    """

    if "MA_Signal" not in df.columns:
        raise ValueError("Missing 'MA_Signal' — run compute_ma_signals first")
    if "RSI" not in df.columns:
        raise ValueError("Missing 'RSI' — run compute_rsi first")

    result = df.copy()

    # Confluence: MA crossover is the trigger, RSI filters out false signals
    result["Confirmed_Signal"] = 0

    # BUY confluence: Golden Cross fires AND RSI confirms market is NOT overbought
    # If RSI > 70, the market is already extended — buying here risks a pullback
    buy_mask = (result["MA_Signal"] == 1) & (result["RSI"] < RSI_OVERBOUGHT)
    result.loc[buy_mask, "Confirmed_Signal"] = 1

    # SELL confluence: Death Cross fires AND RSI confirms market is NOT oversold
    # If RSI < 30, the market is already depressed — selling here risks missing a bounce
    sell_mask = (result["MA_Signal"] == -1) & (result["RSI"] > RSI_OVERSOLD)
    result.loc[sell_mask, "Confirmed_Signal"] = -1

    confirmed_buys = buy_mask.sum()
    confirmed_sells = sell_mask.sum()
    logger.info(
        "Confluence filter: %d confirmed BUYs, %d confirmed SELLs "
        "(filtered from MA + RSI)",
        confirmed_buys,
        confirmed_sells,
    )

    return result
