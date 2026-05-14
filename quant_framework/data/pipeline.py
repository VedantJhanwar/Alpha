"""
Data Pipeline Module — Ingestion and Cleaning
===============================================

This module handles the acquisition and sanitization of historical market data.
In quantitative finance, data quality is the single most critical determinant
of backtest reliability. Garbage-in, garbage-out applies doubly here because
backtesting engines will silently produce misleading results if fed unclean data.

Key design decisions:
    - We use yfinance for free, reproducible access to Yahoo Finance data.
    - Every cleaning step is logged so that an analyst can audit the data pipeline.
    - The 7-year lookback window provides ~1,750 trading days — sufficient for
      statistical significance across multiple market regimes (bull, bear, sideways).

Author: Quant Framework
"""

import logging
from datetime import datetime, timedelta
from typing import Optional

import pandas as pd
import yfinance as yf

# Configure module-level logger
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DEFAULT_TICKER: str = "^NSEI"          # NIFTY 50 — India's benchmark index
FALLBACK_TICKER: str = "^GSPC"         # S&P 500 — US benchmark fallback
LOOKBACK_YEARS: int = 7               # 7 years captures multiple market cycles
REQUIRED_COLUMNS: list[str] = ["Open", "High", "Low", "Close", "Volume"]


def fetch_market_data(
    ticker: str = DEFAULT_TICKER,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> pd.DataFrame:
    """
    Download and clean historical OHLCV data for a given ticker.

    Why each cleaning step matters for backtesting integrity:

    1. **Drop NaN Close prices**: Close is the anchor for all signal calculations
       (SMA, RSI, returns). A missing Close corrupts every downstream computation.
       Unlike Open/High/Low, Close cannot be reasonably interpolated because it
       represents the market's consensus valuation at day's end.

    2. **Forward-fill Open/High/Low**: Minor gaps in intraday columns often arise
       from exchange data-feed issues. Forward-filling is the least biased imputation
       because it assumes "no new information" — the previous day's value persists
       until a genuine update arrives.

    3. **Drop zero-volume rows**: Zero volume indicates non-trading days (holidays,
       exchange closures) that somehow leaked into the dataset. Including them would
       create phantom holding periods and distort return calculations.

    4. **Reset and rename index**: Backtrader and Plotly expect a clean 'Date' column
       with no multi-level index artifacts from yfinance.

    Parameters
    ----------
    ticker : str
        Yahoo Finance ticker symbol (e.g., "^NSEI", "RELIANCE.NS", "^GSPC").
    start_date : str, optional
        Start date in "YYYY-MM-DD" format. Defaults to 7 years ago.
    end_date : str, optional
        End date in "YYYY-MM-DD" format. Defaults to today.

    Returns
    -------
    pd.DataFrame
        Cleaned DataFrame with columns: Date, Open, High, Low, Close, Volume.

    Raises
    ------
    ValueError
        If no data is returned for both the primary and fallback tickers.
    """

    # Compute date range
    if end_date is None:
        end_date = datetime.today().strftime("%Y-%m-%d")
    if start_date is None:
        start_dt = datetime.today() - timedelta(days=LOOKBACK_YEARS * 365)
        start_date = start_dt.strftime("%Y-%m-%d")

    logger.info(
        "Fetching data for ticker=%s from %s to %s", ticker, start_date, end_date
    )

    # -------------------------------------------------------------------
    # Step 1: Download data from Yahoo Finance
    # -------------------------------------------------------------------
    df = _download_with_fallback(ticker, start_date, end_date)

    logger.info("Raw data shape: %s rows × %s columns", df.shape[0], df.shape[1])

    # -------------------------------------------------------------------
    # Step 2: Retain only OHLCV columns
    # -------------------------------------------------------------------
    # yfinance may return extra columns like 'Adj Close'; we discard them
    # to keep the data schema predictable for downstream consumers.
    available_cols = [c for c in REQUIRED_COLUMNS if c in df.columns]
    df = df[available_cols].copy()
    logger.info("Retained columns: %s", available_cols)

    # -------------------------------------------------------------------
    # Step 3: Drop rows where Close is NaN
    # -------------------------------------------------------------------
    rows_before = len(df)
    df.dropna(subset=["Close"], inplace=True)
    rows_dropped = rows_before - len(df)
    logger.info("Dropped %d rows with NaN Close values", rows_dropped)

    # -------------------------------------------------------------------
    # Step 4: Forward-fill missing Open, High, Low values
    # -------------------------------------------------------------------
    fill_cols = ["Open", "High", "Low"]
    for col in fill_cols:
        if col in df.columns:
            nan_count = df[col].isna().sum()
            if nan_count > 0:
                df[col] = df[col].ffill()
                logger.info("Forward-filled %d NaN values in '%s'", nan_count, col)

    # -------------------------------------------------------------------
    # Step 5: Drop rows with zero Volume (non-trading days)
    # -------------------------------------------------------------------
    if "Volume" in df.columns:
        rows_before = len(df)
        df = df[df["Volume"] > 0].copy()
        rows_dropped = rows_before - len(df)
        logger.info("Dropped %d rows with zero Volume", rows_dropped)

    # -------------------------------------------------------------------
    # Step 6: Reset index and ensure clean Date column
    # -------------------------------------------------------------------
    df = df.reset_index()

    # yfinance uses 'Date' or 'Datetime' as the index name; normalize it
    if "Datetime" in df.columns:
        df.rename(columns={"Datetime": "Date"}, inplace=True)

    # Ensure Date column exists and is datetime type
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"])
    else:
        # If no date column exists after reset, something is wrong
        logger.warning("No 'Date' column found after reset_index — check data source")

    logger.info(
        "Clean data ready: %d rows from %s to %s",
        len(df),
        df["Date"].iloc[0].strftime("%Y-%m-%d") if "Date" in df.columns else "N/A",
        df["Date"].iloc[-1].strftime("%Y-%m-%d") if "Date" in df.columns else "N/A",
    )

    return df


def _download_with_fallback(
    ticker: str, start_date: str, end_date: str
) -> pd.DataFrame:
    """
    Attempt to download data for the given ticker; fall back to S&P 500
    if the primary ticker returns empty data (common with ^NSEI outside
    Indian market hours or during Yahoo Finance API issues).

    Parameters
    ----------
    ticker : str
        Primary ticker symbol.
    start_date : str
        Start date string.
    end_date : str
        End date string.

    Returns
    -------
    pd.DataFrame
        Raw OHLCV data from Yahoo Finance.

    Raises
    ------
    ValueError
        If both primary and fallback tickers return empty data.
    """
    try:
        df = yf.download(ticker, start=start_date, end=end_date, progress=False)

        # Handle MultiIndex columns that yfinance sometimes returns
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        if df.empty:
            raise ValueError(f"No data returned for ticker '{ticker}'")

        logger.info("Successfully downloaded data for %s", ticker)
        return df

    except (ValueError, Exception) as e:
        logger.warning(
            "Failed to fetch '%s': %s — falling back to '%s'",
            ticker,
            str(e),
            FALLBACK_TICKER,
        )

        try:
            df = yf.download(
                FALLBACK_TICKER, start=start_date, end=end_date, progress=False
            )
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)

            if df.empty:
                raise ValueError(
                    f"Fallback ticker '{FALLBACK_TICKER}' also returned no data"
                )

            logger.info("Successfully downloaded fallback data for %s", FALLBACK_TICKER)
            return df

        except Exception as fallback_error:
            error_msg = (
                f"Data download failed for both '{ticker}' and "
                f"'{FALLBACK_TICKER}': {fallback_error}"
            )
            logger.error(error_msg)
            raise ValueError(error_msg) from fallback_error
