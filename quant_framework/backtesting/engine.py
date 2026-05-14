"""
Backtesting Engine — Backtrader Integration
=============================================

This module provides a production-grade backtesting engine using the Backtrader
library. It simulates historical trading performance with realistic friction
costs that separate academic backtests from real-world deployable strategies.

Key realism features:
    1. **Slippage** (0.1%): Simulates the price difference between when a trade
       signal is generated and when the order is actually filled. In real markets,
       market orders execute at the next available price, which may differ from
       the signal price — especially in illiquid securities.

    2. **Commission** (0.1%): Simulates brokerage fees, Securities Transaction
       Tax (STT) for Indian markets, or SEC fees for US markets. These costs
       compound across many trades and can erode returns significantly in
       high-frequency strategies.

    These friction costs separate academic backtests from real-world deployable
    strategies. A strategy that looks profitable without them may be unprofitable
    in practice. This is why institutional quant desks ALWAYS backtest with
    realistic transaction costs.

Author: Quant Framework
"""

import logging
from datetime import datetime
from typing import Any, Optional

import backtrader as bt
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants — realistic market friction parameters
# ---------------------------------------------------------------------------
SLIPPAGE_PERCENT: float = 0.001       # 0.1% slippage per trade
COMMISSION_PERCENT: float = 0.001     # 0.1% commission per trade
DEFAULT_CASH_INR: float = 1_000_000   # ₹10,00,000 starting capital (Indian markets)
DEFAULT_CASH_USD: float = 100_000     # $100,000 starting capital (US markets)


class ConfluenceStrategy(bt.Strategy):
    """
    Custom Backtrader strategy that executes trades based on pre-computed
    confluence signals (MA Crossover + RSI agreement).

    Rather than computing indicators inside Backtrader (which would duplicate
    our modular pipeline), we pass in pre-computed signals via a custom data
    feed. This keeps the strategy logic testable and decoupled from the
    backtesting engine.

    Trade logging captures every entry and exit for post-hoc analysis.
    """

    params = (
        ("printlog", True),  # Whether to log individual trades
    )

    def __init__(self) -> None:
        """Initialize strategy and trade tracking."""
        self.order: Optional[bt.Order] = None
        self.trade_log: list[dict[str, Any]] = []
        self.entry_date: Optional[datetime] = None
        self.entry_price: float = 0.0

    def log(self, message: str) -> None:
        """Log a message with the current bar's date."""
        dt = self.datas[0].datetime.date(0)
        logger.info("[%s] %s", dt.isoformat(), message)

    def notify_order(self, order: bt.Order) -> None:
        """Handle order state transitions (submitted → completed/cancelled)."""
        if order.status in [order.Submitted, order.Accepted]:
            return  # Order pending, nothing to do

        if order.status in [order.Completed]:
            if order.isbuy():
                self.entry_date = self.datas[0].datetime.date(0)
                self.entry_price = order.executed.price
                self.log(
                    f"BUY EXECUTED: Price={order.executed.price:.2f}, "
                    f"Size={order.executed.size:.0f}, "
                    f"Cost={order.executed.value:.2f}, "
                    f"Commission={order.executed.comm:.2f}"
                )
            elif order.issell():
                exit_date = self.datas[0].datetime.date(0)
                exit_price = order.executed.price
                pnl = (exit_price - self.entry_price) * abs(order.executed.size)

                # Record the complete trade lifecycle
                self.trade_log.append({
                    "entry_date": str(self.entry_date),
                    "exit_date": str(exit_date),
                    "quantity": abs(order.executed.size),
                    "entry_price": round(self.entry_price, 2),
                    "exit_price": round(exit_price, 2),
                    "pnl": round(pnl, 2),
                    "commission": round(order.executed.comm, 2),
                })

                self.log(
                    f"SELL EXECUTED: Price={exit_price:.2f}, "
                    f"P&L={pnl:.2f}, "
                    f"Commission={order.executed.comm:.2f}"
                )

        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            self.log(f"Order CANCELLED/MARGIN/REJECTED: Status={order.status}")

        self.order = None  # Reset order tracking

    def next(self) -> None:
        """
        Called on every bar. Execute trades based on the pre-computed
        Confirmed_Signal line in the data feed.

        Signal interpretation:
            +1 → BUY (if not already in position)
            -1 → SELL (if currently in position)
             0 → HOLD (do nothing)
        """
        if self.order:
            return  # An order is pending — wait for it to resolve

        signal = self.datas[0].signal[0]

        if not self.position:
            # Not in a position — look for BUY signals
            if signal > 0.5:  # Signal == +1 (using > 0.5 to handle float comparison)
                # Size: invest all available cash
                size = int(self.broker.getcash() / self.datas[0].close[0])
                if size > 0:
                    self.order = self.buy(size=size)
                    self.log(f"BUY ORDER CREATED: Size={size}")
        else:
            # In a position — look for SELL signals
            if signal < -0.5:  # Signal == -1
                self.order = self.sell(size=self.position.size)
                self.log(f"SELL ORDER CREATED: Size={self.position.size}")


class SignalData(bt.feeds.PandasData):
    """
    Custom Backtrader data feed that includes our pre-computed signal column.

    Backtrader's default PandasData only reads OHLCV columns. By extending it,
    we pass our confluence signal directly into the strategy without recomputing
    indicators inside the engine.
    """
    lines = ("signal",)  # Add custom line for our confluence signal

    params = (
        ("signal", -1),  # Column index or name; -1 means auto-detect
    )


def run_backtest(
    df: pd.DataFrame,
    ticker: str = "^NSEI",
    initial_cash: Optional[float] = None,
) -> dict[str, Any]:
    """
    Execute a backtest using Backtrader with realistic market friction.

    These friction costs separate academic backtests from real-world deployable
    strategies. Without them, backtest results are systematically overstated —
    a phenomenon known as "backtest overfitting" in the quant literature.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with columns: Date, Open, High, Low, Close, Volume,
        and Confirmed_Signal (from confluence filter).
    ticker : str
        Ticker symbol (used to determine starting capital currency).
    initial_cash : float, optional
        Starting capital. Defaults to ₹10,00,000 for Indian tickers
        or $100,000 for US tickers.

    Returns
    -------
    dict
        Results dictionary containing:
        - 'final_value': Final portfolio value after all trades
        - 'total_trades': Number of completed round-trip trades
        - 'win_rate': Percentage of profitable trades
        - 'trade_log': List of dicts with trade details
        - 'initial_cash': Starting capital used
        - 'portfolio_values': Daily portfolio value series for drawdown analysis
    """

    if "Confirmed_Signal" not in df.columns:
        raise ValueError(
            "DataFrame must contain 'Confirmed_Signal' column — "
            "run apply_confluence_filter first"
        )

    # -------------------------------------------------------------------
    # Step 1: Determine starting capital based on market
    # -------------------------------------------------------------------
    if initial_cash is None:
        # Indian market tickers end with .NS, .BO, or are ^NSEI / ^NSEBANK
        is_indian = ticker.endswith((".NS", ".BO")) or ticker.startswith("^NS")
        initial_cash = DEFAULT_CASH_INR if is_indian else DEFAULT_CASH_USD

    logger.info(
        "Initializing backtest: ticker=%s, capital=%.2f", ticker, initial_cash
    )

    # -------------------------------------------------------------------
    # Step 2: Prepare data for Backtrader
    # -------------------------------------------------------------------
    bt_df = df[["Date", "Open", "High", "Low", "Close", "Volume", "Confirmed_Signal"]].copy()
    bt_df = bt_df.rename(columns={"Confirmed_Signal": "signal"})
    bt_df["Date"] = pd.to_datetime(bt_df["Date"])
    bt_df = bt_df.set_index("Date")

    # Ensure no NaN values in OHLCV (Backtrader will crash otherwise)
    bt_df = bt_df.dropna(subset=["Open", "High", "Low", "Close"])

    data = SignalData(dataname=bt_df)

    # -------------------------------------------------------------------
    # Step 3: Configure Backtrader engine with realism layers
    # -------------------------------------------------------------------
    cerebro = bt.Cerebro()
    cerebro.adddata(data)
    cerebro.addstrategy(ConfluenceStrategy)

    # Starting capital
    cerebro.broker.setcash(initial_cash)

    # CRITICAL: Slippage — simulates price lag in real execution
    # 0.1% slippage means if signal triggers at ₹1000, execution price
    # may be ₹1001 (buy) or ₹999 (sell), reflecting market impact
    cerebro.broker.set_slippage_perc(
        perc=SLIPPAGE_PERCENT,  # 0.1% price slippage per trade
        slip_open=True,          # Apply slippage on open orders too
        slip_limit=True,         # Apply slippage on limit orders
        slip_match=True,         # Match with volume limitations
        slip_out=False,          # Don't apply on exit (already included in spread)
    )

    # CRITICAL: Commission — simulates brokerage fees + STT/SEC fees
    # 0.1% per trade is a conservative estimate that includes:
    # - Indian markets: brokerage (0.03%) + STT (0.025%) + GST + SEBI charges
    # - US markets: brokerage (varies) + SEC fees + FINRA TAF
    cerebro.broker.setcommission(commission=COMMISSION_PERCENT)

    # -------------------------------------------------------------------
    # Step 4: Run the backtest
    # -------------------------------------------------------------------
    logger.info("Running backtest...")

    # Add a portfolio value observer to track daily values
    cerebro.addobserver(bt.observers.Value)

    results = cerebro.run()
    strategy = results[0]

    # -------------------------------------------------------------------
    # Step 5: Extract results
    # -------------------------------------------------------------------
    final_value = cerebro.broker.getvalue()
    trade_log = strategy.trade_log
    total_trades = len(trade_log)

    # Compute win rate
    if total_trades > 0:
        winning_trades = sum(1 for t in trade_log if t["pnl"] > 0)
        win_rate = (winning_trades / total_trades) * 100
    else:
        win_rate = 0.0

    # Extract daily portfolio values for drawdown analysis
    # We reconstruct from trade log and data
    portfolio_values = _compute_portfolio_values(bt_df, trade_log, initial_cash)

    results_dict = {
        "final_value": round(final_value, 2),
        "total_trades": total_trades,
        "win_rate": round(win_rate, 2),
        "trade_log": trade_log,
        "initial_cash": initial_cash,
        "portfolio_values": portfolio_values,
    }

    logger.info(
        "Backtest complete: Final Value=%.2f, Trades=%d, Win Rate=%.1f%%",
        final_value,
        total_trades,
        win_rate,
    )

    return results_dict


def _compute_portfolio_values(
    df: pd.DataFrame,
    trade_log: list[dict[str, Any]],
    initial_cash: float,
) -> pd.Series:
    """
    Reconstruct daily portfolio values from trade log data.

    This approximation tracks cash and position value day-by-day,
    accounting for trade entries and exits.

    Parameters
    ----------
    df : pd.DataFrame
        Price data indexed by date.
    trade_log : list[dict]
        List of completed trades with entry/exit details.
    initial_cash : float
        Starting capital.

    Returns
    -------
    pd.Series
        Daily portfolio values indexed by date.
    """
    dates = df.index
    portfolio = pd.Series(initial_cash, index=dates, dtype=float)

    cash = initial_cash
    position_size = 0
    entry_price = 0.0

    # Build a lookup of trade events by date
    entries = {}
    exits = {}
    for trade in trade_log:
        entries[trade["entry_date"]] = trade
        exits[trade["exit_date"]] = trade

    for date in dates:
        date_str = str(date.date()) if hasattr(date, "date") else str(date)[:10]

        if date_str in entries:
            trade = entries[date_str]
            entry_price = trade["entry_price"]
            position_size = trade["quantity"]
            cash -= entry_price * position_size

        if date_str in exits:
            trade = exits[date_str]
            exit_price = trade["exit_price"]
            cash += exit_price * position_size
            position_size = 0

        # Portfolio value = cash + mark-to-market value of open position
        current_price = df.loc[date, "Close"]
        portfolio[date] = cash + (position_size * current_price)

    return portfolio
