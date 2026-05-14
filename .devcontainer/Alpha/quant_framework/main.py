"""
Master Orchestrator — Quant Trading & Risk Analytics Framework
================================================================

This is the entry point for the entire quantitative framework. It wires
together all modules in a strict sequential pipeline:

    1. Parse CLI arguments
    2. Fetch and clean market data
    3. Compute MA Crossover signals
    4. Compute RSI signals + confluence filter
    5. Run backtest with slippage + commissions
    6. Compute risk analytics
    7. Generate risk dashboard
    8. Print summary to terminal

Usage:
    python main.py --ticker ^NSEI
    python main.py --ticker ^GSPC --short-window 10 --long-window 30
    python main.py --ticker RELIANCE.NS --start-date 2018-01-01 --end-date 2024-12-31

Author: Quant Framework
"""

import argparse
import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

# Module imports
from data.pipeline import fetch_market_data
from strategy.ma_crossover import compute_ma_signals
from strategy.rsi_signal import apply_confluence_filter, compute_rsi, compute_rsi_signals
from backtesting.engine import run_backtest
from risk.analytics import (
    benchmark_comparison,
    calmar_ratio,
    compute_annualized_return,
    compute_daily_returns,
    maximum_drawdown,
    sharpe_ratio,
)
from dashboard.visualizer import generate_dashboard

# ---------------------------------------------------------------------------
# Logging configuration
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)-30s | %(levelname)-5s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def parse_arguments() -> argparse.Namespace:
    """
    Parse command-line arguments for the quant framework.

    All parameters have sensible defaults so the framework can be run
    with zero configuration: `python main.py` uses NIFTY 50 with
    standard 20/50 MA windows over the last 7 years.

    Returns
    -------
    argparse.Namespace
        Parsed arguments with ticker, dates, and MA window sizes.
    """
    parser = argparse.ArgumentParser(
        description="Quantitative Trading & Risk Analytics Framework",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --ticker ^NSEI
  python main.py --ticker ^GSPC --short-window 10 --long-window 30
  python main.py --ticker RELIANCE.NS --start-date 2018-01-01
        """,
    )

    # Default date range: 7 years ending today
    default_end = datetime.today().strftime("%Y-%m-%d")
    default_start = (datetime.today() - timedelta(days=7 * 365)).strftime("%Y-%m-%d")

    parser.add_argument(
        "--ticker",
        type=str,
        default="^NSEI",
        help="Yahoo Finance ticker symbol (default: ^NSEI for NIFTY 50)",
    )
    parser.add_argument(
        "--start-date",
        type=str,
        default=default_start,
        help=f"Start date in YYYY-MM-DD format (default: {default_start})",
    )
    parser.add_argument(
        "--end-date",
        type=str,
        default=default_end,
        help=f"End date in YYYY-MM-DD format (default: {default_end})",
    )
    parser.add_argument(
        "--short-window",
        type=int,
        default=20,
        help="Short-term SMA window in days (default: 20)",
    )
    parser.add_argument(
        "--long-window",
        type=int,
        default=50,
        help="Long-term SMA window in days (default: 50)",
    )

    return parser.parse_args()


def main() -> None:
    """
    Execute the full quantitative analysis pipeline.

    This function orchestrates all modules in sequence, ensuring that each
    module receives the correct output from the previous step. The pipeline
    is designed to fail fast with clear error messages if any step encounters
    an issue.
    """

    # ==================================================================
    # Step 1: Parse CLI arguments
    # ==================================================================
    args = parse_arguments()
    logger.info("=" * 70)
    logger.info("QUANT FRAMEWORK - Starting analysis")
    logger.info("=" * 70)
    logger.info(
        "Config: ticker=%s, period=%s to %s, MA=%d/%d",
        args.ticker,
        args.start_date,
        args.end_date,
        args.short_window,
        args.long_window,
    )

    try:
        # ==============================================================
        # Step 2: Fetch and clean market data
        # ==============================================================
        logger.info("[1/7] Fetching market data...")
        df = fetch_market_data(
            ticker=args.ticker,
            start_date=args.start_date,
            end_date=args.end_date,
        )
        logger.info("Data loaded: %d rows", len(df))

        # ==============================================================
        # Step 3: Compute MA Crossover signals
        # ==============================================================
        logger.info("[2/7] Computing MA Crossover signals...")
        df = compute_ma_signals(
            df,
            short_window=args.short_window,
            long_window=args.long_window,
        )

        # ==============================================================
        # Step 4: Compute RSI + confluence filter
        # ==============================================================
        logger.info("[3/7] Computing RSI signals...")
        df = compute_rsi(df)
        df = compute_rsi_signals(df)

        logger.info("[4/7] Applying confluence filter...")
        df = apply_confluence_filter(df)

        # ==============================================================
        # Step 5: Run backtest
        # ==============================================================
        logger.info("[5/7] Running backtest with slippage + commissions...")
        bt_results = run_backtest(df, ticker=args.ticker)

        # ==============================================================
        # Step 6: Compute risk analytics
        # ==============================================================
        logger.info("[6/7] Computing risk metrics...")

        portfolio_values = bt_results["portfolio_values"]
        daily_returns = compute_daily_returns(portfolio_values)

        # Determine risk-free rate based on market
        is_indian = args.ticker.endswith((".NS", ".BO")) or args.ticker.startswith("^NS")
        risk_free = 0.065 if is_indian else 0.043  # Indian G-Sec vs US Treasury

        # Sharpe Ratio
        sr = sharpe_ratio(daily_returns, risk_free_rate=risk_free)

        # Maximum Drawdown
        max_dd, max_dd_date = maximum_drawdown(portfolio_values)

        # Annualized Return
        ann_return = compute_annualized_return(portfolio_values)

        # Calmar Ratio
        cr = calmar_ratio(ann_return, max_dd)

        # Benchmark Comparison
        strategy_cum, benchmark_cum = benchmark_comparison(
            daily_returns,
            ticker=args.ticker,
            start_date=args.start_date,
            end_date=args.end_date,
        )

        # Compute return percentages
        strategy_return_pct = (
            (strategy_cum.iloc[-1] - 1) * 100 if len(strategy_cum) > 0 else 0
        )
        benchmark_return_pct = (
            (benchmark_cum.iloc[-1] - 1) * 100 if len(benchmark_cum) > 0 else 0
        )
        alpha = strategy_return_pct - benchmark_return_pct

        # Assemble metrics dictionary
        metrics = {
            "sharpe_ratio": sr,
            "max_drawdown": max_dd,
            "max_dd_date": max_dd_date,
            "calmar_ratio": cr,
            "total_trades": bt_results["total_trades"],
            "win_rate": bt_results["win_rate"],
            "strategy_return": strategy_return_pct,
            "benchmark_return": benchmark_return_pct,
            "alpha": alpha,
            "final_value": bt_results["final_value"],
            "initial_cash": bt_results["initial_cash"],
        }

        # ==============================================================
        # Step 7: Generate risk dashboard
        # ==============================================================
        logger.info("[7/7] Generating risk dashboard...")

        output_dir = Path(__file__).parent
        dashboard_path = str(output_dir / "risk_dashboard.html")

        generate_dashboard(
            df=df,
            strategy_cum_returns=strategy_cum,
            benchmark_cum_returns=benchmark_cum,
            portfolio_values=portfolio_values,
            metrics=metrics,
            output_path=dashboard_path,
        )

        # ==============================================================
        # Step 8: Print terminal summary using Rich
        # ==============================================================
        _print_rich_summary(args, metrics, bt_results)

        logger.info("=" * 70)
        logger.info("Pipeline complete! Dashboard saved to: %s", dashboard_path)
        logger.info("=" * 70)

    except Exception as e:
        logger.error("Pipeline failed: %s", str(e), exc_info=True)
        sys.exit(1)


def _print_rich_summary(
    args: argparse.Namespace,
    metrics: dict,
    bt_results: dict,
) -> None:
    """
    Print a formatted summary table to the terminal using the Rich library.

    Rich provides beautiful terminal output with colors and formatting,
    making the results immediately readable without opening the HTML dashboard.

    Parameters
    ----------
    args : argparse.Namespace
        CLI arguments for context.
    metrics : dict
        Computed risk metrics.
    bt_results : dict
        Backtest results.
    """
    try:
        from rich.console import Console
        from rich.table import Table
        from rich.panel import Panel
        from rich import box

        console = Console(force_terminal=True)

        # Header
        console.print()
        console.print(
            Panel(
                f"[bold cyan]Quant Framework - Results Summary[/bold cyan]\n"
                f"[dim]Ticker: {args.ticker} | "
                f"Period: {args.start_date} to {args.end_date} | "
                f"MA: {args.short_window}/{args.long_window}[/dim]",
                border_style="cyan",
                padding=(1, 2),
            )
        )

        # Metrics table
        table = Table(
            title="Risk Analytics",
            box=box.ROUNDED,
            border_style="bright_blue",
            header_style="bold cyan",
            show_lines=True,
        )

        table.add_column("Metric", style="white", min_width=25)
        table.add_column("Value", justify="right", min_width=15)

        # Format values with color
        sharpe = metrics["sharpe_ratio"]
        sharpe_color = "green" if sharpe > 1 else ("red" if sharpe < 0 else "yellow")
        table.add_row("Sharpe Ratio", f"[{sharpe_color}]{sharpe:.4f}[/{sharpe_color}]")

        max_dd = metrics["max_drawdown"] * 100
        dd_color = "red" if max_dd < -10 else "yellow"
        table.add_row(
            "Maximum Drawdown",
            f"[{dd_color}]{max_dd:.2f}%[/{dd_color}] ({metrics['max_dd_date']})",
        )

        calmar_val = metrics["calmar_ratio"]
        calmar_color = "green" if calmar_val > 1 else "yellow"
        table.add_row(
            "Calmar Ratio",
            f"[{calmar_color}]{calmar_val:.4f}[/{calmar_color}]",
        )

        table.add_row("Total Trades", f"[cyan]{metrics['total_trades']}[/cyan]")

        wr = metrics["win_rate"]
        wr_color = "green" if wr > 50 else ("red" if wr < 40 else "yellow")
        table.add_row("Win Rate", f"[{wr_color}]{wr:.2f}%[/{wr_color}]")

        sr = metrics["strategy_return"]
        sr_sign = "+" if sr >= 0 else ""
        sr_color = "green" if sr > 0 else "red"
        table.add_row(
            "Strategy Return",
            f"[{sr_color}]{sr_sign}{sr:.2f}%[/{sr_color}]",
        )

        br = metrics["benchmark_return"]
        br_sign = "+" if br >= 0 else ""
        br_color = "green" if br > 0 else "red"
        table.add_row(
            "Benchmark Return",
            f"[{br_color}]{br_sign}{br:.2f}%[/{br_color}]",
        )

        a = metrics["alpha"]
        a_sign = "+" if a >= 0 else ""
        a_color = "green" if a > 0 else "red"
        table.add_row(
            "Alpha (Strategy - Benchmark)",
            f"[{a_color}]{a_sign}{a:.2f}%[/{a_color}]",
        )

        table.add_row(
            "Initial Capital",
            f"[white]{metrics['initial_cash']:,.2f}[/white]",
        )
        table.add_row(
            "Final Portfolio Value",
            f"[cyan]{metrics['final_value']:,.2f}[/cyan]",
        )

        console.print(table)
        console.print()

    except ImportError:
        # Fallback if Rich is not installed
        logger.warning("Rich library not found — printing plain text summary")
        print("\n" + "=" * 50)
        print("RISK ANALYTICS SUMMARY")
        print("=" * 50)
        for key, value in metrics.items():
            print(f"  {key}: {value}")
        print("=" * 50)


if __name__ == "__main__":
    main()
