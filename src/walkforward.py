"""Walk-Forward Validation Framework.

Provides tools for robust strategy validation using walk-forward analysis
(purged K-fold cross-validation) instead of single train/test splits.
"""

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Callable, List, Optional

import numpy as np
import polars as pl

from src.backtest import BacktestResult, run_backtest
from src.logging_config import get_logger

logger = get_logger("walkforward")


@dataclass
class WalkForwardWindow:
    """A single walk-forward train/test window."""
    train_start: datetime
    train_end: datetime
    test_start: datetime
    test_end: datetime
    train_bars: pl.DataFrame
    test_bars: pl.DataFrame
    index: int
    
    def __post_init__(self):
        self.duration_train = (self.train_end - self.train_start).days
        self.duration_test = (self.test_end - self.test_start).days


@dataclass
class WalkForwardResult:
    """Results from a walk-forward validation run."""
    windows: List[BacktestResult] = field(default_factory=list)
    metrics_per_window: List[dict] = field(default_factory=list)
    aggregated_metrics: dict = field(default_factory=dict)
    parameter_stability: dict = field(default_factory=dict)
    
    def compute_aggregated_metrics(self) -> dict:
        """Compute aggregated statistics across all windows."""
        if not self.windows:
            return {}
        
        returns = [w.total_return_pct for w in self.windows]
        sharpes = [w.sharpe for w in self.windows if w.sharpe != 0]
        max_dds = [w.max_drawdown_pct for w in self.windows]
        win_rates = [w.win_rate for w in self.windows if w.n_trades > 0]
        n_trades = [w.n_trades for w in self.windows]
        
        metrics = {
            "n_windows": len(self.windows),
            "total_return_pct": {
                "mean": float(np.mean(returns)),
                "std": float(np.std(returns)),
                "median": float(np.median(returns)),
                "min": float(np.min(returns)),
                "max": float(np.max(returns)),
            },
            "sharpe": {
                "mean": float(np.mean(sharpes)) if sharpes else 0.0,
                "std": float(np.std(sharpes)) if sharpes else 0.0,
                "median": float(np.median(sharpes)) if sharpes else 0.0,
            },
            "max_drawdown_pct": {
                "mean": float(np.mean(max_dds)),
                "std": float(np.std(max_dds)),
                "worst": float(np.min(max_dds)),  # Most negative
            },
            "win_rate_pct": {
                "mean": float(np.mean(win_rates)) if win_rates else 0.0,
                "std": float(np.std(win_rates)) if win_rates else 0.0,
            },
            "n_trades_per_window": {
                "mean": float(np.mean(n_trades)),
                "total": int(np.sum(n_trades)),
            },
            "consistency": {
                "pct_positive_windows": float(np.sum(np.array(returns) > 0) / len(returns) * 100),
                "pct_windows_sharpe_gt_1": float(np.sum(np.array(sharpes) > 1.0) / len(sharpes) * 100) if sharpes else 0.0,
                "pct_windows_dd_lt_10": float(np.sum(np.array(max_dds) > -10.0) / len(max_dds) * 100),
            }
        }
        
        self.aggregated_metrics = metrics
        return metrics


class WalkForwardValidator:
    """
    Walk-forward validator using purged K-fold cross-validation.
    
    Key features:
    - Purging: Gap between train and test to prevent data leakage
    - Embargo: Additional gap after test to prevent forward-looking bias
    - Anchored vs sliding: Anchored keeps train start fixed, sliding moves both
    """
    
    def __init__(
        self,
        n_splits: int = 5,
        purge_pct: float = 0.1,
        embargo_pct: float = 0.01,
        anchored: bool = True,
        min_train_days: int = 60,
        min_test_days: int = 14,
    ):
        """
        Args:
            n_splits: Number of walk-forward windows
            purge_pct: Fraction of test window to purge from train (prevents leakage)
            embargo_pct: Fraction of test window to embargo after test
            anchored: If True, train start is fixed; if False, both train and test slide
            min_train_days: Minimum days in training window
            min_test_days: Minimum days in test window
        """
        self.n_splits = n_splits
        self.purge_pct = purge_pct
        self.embargo_pct = embargo_pct
        self.anchored = anchored
        self.min_train_days = min_train_days
        self.min_test_days = min_test_days
    
    def create_windows(
        self,
        bars: pl.DataFrame,
        date_col: str = "t",
    ) -> List[WalkForwardWindow]:
        """
        Create walk-forward windows from bar data.
        
        Args:
            bars: DataFrame with timestamp column
            date_col: Name of timestamp column
            
        Returns:
            List of WalkForwardWindow objects
        """
        # Parse timestamps
        timestamps = bars[date_col].to_list()
        if isinstance(timestamps[0], str):
            dates = [datetime.fromisoformat(ts.replace('Z', '+00:00')) for ts in timestamps]
        else:
            dates = timestamps
        
        start_date = dates[0]
        end_date = dates[-1]
        total_days = (end_date - start_date).days
        
        if total_days < self.min_train_days + self.min_test_days:
            raise ValueError(f"Insufficient data: {total_days} days < min required {self.min_train_days + self.min_test_days}")
        
        # Calculate window sizes
        test_window_days = max(self.min_test_days, total_days // (self.n_splits + 1))
        purge_days = int(test_window_days * self.purge_pct)
        embargo_days = int(test_window_days * self.embargo_pct)
        
        windows = []
        
        if self.anchored:
            # Anchored: train start fixed, train end expands
            train_start = start_date
            for i in range(self.n_splits):
                test_start = start_date + timedelta(days=self.min_train_days + i * test_window_days)
                test_end = test_start + timedelta(days=test_window_days)
                train_end = test_start - timedelta(days=purge_days)
                
                if test_end > end_date:
                    test_end = end_date
                    test_start = test_end - timedelta(days=test_window_days)
                
                # Apply embargo
                embargo_end = test_end + timedelta(days=embargo_days)
                
                train_mask = (pl.col(date_col) >= train_start) & (pl.col(date_col) < train_end)
                test_mask = (pl.col(date_col) >= test_start) & (pl.col(date_col) < test_end)
                
                train_bars = bars.filter(train_mask)
                test_bars = bars.filter(test_mask)
                
                if len(train_bars) >= self.min_train_days and len(test_bars) >= self.min_test_days:
                    windows.append(WalkForwardWindow(
                        train_start=train_start,
                        train_end=train_end,
                        test_start=test_start,
                        test_end=test_end,
                        train_bars=train_bars,
                        test_bars=test_bars,
                        index=i,
                    ))
        else:
            # Sliding: both train and test windows slide forward
            window_total = test_window_days + self.min_train_days
            for i in range(self.n_splits):
                window_start = start_date + timedelta(days=i * test_window_days)
                train_start = window_start
                train_end = train_start + timedelta(days=self.min_train_days) - timedelta(days=purge_days)
                test_start = train_end + timedelta(days=purge_days)
                test_end = test_start + timedelta(days=test_window_days)
                
                if test_end > end_date:
                    break
                
                train_mask = (pl.col(date_col) >= train_start) & (pl.col(date_col) < train_end)
                test_mask = (pl.col(date_col) >= test_start) & (pl.col(date_col) < test_end)
                
                train_bars = bars.filter(train_mask)
                test_bars = bars.filter(test_mask)
                
                if len(train_bars) >= self.min_train_days and len(test_bars) >= self.min_test_days:
                    windows.append(WalkForwardWindow(
                        train_start=train_start,
                        train_end=train_end,
                        test_start=test_start,
                        test_end=test_end,
                        train_bars=train_bars,
                        test_bars=test_bars,
                        index=i,
                    ))
        
        logger.info(f"Created {len(windows)} walk-forward windows (anchored={self.anchored})")
        return windows


async def run_walkforward_validation(
    symbol: str,
    bars: pl.DataFrame,
    n_splits: int = 5,
    purge_pct: float = 0.1,
    embargo_pct: float = 0.01,
    anchored: bool = True,
    backtest_params: dict | None = None,
    progress_callback: Callable[[int, int, BacktestResult], Any] | None = None,
) -> WalkForwardResult:
    """
    Run walk-forward validation for a symbol.
    
    Args:
        symbol: Trading symbol
        bars: Historical OHLCV data
        n_splits: Number of walk-forward windows
        purge_pct: Purge fraction to prevent data leakage
        embargo_pct: Embargo fraction after test window
        anchored: Use anchored (expanding) or sliding windows
        backtest_params: Additional params passed to run_backtest()
        progress_callback: Optional callback(window_idx, n_windows, result) for progress
        
    Returns:
        WalkForwardResult with all window results and aggregated metrics
    """
    if backtest_params is None:
        backtest_params = {}
    
    validator = WalkForwardValidator(
        n_splits=n_splits,
        purge_pct=purge_pct,
        embargo_pct=embargo_pct,
        anchored=anchored,
    )
    
    windows = validator.create_windows(bars)
    if not windows:
        raise ValueError("No valid walk-forward windows created")
    
    result = WalkForwardResult()
    
    for i, window in enumerate(windows):
        logger.info(f"Walk-forward window {i+1}/{len(windows)}: "
                   f"train={window.train_start.date()} to {window.train_end.date()}, "
                   f"test={window.test_start.date()} to {window.test_end.date()}")
        
        # Run backtest on test window
        bt_result = await run_backtest(
            symbol=symbol,
            bars=window.test_bars,
            **backtest_params,
        )
        
        bt_result.n_trades = len(bt_result.trades)  # Ensure it's set
        result.windows.append(bt_result)
        
        # Store window metrics
        result.metrics_per_window.append({
            "window": i,
            "train_start": window.train_start.isoformat(),
            "train_end": window.train_end.isoformat(),
            "test_start": window.test_start.isoformat(),
            "test_end": window.test_end.isoformat(),
            "n_train_bars": len(window.train_bars),
            "n_test_bars": len(window.test_bars),
            "total_return_pct": bt_result.total_return_pct,
            "sharpe": bt_result.sharpe,
            "max_drawdown_pct": bt_result.max_drawdown_pct,
            "win_rate": bt_result.win_rate,
            "n_trades": bt_result.n_trades,
        })
        
        if progress_callback:
            await progress_callback(i + 1, len(windows), bt_result)
    
    result.compute_aggregated_metrics()
    return result


def print_walkforward_summary(result: WalkForwardResult) -> None:
    """Print a formatted summary of walk-forward results."""
    print("\n" + "=" * 80)
    print("WALK-FORWARD VALIDATION SUMMARY")
    print("=" * 80)
    
    if not result.windows:
        print("No windows completed")
        return
    
    agg = result.aggregated_metrics
    print(f"\nWindows: {agg['n_windows']}")
    print(f"Total Trades: {agg['n_trades_per_window']['total']}")
    print(f"Avg Trades/Window: {agg['n_trades_per_window']['mean']:.1f}")
    
    print(f"\n--- Returns ---")
    ret = agg['total_return_pct']
    print(f"  Mean: {ret['mean']:.2f}%  Std: {ret['std']:.2f}%  Median: {ret['median']:.2f}%")
    print(f"  Range: [{ret['min']:.2f}%, {ret['max']:.2f}%]")
    
    print(f"\n--- Sharpe ---")
    shp = agg['sharpe']
    print(f"  Mean: {shp['mean']:.3f}  Std: {shp['std']:.3f}  Median: {shp['median']:.3f}")
    
    print(f"\n--- Max Drawdown ---")
    dd = agg['max_drawdown_pct']
    print(f"  Mean: {dd['mean']:.2f}%  Std: {dd['std']:.2f}%  Worst: {dd['worst']:.2f}%")
    
    print(f"\n--- Win Rate ---")
    wr = agg['win_rate_pct']
    print(f"  Mean: {wr['mean']:.1f}%  Std: {wr['std']:.1f}%")
    
    print(f"\n--- Consistency ---")
    cons = agg['consistency']
    print(f"  % Positive Windows: {cons['pct_positive_windows']:.1f}%")
    print(f"  % Windows Sharpe > 1: {cons['pct_windows_sharpe_gt_1']:.1f}%")
    print(f"  % Windows DD < 10%: {cons['pct_windows_dd_lt_10']:.1f}%")
    
    print(f"\n--- Per-Window Details ---")
    for wm in result.metrics_per_window:
        print(f"  W{wm['window']}: ret={wm['total_return_pct']:.2f}%  "
              f"sharpe={wm['sharpe']:.2f}  dd={wm['max_drawdown_pct']:.2f}%  "
              f"wr={wm['win_rate']:.1f}%  trades={wm['n_trades']}")
    
    print("=" * 80)