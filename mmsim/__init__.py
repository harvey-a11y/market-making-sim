"""mmsim: Avellaneda-Stoikov market-making simulator.

Simulates a market maker quoting a single asset over one trading session
(arithmetic Brownian midprice, exponential fill intensities) and compares the
Avellaneda-Stoikov inventory strategy against a naive symmetric quoter at
matched average spread, using paired random seeds.
"""

from .dynamics import fill_intensity, fill_probability, generate_price_path
from .metrics import (
    PairedComparison,
    StrategyMetrics,
    bootstrap_std_reduction_ci,
    comparison_table,
    paired_comparison,
    paired_comparison_block,
    paired_pnl_difference,
    summarize,
)
from .simulate import PairedRun, SimConfig, StrategyRun, run_paired
from .strategies import AvellanedaStoikov, NaiveSymmetric, matched_naive_half_spread

__version__ = "0.1.0"

__all__ = [
    "AvellanedaStoikov",
    "NaiveSymmetric",
    "PairedComparison",
    "PairedRun",
    "SimConfig",
    "StrategyMetrics",
    "StrategyRun",
    "bootstrap_std_reduction_ci",
    "comparison_table",
    "fill_intensity",
    "fill_probability",
    "generate_price_path",
    "matched_naive_half_spread",
    "paired_comparison",
    "paired_comparison_block",
    "paired_pnl_difference",
    "run_paired",
    "summarize",
]
