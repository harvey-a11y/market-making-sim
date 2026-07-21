"""Summary metrics and the console comparison table."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .simulate import StrategyRun

__all__ = ["StrategyMetrics", "summarize", "comparison_table"]


@dataclass(frozen=True)
class StrategyMetrics:
    """Aggregate metrics for one strategy across a Monte Carlo run."""

    name: str
    mean_pnl: float
    std_pnl: float
    mean_abs_terminal_inventory: float
    mean_inventory_variance: float
    mean_max_abs_inventory: float
    mean_fills: float
    adverse_selection: float


def summarize(run: StrategyRun) -> StrategyMetrics:
    """Reduce per-episode results to the reported metrics.

    The adverse-selection proxy is the mean, pooled over every fill, of the
    signed midprice move over the next ``adverse_horizon`` steps from the
    position's perspective (+ after a buy if the mid rose, + after a sell if
    the mid fell). Negative values indicate fills that were adversely
    selected.
    """
    total_fills = int(run.fills.sum())
    adverse = float(run.adverse_sum.sum() / total_fills) if total_fills else 0.0
    return StrategyMetrics(
        name=run.name,
        mean_pnl=float(run.pnl.mean()),
        std_pnl=float(run.pnl.std(ddof=1)),
        mean_abs_terminal_inventory=float(np.abs(run.terminal_inventory).mean()),
        mean_inventory_variance=float(run.inventory_variance.mean()),
        mean_max_abs_inventory=float(run.max_abs_inventory.mean()),
        mean_fills=float(run.fills.mean()),
        adverse_selection=adverse,
    )


_ROWS = [
    ("Mean terminal P&L", "mean_pnl", "{:+.3f}"),
    ("Std terminal P&L", "std_pnl", "{:.3f}"),
    ("Mean |terminal inventory|", "mean_abs_terminal_inventory", "{:.2f}"),
    ("Inventory variance (path mean)", "mean_inventory_variance", "{:.2f}"),
    ("Mean max |inventory|", "mean_max_abs_inventory", "{:.2f}"),
    ("Fills per episode", "mean_fills", "{:.1f}"),
    ("Adverse-selection proxy", "adverse_selection", "{:+.4f}"),
]


def comparison_table(*metrics: StrategyMetrics) -> str:
    """Render a fixed-width comparison table (one column per strategy)."""
    label_w = max(len(label) for label, _, _ in _ROWS) + 2
    col_w = max(max(len(m.name) for m in metrics) + 3, 20)
    header = "Metric".ljust(label_w) + "".join(m.name.rjust(col_w) for m in metrics)
    lines = [header, "-" * len(header)]
    for label, attr, fmt in _ROWS:
        cells = "".join(fmt.format(getattr(m, attr)).rjust(col_w) for m in metrics)
        lines.append(label.ljust(label_w) + cells)
    return "\n".join(lines)
