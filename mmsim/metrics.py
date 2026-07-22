"""Summary metrics, paired-difference inference, and the console table.

The simulator runs both strategies against the same random draws (common
random numbers), so the per-episode arrays are *paired*. Inference on the
comparison should therefore be paired too: the mean P&L difference carries a
t-based confidence interval computed from the per-episode differences, and
the headline std-reduction ratio carries a paired-bootstrap percentile CI.

No scipy dependency: the Student-t quantile is computed exactly via the
regularized incomplete beta function (continued-fraction evaluation).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .simulate import StrategyRun

__all__ = [
    "StrategyMetrics",
    "PairedComparison",
    "summarize",
    "comparison_table",
    "paired_pnl_difference",
    "bootstrap_std_reduction_ci",
    "paired_comparison",
    "paired_comparison_block",
]


# --------------------------------------------------------------------------
# Student-t quantile (exact, via the regularized incomplete beta function)
# --------------------------------------------------------------------------

def _betacf(a: float, b: float, x: float) -> float:
    """Continued fraction for the incomplete beta function (Lentz's method)."""
    max_iter = 300
    eps = 3e-15
    fpmin = 1e-300
    qab = a + b
    qap = a + 1.0
    qam = a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < fpmin:
        d = fpmin
    d = 1.0 / d
    h = d
    for m in range(1, max_iter + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < fpmin:
            d = fpmin
        c = 1.0 + aa / c
        if abs(c) < fpmin:
            c = fpmin
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < fpmin:
            d = fpmin
        c = 1.0 + aa / c
        if abs(c) < fpmin:
            c = fpmin
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < eps:
            break
    return h


def _betainc(a: float, b: float, x: float) -> float:
    """Regularized incomplete beta function I_x(a, b)."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    ln_front = (
        math.lgamma(a + b)
        - math.lgamma(a)
        - math.lgamma(b)
        + a * math.log(x)
        + b * math.log1p(-x)
    )
    front = math.exp(ln_front)
    if x < (a + 1.0) / (a + b + 2.0):
        return front * _betacf(a, b, x) / a
    return 1.0 - front * _betacf(b, a, 1.0 - x) / b


def _student_t_cdf(t: float, df: float) -> float:
    """CDF of Student's t with ``df`` degrees of freedom."""
    x = df / (df + t * t)
    tail = 0.5 * _betainc(0.5 * df, 0.5, x)
    return 1.0 - tail if t >= 0.0 else tail


def _t_ppf(p: float, df: int) -> float:
    """Quantile (inverse CDF) of Student's t, by bisection on the exact CDF."""
    if not 0.0 < p < 1.0:
        raise ValueError("p must be in the open interval (0, 1)")
    if df < 1:
        raise ValueError("df must be >= 1")
    if p == 0.5:
        return 0.0
    if p < 0.5:
        return -_t_ppf(1.0 - p, df)
    lo, hi = 0.0, 1.0
    while _student_t_cdf(hi, df) < p:
        hi *= 2.0
        if hi > 1e12:  # pragma: no cover - p astronomically close to 1
            break
    for _ in range(120):
        mid = 0.5 * (lo + hi)
        if _student_t_cdf(mid, df) < p:
            lo = mid
        else:
            hi = mid
        if hi - lo < 1e-12 * max(1.0, hi):
            break
    return 0.5 * (lo + hi)


# --------------------------------------------------------------------------
# Per-strategy summary
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class StrategyMetrics:
    """Aggregate metrics for one strategy across a Monte Carlo run."""

    name: str
    mean_pnl: float
    std_pnl: float
    mean_over_std: float  # mean/std of terminal P&L (episode P&Ls, NOT a Sharpe ratio)
    mean_abs_terminal_inventory: float
    mean_inventory_variance: float
    mean_max_abs_inventory: float
    mean_fills: float
    adverse_selection: float


def summarize(run: StrategyRun) -> StrategyMetrics:
    """Reduce per-episode results to the reported metrics.

    The adverse-selection proxy is the mean, pooled over fills, of the signed
    midprice move over the next ``adverse_horizon`` steps from the position's
    perspective (+ after a buy if the mid rose, + after a sell if the mid
    fell). Fills within ``adverse_horizon`` steps of session end have no
    complete lookahead window and are excluded from the average (truncating
    the window at the final mid would bias late-fill contributions toward
    zero). Negative values indicate fills that were adversely selected.

    ``mean_over_std`` is the ratio of mean to standard deviation of terminal
    episode P&L. It is deliberately not labelled "Sharpe": these are episode
    P&Ls under paired seeds, not a returns time series.
    """
    horizon_fills = int(run.adverse_fills.sum())
    adverse = float(run.adverse_sum.sum() / horizon_fills) if horizon_fills else 0.0
    mean_pnl = float(run.pnl.mean())
    std_pnl = float(run.pnl.std(ddof=1))
    return StrategyMetrics(
        name=run.name,
        mean_pnl=mean_pnl,
        std_pnl=std_pnl,
        mean_over_std=mean_pnl / std_pnl if std_pnl > 0.0 else math.nan,
        mean_abs_terminal_inventory=float(np.abs(run.terminal_inventory).mean()),
        mean_inventory_variance=float(run.inventory_variance.mean()),
        mean_max_abs_inventory=float(run.max_abs_inventory.mean()),
        mean_fills=float(run.fills.mean()),
        adverse_selection=adverse,
    )


# --------------------------------------------------------------------------
# Paired inference
# --------------------------------------------------------------------------

def paired_pnl_difference(
    pnl_a: np.ndarray, pnl_b: np.ndarray, confidence: float = 0.95
) -> tuple[float, float, tuple[float, float]]:
    """Paired-difference inference on per-episode P&L.

    Returns ``(mean_diff, se, (ci_low, ci_high))`` for the per-episode
    differences ``pnl_a - pnl_b``: the mean difference, its standard error
    (sample std with ``ddof=1`` over sqrt(n)), and the t-based confidence
    interval with ``n - 1`` degrees of freedom.
    """
    a = np.asarray(pnl_a, dtype=float)
    b = np.asarray(pnl_b, dtype=float)
    if a.shape != b.shape or a.ndim != 1:
        raise ValueError("pnl arrays must be 1-D and of equal length (paired)")
    n = a.size
    if n < 2:
        raise ValueError("need at least 2 paired episodes")
    diff = a - b
    mean = float(diff.mean())
    se = float(diff.std(ddof=1) / math.sqrt(n))
    tcrit = _t_ppf(0.5 + confidence / 2.0, n - 1)
    return mean, se, (mean - tcrit * se, mean + tcrit * se)


def bootstrap_std_reduction_ci(
    pnl_a: np.ndarray,
    pnl_b: np.ndarray,
    rng: np.random.Generator,
    n_resamples: int = 10_000,
    confidence: float = 0.95,
) -> tuple[float, float]:
    """Paired-bootstrap percentile CI on the std reduction ``1 - std_a/std_b``.

    Episodes are resampled with replacement using the *same* indices for both
    strategies (the design is paired), and the ratio statistic is recomputed
    on each resample (``ddof=1`` throughout). ``rng`` is required: pass a
    seeded ``numpy.random.Generator`` so the interval is reproducible.
    """
    a = np.asarray(pnl_a, dtype=float)
    b = np.asarray(pnl_b, dtype=float)
    if a.shape != b.shape or a.ndim != 1:
        raise ValueError("pnl arrays must be 1-D and of equal length (paired)")
    n = a.size
    if n < 2:
        raise ValueError("need at least 2 paired episodes")
    stats = np.empty(n_resamples)
    batch = max(1, min(1000, n_resamples))
    done = 0
    while done < n_resamples:
        m = min(batch, n_resamples - done)
        idx = rng.integers(0, n, size=(m, n))
        std_a = a[idx].std(axis=1, ddof=1)
        std_b = b[idx].std(axis=1, ddof=1)
        stats[done : done + m] = 1.0 - std_a / std_b
        done += m
    alpha = 1.0 - confidence
    lo, hi = np.quantile(stats, [alpha / 2.0, 1.0 - alpha / 2.0])
    return float(lo), float(hi)


@dataclass(frozen=True)
class PairedComparison:
    """Paired inference on two strategies run under common random numbers."""

    name_a: str
    name_b: str
    n_episodes: int
    confidence: float
    mean_diff: float       # mean per-episode (pnl_a - pnl_b)
    se_diff: float         # standard error of the mean difference (ddof=1)
    ci_low: float          # t-based CI on the mean difference
    ci_high: float
    std_reduction: float   # 1 - std_a/std_b on the full sample
    std_reduction_ci_low: float   # paired-bootstrap percentile CI
    std_reduction_ci_high: float
    n_resamples: int


def paired_comparison(
    run_a: StrategyRun,
    run_b: StrategyRun,
    rng: np.random.Generator,
    n_resamples: int = 10_000,
    confidence: float = 0.95,
) -> PairedComparison:
    """Full paired comparison of ``run_a`` against ``run_b`` (a minus b)."""
    mean, se, (lo, hi) = paired_pnl_difference(run_a.pnl, run_b.pnl, confidence)
    std_a = float(np.asarray(run_a.pnl, dtype=float).std(ddof=1))
    std_b = float(np.asarray(run_b.pnl, dtype=float).std(ddof=1))
    boot_lo, boot_hi = bootstrap_std_reduction_ci(
        run_a.pnl, run_b.pnl, rng=rng, n_resamples=n_resamples, confidence=confidence
    )
    return PairedComparison(
        name_a=run_a.name,
        name_b=run_b.name,
        n_episodes=int(np.asarray(run_a.pnl).size),
        confidence=confidence,
        mean_diff=mean,
        se_diff=se,
        ci_low=lo,
        ci_high=hi,
        std_reduction=1.0 - std_a / std_b,
        std_reduction_ci_low=boot_lo,
        std_reduction_ci_high=boot_hi,
        n_resamples=n_resamples,
    )


def paired_comparison_block(pc: PairedComparison) -> str:
    """Render the paired-inference lines printed under the comparison table."""
    pct = 100.0 * pc.confidence
    label_w = 34
    lines = [
        f"Paired inference ({pc.name_a} minus {pc.name_b}, "
        f"{pc.n_episodes} paired episodes):",
        "{}  {:+.3f}  (SE {:.3f})".format(
            "Mean terminal P&L difference".ljust(label_w), pc.mean_diff, pc.se_diff
        ),
        "{}  [{:+.3f}, {:+.3f}]".format(
            f"{pct:.0f}% CI, t-based (df={pc.n_episodes - 1})".ljust(label_w),
            pc.ci_low,
            pc.ci_high,
        ),
        "{}  {:.1%}".format(
            "Std reduction (1 - std_AS/std_nv)".ljust(label_w), pc.std_reduction
        ),
        "{}  [{:.1%}, {:.1%}]".format(
            f"Bootstrap {pct:.0f}% CI ({pc.n_resamples} resamples)".ljust(label_w),
            pc.std_reduction_ci_low,
            pc.std_reduction_ci_high,
        ),
    ]
    return "\n".join(lines)


# --------------------------------------------------------------------------
# Console table
# --------------------------------------------------------------------------

_ROWS = [
    ("Mean terminal P&L", "mean_pnl", "{:+.3f}"),
    ("Std terminal P&L", "std_pnl", "{:.3f}"),
    ("Mean/std of terminal P&L", "mean_over_std", "{:.2f}"),
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
