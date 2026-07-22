"""Command-line interface.

    python -m mmsim run --episodes 1000 --gamma 0.1 --seed 42 \
        --plot examples/comparison.png --plot-inventory examples/inventory_paths.png
"""

from __future__ import annotations

import argparse
import time

import numpy as np

from .metrics import (
    comparison_table,
    paired_comparison,
    paired_comparison_block,
    summarize,
)
from .simulate import PairedRun, SimConfig, run_paired

# Chart palette: validated categorical slots 1-2 plus chart chrome (light mode).
_SURFACE = "#fcfcfb"
_INK = "#0b0b0b"
_INK_SECONDARY = "#52514e"
_MUTED = "#898781"
_GRID = "#e1e0d9"
_BASELINE = "#c3c2b7"
_SERIES_AS = "#2a78d6"      # blue: Avellaneda-Stoikov
_SERIES_NAIVE = "#eb6834"   # orange: naive symmetric


def _style_axis(ax) -> None:
    ax.set_facecolor(_SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(_BASELINE)
    ax.tick_params(colors=_MUTED, labelsize=9)
    ax.yaxis.grid(True, color=_GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def plot_pnl_distribution(result: PairedRun, path: str) -> None:
    """Overlaid terminal P&L histograms for the two strategies."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    as_pnl = result.avellaneda_stoikov.pnl
    nv_pnl = result.naive.pnl
    lo = float(min(as_pnl.min(), nv_pnl.min()))
    hi = float(max(as_pnl.max(), nv_pnl.max()))
    bins = np.linspace(lo, hi, 61)

    fig, ax = plt.subplots(figsize=(8.0, 4.5), dpi=150)
    fig.patch.set_facecolor(_SURFACE)
    _style_axis(ax)

    for pnl, color, run_name in (
        (nv_pnl, _SERIES_NAIVE, result.naive.name),
        (as_pnl, _SERIES_AS, result.avellaneda_stoikov.name),
    ):
        label = f"{run_name}  (std {pnl.std(ddof=1):.2f})"
        ax.hist(pnl, bins=bins, color=color, alpha=0.35)
        ax.hist(pnl, bins=bins, histtype="step", color=color, linewidth=1.6, label=label)

    ax.set_xlabel("Terminal P&L per episode", color=_INK_SECONDARY, fontsize=10)
    ax.set_ylabel("Episodes", color=_INK_SECONDARY, fontsize=10)
    ax.set_title(
        f"Terminal P&L, {result.config.episodes} paired episodes "
        f"(seed {result.config.seed}, matched mean spread)",
        color=_INK,
        fontsize=11,
        loc="left",
    )
    ax.legend(frameon=False, fontsize=9, labelcolor=_INK_SECONDARY)
    fig.tight_layout()
    fig.savefig(path, facecolor=_SURFACE)
    plt.close(fig)


def plot_inventory_paths(result: PairedRun, path: str) -> None:
    """Sample inventory paths, one panel per strategy on a shared y-scale."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(
        2, 1, figsize=(8.0, 5.5), dpi=150, sharex=True, sharey=True
    )
    fig.patch.set_facecolor(_SURFACE)

    panels = (
        (result.avellaneda_stoikov, _SERIES_AS),
        (result.naive, _SERIES_NAIVE),
    )
    alphas = (1.0, 0.62, 0.38)
    for ax, (run, color) in zip(axes, panels):
        _style_axis(ax)
        ax.axhline(0.0, color=_BASELINE, linewidth=1.0)
        for j, inv in enumerate(run.sample_inventory_paths):
            t = np.linspace(0.0, 1.0, inv.size)
            ax.plot(t, inv, color=color, linewidth=1.6, alpha=alphas[j % len(alphas)])
        ax.set_title(run.name, color=_INK, fontsize=10, loc="left")
        ax.set_ylabel("Inventory (units)", color=_INK_SECONDARY, fontsize=9)
    axes[-1].set_xlabel("Session time t / T", color=_INK_SECONDARY, fontsize=10)
    fig.suptitle(
        f"Sample inventory paths, first {len(panels[0][0].sample_inventory_paths)} "
        "paired episodes (shared scale)",
        color=_INK,
        fontsize=11,
        x=0.02,
        ha="left",
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))
    fig.savefig(path, facecolor=_SURFACE)
    plt.close(fig)


def _cmd_run(args: argparse.Namespace) -> int:
    cfg = SimConfig(
        sigma=args.sigma,
        arrival_a=args.arrival_a,
        arrival_k=args.arrival_k,
        gamma=args.gamma,
        n_steps=args.steps,
        episodes=args.episodes,
        seed=args.seed,
    )
    keep_paths = 3 if args.plot_inventory else 0

    start = time.perf_counter()
    result = run_paired(cfg, keep_paths=keep_paths)
    elapsed = time.perf_counter() - start

    print("mmsim: Avellaneda-Stoikov vs naive symmetric quoting (paired seeds)")
    print(
        f"episodes={cfg.episodes}  steps/episode={cfg.n_steps}  seed={cfg.seed}  "
        f"sigma={cfg.sigma}  A={cfg.arrival_a}  k={cfg.arrival_k}  gamma={cfg.gamma}"
    )
    print(
        f"matched naive half-spread = {result.naive_half_spread:.4f} "
        "(equals the A-S time-average half-spread)"
    )
    print()
    print(
        comparison_table(
            summarize(result.avellaneda_stoikov), summarize(result.naive)
        )
    )
    print()
    # Paired inference under common random numbers. The bootstrap RNG is
    # seeded from --seed (independently of the simulation draws) so the
    # reported intervals are reproducible.
    inference = paired_comparison(
        result.avellaneda_stoikov,
        result.naive,
        rng=np.random.default_rng(cfg.seed),
    )
    print(paired_comparison_block(inference))
    print()
    print(f"elapsed: {elapsed:.1f}s")

    if args.plot:
        plot_pnl_distribution(result, args.plot)
        print(f"wrote {args.plot}")
    if args.plot_inventory:
        plot_inventory_paths(result, args.plot_inventory)
        print(f"wrote {args.plot_inventory}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="mmsim",
        description="Avellaneda-Stoikov market-making simulator",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    run_p = sub.add_parser(
        "run", help="run a paired Monte Carlo comparison and print a table"
    )
    run_p.add_argument("--episodes", type=int, default=1000)
    run_p.add_argument("--gamma", type=float, default=0.1, help="risk aversion")
    run_p.add_argument("--seed", type=int, default=42)
    run_p.add_argument("--sigma", type=float, default=2.0, help="volatility per session")
    run_p.add_argument("--arrival-a", type=float, default=140.0, help="arrival intensity A")
    run_p.add_argument("--arrival-k", type=float, default=1.5, help="arrival decay k")
    run_p.add_argument("--steps", type=int, default=3600, help="steps per episode")
    run_p.add_argument("--plot", metavar="PATH", default=None,
                       help="write the P&L distribution comparison PNG here")
    run_p.add_argument("--plot-inventory", metavar="PATH", default=None,
                       help="write the sample inventory paths PNG here")

    args = parser.parse_args(argv)
    if args.command == "run":
        return _cmd_run(args)
    parser.error("unknown command")
    return 2
