"""Paired-seed Monte Carlo simulation of quoting strategies.

Each episode draws one midprice path and one pair of uniform arrays (bid-side
and ask-side arrival randomness). Every strategy is then run against the
*same* draws (common random numbers), so cross-strategy differences are
attributable to the quoting rule rather than to sampling noise.

Bookkeeping per fill (unit size, quotes refreshed every step):

    bid filled:  q += 1, cash -= bid_price      (we buy)
    ask filled:  q -= 1, cash += ask_price      (we sell)

Terminal inventory is liquidated at the final mid with no penalty beyond
mark-to-market: ``pnl = cash + q_T * S_T``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .dynamics import generate_price_path
from .strategies import AvellanedaStoikov, NaiveSymmetric, matched_naive_half_spread

__all__ = ["SimConfig", "StrategyRun", "PairedRun", "run_paired"]


@dataclass(frozen=True)
class SimConfig:
    """Simulation parameters. Defaults follow the project specification."""

    sigma: float = 2.0          # midprice volatility per session (T = 1)
    arrival_a: float = 140.0    # baseline arrival intensity A, per unit time
    arrival_k: float = 1.5      # arrival decay k
    gamma: float = 0.1          # risk aversion
    horizon: float = 1.0        # session length T (normalised)
    n_steps: int = 3600         # steps per episode
    s0: float = 100.0           # initial midprice
    episodes: int = 1000
    seed: int = 42
    adverse_horizon: int = 60   # look-ahead steps for the adverse-selection proxy

    @property
    def dt(self) -> float:
        return self.horizon / self.n_steps


@dataclass
class StrategyRun:
    """Per-episode results for one strategy across the Monte Carlo run."""

    name: str
    pnl: np.ndarray                  # terminal P&L (cash + q_T * S_T)
    terminal_inventory: np.ndarray
    inventory_variance: np.ndarray   # Var(q_t) along each episode's path
    max_abs_inventory: np.ndarray
    fills: np.ndarray                # fills per episode, both sides
    adverse_sum: np.ndarray          # per-episode sum of side * (S_{t+h} - S_t)
    sample_inventory_paths: list = field(default_factory=list)


@dataclass
class PairedRun:
    """Results of a paired-seed run of both strategies."""

    config: SimConfig
    avellaneda_stoikov: StrategyRun
    naive: StrategyRun
    naive_half_spread: float         # matched to the A-S time-average


def _run_episode(skew, half, path, u_bid, u_ask, cfg: SimConfig, keep_path: bool):
    """Run one episode given precomputed quote schedules (plain lists for speed).

    ``skew[i]`` multiplies inventory to shift the quote centre off the mid;
    ``half[i]`` is the half-spread at step ``i``.
    """
    a = cfg.arrival_a
    k = cfg.arrival_k
    dt = cfg.dt
    h = cfg.adverse_horizon
    n = cfg.n_steps
    exp = math.exp

    q = 0
    cash = 0.0
    fills = 0
    adverse = 0.0
    inventory = [0] * (n + 1)

    for i in range(n):
        s = path[i]
        centre = s - q * skew[i]
        d = half[i]
        bid = centre - d
        ask = centre + d
        # Quote distances from the mid feed the arrival intensity; a distance
        # may go negative when inventory skew pushes a quote through the mid
        # (an aggressive quote that fills with near-certainty).
        p_bid = 1.0 - exp(-a * exp(-k * (s - bid)) * dt)
        p_ask = 1.0 - exp(-a * exp(-k * (ask - s)) * dt)
        if u_bid[i] < p_bid:            # our bid is lifted: we buy one unit
            q += 1
            cash -= bid
            fills += 1
            j = i + h
            adverse += (path[j] if j <= n else path[n]) - s
        if u_ask[i] < p_ask:            # our ask is hit: we sell one unit
            q -= 1
            cash += ask
            fills += 1
            j = i + h
            adverse -= (path[j] if j <= n else path[n]) - s
        inventory[i + 1] = q

    inv = np.asarray(inventory, dtype=np.int64)
    pnl = cash + q * path[n]            # liquidate terminal inventory at final mid
    return (
        pnl,
        q,
        float(inv.var()),
        int(np.abs(inv).max()),
        fills,
        adverse,
        inv if keep_path else None,
    )


def run_paired(cfg: SimConfig | None = None, keep_paths: int = 0) -> PairedRun:
    """Run both strategies over ``cfg.episodes`` episodes with paired seeds.

    ``keep_paths`` retains the full inventory path of the first that many
    episodes per strategy (for plotting).
    """
    if cfg is None:
        cfg = SimConfig()

    strategy = AvellanedaStoikov(
        gamma=cfg.gamma, sigma=cfg.sigma, k=cfg.arrival_k, horizon=cfg.horizon
    )
    naive_half = matched_naive_half_spread(strategy, cfg.n_steps, cfg.dt)
    naive = NaiveSymmetric(half_spread_value=naive_half)

    schedules = []
    for strat in (strategy, naive):
        skew, half = strat.schedule(cfg.n_steps, cfg.dt)
        schedules.append((strat.name, skew.tolist(), half.tolist()))

    n_metrics = 6
    acc = [[[] for _ in range(n_metrics)] for _ in schedules]
    samples: list[list] = [[] for _ in schedules]

    seed_children = np.random.SeedSequence(cfg.seed).spawn(cfg.episodes)
    for episode, child in enumerate(seed_children):
        rng = np.random.default_rng(child)
        path = generate_price_path(cfg.s0, cfg.sigma, cfg.dt, cfg.n_steps, rng)
        u_bid = rng.random(cfg.n_steps).tolist()
        u_ask = rng.random(cfg.n_steps).tolist()
        path_list = path.tolist()
        keep = episode < keep_paths
        for idx, (_, skew, half) in enumerate(schedules):
            out = _run_episode(skew, half, path_list, u_bid, u_ask, cfg, keep)
            for m in range(n_metrics):
                acc[idx][m].append(out[m])
            if keep:
                samples[idx].append(out[6])

    def build(idx: int, name: str) -> StrategyRun:
        a = acc[idx]
        return StrategyRun(
            name=name,
            pnl=np.asarray(a[0]),
            terminal_inventory=np.asarray(a[1], dtype=np.int64),
            inventory_variance=np.asarray(a[2]),
            max_abs_inventory=np.asarray(a[3], dtype=np.int64),
            fills=np.asarray(a[4], dtype=np.int64),
            adverse_sum=np.asarray(a[5]),
            sample_inventory_paths=samples[idx],
        )

    return PairedRun(
        config=cfg,
        avellaneda_stoikov=build(0, schedules[0][0]),
        naive=build(1, schedules[1][0]),
        naive_half_spread=naive_half,
    )
