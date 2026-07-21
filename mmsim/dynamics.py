"""Midprice dynamics and the order-arrival (fill) model.

The midprice follows arithmetic Brownian motion, dS = sigma dW, simulated on a
discrete grid of ``n_steps`` steps over a horizon T (one trading session,
normalised to T = 1). Market orders arrive on each side of the book as a
Poisson process whose intensity against a quote at distance ``delta`` from the
mid is

    lambda(delta) = A * exp(-k * delta)

so tighter quotes fill more often. Over one step of length ``dt`` the
probability of at least one fill is ``1 - exp(-lambda(delta) * dt)``.
"""

from __future__ import annotations

import numpy as np

__all__ = ["generate_price_path", "fill_intensity", "fill_probability"]


def generate_price_path(
    s0: float,
    sigma: float,
    dt: float,
    n_steps: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Simulate an arithmetic Brownian midprice path.

    Returns an array of length ``n_steps + 1``; element ``i`` is the midprice
    at time ``i * dt``.
    """
    increments = sigma * np.sqrt(dt) * rng.standard_normal(n_steps)
    path = np.empty(n_steps + 1)
    path[0] = s0
    np.cumsum(increments, out=path[1:])
    path[1:] += s0
    return path


def fill_intensity(delta, a: float, k: float):
    """Poisson fill intensity ``lambda(delta) = A * exp(-k * delta)``.

    ``delta`` is the quote's distance from the mid (may be an array). The
    intensity is per unit time; it is strictly decreasing in ``delta``.
    """
    return a * np.exp(-k * np.asarray(delta, dtype=float))


def fill_probability(delta, a: float, k: float, dt: float):
    """Probability of at least one fill during a step of length ``dt``."""
    return 1.0 - np.exp(-fill_intensity(delta, a, k) * dt)
