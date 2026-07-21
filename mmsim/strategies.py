"""Quoting strategies.

``AvellanedaStoikov`` implements the closed-form inventory strategy of
Avellaneda & Stoikov (2008) for a market maker with exponential utility
quoting into exponential arrival flow:

    reservation price   r(S, q, t) = S - q * gamma * sigma^2 * (T - t)
    half-spread         delta(t)   = gamma * sigma^2 * (T - t) / 2
                                     + (1 / gamma) * ln(1 + gamma / k)

with the bid quoted at ``r - delta`` and the ask at ``r + delta``. Holding
inventory shifts the *centre* of the quotes away from the mid (long inventory
lowers both quotes, making a sell more likely), while the half-spread itself
is inventory-independent and decays linearly towards its terminal value as
t -> T.

``NaiveSymmetric`` quotes a fixed symmetric half-spread around the mid with no
inventory management. ``matched_naive_half_spread`` computes the constant that
makes the naive quoter's time-average quoted spread equal to the
Avellaneda-Stoikov strategy's over the same discrete quote grid, so the two
strategies are compared at matched average spread and differ only in *where*
the spread is centred.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

__all__ = ["AvellanedaStoikov", "NaiveSymmetric", "matched_naive_half_spread"]


@dataclass(frozen=True)
class AvellanedaStoikov:
    """Closed-form Avellaneda-Stoikov quoting rule."""

    gamma: float = 0.1
    sigma: float = 2.0
    k: float = 1.5
    horizon: float = 1.0
    name: str = "Avellaneda-Stoikov"

    def reservation_price(self, mid: float, inventory: int, t: float) -> float:
        """r = S - q * gamma * sigma^2 * (T - t)."""
        return mid - inventory * self.gamma * self.sigma**2 * (self.horizon - t)

    def half_spread(self, t: float) -> float:
        """delta = gamma * sigma^2 * (T - t) / 2 + (1 / gamma) * ln(1 + gamma / k)."""
        return (
            self.gamma * self.sigma**2 * (self.horizon - t) / 2.0
            + math.log(1.0 + self.gamma / self.k) / self.gamma
        )

    def quotes(self, mid: float, inventory: int, t: float) -> tuple[float, float]:
        """Return (bid, ask) centred on the reservation price."""
        r = self.reservation_price(mid, inventory, t)
        d = self.half_spread(t)
        return r - d, r + d

    def schedule(self, n_steps: int, dt: float) -> tuple[np.ndarray, np.ndarray]:
        """Precompute (skew, half_spread) arrays over the quote grid.

        ``skew[i]`` multiplies inventory to shift the quote centre off the mid
        (``gamma * sigma^2 * (T - t_i)``); ``half[i]`` is the half-spread at
        step ``i``. Quotes at step ``i`` are ``S - q * skew[i] -/+ half[i]``.
        """
        t = np.arange(n_steps) * dt
        skew = self.gamma * self.sigma**2 * (self.horizon - t)
        half = skew / 2.0 + math.log(1.0 + self.gamma / self.k) / self.gamma
        return skew, half


@dataclass(frozen=True)
class NaiveSymmetric:
    """Fixed symmetric half-spread around the mid; no inventory management."""

    half_spread_value: float
    name: str = "Naive symmetric"

    def quotes(self, mid: float, inventory: int, t: float) -> tuple[float, float]:
        return mid - self.half_spread_value, mid + self.half_spread_value

    def schedule(self, n_steps: int, dt: float) -> tuple[np.ndarray, np.ndarray]:
        return np.zeros(n_steps), np.full(n_steps, self.half_spread_value)


def matched_naive_half_spread(
    strategy: AvellanedaStoikov, n_steps: int, dt: float
) -> float:
    """Half-spread that matches the A-S time-average spread on the quote grid.

    The A-S quoted spread is ``2 * delta(t)`` regardless of inventory (skew
    only recentres the quotes), so matching the mean half-spread matches the
    mean quoted spread exactly.
    """
    _, half = strategy.schedule(n_steps, dt)
    return float(half.mean())
