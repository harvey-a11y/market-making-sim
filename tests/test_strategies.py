import math

import numpy as np
import pytest

from mmsim.strategies import (
    AvellanedaStoikov,
    NaiveSymmetric,
    matched_naive_half_spread,
)

STRAT = AvellanedaStoikov(gamma=0.1, sigma=2.0, k=1.5, horizon=1.0)


def test_reservation_price_below_mid_when_long():
    mid = 100.0
    assert STRAT.reservation_price(mid, inventory=3, t=0.25) < mid


def test_reservation_price_equals_mid_when_flat():
    mid = 100.0
    assert STRAT.reservation_price(mid, inventory=0, t=0.25) == pytest.approx(mid)


def test_reservation_price_skew_is_symmetric():
    mid = 100.0
    down = mid - STRAT.reservation_price(mid, inventory=4, t=0.3)
    up = STRAT.reservation_price(mid, inventory=-4, t=0.3) - mid
    assert down > 0.0
    assert down == pytest.approx(up)


def test_half_spread_shrinks_as_t_approaches_terminal():
    times = np.linspace(0.0, 1.0, 200)
    spreads = np.array([STRAT.half_spread(t) for t in times])
    assert np.all(np.diff(spreads) < 0.0)
    # At t = T only the arrival-driven term remains.
    assert spreads[-1] == pytest.approx(math.log(1.0 + 0.1 / 1.5) / 0.1)


def test_quotes_bracket_reservation_price():
    bid, ask = STRAT.quotes(mid=100.0, inventory=2, t=0.5)
    r = STRAT.reservation_price(100.0, 2, 0.5)
    d = STRAT.half_spread(0.5)
    assert bid == pytest.approx(r - d)
    assert ask == pytest.approx(r + d)


def test_matched_naive_half_spread_equals_time_average():
    n_steps, dt = 3600, 1.0 / 3600.0
    naive_half = matched_naive_half_spread(STRAT, n_steps, dt)
    _, as_half = STRAT.schedule(n_steps, dt)
    assert naive_half == pytest.approx(float(as_half.mean()), abs=1e-12)

    naive = NaiveSymmetric(half_spread_value=naive_half)
    _, naive_half_sched = naive.schedule(n_steps, dt)
    assert float(naive_half_sched.mean()) == pytest.approx(float(as_half.mean()))
