import numpy as np

from mmsim.dynamics import fill_intensity, fill_probability, generate_price_path


def test_fill_intensity_strictly_decreasing_in_distance():
    deltas = np.linspace(0.0, 5.0, 501)
    lam = fill_intensity(deltas, a=140.0, k=1.5)
    assert np.all(np.diff(lam) < 0.0)


def test_fill_intensity_at_zero_distance_equals_a():
    assert fill_intensity(0.0, a=140.0, k=1.5) == 140.0


def test_fill_probability_bounded_in_unit_interval():
    deltas = np.linspace(-2.0, 5.0, 101)
    p = fill_probability(deltas, a=140.0, k=1.5, dt=1.0 / 3600.0)
    assert np.all(p > 0.0)
    assert np.all(p < 1.0)


def test_price_path_shape_and_start():
    rng = np.random.default_rng(0)
    path = generate_price_path(100.0, 2.0, 1.0 / 3600.0, 3600, rng)
    assert path.shape == (3601,)
    assert path[0] == 100.0
