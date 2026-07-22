import numpy as np
import pytest

from mmsim.simulate import SimConfig, _run_episode, run_paired


def test_paired_run_as_reduces_inventory_and_pnl_variance():
    """At matched average spread, A-S must carry less inventory risk."""
    cfg = SimConfig(episodes=200, seed=7)
    result = run_paired(cfg)
    as_run = result.avellaneda_stoikov
    nv_run = result.naive
    assert as_run.inventory_variance.mean() < nv_run.inventory_variance.mean()
    assert as_run.pnl.std(ddof=1) < nv_run.pnl.std(ddof=1)


def test_same_seed_reproduces_identical_results():
    cfg = SimConfig(episodes=25, seed=123)
    r1 = run_paired(cfg)
    r2 = run_paired(cfg)
    for a, b in (
        (r1.avellaneda_stoikov, r2.avellaneda_stoikov),
        (r1.naive, r2.naive),
    ):
        assert np.array_equal(a.pnl, b.pnl)
        assert np.array_equal(a.terminal_inventory, b.terminal_inventory)
        assert np.array_equal(a.inventory_variance, b.inventory_variance)
        assert np.array_equal(a.max_abs_inventory, b.max_abs_inventory)
        assert np.array_equal(a.fills, b.fills)
        assert np.array_equal(a.adverse_sum, b.adverse_sum)
        assert np.array_equal(a.adverse_fills, b.adverse_fills)
    assert r1.naive_half_spread == r2.naive_half_spread


def test_fills_occur_on_every_episode():
    cfg = SimConfig(episodes=10, seed=1)
    result = run_paired(cfg)
    assert result.avellaneda_stoikov.fills.min() > 0
    assert result.naive.fills.min() > 0


def test_keep_paths_retains_requested_inventory_paths():
    cfg = SimConfig(episodes=5, seed=2)
    result = run_paired(cfg, keep_paths=3)
    for run in (result.avellaneda_stoikov, result.naive):
        assert len(run.sample_inventory_paths) == 3
        for inv in run.sample_inventory_paths:
            assert inv.shape == (cfg.n_steps + 1,)
            assert inv[0] == 0


def test_adverse_proxy_excludes_incomplete_horizon_fills():
    """Fills in the last h-1 steps have no complete lookahead window and must
    not enter the proxy at all. On a linear path every complete window moves
    exactly h * slope, so the episode total pins the exclusion: truncating
    the window at the session end (the old behaviour) would add smaller,
    biased-toward-zero contributions from the late fills."""
    cfg = SimConfig(n_steps=50, adverse_horizon=10, episodes=1)
    n, h = cfg.n_steps, cfg.adverse_horizon
    slope = 0.01
    path = [100.0 + slope * i for i in range(n + 1)]
    u_bid = [0.0] * n           # 0 < p_bid always: the bid fills every step
    u_ask = [1.0] * n           # 1 > p_ask always: the ask never fills
    skew = [0.0] * n
    half = [0.5] * n
    out = _run_episode(skew, half, path, u_bid, u_ask, cfg, keep_path=False)
    fills, adverse, adverse_fills = out[4], out[5], out[6]
    assert fills == n
    # Complete windows exist for fill steps i with i + h <= n: steps 0..n-h.
    assert adverse_fills == n - h + 1
    assert adverse == pytest.approx((n - h + 1) * h * slope)
    # The per-fill proxy average is then exactly the h-step drift.
    assert adverse / adverse_fills == pytest.approx(h * slope)
