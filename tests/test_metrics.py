"""Tests for the paired-difference inference in mmsim.metrics."""

from types import SimpleNamespace

import numpy as np
import pytest

from mmsim.metrics import (
    _t_ppf,
    bootstrap_std_reduction_ci,
    comparison_table,
    paired_comparison,
    paired_comparison_block,
    paired_pnl_difference,
)


def test_t_quantile_matches_known_critical_values():
    """Spot-check the scipy-free t quantile against standard table values."""
    assert _t_ppf(0.975, 10) == pytest.approx(2.228139, abs=1e-4)
    assert _t_ppf(0.975, 30) == pytest.approx(2.042272, abs=1e-4)
    assert _t_ppf(0.975, 999) == pytest.approx(1.962341, abs=1e-4)
    assert _t_ppf(0.995, 20) == pytest.approx(2.845340, abs=1e-4)
    assert _t_ppf(0.025, 10) == pytest.approx(-2.228139, abs=1e-4)
    assert _t_ppf(0.5, 7) == 0.0


def test_paired_difference_recovers_known_effect():
    """Synthetic paired data with a known mean shift: the CI must contain the
    true effect, exclude zero, and equal mean +/- t * SE exactly."""
    rng = np.random.default_rng(0)
    n = 800
    common = rng.normal(0.0, 5.0, n)          # large shared paired noise
    true_effect = 1.5
    a = common + true_effect + rng.normal(0.0, 0.5, n)
    b = common
    mean, se, (lo, hi) = paired_pnl_difference(a, b)
    assert mean == pytest.approx(true_effect, abs=0.1)
    # The common component cancels in the pairing: SE ~ 0.5 / sqrt(n), far
    # smaller than an unpaired analysis (~ 5 * sqrt(2/n)) would give.
    assert se == pytest.approx(0.5 / np.sqrt(n), rel=0.15)
    assert lo < true_effect < hi
    assert lo > 0.0
    tcrit = _t_ppf(0.975, n - 1)
    assert lo == pytest.approx(mean - tcrit * se, abs=1e-12)
    assert hi == pytest.approx(mean + tcrit * se, abs=1e-12)


def test_paired_difference_rejects_mismatched_lengths():
    with pytest.raises(ValueError):
        paired_pnl_difference(np.zeros(10), np.zeros(11))


def test_bootstrap_ci_deterministic_given_seed():
    """Same seeded RNG -> bit-identical CI; different seed -> different CI."""
    rng = np.random.default_rng(0)
    a = rng.normal(0.0, 1.0, 300)
    b = rng.normal(0.0, 2.0, 300)
    ci1 = bootstrap_std_reduction_ci(a, b, rng=np.random.default_rng(42), n_resamples=2000)
    ci2 = bootstrap_std_reduction_ci(a, b, rng=np.random.default_rng(42), n_resamples=2000)
    assert ci1 == ci2
    ci3 = bootstrap_std_reduction_ci(a, b, rng=np.random.default_rng(43), n_resamples=2000)
    assert ci1 != ci3


def test_bootstrap_ci_covers_known_std_reduction():
    """std_a = 1, std_b = 2: the true reduction 0.5 must sit inside a
    reasonably tight percentile interval."""
    rng = np.random.default_rng(7)
    n = 2000
    a = rng.normal(0.0, 1.0, n)
    b = rng.normal(0.0, 2.0, n)
    lo, hi = bootstrap_std_reduction_ci(
        a, b, rng=np.random.default_rng(11), n_resamples=3000
    )
    assert lo < 0.5 < hi
    assert lo > 0.0
    assert hi - lo < 0.2


def test_paired_comparison_bundles_consistent_numbers():
    rng = np.random.default_rng(3)
    n = 500
    common = rng.normal(0.0, 3.0, n)
    pnl_a = common + rng.normal(-0.5, 1.0, n)
    pnl_b = 2.0 * common + rng.normal(0.0, 1.0, n)
    run_a = SimpleNamespace(name="A", pnl=pnl_a)
    run_b = SimpleNamespace(name="B", pnl=pnl_b)
    pc = paired_comparison(run_a, run_b, rng=np.random.default_rng(9), n_resamples=1000)
    mean, se, (lo, hi) = paired_pnl_difference(pnl_a, pnl_b)
    assert pc.n_episodes == n
    assert pc.mean_diff == pytest.approx(mean)
    assert pc.se_diff == pytest.approx(se)
    assert (pc.ci_low, pc.ci_high) == pytest.approx((lo, hi))
    expected_reduction = 1.0 - pnl_a.std(ddof=1) / pnl_b.std(ddof=1)
    assert pc.std_reduction == pytest.approx(expected_reduction)
    assert pc.std_reduction_ci_low < pc.std_reduction < pc.std_reduction_ci_high
    block = paired_comparison_block(pc)
    assert "A minus B" in block
    assert "t-based" in block


def test_table_labels_mean_over_std_not_sharpe():
    """Episode P&Ls are not a returns series; the ratio row must not claim to
    be a Sharpe ratio."""
    from mmsim.simulate import SimConfig, run_paired
    from mmsim.metrics import summarize

    result = run_paired(SimConfig(episodes=10, seed=5))
    m_as = summarize(result.avellaneda_stoikov)
    m_nv = summarize(result.naive)
    table = comparison_table(m_as, m_nv)
    assert "Mean/std of terminal P&L" in table
    assert "Sharpe" not in table
    assert m_as.mean_over_std == pytest.approx(m_as.mean_pnl / m_as.std_pnl)
