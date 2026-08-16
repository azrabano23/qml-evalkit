"""Validation against ground truth — same discipline as evalkit's test suite."""

import numpy as np
import pytest

from qml_evalkit import (
    bootstrap_ci,
    mcnemar_exact,
    paired_diff_ci,
    seed_report,
    wilson_ci,
)


def test_mcnemar_known_value():
    # n=6 discordant, min side 1: p = 2 * (C(6,0)+C(6,1)) / 2^6 = 2 * 7/64.
    assert mcnemar_exact(5, 1) == pytest.approx(14 / 64)
    # No discordance carries no evidence of a difference.
    assert mcnemar_exact(0, 0) == 1.0
    # Symmetry.
    assert mcnemar_exact(3, 9) == mcnemar_exact(9, 3)


def test_wilson_matches_hand_calc():
    ci = wilson_ci(50, 100)
    assert ci.point == pytest.approx(0.5)
    # Wilson at p=0.5 is symmetric about 0.5.
    assert (0.5 - ci.low) == pytest.approx(ci.high - 0.5, abs=1e-12)
    # Known value for z=1.96-ish: half-width ~0.096 at n=100, p=0.5.
    assert ci.high - ci.low == pytest.approx(0.192, abs=0.005)


def test_bootstrap_ci_coverage_calibrated():
    # 95% CI should cover the true mean ~95% of the time. Loose gate to keep
    # the test fast and non-flaky.
    rng = np.random.default_rng(1)
    true_p, n, trials = 0.6, 100, 200
    covered = 0
    for i in range(trials):
        sample = (rng.random(n) < true_p).astype(float)
        ci = bootstrap_ci(sample, n_boot=1000, seed=i)
        covered += ci.low <= true_p <= ci.high
    assert 0.88 <= covered / trials <= 0.99


def test_paired_diff_ci_detects_zero_and_real_gaps():
    rng = np.random.default_rng(2)
    same = (rng.random(200) < 0.7).astype(float)
    assert paired_diff_ci(same, same).contains(0.0)

    a = np.ones(200)
    b = np.zeros(200)
    ci = paired_diff_ci(a, b)
    assert ci.point == pytest.approx(1.0)
    assert not ci.contains(0.0)


def test_paired_diff_requires_same_items():
    with pytest.raises(ValueError):
        paired_diff_ci([1, 0, 1], [1, 0])


def test_seed_report_stats_and_guardrail():
    report = seed_report({"q": [0.6, 0.7, 0.8], "c": [0.70, 0.71, 0.72]})
    q = next(s for s in report.summaries if s.name == "q")
    assert q.mean == pytest.approx(0.7)
    assert q.std == pytest.approx(np.std([0.6, 0.7, 0.8], ddof=1))
    assert q.sem == pytest.approx(q.std / np.sqrt(3))
    # One seed is exactly the malpractice this package exists to end.
    with pytest.raises(ValueError):
        seed_report({"q": [0.99]})
