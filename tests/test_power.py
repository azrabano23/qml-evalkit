"""Power estimates must behave like power: alpha under H0, rising with n."""

import math

import pytest

from qml_evalkit import min_detectable_diff, power_paired


def test_null_rejection_rate_near_alpha():
    # With no true difference, rejections should track alpha (exact McNemar is
    # conservative, so allow the low side).
    p = power_paired(n=100, acc_a=0.7, acc_b=0.7, n_sim=1000, seed=3)
    assert p <= 0.08


def test_power_increases_with_n_and_effect():
    small_n = power_paired(n=50, acc_a=0.85, acc_b=0.70, n_sim=500, seed=4)
    large_n = power_paired(n=400, acc_a=0.85, acc_b=0.70, n_sim=500, seed=4)
    assert large_n > small_n
    assert large_n > 0.9

    small_fx = power_paired(n=100, acc_a=0.72, acc_b=0.70, n_sim=500, seed=5)
    large_fx = power_paired(n=100, acc_a=0.90, acc_b=0.70, n_sim=500, seed=5)
    assert large_fx > small_fx


def test_mdd_shrinks_with_more_data():
    mdd_50 = min_detectable_diff(n=50, base_acc=0.70, n_sim=300, seed=6)
    mdd_400 = min_detectable_diff(n=400, base_acc=0.70, n_sim=300, seed=6)
    assert not math.isnan(mdd_50)
    assert mdd_400 < mdd_50
    # The demo's headline: a 50-item test set cannot resolve a ~6-point gap.
    assert mdd_50 > 0.06


def test_input_validation():
    with pytest.raises(ValueError):
        power_paired(n=0, acc_a=0.5, acc_b=0.5)
    with pytest.raises(ValueError):
        power_paired(n=10, acc_a=1.5, acc_b=0.5)
