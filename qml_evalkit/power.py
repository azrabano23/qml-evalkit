"""Small-sample power checks — can your test set resolve the effect you claim?

Before celebrating a 5-point quantum advantage on 50 test samples, ask whether
that test set could have detected a 5-point difference at all. Power is
estimated by seeded Monte Carlo over paired Bernoulli outcomes with an explicit
between-model correlation, judged by exact McNemar — the same test the final
comparison should use. No closed form is pretended where none is honest.
"""

from __future__ import annotations

import numpy as np

from .stats import mcnemar_exact


def power_paired(
    n: int,
    acc_a: float,
    acc_b: float,
    corr: float = 0.5,
    alpha: float = 0.05,
    n_sim: int = 2000,
    seed: int = 0,
) -> float:
    """Estimated power of exact McNemar to detect acc_a vs acc_b at test size n.

    `corr` is the correlation between the two models' per-item correctness
    (models that find the same items hard have high corr, which *increases*
    power by shrinking discordance noise). 0.5 is a reasonable default for two
    competent models on one dataset; pass your measured value when you have one.
    """
    if not 0 < n:
        raise ValueError("n must be positive")
    for name, p in (("acc_a", acc_a), ("acc_b", acc_b)):
        if not 0.0 <= p <= 1.0:
            raise ValueError(f"{name} must be in [0, 1]")
    rng = np.random.default_rng(seed)
    rejections = 0
    for _ in range(n_sim):
        a, b = _paired_bernoulli(rng, n, acc_a, acc_b, corr)
        disc_b = int(np.sum(a & ~b))
        disc_c = int(np.sum(~a & b))
        if mcnemar_exact(disc_b, disc_c) < alpha:
            rejections += 1
    return rejections / n_sim


def min_detectable_diff(
    n: int,
    base_acc: float,
    power: float = 0.8,
    corr: float = 0.5,
    alpha: float = 0.05,
    n_sim: int = 1000,
    seed: int = 0,
) -> float:
    """Smallest accuracy gain over `base_acc` detectable with the given power.

    Bisects the advantage until `power_paired` reaches the target. If the
    returned value is larger than the advantage a paper claims, the experiment
    was not powered to support the claim.
    """
    lo, hi = 0.0, 1.0 - base_acc
    if power_paired(n, base_acc + hi, base_acc, corr, alpha, n_sim, seed) < power:
        return float("nan")  # even a perfect model is not reliably detectable
    for _ in range(12):
        mid = (lo + hi) / 2
        if power_paired(n, base_acc + mid, base_acc, corr, alpha, n_sim, seed) < power:
            lo = mid
        else:
            hi = mid
    return hi


def _paired_bernoulli(rng, n: int, p_a: float, p_b: float, corr: float):
    """Draw n paired correctness outcomes with target marginals and correlation.

    Gaussian-copula construction: correlated normals thresholded at each
    marginal. The realized point-biserial correlation approximates `corr`;
    exactness is not needed for a power *estimate*.
    """
    cov = np.array([[1.0, corr], [corr, 1.0]])
    z = rng.multivariate_normal([0.0, 0.0], cov, size=n)
    # Thresholds put P(z < t) = p, so "correct" = z below threshold.
    from math import sqrt

    from .stats import _erfinv

    def _ppf(p):
        # Inverse normal CDF via erfinv: Phi^{-1}(p) = sqrt(2) * erfinv(2p - 1).
        if p <= 0.0:
            return -np.inf
        if p >= 1.0:
            return np.inf
        sign = 1.0 if p >= 0.5 else -1.0
        return sign * sqrt(2) * _erfinv(abs(2 * p - 1))

    a = z[:, 0] < _ppf(p_a)
    b = z[:, 1] < _ppf(p_b)
    return a, b
