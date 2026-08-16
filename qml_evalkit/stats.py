"""The statistics engine — qml-evalkit's scientific core.

A benchmark number without an interval is a rumor, and nowhere more so than in
quantum ML, where the typical result is a single-seed accuracy on a ~50-sample
test set. Everything here turns raw correctness arrays into claims you can
defend: confidence intervals, and paired significance tests for "the quantum
model beats the classical baseline". Pure numpy, deterministic (seeded).

Ported from evalkit (github.com/azrabano23/evalkit), where the same functions
are validated against analytic ground truth (calibrated CI coverage, known
McNemar values); the port re-runs those checks in `tests/`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass
class Interval:
    point: float
    low: float
    high: float
    confidence: float = 0.95

    def __str__(self) -> str:
        pct = int(self.confidence * 100)
        return f"{self.point:.4f} [{self.low:.4f}, {self.high:.4f}] ({pct}% CI)"

    def contains(self, value: float) -> bool:
        return self.low <= value <= self.high


def bootstrap_ci(
    values,
    statistic=np.mean,
    n_boot: int = 10000,
    confidence: float = 0.95,
    seed: int = 0,
) -> Interval:
    """Percentile bootstrap CI for any statistic of a 1-D sample.

    The default `statistic=mean` over a 0/1 array gives an accuracy CI without
    assuming normality — appropriate for the small, skewed n of most QML test
    sets.
    """
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        raise ValueError("cannot bootstrap an empty sample")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, arr.size, size=(n_boot, arr.size))
    boot = statistic(arr[idx], axis=1)
    alpha = 1.0 - confidence
    lo, hi = np.quantile(boot, [alpha / 2, 1 - alpha / 2])
    return Interval(float(statistic(arr)), float(lo), float(hi), confidence)


def wilson_ci(correct: int, n: int, confidence: float = 0.95) -> Interval:
    """Wilson score interval for a proportion — well-behaved at extreme rates.

    A closed-form sanity check on the bootstrap (and what you want when accuracy
    is 0% or 100% and the bootstrap CI collapses to a point).
    """
    if n == 0:
        raise ValueError("n must be > 0")
    # Two-sided normal quantile for the requested confidence.
    z = math.sqrt(2) * _erfinv(confidence)
    p = correct / n
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    half = (z / denom) * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2))
    return Interval(p, max(0.0, center - half), min(1.0, center + half), confidence)


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact McNemar p-value on discordant counts.

    `b` = items model A got right and B got wrong; `c` = A wrong, B right.
    Concordant items carry no information about a *difference*, so the test
    conditions on the n=b+c discordant pairs, each ~Bernoulli(0.5) under H0.
    This is the correct significance test for comparing a quantum model and a
    classical baseline on the *same* test items.
    """
    n = b + c
    if n == 0:
        return 1.0
    m = min(b, c)
    tail = sum(math.comb(n, i) for i in range(m + 1)) * (0.5**n)
    return float(min(1.0, 2.0 * tail))


def paired_diff_ci(
    correct_a, correct_b, n_boot: int = 10000, confidence: float = 0.95, seed: int = 0
) -> Interval:
    """Bootstrap CI for the accuracy *difference* (A - B) on paired items.

    Resamples item indices jointly so the pairing (and its correlation) is
    preserved — the whole point of evaluating both models on the same set. If
    this interval contains zero, the claimed quantum advantage is not resolved
    by the data.
    """
    a = np.asarray(correct_a, dtype=float)
    b = np.asarray(correct_b, dtype=float)
    if a.shape != b.shape:
        raise ValueError("A and B must be evaluated on the same items (equal length)")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, a.size, size=(n_boot, a.size))
    diff = a[idx].mean(axis=1) - b[idx].mean(axis=1)
    alpha = 1.0 - confidence
    lo, hi = np.quantile(diff, [alpha / 2, 1 - alpha / 2])
    return Interval(float(a.mean() - b.mean()), float(lo), float(hi), confidence)


def _erfinv(c: float) -> float:
    """Inverse error function at the two-sided quantile, via bisection (no scipy)."""
    # We want z such that erf(z/sqrt2) = c, i.e. erf(x)=c with x=z/sqrt2.
    target = c
    lo, hi = 0.0, 5.0
    for _ in range(100):
        mid = (lo + hi) / 2
        if math.erf(mid) < target:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2
