"""qml-evalkit — rigorous statistical evaluation for quantum machine learning.

Working today: the numpy-only stats core (intervals, paired tests, multi-seed
variance reports, power checks, reproducibility manifests). On the roadmap:
Qiskit and PennyLane experiment adapters and a backend-variance module — see
ROADMAP.md.
"""

from .manifest import capture, save
from .power import min_detectable_diff, power_paired
from .stats import (
    Interval,
    bootstrap_ci,
    mcnemar_exact,
    paired_diff_ci,
    wilson_ci,
)
from .variance import SeedReport, SeedSummary, seed_report

__version__ = "0.1.0.dev0"

__all__ = [
    "Interval",
    "SeedReport",
    "SeedSummary",
    "bootstrap_ci",
    "capture",
    "mcnemar_exact",
    "min_detectable_diff",
    "paired_diff_ci",
    "power_paired",
    "save",
    "seed_report",
    "wilson_ci",
    "__version__",
]
