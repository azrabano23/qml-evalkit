"""Multi-seed variance reports — mean ± std ± SEM, per method.

The single most common failure in QML papers is reporting one seed. Variational
models are optimization-path-sensitive: the same circuit on the same data can
land 10+ accuracy points apart across seeds, optimizers, and software versions,
while classical baselines barely move. A result is the *distribution*, not the
best draw.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class SeedSummary:
    name: str
    accuracies: list[float]
    mean: float
    std: float
    sem: float

    def __str__(self) -> str:
        return (
            f"{self.name}: {self.mean:.3f} ± {self.std:.3f} (std) "
            f"± {self.sem:.3f} (SEM), n={len(self.accuracies)} seeds, "
            f"range [{min(self.accuracies):.3f}, {max(self.accuracies):.3f}]"
        )


@dataclass
class SeedReport:
    summaries: list[SeedSummary] = field(default_factory=list)

    def __str__(self) -> str:
        lines = [str(s) for s in self.summaries]
        spread = max(self.summaries, key=lambda s: s.std)
        stable = min(self.summaries, key=lambda s: s.std)
        if spread.std > 3 * max(stable.std, 1e-9):
            lines.append(
                f"⚠ {spread.name} is {spread.std / max(stable.std, 1e-9):.1f}x more "
                f"seed-sensitive than {stable.name} — report the distribution, "
                f"not the best seed."
            )
        return "\n".join(lines)


def seed_report(results: dict[str, list[float]]) -> SeedReport:
    """Summarize per-seed accuracies for each method.

    `results` maps method name -> list of accuracies, one per seed. Two seeds is
    the minimum for a std; fewer raises, because a single seed is exactly the
    practice this package exists to end.
    """
    summaries = []
    for name, accs in results.items():
        arr = np.asarray(accs, dtype=float)
        if arr.size < 2:
            raise ValueError(
                f"{name}: need >= 2 seeds to estimate variance (got {arr.size})"
            )
        summaries.append(
            SeedSummary(
                name=name,
                accuracies=[float(a) for a in arr],
                mean=float(arr.mean()),
                std=float(arr.std(ddof=1)),
                sem=float(arr.std(ddof=1) / np.sqrt(arr.size)),
            )
        )
    return SeedReport(summaries)
