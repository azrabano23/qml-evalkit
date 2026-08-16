"""The pitch, as a program: watch a naive QML evaluation fall apart.

Four acts, pure numpy, deterministic, instant. The numbers are synthetic but
the shape is real — it is the shape of most quantum-advantage claims in the
current QML literature (single seed, ~50 test samples, no interval, no paired
test), and of the author's own first paper before reviewers caught it.

Run: python -m qml_evalkit.demo
"""

from __future__ import annotations

import numpy as np

from .manifest import capture
from .power import min_detectable_diff
from .stats import mcnemar_exact, paired_diff_ci, wilson_ci
from .variance import seed_report


def main() -> None:
    print(__doc__)

    # A synthetic-but-typical result table. On a 50-item test set, the quantum
    # model and the classical baseline agree on 37 items and disagree on 13.
    both_right, q_only, c_only, both_wrong = 30, 8, 5, 7
    q = np.array([1] * both_right + [1] * q_only + [0] * c_only + [0] * both_wrong)
    c = np.array([1] * both_right + [0] * q_only + [1] * c_only + [0] * both_wrong)
    n = q.size

    print("=" * 72)
    print("ACT 1 — The headline (how the result is usually reported)")
    print("=" * 72)
    print(f"  Quantum model:      {q.mean():.0%}  (n={n} test samples, 1 seed)")
    print(f"  Classical baseline: {c.mean():.0%}")
    print(f"  Claim: quantum advantage of +{q.mean() - c.mean():.0%}.\n")

    print("=" * 72)
    print("ACT 2 — The same numbers, with uncertainty attached")
    print("=" * 72)
    print(f"  Quantum accuracy (Wilson): {wilson_ci(int(q.sum()), n)}")
    print(f"  Gap, paired bootstrap:     {paired_diff_ci(q, c)}")
    p = mcnemar_exact(q_only, c_only)
    print(f"  Exact McNemar on the {q_only + c_only} discordant items: p = {p:.3f}")
    print(
        "  The gap's CI contains zero and the paired test is far from\n"
        "  significant: this data does not support the +6-point claim.\n"
    )

    print("=" * 72)
    print("ACT 3 — What multiple seeds reveal")
    print("=" * 72)
    # Variational models are optimization-path-sensitive; classical baselines
    # are not. Ten seeds of each (synthetic, but the spread ratio is faithful
    # to what the author measured on a real VQC-vs-CNN medical imaging task).
    report = seed_report(
        {
            "quantum (VQC)": [0.76, 0.62, 0.70, 0.58, 0.74, 0.66, 0.72, 0.60, 0.68, 0.64],
            "classical (CNN)": [0.70, 0.71, 0.69, 0.70, 0.72, 0.70, 0.69, 0.71, 0.70, 0.70],
        }
    )
    for line in str(report).splitlines():
        print(f"  {line}")
    print("  The single-seed 76% in Act 1 was the best draw, not the result.\n")

    print("=" * 72)
    print("ACT 4 — Could this experiment ever have worked?")
    print("=" * 72)
    mdd = min_detectable_diff(n=n, base_acc=0.70)
    print(
        f"  Minimum advantage detectable at 80% power with n={n}, base 70%:\n"
        f"  ~{mdd:.0%}. The experiment claimed +6% — it was never powered to\n"
        f"  support that claim. Fix: more test items, or stop claiming.\n"
    )

    print("=" * 72)
    print("Every rigorous result ships with its environment:")
    print("=" * 72)
    m = capture(seeds=[42, 7, 123], backend="statevector_simulator (demo)")
    for key in ("python", "platform", "packages", "seeds", "backend"):
        print(f"  {key}: {m.get(key)}")


if __name__ == "__main__":
    main()
