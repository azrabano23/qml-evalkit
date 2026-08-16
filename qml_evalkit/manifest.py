"""Reproducibility manifest — capture the environment your numbers came from.

QML results are platform-sensitive: floating-point backends, framework
versions, and optimizer paths all move variational-model accuracy. A result
without its environment is unreproducible by construction. `capture()` records
what a reader needs to rerun the experiment; emit it alongside every results
file.
"""

from __future__ import annotations

import json
import platform
import subprocess
import sys
from importlib import metadata

# Packages whose versions move QML results when they change.
_TRACKED = (
    "numpy",
    "scipy",
    "qiskit",
    "qiskit-aer",
    "qiskit-machine-learning",
    "pennylane",
    "torch",
    "scikit-learn",
)


def capture(seeds=None, backend: str | None = None, extra: dict | None = None) -> dict:
    """Snapshot python/platform/package versions plus experiment identifiers.

    `seeds` and `backend` are the experiment's own knobs — passing them here
    keeps the manifest the single place a reader looks to rerun the result.
    """
    packages = {}
    for name in _TRACKED:
        try:
            packages[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            continue
    manifest = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "machine": platform.machine(),
        "packages": packages,
        "git_commit": _git_commit(),
    }
    if seeds is not None:
        manifest["seeds"] = list(seeds)
    if backend is not None:
        manifest["backend"] = backend
    if extra:
        manifest["extra"] = extra
    return manifest


def save(manifest: dict, path: str) -> None:
    with open(path, "w") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
        f.write("\n")


def _git_commit() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    return out.stdout.strip() if out.returncode == 0 else None
