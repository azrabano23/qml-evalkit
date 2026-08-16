# qml-evalkit

**Rigorous statistical evaluation for quantum machine learning experiments.**

> **Status: early-stage.** The numpy-only stats core below works and is tested.
> The Qiskit/PennyLane adapters and real-hardware backend-variance module are
> the current build (see [ROADMAP.md](ROADMAP.md)); this repo accompanies a
> Unitary Foundation microgrant proposal to fund exactly that work.

## Why

The typical QML result is a single-seed accuracy on a ~50-sample test set: no
confidence interval, no significance test against the classical baseline, no
variance across seeds or backends. At that sample size, a 5-point "quantum
advantage" is usually statistically indistinguishable from zero — and
variational models are so optimization-path-sensitive that the same circuit can
land 10+ points apart across seeds, framework versions, and machines.

I know because it happened to me. In my own quantum-vs-classical comparison
(VQC and Equilibrium Propagation vs CNN on AML blood-cell classification,
[arXiv:2601.18710](https://arxiv.org/abs/2601.18710)), the VQC's accuracy
shifted by more than 10 points across machines and Qiskit versions while the
CNN barely moved — and reviewers correctly noted that 3 seeds with no mean/std
proved nothing. qml-evalkit is the tool that would have caught all of it before
submission.

Mitiq raised the floor for error mitigation; Metriq for benchmark transparency.
This aims to do the same for empirical claims in QML.

## What works today

A pure-numpy stats core, ported from my LLM-eval toolkit
[evalkit](https://github.com/azrabano23/evalkit) and validated against ground
truth (known McNemar values, calibrated bootstrap coverage):

- **`bootstrap_ci` / `wilson_ci`** — accuracy intervals honest at small n and
  extreme rates.
- **`paired_diff_ci` / `mcnemar_exact`** — the correct way to compare a quantum
  model and a classical baseline *on the same test items*. If the gap's CI
  contains zero, the advantage claim is not supported.
- **`seed_report`** — mean ± std ± SEM per method across seeds, with a
  guardrail that flags optimization-path-sensitive models (and refuses
  single-seed input outright).
- **`power_paired` / `min_detectable_diff`** — seeded Monte Carlo power checks:
  could your test set have detected the effect you claim, at all?
- **`manifest.capture`** — a reproducibility manifest (Python, platform,
  framework versions, seeds, backend, git commit) to emit alongside every
  results file.

## Quickstart

```bash
pip install numpy && pip install -e .
python -m qml_evalkit.demo   # 4-act demo: watch a naive QML evaluation fall apart
./run.sh test                # validate the stats against ground truth
```

```python
import qml_evalkit as qe

# q, c: 0/1 correctness of the quantum model and classical baseline on the SAME items
print(qe.paired_diff_ci(q, c))          # gap with CI — does it contain zero?
print(qe.mcnemar_exact(b=8, c=5))       # exact p on discordant items
print(qe.seed_report({"VQC": vqc_accs, "CNN": cnn_accs}))
print(qe.min_detectable_diff(n=50, base_acc=0.70))  # what n=50 can even resolve
qe.save(qe.capture(seeds=[42, 7], backend="aer statevector"), "manifest.json")
```

## Case study (in progress)

The flagship worked example will be my own AML comparison
([quantum-blood-cell-classification](https://github.com/azrabano23/quantum-blood-cell-classification))
re-evaluated with this toolkit: the same experiment, naive vs rigorous, showing
how the naive version overstates the quantum result.

## Honesty

- These tools attach uncertainty to numbers; they cannot rescue a bad
  experimental design, a leaky split, or a contaminated dataset.
- `power_paired` is a Monte Carlo estimate under a Gaussian-copula pairing
  model — the correlation knob is explicit and yours to justify.
- Nothing here requires quantum hardware; that is the point (the stats are
  framework-agnostic) and the limitation (the backend-variance module, which
  needs real NISQ runs, is roadmap, not code).

MIT license.
