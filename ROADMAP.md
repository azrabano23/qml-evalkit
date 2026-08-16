# Roadmap

Mirrors the Unitary Foundation microgrant proposal timeline (3–4 months).

## Month 1 — Core + Qiskit
- [x] Port and validate the evalkit stats core (intervals, McNemar, paired bootstrap)
- [x] Multi-seed variance reports (mean ± std ± SEM, spread guardrail)
- [x] Monte Carlo power checks (`power_paired`, `min_detectable_diff`)
- [x] Reproducibility manifest
- [ ] Qiskit adapter: wrap a `(circuit, dataset, optimizer)` experiment; run it
      across a seed grid; return correctness arrays ready for the stats core

## Month 2 — PennyLane + backend variance
- [ ] PennyLane adapter (same experiment interface)
- [ ] Backend-variance module: same experiment across simulators
      (statevector, density-matrix, noise-model) and real NISQ hardware;
      variance decomposition seed × backend
- [ ] Real-hardware runs (IBM Quantum / Braket — grant-funded cloud credits)

## Month 3 — Case study + release
- [ ] AML case-study notebook: [arXiv:2601.18710](https://arxiv.org/abs/2601.18710)
      re-evaluated naive vs rigorous
- [ ] Docs site, tutorial
- [ ] v0.1 on PyPI

## Month 4 — Community (buffer)
- [ ] Feedback pass, issue triage
- [ ] Present at a community call / QOSF-adjacent venue
