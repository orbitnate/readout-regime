# 04 · Basis-privilege battery + off-axis attractor

**Paper:** §6, Appendix B.9 · **Tier:** 📦 Archived + reference script

The falsification battery that isolates *which* write direction is privileged: only
the genuine `lm_head` row of the target token flips the argmax. The pre-registered
**orthogonal-A control** (project out the `lm_head[t]` component, renorm) collapses
the effect toward random-A on **4/4 architectures**, and a cosine-dosage sweep maps
the off-axis residual *attractor* (Figure 2) — on instruction-tuned models a
norm-matched random install still lifts the target logit, so the privilege is at the
**argmax-flip** level, not “a wrong direction moves the logit by zero.”

## Files

- `basis_mechanism_robustness_probe.py`, `orthogonal_a_pythia_probe.py`,
  `random_a_probe.py` — the **authentic** probe scripts. They import the `pitwm`
  dispatcher/harness (not vendored here), so treat them as reference; the committed
  artifacts are the runs reported in the paper.
- `figures/make_figures.py` — regenerates the off-axis attractor figure from the
  verified seed medians (`python figures/make_figures.py`, needs matplotlib).
  `figures/fig_attractor.{png,pdf}` are the committed renders.

## Committed artifacts (`results/`)

Orthogonal-A vs canonical at `mult=4`, the basis-privilege collapse:

| Arch | orthogonal-A Δlogp | as % of canonical |
|---|---|---|
| Pythia-1.4B | +0.16 nats | **1.2%** |
| Mistral-7B | +0.90 nats | **5.8%** |
| Llama-3.1-8B | +2.49 nats | **18%** |
| Qwen2.5-7B | +7.00 nats | **38%** |

`dose_response_pythia1_4b_n10_canonical_mult4.json` holds the canonical-direction
dosage baseline. The 0.940→0.000 paraphrased-recall collapse under random-A is the
reference expected-value used by the recall harness (see experiment 06).
