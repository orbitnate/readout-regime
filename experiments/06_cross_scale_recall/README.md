# 06 · Cross-scale single-token recall (bias suffices)

**Paper:** §7 (Tier B, late-layer positive) · **Tier:** 📦 Archived (GPU; 72B multi-GPU)

When the base prior is flat, a single `lm_head`-row install at a late layer
(L27 Qwen / L30 Llama·Mistral) is enough to recall the right token — across **6
models from 1.5B to 72B and 3 families**. This is the “bias-only tasks succeed”
half of the Tier-B picture: no computation is required, so the readout-regime bias
is sufficient, at every scale. The frozen-model no-harm sweep confirms installs do
not damage general capability.

## Files (reference)

- `expH_7b_transfer.py` — the recall/scoring harness (imports the `pitwm` package).
- `run_scale_test.sh` — the scale-sweep driver:
  `bash run_scale_test.sh Qwen/Qwen2.5-72B-Instruct 72b` (auto-detects layers).

Reproducing the large checkpoints needs real GPU (72B needs multi-GPU); the
committed artifacts are the original runs.

## Committed artifacts (`results/`)

| Artifact | Model | oracle ρ | full ρ |
|---|---|---|---|
| `task6_headline.json` | Qwen2.5-7B | 1.000 | 0.940 |
| `task1a_llama_headline.json` | Llama-3.1-8B | 1.000 | 0.940 |
| `task1a2_mistral_headline.json` | Mistral-7B-v0.3 | 1.000 | 0.947 |
| `task1c_14b_headline_v2.json` | Qwen2.5-14B | 1.000 | (see file) |
| `task1c_72b_headline.json` | Qwen2.5-72B | 1.000 | (see file) |
| `no_harm_7b_N100.json` | Qwen2.5-7B | — | frozen-model no-harm sweep (MMLU/GSM8K/HumanEval): no spurious activations |

The gap from full ρ to oracle ρ is a **routing** gap (which module to fire), not a
content gap — see the paper’s §7. The `task1c_14b_headline_v2.json` is the canonical
14B run (an earlier v1 sweep was weaker; use v2).
