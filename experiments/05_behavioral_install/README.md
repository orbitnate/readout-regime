# 05 · Behavioral install — a final-residual “tip”

**Paper:** §7 (Tier A), C2 · **Tier:** 📦 Archived + reference script

A worked example of the one job the readout regime is *right* for: tipping a
propensity. A **pure-geometric** install (compute a per-prompt scalar `c` at the
gate layer, then add `c·A` at the **final-norm pre-hook** — a genuine final-residual,
Tier-A install) shifts a refusal decision cleanly, cross-architecture, with no
leakage onto benign prompts, and scales past the mid-layer “coherence wall.”

## File

- `behavioral_install_pure_geometric_probe.py` — the **authentic** probe. It imports
  `pitwm.dispatcher` (`UniversalDispatcher`, the gate hook), which is not vendored
  here, so it is included as a reference for the exact mechanism; the committed JSONs
  are the runs reported in the paper.

## Committed artifacts (`results/`)

| Model | mult | Δlogp(target) | trigger flip | benign / HN leakage |
|---|---|---|---|---|
| Llama-3.1-8B | 4 | **+24.61** nats | 100% | 0/20 |
| Mistral-7B-v0.3 | 6 | **+20.40** nats | 100% (n=20) | 0/20 |
| Qwen2.5-7B | 0.5 | **+17.25** nats | 100% (n=5 smoke) | 0 |

**Caveats (from the source notes):** leakage is measured on training-set
benigns/hard-negatives; the install is at the first generated position (full-sequence
generation untested). This is a *tip*, not an override — see C1/C2 in the paper.
