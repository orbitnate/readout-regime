# 07 · Tier-B walls (what the readout regime cannot do)

**Paper:** §7, Appendix B.6–B.8 · **Tier:** 📦 Archived (GPU / Spark)

The “compute-tasks wall” half of the Tier-B picture, supporting corollaries C1 (no
hard override) and C3 (cannot compute a hidden intermediate). Same late-layer site
as the recall positives — but here the task needs *computation or chaining*, and it
fails.

## Files (reference) and artifacts

**Single-pass multi-hop (C3).**
- `probe_install_then_remove.py`, `analyze_install_then_remove.py` — the authentic
  install-then-continue probe (no public CLI; the reported run was on a GPU pod with
  a cached first-hop delta).
- `results/tau10_alpha4_analysis.json` — Llama-3.1-8B single-pass multi-hop
  **0.183 vs a 0.40 bar** (n=372, MQuAKE-CF). *Caveat:* installs at the subject
  positions and re-encodes the bridge, so this is a **locus-confounded near-miss**,
  not a pure final-residual wall.
- `results/framing2_gate.json` — the Llama→Mistral cross-host install-then-continue
  gate, **0.120 vs a 0.70 bar** (n=50).

**Composition with a visible operand (C2 boundary).**
- `run_phase_5_72_track_a_consumer.py` — the forward-pass composition consumer
  (frozen Qwen; imports the v7 `lrm` harness).
- `results/gates_v73_t18-21_forced_choice.json` — **272/272 at the oracle /
  exact-delta upper bound**, while the **learned hidden route reaches only
  189/272 = 0.695 and fails its pre-registered gate** (it is K=4-codebook regression
  error, **not** module recovery). Controls: no-module 0.2684, wrong-delta 0.0551.

**Not in this repo (Spark-only / off-repo):** the bounded readout-wall single-hop
diagnostic (oracle-correct content moves a peaked logit only +0.016…+0.136, stays
8/32), the mid-layer CCA-substrate read (cosine 0.9918, payload-swap 0/3), and the
hidden second-hop surface (8/32, a failed positive control reported as a broken
construction, **not** as evidence). These are described in the paper; their raw
artifacts live on the original compute host.
