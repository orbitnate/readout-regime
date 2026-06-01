# 03 · K-target capacity (within-decision winner-take-all)

**Paper:** §5, Appendix B.7 · **Tier:** ✅ Runnable (single 7B, forward-only)

Co-installs `K` target rows at one final-residual position and asks how many can
*co-win* the argmax. The result: the readout is **winner-take-all**. At K=2 at least
one of the two installed targets reliably wins, but both seldom do; K≥3 collapses.
Each run is scored against a norm-matched random-row control.

## Run

```bash
# Qwen2.5-7B (α = 64)
python pvra_k_packet_probe.py \
  --model Qwen/Qwen2.5-7B-Instruct \
  --validation_json data/pvra_validation_qwen.json \
  --chains_path data/chains_natural_n5000.json \
  --alpha 64 --Ks 2,4,6 --n_random 200 --seed 0 \
  --out results/pvra_k_packet_qwen.json

# Mistral-7B-v0.3 (α = 24).  Llama uses its own calibrated α — see the
# config block of the committed results/pvra_k_packet_llama.json.
python pvra_k_packet_probe.py \
  --model mistralai/Mistral-7B-Instruct-v0.3 \
  --validation_json data/pvra_validation_mistral.json \
  --chains_path data/chains_natural_n5000.json \
  --alpha 24 --Ks 2,4,6 --n_random 200 --seed 0 \
  --out results/pvra_k_packet_mistral.json
```

## Expected (committed artifacts under `results/`)

| Arch | ≥1-of-2 wins | strict both-win | random control | K≥3 |
|---|---|---|---|---|
| Llama-3.1-8B | 14/15 | 3/15 | 0/1600 | collapses |
| Mistral-7B-v0.3 | 15/15 | 2/15 | 0/1600 | collapses |
| Qwen2.5-7B | 11/15 | 1/15 | 0/1600 | collapses |

**Reading (from the paper):** K=2 “passes” only at the ≥1-of-2 criterion — that is
*chain-competition dominance* (one of the two reliably wins), **not** additive
co-installation. The clean single-decision argmax ceiling is ≈2.
