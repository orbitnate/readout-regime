# 01 · Closed-form readout transform (T1)

**Paper:** §3, Appendix B.4 · **Tier:** ✅ Runnable (single 7B, forward-only)

Verifies that a final-residual install `δ = α·U[t]` induces an *exactly* predictable
logit bias under RMSNorm: a rank-1 term along the fixed direction `u_t = U(γ⊙U[t])`
plus a scalar temperature `s = r/r′`. The script captures the pre-norm residual `h`
by a forward hook, forms the bias two ways — **direct** (two forward passes,
`logits(h+δ) − logits(h)`) and **closed-form** (the formula) — and reports their max
difference over the vocabulary.

## Run

```bash
# Qwen2.5-7B (α = 64)
python verify_closed_form_induced_bias.py \
  --model Qwen/Qwen2.5-7B-Instruct \
  --validation_json data/pvra_validation_qwen.json \
  --chains_path data/chains_natural_n5000.json \
  --alpha 64 --out results/closed_form_induced_bias_qwen.json

# Mistral-7B-v0.3 (α = 24)
python verify_closed_form_induced_bias.py \
  --model mistralai/Mistral-7B-Instruct-v0.3 \
  --validation_json data/pvra_validation_mistral.json \
  --chains_path data/chains_natural_n5000.json \
  --alpha 24 --out results/closed_form_induced_bias_mistral.json
```

## Expected (committed artifacts under `results/`)

| Model | α | `max│direct − closed│` | rank-1 share of ‖b‖ | argmax preserved by rank-1 term |
|---|---|---|---|---|
| Qwen2.5-7B | 64 | **7.15e-6** | **96.83%** | 6/6 chains |
| Mistral-7B-v0.3 | 24 | **4.77e-6** | **96.1%** | 6/6 chains |

The residual is at fp32 round-off — the transform is an identity, not a fit. For a
model-free, machine-precision (`~1e-15`) version of the same check, see
[`../../tests/test_t1_closed_form.py`](../../tests/test_t1_closed_form.py).

`data/` holds the six natural-language chains and the per-chain validation metadata
(target token ids) used by the run.
