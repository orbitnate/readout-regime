# 02 · Effective rank of the output projection

**Paper:** §6 (probe 7), §5 reconciliation · **Tier:** ✅ Runnable (loads `lm_head` only)

Measures the singular spectrum and pairwise geometry of the unembedding matrix
`lm_head.weight`. This is the “≈918” figure the paper contrasts against the ≈2
within-decision co-install ceiling: the projection has plenty of nominal room, yet
softmax winner-take-all over a near-orthogonal Gram still limits co-winnable targets
to ≈2. Computation is a sampled-row SVD — no forward pass, no GPU needed beyond
loading the weights.

## Run

```bash
python lm_head_orthogonality.py \
  --model-id Qwen/Qwen2.5-7B-Instruct \
  --sample-size 2000 --seed 0 \
  --out results/lm_head_orthogonality.json
```

> Note: the script’s default `--out` points at an absolute path from the original
> machine; pass `--out results/lm_head_orthogonality.json` as above.

## Expected (committed `results/lm_head_orthogonality.json`, Qwen2.5-7B)

| Quantity | Value |
|---|---|
| entropy effective rank | **917.83 (≈918)** |
| stable rank | **10.04** |
| participation ratio | **93.45** |
| mean \|cos\| (pairwise rows) | **0.082** (p95 0.233) |
| condition number | **2.2e6** |
