#!/usr/bin/env python3
"""Measure lm_head row orthogonality on Qwen2.5-7B-Instruct.

Why: the WRIT/PITWM lm_head-basis theorem implies the privileged write basis
is `span(lm_head)`. The capacity ceiling of that bus depends on how many
near-orthogonal directions actually exist in the lm_head row matrix. If the
rows are near-orthonormal, capacity ≈ vocab size (~152K for Qwen). If they're
not, capacity is bounded by the effective rank.

Strategy: load just the lm_head weights (~1GB at bf16) on CPU, compute
pairwise cosine on a stratified sample of vocabulary indices (the full
152K x 152K cosine matrix would be 92 GB). Report:
  - mean / median / std of |cos(lm_head[i], lm_head[j])| for i != j
  - 90th, 95th, 99th percentiles
  - effective rank via singular value spectrum of a sampled rectangular slice
  - row norm distribution (the lm_head[t] effect strength scales with its norm)
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-id", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--sample-size", type=int, default=2000,
                    help="number of vocabulary tokens to sample for cosine pairs")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path,
                    default=Path("results/lm_head_orthogonality.json"))
    args = ap.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)

    print(f"loading {args.model_id} on CPU (lm_head only)...")
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(args.model_id)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id, torch_dtype=torch.bfloat16, device_map="cpu"
    )
    print(f"  model loaded in {time.time()-t0:.1f}s")

    # lm_head is the output projection. In Qwen2.5 it's tied to embed_tokens
    # by default unless tie_word_embeddings=False.
    lm = model.lm_head.weight.detach().to(torch.float32)
    vocab_size, d_model = lm.shape
    print(f"lm_head shape: {tuple(lm.shape)} (vocab={vocab_size}, d_model={d_model})")
    print(f"tie_word_embeddings: {model.config.tie_word_embeddings}")

    # Row norms
    row_norms = lm.norm(dim=-1)
    print(f"\nrow norm distribution:")
    print(f"  min={row_norms.min():.4f}  median={row_norms.median():.4f}  max={row_norms.max():.4f}")
    print(f"  mean={row_norms.mean():.4f}  std={row_norms.std():.4f}")

    # Sampled pairwise cosine (full matrix would be ~92 GB)
    print(f"\nsampling {args.sample_size} tokens for pairwise cosine...")
    g = torch.Generator().manual_seed(args.seed)
    idx = torch.randperm(vocab_size, generator=g)[:args.sample_size]
    sub = lm[idx]  # [sample_size, d_model]
    sub_n = sub / sub.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    cos = sub_n @ sub_n.T  # [sample_size, sample_size]
    # Off-diagonal upper triangle
    iu = torch.triu_indices(args.sample_size, args.sample_size, offset=1)
    off = cos[iu[0], iu[1]]
    abs_off = off.abs()

    print(f"\npairwise |cos(row_i, row_j)| (n={off.numel()} pairs):")
    print(f"  mean   = {abs_off.mean().item():.6f}")
    print(f"  median = {abs_off.median().item():.6f}")
    print(f"  std    = {abs_off.std().item():.6f}")
    print(f"  90th   = {torch.quantile(abs_off, 0.90).item():.6f}")
    print(f"  95th   = {torch.quantile(abs_off, 0.95).item():.6f}")
    print(f"  99th   = {torch.quantile(abs_off, 0.99).item():.6f}")
    print(f"  max    = {abs_off.max().item():.6f}")

    # Signed cosine (do rows tend to be positively or negatively correlated?)
    print(f"\npairwise cos (signed):")
    print(f"  mean   = {off.mean().item():.6f}  (0 = isotropic, !=0 = directional bias)")
    print(f"  median = {off.median().item():.6f}")

    # Effective rank of the sample (singular value spectrum)
    print("\ncomputing singular values of sampled lm_head slice...")
    t0 = time.time()
    # SVD of [sample_size x d_model] is fast (sample_size << d_model? no, larger usually)
    # Use torch.linalg.svdvals
    s = torch.linalg.svdvals(sub)
    s2 = s ** 2
    s2_sum = s2.sum()
    p = s2 / s2_sum
    # Participation ratio (effective dimension)
    pr = (s2.sum() ** 2) / (s2 ** 2).sum()
    # Stable rank
    sr = (s.sum() ** 2) / (s ** 2).sum() if False else (s2.sum() / (s.max() ** 2))
    # Entropy-based effective rank
    p_safe = p.clamp_min(1e-30)
    H = -(p_safe * p_safe.log()).sum()
    eff_rank = torch.exp(H).item()
    print(f"  SVD time: {time.time()-t0:.1f}s")
    print(f"  d_model       = {d_model}")
    print(f"  sample_size   = {args.sample_size}")
    print(f"  participation ratio = {pr.item():.1f}")
    print(f"  stable rank         = {sr.item():.1f}  (||S||_F^2 / ||S||_op^2)")
    print(f"  entropy eff rank    = {eff_rank:.1f}  (exp(H[p_i = s_i^2/sum]))")
    print(f"  s[0] (largest)      = {s[0].item():.4f}")
    print(f"  s[d_model-1] (smallest) = {s[-1].item():.4f}")
    print(f"  condition number    = {(s[0]/s[-1].clamp_min(1e-12)).item():.2e}")

    # Save full payload
    out_payload = {
        "model_id": args.model_id,
        "vocab_size": int(vocab_size),
        "d_model": int(d_model),
        "tie_word_embeddings": bool(model.config.tie_word_embeddings),
        "sample_size": int(args.sample_size),
        "seed": int(args.seed),
        "row_norms": {
            "min": float(row_norms.min()),
            "median": float(row_norms.median()),
            "max": float(row_norms.max()),
            "mean": float(row_norms.mean()),
            "std": float(row_norms.std()),
        },
        "pairwise_abs_cos": {
            "n_pairs": int(off.numel()),
            "mean": float(abs_off.mean()),
            "median": float(abs_off.median()),
            "std": float(abs_off.std()),
            "p90": float(torch.quantile(abs_off, 0.90)),
            "p95": float(torch.quantile(abs_off, 0.95)),
            "p99": float(torch.quantile(abs_off, 0.99)),
            "max": float(abs_off.max()),
        },
        "pairwise_signed_cos": {
            "mean": float(off.mean()),
            "median": float(off.median()),
        },
        "spectral": {
            "participation_ratio": float(pr),
            "stable_rank": float(sr),
            "entropy_effective_rank": float(eff_rank),
            "s_max": float(s[0]),
            "s_min": float(s[-1]),
            "condition_number": float(s[0] / s[-1].clamp_min(1e-12)),
        },
    }
    args.out.write_text(json.dumps(out_payload, indent=2))
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
