#!/usr/bin/env python3
"""Verify the closed-form induced-bias decomposition under RMSNorm.

Claim: for any α, h, t, the induced bias from residual install
δ = α·W[t] at pre-final-norm position decomposes EXACTLY as:

    b_induced  =  (α/r') · (W · (γ ⊙ W[t]))                         [rank-1 along γ-weighted W[t]]
                + ((r − r')/(r·r')) · (W · (γ ⊙ h))                 [global rescaling of pre-norm-baseline logits]

where r = √(mean(h²) + ε), r' = √(mean((h+δ)²) + ε).

Equivalent form using logits_baseline = W·LN(h) = W·γ⊙h / r:

    b_induced  =  (α/r') · u_t                                      [rank-1]
                + ((r − r')/r') · logits_baseline                   [scalar multiplier on logits_baseline]

Equivalently:

    b_induced  =  logits_baseline_scaled_diff  +  rank1_install_scaled

Verify on Mistral-7B-Instruct against the per-chain h captured from a
forward pass. The two terms should sum to the directly-computed
b_induced = W·LN(h+δ) − W·LN(h) at fp tolerance.

Usage:
  python scripts/lm_head_basis_probes/verify_closed_form_induced_bias.py \\
      --model mistralai/Mistral-7B-Instruct-v0.3 \\
      --validation_json results/lm_head_basis_probes/pvra_validation_mistral.json \\
      --chains_path data/expN/chains_natural_n5000.json \\
      --alpha 24 \\
      --out results/lm_head_basis_probes/closed_form_induced_bias_mistral.json
"""
from __future__ import annotations

import argparse, json, sys
from pathlib import Path
from typing import Any, Dict, List

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def _get_lm_head(m):
    return m.lm_head.weight.detach() if not hasattr(m, "embed_out") else m.embed_out.weight.detach()


def _get_final_norm(m):
    return m.model.norm if not hasattr(m, "gpt_neox") else m.gpt_neox.final_layer_norm


def _rms_eps(fn):
    if hasattr(fn, "variance_epsilon"):
        return float(fn.variance_epsilon)
    if hasattr(fn, "eps"):
        return float(fn.eps)
    return 1e-6


def rmsnorm(h, gamma, eps):
    """RMSNorm: γ ⊙ h / √(mean(h²) + ε)"""
    r = torch.sqrt((h * h).mean(dim=-1, keepdim=True) + eps)
    return gamma * (h / r)


def r_of(h, eps):
    return float(torch.sqrt((h * h).mean() + eps).item())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--validation_json", required=True)
    ap.add_argument("--chains_path", required=True)
    ap.add_argument("--alpha", type=float, default=24.0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    print(f"[load-chains] {args.chains_path}")
    with open(args.chains_path) as f:
        chains = {c["chain_id"]: c for c in json.load(f)["chains"]}
    print(f"[load-validation] {args.validation_json}")
    with open(args.validation_json) as f:
        vd = json.load(f)
    recs = [r for r in vd["per_chain"] if "target_z_id" in r]
    print(f"  {len(recs)} chains, α={args.alpha}")

    print(f"[load-model] {args.model}")
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map="auto"
    )
    model.eval()
    device = next(model.parameters()).device

    W = _get_lm_head(model).float().to(device)             # (V, d)
    fn = _get_final_norm(model)
    gamma = fn.weight.detach().float().to(device)           # (d,)
    eps = _rms_eps(fn)
    print(f"  W: {tuple(W.shape)}, γ-shape: {tuple(gamma.shape)}, ε: {eps}")

    # Precompute W ⊙ γ as W·γ:  (V, d) with each column scaled by γ.
    # Per-token vector: W[t] is row t of (V,d). γ ⊙ W[t] is element-wise scale.
    # logits_from_h = (W · γ_diag) · h = (W * γ) · h  (row-broadcast).
    W_gamma = W * gamma  # (V, d)

    per_chain: List[Dict[str, Any]] = []
    for ci, rec in enumerate(recs):
        cid = rec["chain_id"]
        chain = chains[cid]
        t_id = int(rec["target_z_id"])

        prompt_ids = tok(chain["chain_prompt"], return_tensors="pt",
                         add_special_tokens=False).input_ids.to(device)

        # Capture h pre-final-norm at last_pos.
        cap = []
        def cap_hook(_m, args_, _kw):
            cap.append(args_[0][0, -1, :].detach().float().clone())
            return None
        handle = fn.register_forward_pre_hook(cap_hook, with_kwargs=True)
        try:
            with torch.no_grad():
                out = model(input_ids=prompt_ids)
        finally:
            handle.remove()
        h = cap[0].to(device)                    # (d,)
        baseline_logits_full = out.logits[0, -1].float()   # (V,)

        # Direct b_induced
        delta = args.alpha * W[t_id]                       # (d,)
        h_inj = h + delta
        ln_h = rmsnorm(h.unsqueeze(0), gamma, eps).squeeze(0)
        ln_hinj = rmsnorm(h_inj.unsqueeze(0), gamma, eps).squeeze(0)
        logits_direct_base = W @ ln_h.float()              # (V,)
        logits_direct_inj = W @ ln_hinj.float()            # (V,)
        b_direct = logits_direct_inj - logits_direct_base  # (V,)

        # Closed-form b_induced (corrected algebra):
        #   b = (α/r') · (W · γ ⊙ W[t]) + ((r-r')/(r·r')) · (W · γ ⊙ h)
        r = r_of(h, eps)
        r_prime = r_of(h_inj, eps)
        rank1_coeff = args.alpha / r_prime
        rescale_coeff_wgh = (r - r_prime) / (r * r_prime)   # multiplies W·γ⊙h
        rank1_term = rank1_coeff * (W_gamma @ W[t_id].float())              # (V,)
        baseline_rescale_term = rescale_coeff_wgh * (W_gamma @ h.float())   # (V,)
        b_closed = rank1_term + baseline_rescale_term

        # Comparisons
        # 1. baseline_logits_full vs logits_direct_base — should be ≈ (model rounding)
        baseline_match_max_diff = float((baseline_logits_full - logits_direct_base).abs().max().item())
        # 2. b_direct vs b_closed — the load-bearing check
        b_max_diff = float((b_direct - b_closed).abs().max().item())
        b_rel = float((b_direct - b_closed).norm().item() / (b_direct.norm().item() + 1e-12))

        # Rank-1 term sanity: it's just α scaled by r/r' times γ-weighted W[t]'s logit row.
        rank1_norm = float(rank1_term.norm().item())
        baseline_rescale_norm = float(baseline_rescale_term.norm().item())
        b_direct_norm = float(b_direct.norm().item())

        # The rescaling term's contribution to argmax: since it's a scalar multiple of the
        # baseline logit vector, it shifts all logits proportionally. Under raw argmax,
        # invisible. Under top-k or temperature, may matter.
        # Compute KL between b_direct vs rank1_term-only.
        # That tells us how much of the install's logit-bias structure is captured by
        # rank-1 alone (i.e., dropping the baseline-rescale term).
        from torch.nn.functional import log_softmax
        log_p_direct = log_softmax(baseline_logits_full + b_direct, dim=-1)
        log_p_rank1_only = log_softmax(baseline_logits_full + rank1_term, dim=-1)
        p_direct = log_p_direct.exp()
        kl_drop_rescale = float((p_direct * (log_p_direct - log_p_rank1_only)).sum().item())

        # Argmax under each
        argmax_direct = int(torch.argmax(baseline_logits_full + b_direct).item())
        argmax_rank1_only = int(torch.argmax(baseline_logits_full + rank1_term).item())
        argmax_baseline = int(torch.argmax(baseline_logits_full).item())

        per_chain.append({
            "chain_id": cid,
            "target_z": chain["target_z"],
            "target_z_id": t_id,
            "r": r,
            "r_prime": r_prime,
            "delta_r": r_prime - r,
            "rank1_coeff_alpha_over_r_prime": rank1_coeff,
            "rescale_coeff_for_W_gamma_h": rescale_coeff_wgh,
            "rank1_term_norm": rank1_norm,
            "baseline_rescale_term_norm": baseline_rescale_norm,
            "b_direct_norm": b_direct_norm,
            "rank1_share_of_b_norm": rank1_norm / (b_direct_norm + 1e-12),
            "rescale_share_of_b_norm": baseline_rescale_norm / (b_direct_norm + 1e-12),
            "b_direct_minus_b_closed_max_abs": b_max_diff,
            "b_direct_minus_b_closed_rel": b_rel,
            "baseline_logits_match_max_diff": baseline_match_max_diff,
            "kl_drop_rescale_term": kl_drop_rescale,
            "argmax_baseline": argmax_baseline,
            "argmax_b_direct": argmax_direct,
            "argmax_b_rank1_only": argmax_rank1_only,
            "argmax_match_with_rank1_only": argmax_direct == argmax_rank1_only,
        })

        print(f"  [chain {ci+1}/{len(recs)}] {cid} t={chain['target_z']!r}")
        print(f"    r={r:.4f}  r'={r_prime:.4f}  α/r'={rank1_coeff:.5f}  (r-r')/(r·r')={rescale_coeff_wgh:.5f}")
        print(f"    ||rank1_term||={rank1_norm:.3f}  ||rescale_term||={baseline_rescale_norm:.3f}  ||b_direct||={b_direct_norm:.3f}")
        print(f"    b_direct − b_closed: max|·|={b_max_diff:.2e}  rel={b_rel:.2e}")
        print(f"    KL(p_direct || p_rank1_only) = {kl_drop_rescale:.4f} nats")
        print(f"    argmax: baseline={argmax_baseline}  direct={argmax_direct}  rank1_only={argmax_rank1_only}  match={argmax_direct==argmax_rank1_only}")
        print()

    # Aggregate
    b_max_diffs = [c["b_direct_minus_b_closed_max_abs"] for c in per_chain]
    rank1_shares = [c["rank1_share_of_b_norm"] for c in per_chain]
    rescale_shares = [c["rescale_share_of_b_norm"] for c in per_chain]
    kl_drops = [c["kl_drop_rescale_term"] for c in per_chain]
    argmax_matches = [c["argmax_match_with_rank1_only"] for c in per_chain]

    summary = {
        "n_chains": len(per_chain),
        "max_b_direct_minus_b_closed_across_chains": float(max(b_max_diffs)),
        "median_b_direct_minus_b_closed_across_chains": float(np.median(b_max_diffs)),
        "median_rank1_share_of_b_norm": float(np.median(rank1_shares)),
        "median_rescale_share_of_b_norm": float(np.median(rescale_shares)),
        "median_KL_drop_rescale_term": float(np.median(kl_drops)),
        "argmax_match_with_rank1_only_count": sum(argmax_matches),
        "theorem_verified_at_fp": float(max(b_max_diffs)) < 1e-2,
    }

    print(f"\n=== Closed-form decomposition summary ===")
    print(f"  max |b_direct − b_closed| across chains: {summary['max_b_direct_minus_b_closed_across_chains']:.2e}")
    print(f"  median rank1 share of ||b||:   {summary['median_rank1_share_of_b_norm']:.4f}")
    print(f"  median rescale share of ||b||: {summary['median_rescale_share_of_b_norm']:.4f}")
    print(f"  median KL when rescale dropped: {summary['median_KL_drop_rescale_term']:.4f} nats")
    print(f"  argmax-1 matches rank1-only:    {summary['argmax_match_with_rank1_only_count']}/{len(per_chain)}")
    print(f"  theorem verified at fp:         {summary['theorem_verified_at_fp']}")

    out = {
        "probe": "closed_form_induced_bias_decomposition",
        "model": args.model,
        "alpha": args.alpha,
        "decomposition": {
            "b_induced(α,h,t) = (r/r') · α · (W · (γ ⊙ W[t]))    [rank-1 along γ-weighted W[t]]": True,
            "                 + ((r-r')/r') · (W · (γ ⊙ h))      [global rescaling of pre-norm-baseline logits]": True,
        },
        "summary": summary,
        "per_chain": per_chain,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[saved] {args.out}")


if __name__ == "__main__":
    main()
