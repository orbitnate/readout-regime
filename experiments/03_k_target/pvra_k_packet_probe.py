#!/usr/bin/env python3
"""PVRA K-suppression-substitution packet probe (post-PVRA pretest §3).

Tests whether K co-installed face translations at the final residual
preserve their individual target_z installs without cross-talk above the
edge-Gram-predicted interference floor. This is the 2-hour AIAYN candidate
from POST_PVRA_PRETEST_HANDOFF_2026-05-12.md §3, dispatched after tonight's
cell-distance gate FAIL × 3 archs reframed PVRA's mechanism as
suppression-substitution at the readout (atlas decoration, NOT load-bearing).

Pre-registered pass bars (handoff §3.1):
  - Canonical: ≥ K-1 of K target_zᵢ tokens reach top-1 on most subsets
  - Substrate-clean: ≤ 1/n_random spectrum-matched random K-packets pass
  - Cross-arch: ≥ 2/3 archs pass at K=4 (PVRA-strict)

If canonical passes and random fails, output type changes from
`Vector` (single face translation) to `Set[FaceTranslation]`,
clearing §3/FL-59.

Probe structure:
  For each chain × arch (6 chains from §1.17 PVRA validation):
    1. Capture h_baseline_i pre-final-norm at last_pos.
    2. For each K ∈ Ks:
       - Canonical: enumerate C(6,K) chain subsets. For each subset S:
           δ_canon = Σ_{j∈S} α · (W[t_j] − W[s_j])
         For each chain i ∈ S, apply δ_canon to h_baseline_i;
         check if target_z_i reaches top-1 in W·rmsnorm(h_baseline + δ).
         Packet passes if ≥ K−1 of K chains hit.
       - Random: n_random K-packets sampled from norm-stratified pools
         (anisotropy-matched per FL-39). For each random packet, sample a
         random subset of K chains and apply / score the same way.

Usage:
  python scripts/lm_head_basis_probes/pvra_k_packet_probe.py \\
      --model NousResearch/Meta-Llama-3.1-8B-Instruct \\
      --validation_json results/lm_head_basis_probes/pvra_validation_llama.json \\
      --chains_path data/expN/chains_natural_n5000.json \\
      --alpha 8.0 \\
      --Ks 2,4,6 \\
      --n_random 200 \\
      --out results/lm_head_basis_probes/pvra_k_packet_llama.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from itertools import combinations
from pathlib import Path
from typing import Any, Dict, List, Tuple

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


# ----------------------------- model helpers -----------------------------

def _get_lm_head_weight(model) -> torch.Tensor:
    if hasattr(model, "embed_out"):
        return model.embed_out.weight.detach()
    return model.lm_head.weight.detach()


def _get_final_norm(model):
    if hasattr(model, "gpt_neox"):
        return model.gpt_neox.final_layer_norm
    return model.model.norm


def _get_final_norm_params(model):
    fn = _get_final_norm(model)
    weight = fn.weight.detach().float() if hasattr(fn, "weight") else None
    if hasattr(fn, "variance_epsilon"):
        eps = float(fn.variance_epsilon)
    elif hasattr(fn, "eps"):
        eps = float(fn.eps)
    else:
        eps = 1e-6
    return weight, eps


def _rmsnorm(h: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    rms = torch.sqrt((h * h).mean(dim=-1, keepdim=True) + eps)
    out = h / rms
    if weight is not None:
        out = out * weight
    return out


def _rank_of(logits: torch.Tensor, token_id: int) -> int:
    return int((logits > logits[token_id]).sum().item())


def _capture_baseline(model, tok, prompt: str, device):
    """Pre-final-norm residual at last_pos."""
    final_norm = _get_final_norm(model)
    captured: List[torch.Tensor] = []

    def hook(module, args, kwargs):
        h = args[0]
        captured.append(h[0, -1, :].detach().float().clone())
        return None

    handle = final_norm.register_forward_pre_hook(hook, with_kwargs=True)
    try:
        ids = tok(prompt, return_tensors="pt", add_special_tokens=False).to(device)
        with torch.no_grad():
            _ = model(**ids)
    finally:
        handle.remove()
    return captured[0]


# ----------------------------- driver -----------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--validation_json", required=True,
                    help="pvra_validation_*.json with per-chain target_z_id, runner_baseline_id")
    ap.add_argument("--chains_path", required=True)
    ap.add_argument("--alpha", type=float, required=True,
                    help="arch-specific saturating α (Llama=8, Mistral=24, Qwen=64)")
    ap.add_argument("--Ks", default="2,4,6",
                    help="comma-separated K values to test")
    ap.add_argument("--n_random", type=int, default=200,
                    help="random K-packets sampled per K")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    Ks = [int(k) for k in args.Ks.split(",")]

    print(f"[load-chains] {args.chains_path}")
    with open(args.chains_path) as f:
        chains_data = json.load(f)
    chains = chains_data["chains"]
    chain_by_id = {c["chain_id"]: c for c in chains}

    print(f"[load-validation] {args.validation_json}")
    with open(args.validation_json) as f:
        vd = json.load(f)
    val_by_chain = {
        r["chain_id"]: r
        for r in vd["per_chain"]
        if "target_z_id" in r and "runner_baseline_id" in r
    }
    ordered_cids = [
        r["chain_id"] for r in vd["per_chain"]
        if "target_z_id" in r and "runner_baseline_id" in r
    ]
    print(f"  {len(ordered_cids)} chains: {ordered_cids}")

    print(f"\n[load-model] {args.model}")
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map="auto",
    )
    model.eval()
    device = next(model.parameters()).device
    print(f"  loaded in {time.time() - t0:.1f}s on {device}")

    W = _get_lm_head_weight(model).float().to(device)
    final_norm_w, final_norm_eps = _get_final_norm_params(model)
    if final_norm_w is not None:
        final_norm_w = final_norm_w.to(device)
    V, D = W.shape
    W_norms = W.norm(dim=1)
    print(f"  W: {V}x{D}, ‖row‖ median = {W_norms.median().item():.3f}")
    print(f"  α = {args.alpha}")

    # --- Capture baselines per chain ---
    chain_data: List[Dict[str, Any]] = []
    for cid in ordered_cids:
        chain = chain_by_id[cid]
        val = val_by_chain[cid]
        target_z_id = val["target_z_id"]
        runner_baseline_id = val["runner_baseline_id"]
        h_base = _capture_baseline(model, tok, chain["chain_prompt"], device).to(device)
        x_base = _rmsnorm(h_base, final_norm_w, final_norm_eps)
        logits_base = (W @ x_base.to(torch.float32))
        base_rank_tz = _rank_of(logits_base, target_z_id)
        base_top1 = int(torch.argmax(logits_base).item())
        print(f"  [chain] {cid}  target_z={chain['target_z']!r}({target_z_id})  "
              f"runner={tok.decode([runner_baseline_id])!r}({runner_baseline_id})  "
              f"base_rank_tz={base_rank_tz}")
        chain_data.append({
            "chain_id": cid,
            "chain_prompt": chain["chain_prompt"],
            "bridge": chain["bridge"],
            "target_z": chain["target_z"],
            "target_z_id": target_z_id,
            "runner_baseline_id": runner_baseline_id,
            "runner_baseline_decoded": tok.decode([runner_baseline_id]),
            "h_baseline": h_base,
            "h_baseline_norm": float(h_base.norm().item()),
            "baseline_target_z_rank": base_rank_tz,
            "baseline_top1": base_top1,
        })

    n_chains = len(chain_data)
    assert n_chains == 6, f"expected 6 chains, got {n_chains}"

    # --- Norm-stratified candidate pools (anisotropy-matched random control) ---
    t_norms_actual = [float(W_norms[c["target_z_id"]]) for c in chain_data]
    s_norms_actual = [float(W_norms[c["runner_baseline_id"]]) for c in chain_data]
    t_norm_med = float(np.median(t_norms_actual))
    s_norm_med = float(np.median(s_norms_actual))
    t_lo, t_hi = t_norm_med * 0.5, t_norm_med * 2.0
    s_lo, s_hi = s_norm_med * 0.5, s_norm_med * 2.0
    t_pool = ((W_norms >= t_lo) & (W_norms <= t_hi)).nonzero(as_tuple=True)[0]
    s_pool = ((W_norms >= s_lo) & (W_norms <= s_hi)).nonzero(as_tuple=True)[0]
    if len(t_pool) < 100:
        t_pool = torch.arange(V, device=device)
    if len(s_pool) < 100:
        s_pool = torch.arange(V, device=device)
    print(f"  norm-stratified pools: |t_pool|={len(t_pool)} (med={t_norm_med:.3f}), "
          f"|s_pool|={len(s_pool)} (med={s_norm_med:.3f})")

    # --- K-packet primitive ---
    def packet_delta(pairs: List[Tuple[int, int]], alpha: float) -> torch.Tensor:
        delta = torch.zeros(D, device=device, dtype=torch.float32)
        for s, t in pairs:
            delta = delta + alpha * (W[t] - W[s])
        return delta

    def apply_to_chain(delta: torch.Tensor, chain_idx: int) -> Tuple[int, int, float]:
        """Return (target_z_rank, top1_id, delta_norm) after applying δ to chain's h_base."""
        c = chain_data[chain_idx]
        h_new = c["h_baseline"] + delta
        x_new = _rmsnorm(h_new, final_norm_w, final_norm_eps)
        logits_new = (W @ x_new.to(torch.float32))
        r = _rank_of(logits_new, c["target_z_id"])
        top1 = int(torch.argmax(logits_new).item())
        return r, top1, float(delta.norm().item())

    gen = torch.Generator(device="cpu").manual_seed(args.seed)

    per_K_results: Dict[str, Any] = {}
    for K in Ks:
        print(f"\n[K = {K}] α = {args.alpha}")
        all_subsets = list(combinations(range(n_chains), K))
        print(f"  canonical subsets: {len(all_subsets)} (C({n_chains},{K}))")

        # --- Canonical ---
        canonical_records: List[Dict[str, Any]] = []
        canon_chain_hits = 0
        canon_chain_total = 0
        canon_packet_passes = 0
        canon_delta_norms: List[float] = []
        for subset in all_subsets:
            pairs = [
                (chain_data[i]["runner_baseline_id"], chain_data[i]["target_z_id"])
                for i in subset
            ]
            delta = packet_delta(pairs, args.alpha)
            per_chain: List[Dict[str, Any]] = []
            n_hit = 0
            for i in subset:
                r, top1, dn = apply_to_chain(delta, i)
                is_hit = (r == 0)
                if is_hit:
                    n_hit += 1
                canon_chain_total += 1
                canon_chain_hits += int(is_hit)
                per_chain.append({
                    "chain_idx": i,
                    "chain_id": chain_data[i]["chain_id"],
                    "target_z_rank": r,
                    "is_top1": is_hit,
                    "top1_id": top1,
                })
            canon_delta_norms.append(float(delta.norm().item()))
            is_packet_pass = (n_hit >= K - 1)
            if is_packet_pass:
                canon_packet_passes += 1
            canonical_records.append({
                "subset": list(subset),
                "n_hit": n_hit,
                "is_packet_pass": is_packet_pass,
                "delta_norm": canon_delta_norms[-1],
                "per_chain": per_chain,
            })

        canon_chain_hit_rate = canon_chain_hits / max(canon_chain_total, 1)
        canon_packet_pass_rate = canon_packet_passes / max(len(all_subsets), 1)
        print(f"  canonical per-chain top-1: {canon_chain_hits}/{canon_chain_total} = {canon_chain_hit_rate:.3f}")
        print(f"  canonical packet pass (≥{K-1}/{K}): {canon_packet_passes}/{len(all_subsets)} = {canon_packet_pass_rate:.3f}")
        print(f"  canonical δ-norm median: {np.median(canon_delta_norms):.2f}")

        # --- Random control ---
        random_records: List[Dict[str, Any]] = []
        rand_chain_hits = 0
        rand_chain_total = 0
        rand_packet_passes = 0
        rand_delta_norms: List[float] = []
        for p in range(args.n_random):
            pairs_r: List[Tuple[int, int]] = []
            for _ in range(K):
                t_pos = int(torch.randint(0, len(t_pool), (1,), generator=gen).item())
                s_pos = int(torch.randint(0, len(s_pool), (1,), generator=gen).item())
                pairs_r.append((int(s_pool[s_pos]), int(t_pool[t_pos])))
            delta_r = packet_delta(pairs_r, args.alpha)
            rand_delta_norms.append(float(delta_r.norm().item()))
            # Random K-subset of chains
            perm = torch.randperm(n_chains, generator=gen).tolist()
            subset_r = tuple(sorted(perm[:K]))
            n_hit = 0
            per_chain_r: List[Dict[str, Any]] = []
            for i in subset_r:
                r, top1, _ = apply_to_chain(delta_r, i)
                is_hit = (r == 0)
                if is_hit:
                    n_hit += 1
                rand_chain_total += 1
                rand_chain_hits += int(is_hit)
                per_chain_r.append({
                    "chain_idx": i,
                    "target_z_rank": r,
                    "is_top1": is_hit,
                })
            is_packet_pass = (n_hit >= K - 1)
            if is_packet_pass:
                rand_packet_passes += 1
            random_records.append({
                "subset": list(subset_r),
                "pairs": pairs_r,
                "n_hit": n_hit,
                "is_packet_pass": is_packet_pass,
                "per_chain": per_chain_r,
            })

        rand_chain_hit_rate = rand_chain_hits / max(rand_chain_total, 1)
        rand_packet_pass_rate = rand_packet_passes / max(args.n_random, 1)
        print(f"  random per-chain top-1: {rand_chain_hits}/{rand_chain_total} = {rand_chain_hit_rate:.4f}")
        print(f"  random packet pass: {rand_packet_passes}/{args.n_random} = {rand_packet_pass_rate:.4f}")
        print(f"  random δ-norm median: {np.median(rand_delta_norms):.2f}")

        # Pre-registered: canonical packet pass rate ≥ 2/3 of subsets AND random passes ≤ 1
        canonical_clears = canon_packet_pass_rate >= (2.0 / 3.0)
        random_clears = rand_packet_passes <= 1
        verdict = (
            "PASS_K_PACKET" if (canonical_clears and random_clears) else
            "PARTIAL" if (canonical_clears or random_clears) else
            "FAIL"
        )
        print(f"  → canonical_clears (≥2/3 subsets pass): {canonical_clears}")
        print(f"  → random_clears (≤1/{args.n_random} passes): {random_clears}")
        print(f"  → verdict K={K}: {verdict}")

        per_K_results[str(K)] = {
            "K": K,
            "n_subsets": len(all_subsets),
            "canonical_chain_hits": canon_chain_hits,
            "canonical_chain_total": canon_chain_total,
            "canonical_chain_hit_rate": canon_chain_hit_rate,
            "canonical_packet_passes": canon_packet_passes,
            "canonical_packet_pass_rate": canon_packet_pass_rate,
            "canonical_delta_norm_median": float(np.median(canon_delta_norms)),
            "random_chain_hits": rand_chain_hits,
            "random_chain_total": rand_chain_total,
            "random_chain_hit_rate": rand_chain_hit_rate,
            "random_packet_passes": rand_packet_passes,
            "random_packet_pass_rate": rand_packet_pass_rate,
            "random_delta_norm_median": float(np.median(rand_delta_norms)),
            "n_random": args.n_random,
            "canonical_clears": canonical_clears,
            "random_clears": random_clears,
            "verdict": verdict,
            "canonical_records": canonical_records,
            "random_records": random_records,
        }

    # --- Cross-K summary ---
    print(f"\n[summary K → verdict]")
    print(f"  {'K':>3} | canon top-1 (per-chain) | rand top-1 (per-chain) | "
          f"canon packet-pass | rand packet-pass | verdict")
    for K in Ks:
        d = per_K_results[str(K)]
        print(f"  {K:>3} | {d['canonical_chain_hits']:>3}/{d['canonical_chain_total']:<3} "
              f"({d['canonical_chain_hit_rate']:.3f})    | "
              f"{d['random_chain_hits']:>4}/{d['random_chain_total']:<4} "
              f"({d['random_chain_hit_rate']:.4f}) | "
              f"{d['canonical_packet_passes']:>3}/{d['n_subsets']:<3} "
              f"({d['canonical_packet_pass_rate']:.3f}) | "
              f"{d['random_packet_passes']:>3}/{d['n_random']:<3} "
              f"({d['random_packet_pass_rate']:.4f}) | "
              f"{d['verdict']}")

    out = {
        "probe": "pvra_k_packet",
        "model": args.model,
        "alpha": args.alpha,
        "Ks": Ks,
        "n_random": args.n_random,
        "seed": args.seed,
        "n_chains": n_chains,
        "ordered_chain_ids": ordered_cids,
        "chains_meta": [
            {k: v for k, v in c.items() if k != "h_baseline"}
            for c in chain_data
        ],
        "norm_stratified_pool_size": {
            "t_pool": int(len(t_pool)),
            "s_pool": int(len(s_pool)),
        },
        "per_K": per_K_results,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n[saved] {args.out}")


if __name__ == "__main__":
    main()
