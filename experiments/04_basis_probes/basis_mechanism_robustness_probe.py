#!/usr/bin/env python3
"""Basis-mechanism robustness probe — multi-seed + cosine-dosage on top of orth-A.

Three extensions over `orthogonal_a_pythia_probe.py`:

  (B) Multi-seed for `random_A` and `orthogonal_A`. `--seeds 0,1,2,3,4`
      generates n_seeds cells per (mode, mult) combination.
  (C) Cosine-dosage mode `cos_dose`: A = cos·ê_canon + √(1-cos²)·R_perp_unit,
      renormed to ||lm_head[t]||. Tests whether response is smooth and
      monotone in |cos(A, lm_head[t])| — if so, no hidden mechanism.
  (D) Architecture-agnostic via the underlying probe's _detect_arch
      (gpt_neox vs llama_family).

Usage:
  # (B) + (C) on Pythia-1.4B:
  python scripts/lm_head_basis_probes/basis_mechanism_robustness_probe.py \\
      --model EleutherAI/pythia-1.4b --layer 22 --frac 0.005 \\
      --config data/behavioral/refusal.json \\
      --mults 1,4 --modes random_A,orthogonal_A \\
      --seeds 0,1,2,3,4 \\
      --cos_targets 0.0,0.1,0.3,0.5,0.7,1.0 --cos_mult 4 \\
      --n_triggers 10 --n_benign 10 \\
      --out_dir results/lm_head_basis_probes \\
      --tag pythia1_4b_n10_robust

  # (D) cross-arch (chat-template):
  python scripts/lm_head_basis_probes/basis_mechanism_robustness_probe.py \\
      --model Qwen/Qwen2.5-7B-Instruct --layer 27 --frac 0.005 \\
      --config data/behavioral/refusal.json \\
      --mults 1,4 --modes orthogonal_A,random_A \\
      --seeds 0,1,2,3,4 --use_chat_template \\
      --n_triggers 10 --n_benign 10 \\
      --out_dir results/lm_head_basis_probes \\
      --tag qwen7b_n10_robust
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from statistics import mean, median
from typing import Any, Dict, List, Optional

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from scripts.lm_head_basis_probes.dose_response_pythia_probe import (
    _detect_arch,
    _get_down_proj,
    _get_lm_head_weight,
    capture_x_at_last,
    last_pos_logits,
    topk_baseline,
    eval_target_dispatched_with_topk,
)
from pitwm.dispatcher import InjectionBundle, UniversalDispatcher


def _baseline_rec(model, tok, prompt: str, target_id: int, device, k: int) -> Dict[str, Any]:
    logits = last_pos_logits(model, tok, prompt, device)
    logp = F.log_softmax(logits, dim=-1)
    base_lp = logp[target_id].item()
    base_argmax = int(logits.argmax().item())
    topk_ids, topk_logp = topk_baseline(model, tok, prompt, target_id, device, k=k)
    return {
        "prompt": prompt,
        "baseline_logp_target": base_lp,
        "baseline_argmax_id": base_argmax,
        "baseline_argmax_decoded": tok.decode([base_argmax]),
        "_topk_ids": topk_ids.tolist(),
        "_baseline_topk_logp": topk_logp.tolist(),
    }


def _summarize(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not records:
        return {"n": 0}
    flips = [int(r["argmax_is_target"]) for r in records]
    fires = [int(r["gate_fired"]) for r in records]
    deltas = [r["delta_logp_target"] for r in records]
    post_logps = [r["logp_target"] for r in records]
    base_logps = [r["baseline_logp_target"] for r in records]
    C_vals = [-bp for bp in base_logps]
    ys = [d / c for d, c in zip(deltas, C_vals) if c > 0]
    drifts = [r["offtarget_drift_topk"] for r in records]
    return {
        "n": len(records),
        "flip_to_target_rate": sum(flips) / len(records),
        "flip_count": sum(flips),
        "gate_fire_rate": sum(fires) / len(records),
        "gate_fires": sum(fires),
        "delta_logp_target_mean": mean(deltas),
        "delta_logp_target_median": median(deltas),
        "post_logp_target_mean": mean(post_logps),
        "post_logp_target_median": median(post_logps),
        "baseline_logp_target_mean": mean(base_logps),
        "baseline_logp_target_median": median(base_logps),
        "C_median": median(C_vals),
        "y_median": median(ys) if ys else 0.0,
        "y_mean": mean(ys) if ys else 0.0,
        "offtarget_drift_mean": mean(drifts),
        "offtarget_drift_median": median(drifts),
    }


def _build_A(
    mode: str,
    *,
    A_canonical: torch.Tensor,
    a_norm_canonical: float,
    seed: int,
    cos_target: Optional[float],
    device,
) -> tuple[torch.Tensor, Dict[str, Any]]:
    """Return (A, info_dict). info_dict carries diagnostics for the JSON."""
    unit_canon = A_canonical / A_canonical.norm().clamp_min(1e-12)
    info: Dict[str, Any] = {"seed": seed, "cos_target": cos_target}

    if mode == "canonical":
        A = A_canonical.clone()
        info["cos_with_canon"] = 1.0
        return A, info

    # All non-canonical modes derive from a Gaussian R; seed determines R
    gen = torch.Generator(device=device).manual_seed(seed)
    R = torch.randn(
        A_canonical.shape, generator=gen, device=device, dtype=torch.float32
    )

    if mode == "random_A":
        A = R * (a_norm_canonical / R.norm().clamp_min(1e-12))
        info["cos_with_canon"] = (A @ unit_canon).item() / a_norm_canonical
        return A, info

    if mode == "orthogonal_A":
        proj = (R * unit_canon).sum()
        R_perp = R - proj * unit_canon
        A = R_perp * (a_norm_canonical / R_perp.norm().clamp_min(1e-12))
        cos = (A @ unit_canon).item() / a_norm_canonical
        info["cos_with_canon"] = cos
        assert abs(cos) < 1e-4, f"orth-A cosine sanity fail: {cos:.3e}"
        return A, info

    if mode == "cos_dose":
        assert cos_target is not None, "cos_dose mode requires --cos_targets"
        c = float(cos_target)
        # Build R_perp_unit: unit vector in lm_head[t]-orthogonal subspace
        proj = (R * unit_canon).sum()
        R_perp = R - proj * unit_canon
        R_perp_unit = R_perp / R_perp.norm().clamp_min(1e-12)
        # A_unit = c · ê_canon + √(1-c²) · R_perp_unit  (unit norm by construction)
        s = math.sqrt(max(0.0, 1.0 - c * c))
        A_unit = c * unit_canon + s * R_perp_unit
        A = A_unit * a_norm_canonical
        cos = (A @ unit_canon).item() / a_norm_canonical
        info["cos_with_canon"] = cos
        # Verify unit-A interpolation: cos should be ≈ c (up to fp32)
        assert abs(cos - c) < 1e-3, f"cos_dose cos sanity fail: target {c} got {cos:.3e}"
        return A, info

    raise ValueError(f"unknown mode {mode}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--layer", type=int, required=True)
    ap.add_argument("--frac", type=float, default=0.005)
    ap.add_argument("--mults", default="1,4",
                    help="Comma-separated mult values to sweep (for canonical/random_A/orth_A)")
    ap.add_argument("--modes", default="random_A,orthogonal_A",
                    help="Comma-separated of: canonical,random_A,orthogonal_A,cos_dose")
    ap.add_argument("--seeds", default="0,1,2,3,4",
                    help="Comma-separated seeds for random_A/orth_A/cos_dose")
    ap.add_argument("--cos_targets", default="",
                    help="Comma-separated cos values for cos_dose mode (e.g. 0.0,0.1,0.3,0.5,0.7,1.0)")
    ap.add_argument("--cos_mult", type=float, default=4.0,
                    help="Mult to use for cos_dose cells (cos_dose sweeps cos at fixed mult)")
    ap.add_argument("--config", required=True)
    ap.add_argument("--n_triggers", type=int, default=10)
    ap.add_argument("--n_benign", type=int, default=10)
    ap.add_argument("--n_hard_neg", type=int, default=0)
    ap.add_argument("--target_token_id", type=int, default=-1)
    ap.add_argument("--gate_type", choices=["softmax", "mlp"], default="softmax")
    ap.add_argument("--use_chat_template", action="store_true")
    ap.add_argument("--topk_drift", type=int, default=100)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--tag", required=True)
    args = ap.parse_args()

    mults = [float(x) for x in args.mults.split(",") if x.strip()]
    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    cos_targets = (
        [float(c) for c in args.cos_targets.split(",") if c.strip()]
        if args.cos_targets
        else []
    )
    valid_modes = {"canonical", "random_A", "orthogonal_A", "cos_dose"}
    for m in modes:
        if m not in valid_modes:
            raise ValueError(f"unknown mode {m}; valid: {valid_modes}")
    if "cos_dose" in modes and not cos_targets:
        raise ValueError("cos_dose mode requires --cos_targets")
    print(f"[robust] mults={mults}  modes={modes}  seeds={seeds}  cos_targets={cos_targets}")

    with open(args.config) as f:
        cfg = json.load(f)
    triggers_all: List[str] = list(cfg["trigger_prompts"])
    benigns_all: List[str] = list(cfg["benign_prompts"])
    hard_negs_all: List[str] = list(cfg.get("hard_negative_prompts", []))
    target_str: str = cfg["target_str"]
    triggers = triggers_all[: args.n_triggers] if args.n_triggers > 0 else triggers_all
    benigns = benigns_all[: args.n_benign] if args.n_benign > 0 else benigns_all
    if args.n_hard_neg == 0:
        hard_negs: List[str] = []
    elif args.n_hard_neg > 0:
        hard_negs = hard_negs_all[: args.n_hard_neg]
    else:
        hard_negs = hard_negs_all

    print(f"[load-model] {args.model}")
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map="auto"
    )
    model.eval()
    device = next(model.parameters()).device
    arch = _detect_arch(model)
    print(f"  loaded in {time.time() - t0:.1f}s on {device}; arch={arch}")

    if args.target_token_id >= 0:
        target_id = args.target_token_id
    else:
        target_ids = tok.encode(target_str, add_special_tokens=False)
        target_id = target_ids[0]
    target_decoded = tok.decode([target_id])
    print(f"  target_id={target_id}  decoded={target_decoded!r}")

    if args.use_chat_template and getattr(tok, "chat_template", None) is not None:
        def _wrap(p: str) -> str:
            return tok.apply_chat_template(
                [{"role": "user", "content": p}],
                tokenize=False, add_generation_prompt=True,
            )
        triggers = [_wrap(p) for p in triggers]
        benigns = [_wrap(p) for p in benigns]
        hard_negs = [_wrap(p) for p in hard_negs]

    down_proj = _get_down_proj(model, args.layer)
    w_norm = down_proj.weight.data.float().norm().item()
    dtype = down_proj.weight.dtype
    print(f"[setup] layer={args.layer}  ||W||={w_norm:.2f}  dtype={dtype}")

    lm_w = _get_lm_head_weight(model)
    A_canonical = lm_w[target_id].to(device=device, dtype=torch.float32)
    a_norm_canonical = A_canonical.norm().item()

    B_canonical = capture_x_at_last(model, tok, triggers[0], args.layer, device).to(
        device=device, dtype=torch.float32
    )

    print(f"\n[baselines] target logp + top-{args.topk_drift}")
    t0 = time.time()
    base_triggers = [_baseline_rec(model, tok, p, target_id, device, args.topk_drift) for p in triggers]
    base_benigns = [_baseline_rec(model, tok, p, target_id, device, args.topk_drift) for p in benigns]
    base_hardnegs = [_baseline_rec(model, tok, p, target_id, device, args.topk_drift) for p in hard_negs]
    print(f"  baselines done in {time.time() - t0:.1f}s")

    controls = list(benigns) + list(hard_negs)
    print(f"\n[gate] n_triggers={len(triggers)}  n_controls={len(controls)}")
    t0 = time.time()
    sentinel_bundle = InjectionBundle(
        name="sentinel",
        slots=[(A_canonical.unsqueeze(1).to(dtype), B_canonical.unsqueeze(0).to(dtype))],
        alpha=0.0,
        trigger_prompts=triggers,
        target_ids=[target_id],
    )
    base_dispatcher = UniversalDispatcher(model, tok, layer_idx=args.layer)
    base_dispatcher.add_bundle(sentinel_bundle)
    gate_info = base_dispatcher.train_gate(controls, gate_type=args.gate_type)
    gate_info_brief = {k: v for k, v in gate_info.items() if k != "sweep"}
    print(f"  trained in {time.time() - t0:.1f}s  info={gate_info_brief}")
    trained_gate = base_dispatcher.gate

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    def _eval_cell(
        mode: str, mult: float, seed: int, cos_target: Optional[float]
    ) -> Dict[str, Any]:
        A, ainfo = _build_A(
            mode,
            A_canonical=A_canonical,
            a_norm_canonical=a_norm_canonical,
            seed=seed,
            cos_target=cos_target,
            device=device,
        )
        B = B_canonical
        a_norm = A.norm().item()
        b_norm = B.norm().item()
        ab_norm = a_norm * b_norm
        alpha = mult * args.frac * w_norm / ab_norm if ab_norm > 0 else 0.0
        S = alpha * (a_norm**2) * (b_norm**2)

        A_slot = A.unsqueeze(1).to(dtype).contiguous()
        B_slot = B.unsqueeze(0).to(dtype).contiguous()
        cos_tag = ""
        if cos_target is not None:
            cos_tag = f"cos{cos_target:.2f}".replace(".", "p")
        bundle = InjectionBundle(
            name=f"robust_{mode}_seed{seed}_{cos_tag}_mult{mult}",
            slots=[(A_slot, B_slot)],
            alpha=alpha,
            trigger_prompts=triggers,
            target_ids=[target_id],
        )
        disp = UniversalDispatcher(model, tok, layer_idx=args.layer)
        disp.add_bundle(bundle)
        disp.gate = trained_gate

        trig_recs = [dict(r) for r in base_triggers]
        ben_recs = [dict(r) for r in base_benigns]
        hn_recs = [dict(r) for r in base_hardnegs]

        def _eval_group(records: List[Dict[str, Any]]) -> None:
            for rec in records:
                topk_ids = torch.tensor(rec["_topk_ids"], dtype=torch.long)
                ev = eval_target_dispatched_with_topk(
                    disp, model, tok, rec["prompt"], target_id, device, topk_ids,
                )
                rec.update(ev)
                rec["delta_logp_target"] = ev["logp_target"] - rec["baseline_logp_target"]
                rec["argmax_decoded"] = tok.decode([ev["argmax_id"]])
                post_topk = torch.tensor(ev["post_topk_logp"], dtype=torch.float32)
                base_topk = torch.tensor(rec["_baseline_topk_logp"], dtype=torch.float32)
                rec["offtarget_drift_topk"] = (post_topk - base_topk).abs().sum().item()
                del rec["_topk_ids"]
                del rec["_baseline_topk_logp"]
                del rec["post_topk_logp"]

        with disp:
            _eval_group(trig_recs)
            _eval_group(ben_recs)
            _eval_group(hn_recs)

        agg = {
            "trigger": _summarize(trig_recs),
            "benign": _summarize(ben_recs),
            "hard_neg": _summarize(hn_recs),
        }
        C_med = agg["trigger"]["C_median"] if agg["trigger"]["n"] else 0.0
        x = S / C_med if C_med > 0 else float("nan")
        y_med = agg["trigger"]["y_median"]

        summary = {
            "model": args.model,
            "arch": arch,
            "layer": args.layer,
            "frac": args.frac,
            "mult": mult,
            "mode": mode,
            "seed": seed,
            "cos_target": cos_target,
            "cos_with_canon": ainfo.get("cos_with_canon"),
            "target_str": target_str,
            "target_token_id": target_id,
            "target_decoded": target_decoded,
            "use_chat_template": args.use_chat_template,
            "n_triggers": len(triggers),
            "n_benign": len(benigns),
            "n_hard_neg": len(hard_negs),
            "gate_type": args.gate_type,
            "config_path": args.config,
            "behavior": cfg.get("behavior", "unknown"),
            "alpha": alpha,
            "a_norm": a_norm,
            "b_norm": b_norm,
            "w_norm": w_norm,
            "S": S,
            "C_med_triggers": C_med,
            "x": x,
            "y_median_triggers": y_med,
            "gate_info": gate_info_brief,
            "aggregate": agg,
            "trigger_records": trig_recs,
            "benign_records": ben_recs,
            "hardneg_records": hn_recs,
        }
        mult_str = f"{mult:.4f}".rstrip("0").rstrip(".").replace(".", "p")
        seed_str = f"s{seed}"
        cos_str = f"_{cos_tag}" if cos_tag else ""
        fname = f"robust_{args.tag}_{mode}_{seed_str}{cos_str}_mult{mult_str}.json"
        out_path = out_dir / fname
        out_path.write_text(json.dumps(summary, indent=2))
        cos_disp = f"cos={ainfo.get('cos_with_canon', 0):+.3f}" if ainfo.get("cos_with_canon") is not None else ""
        print(
            f"  [{mode:>13}  s={seed} {cos_disp:>14}  mult={mult:>5.2f}]  "
            f"α={alpha:.3e}  Δ_med={agg['trigger']['delta_logp_target_median']:+7.3f}  "
            f"trig_flip={agg['trigger']['flip_count']}/{agg['trigger']['n']}  "
            f"ben_drift={agg['benign']['offtarget_drift_median']:.2f}  "
            f"-> {fname}"
        )
        return summary

    # Enumerate cells
    n_cells = 0
    for mode in modes:
        if mode == "cos_dose":
            for cos_t in cos_targets:
                for seed in seeds:
                    n_cells += 1
        elif mode == "canonical":
            for mult in mults:
                n_cells += 1  # seed-invariant
        else:
            for seed in seeds:
                for mult in mults:
                    n_cells += 1
    print(f"\n[sweep] {n_cells} cells total")
    t_sweep = time.time()
    for mode in modes:
        if mode == "cos_dose":
            for cos_t in cos_targets:
                for seed in seeds:
                    _eval_cell(mode, args.cos_mult, seed, cos_t)
        elif mode == "canonical":
            for mult in mults:
                _eval_cell(mode, mult, seeds[0], None)
        else:
            for seed in seeds:
                for mult in mults:
                    _eval_cell(mode, mult, seed, None)
    print(f"\n[sweep] done in {time.time() - t_sweep:.1f}s")


if __name__ == "__main__":
    main()
