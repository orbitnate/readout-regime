#!/usr/bin/env python3
"""Orthogonal-A control probe — substrate-clean basis-mechanism test (Pythia ladder).

Clone of dose_response_pythia_sweep.py with one additional mode, `orthogonal_A`:

    A_orth = (R - (R · ê_canon) ê_canon),  renormed to ||A_canonical||

where R is a Gaussian random direction and ê_canon = A_canonical / ||A_canonical||.

The orthogonal-A condition shares matched ||A|| with random_A but has its
lm_head[t] component algebraically zeroed. If basis-privilege is purely
DLA arithmetic (canonical_A's projection onto lm_head[t] doing all the work),
A_orth collapses to ~random_A. If basis-privilege is mechanism, A_orth
remains close to canonical.

Reuses canonical / random_A / wrong_B cells from Tier 25 (do NOT re-run).

Usage:
  python scripts/lm_head_basis_probes/orthogonal_a_pythia_probe.py \\
      --model EleutherAI/pythia-1.4b --layer 22 --frac 0.005 \\
      --config data/behavioral/refusal.json \\
      --mults 0.5,1,2,4 \\
      --modes orthogonal_A \\
      --n_triggers 10 --n_benign 10 --n_hard_neg 0 \\
      --out_dir results/lm_head_basis_probes \\
      --tag pythia1_4b_n10
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from statistics import mean, median
from typing import Any, Dict, List

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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--layer", type=int, required=True)
    ap.add_argument("--frac", type=float, default=0.005)
    ap.add_argument("--mults", default="0.5,1,2,4",
                    help="Comma-separated mult values to sweep")
    ap.add_argument("--modes", default="orthogonal_A",
                    help="Comma-separated modes: canonical,random_A,wrong_B,orthogonal_A")
    ap.add_argument("--config", required=True)
    ap.add_argument("--n_triggers", type=int, default=10)
    ap.add_argument("--n_benign", type=int, default=10)
    ap.add_argument("--n_hard_neg", type=int, default=0)
    ap.add_argument("--target_token_id", type=int, default=-1)
    ap.add_argument("--gate_type", choices=["softmax", "mlp"], default="softmax")
    ap.add_argument("--use_chat_template", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--topk_drift", type=int, default=100)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--tag", required=True, help="Filename prefix, e.g. pythia1_4b_n10")
    args = ap.parse_args()

    mults = [float(x) for x in args.mults.split(",") if x.strip()]
    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    valid_modes = {"canonical", "random_A", "wrong_B", "orthogonal_A"}
    for m in modes:
        if m not in valid_modes:
            raise ValueError(f"unknown mode {m}; valid: {valid_modes}")
    print(f"[sweep] mults={mults}  modes={modes}")

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

    if "wrong_B" in modes:
        gen = torch.Generator(device="cpu").manual_seed(args.seed)
        wrongB_idx = int(torch.randint(0, len(benigns), (1,), generator=gen).item())
        wrongB_prompt = benigns[wrongB_idx]
        B_wrong = capture_x_at_last(model, tok, wrongB_prompt, args.layer, device).to(
            device=device, dtype=torch.float32
        )
        print(f"  [wrong_B prompt] benigns[{wrongB_idx}] = {wrongB_prompt[:90]!r}")
    else:
        wrongB_idx = -1
        wrongB_prompt = ""
        B_wrong = None

    # ---- Random_A: matched-norm Gaussian ----
    if "random_A" in modes:
        gen = torch.Generator(device=device).manual_seed(args.seed)
        A_rand_raw = torch.randn(
            A_canonical.shape, generator=gen, device=device, dtype=torch.float32
        )
        A_rand = A_rand_raw * (a_norm_canonical / A_rand_raw.norm().clamp_min(1e-12))
        print(f"  [random_A] ||A_rand||={A_rand.norm().item():.3f}")
    else:
        A_rand = None

    # ---- Orthogonal_A: matched-norm random direction with lm_head[t] component zeroed ----
    if "orthogonal_A" in modes:
        unit_canon = A_canonical / A_canonical.norm().clamp_min(1e-12)
        # Use a different seed than random_A so the two controls are not the SAME
        # underlying random vector with one minus a projection -- they should be
        # independent draws of "matched-norm direction other than lm_head[t]".
        gen_o = torch.Generator(device=device).manual_seed(args.seed + 1)
        R = torch.randn(
            A_canonical.shape, generator=gen_o, device=device, dtype=torch.float32
        )
        proj_scalar = (R * unit_canon).sum()
        R_perp = R - proj_scalar * unit_canon
        A_orth = R_perp * (a_norm_canonical / R_perp.norm().clamp_min(1e-12))
        # Verify orthogonality
        cos_canon = (A_orth @ unit_canon).item()
        cos_check = (A_orth @ A_canonical).item() / (
            A_orth.norm().item() * A_canonical.norm().item() + 1e-12
        )
        a_orth_norm = A_orth.norm().item()
        print(
            f"  [orthogonal_A] ||A_orth||={a_orth_norm:.6f}  "
            f"||A_canon||={a_norm_canonical:.6f}  "
            f"(A_orth·ê_canon)={cos_canon:+.2e}  "
            f"cos(A_orth, A_canon)={cos_check:+.2e}"
        )
        # Sanity: orthogonality must be tight; allow 1e-4 for fp32 noise
        assert abs(cos_canon) < 1e-4, (
            f"orthogonality assertion failed: |A_orth·ê_canon| = {abs(cos_canon):.3e} "
            f">= 1e-4"
        )
        assert abs(cos_check) < 1e-5, (
            f"cos(A_orth,A_canon) = {abs(cos_check):.3e} >= 1e-5"
        )
        # Norm-match assertion (within 1e-4 relative)
        rel = abs(a_orth_norm - a_norm_canonical) / max(a_norm_canonical, 1e-12)
        assert rel < 1e-4, f"||A_orth|| not matched: rel err {rel:.3e}"
    else:
        A_orth = None

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
    gate_sweep = gate_info.get("sweep", [])
    print(f"  trained in {time.time() - t0:.1f}s  info={gate_info_brief}")
    trained_gate = base_dispatcher.gate

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    def _eval_cell(mode: str, mult: float) -> Dict[str, Any]:
        if mode == "canonical":
            A, B = A_canonical, B_canonical
            b_source = "triggers[0]"
        elif mode == "random_A":
            A, B = A_rand, B_canonical
            b_source = "triggers[0]"
        elif mode == "wrong_B":
            A, B = A_canonical, B_wrong
            b_source = f"benigns[{wrongB_idx}]"
        elif mode == "orthogonal_A":
            A, B = A_orth, B_canonical
            b_source = "triggers[0]"
        else:
            raise ValueError(f"unknown mode {mode}")

        a_norm = A.norm().item()
        b_norm = B.norm().item()
        ab_norm = a_norm * b_norm
        alpha = mult * args.frac * w_norm / ab_norm if ab_norm > 0 else 0.0
        S = alpha * (a_norm**2) * (b_norm**2)

        A_slot = A.unsqueeze(1).to(dtype).contiguous()
        B_slot = B.unsqueeze(0).to(dtype).contiguous()
        bundle = InjectionBundle(
            name=f"orthogonal_a_{mode}_{mult}",
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
            "b_source": b_source,
            "seed": args.seed,
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
            "gate_sweep": gate_sweep,
            "aggregate": agg,
            "trigger_records": trig_recs,
            "benign_records": ben_recs,
            "hardneg_records": hn_recs,
        }
        mult_str = f"{mult:.4f}".rstrip("0").rstrip(".").replace(".", "p")
        fname = f"orthogonal_a_{args.tag}_{mode}_mult{mult_str}.json"
        out_path = out_dir / fname
        out_path.write_text(json.dumps(summary, indent=2))
        print(
            f"  [{mode:>13}  mult={mult:>7.4f}]  α={alpha:.4e}  S={S:>7.3f}  "
            f"x={x:>7.3f}  Δ_med={agg['trigger']['delta_logp_target_median']:+7.3f}  "
            f"y={y_med:+7.3f}  trig_flip={agg['trigger']['flip_count']}/{agg['trigger']['n']}  "
            f"ben_drift={agg['benign']['offtarget_drift_median']:.2f}  "
            f"-> {fname}"
        )
        return summary

    print(f"\n[sweep] {len(modes)} modes × {len(mults)} mults = {len(modes)*len(mults)} cells")
    t_sweep = time.time()
    for mode in modes:
        for mult in mults:
            _eval_cell(mode, mult)
    print(f"\n[sweep] done in {time.time() - t_sweep:.1f}s")


if __name__ == "__main__":
    main()
