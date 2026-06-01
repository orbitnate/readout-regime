#!/usr/bin/env python3
"""Behavioral install — PURE-GEOMETRIC variant (final-residual injection).

Motivated by `docs/results/TIER1_ATTN_ABLATION_RESULTS.md` (Frame B verdict).
Tier 1 showed that for the bridge-transport line, injecting `A = lm_head[t]`
directly into the residual at last_pos right before final_norm (no L_a
injection, no mid-layer transformation) matches or beats the standard L_a
injection for moving target_z up the readout.

The behavioral install line has the SAME mechanistic structure as the
bridge-transport line: position-0 measurement of how strongly `lm_head[t]`
injection lifts `logp(t)` at the output. In `docs/results/BEHAVIORAL_INSTALL_RESULTS.md`
Phase 2 cross-arch, Llama and Mistral get **0 % argmax-flip** at chat-template
mult=0.50 because the baseline gap between argmax (`'I'`) and target (`' sorry'`)
is 22-27 nats — exceeding what the standard L_a=27 install can close inside
the architecture-dependent coherence wall. Llama's mult sweep peaks at +10 nats
at mult=4-6 then REGRESSES at mult=8; Mistral goes NEGATIVE at mult=8.

Hypothesis (direct consequence of Tier 1 Frame B): a pure-geometric install
(add `c · A` to residual at last_pos right before final_norm, with the gate
deciding whether to inject) should bypass the coherence wall. The model's
late-layer computation runs normally; only the readout sees the direct logit
boost. This may close the cross-arch argmax-flip gap.

Construction:
  A = lm_head[target_token]                              [same as standard]
  B = capture(x at L_a's mlp.down_proj input on triggers[0])  [same as standard]
  alpha = mult * frac * ||W_La||_F / (||A|| * ||B||)     [same as standard]
  gate trained on (triggers, benigns ∪ hard_negs) at L_a [same as standard]

Difference from standard probe:
  Standard: at L_a's down_proj, when gate fires, ADD `alpha * (A @ B) · x_last`
            to the down_proj output. The signal flows through layers L_a+1..N
            and gets attenuated by the coherence wall.
  Pure-geo: at L_a's down_proj, only compute the gate decision + the scalar
            c = alpha * (B · x_last). Do NOT add anything to L_a's output.
            At final_norm pre-hook, when gate fired, ADD `c * A` to the
            residual at last_pos directly.

This preserves the per-prompt scalar (so the magnitude tracks x_at_L_a · B
just like the standard install would), but skips the 14+ layers of mid-layer
processing entirely.

Usage:
  python scripts/lm_head_basis_probes/behavioral_install_pure_geometric_probe.py \\
      --model meta-llama/Meta-Llama-3.1-8B-Instruct --layer 27 --frac 0.005 --mult 0.50 \\
      --config data/behavioral/refusal.json \\
      --n_triggers 20 --n_benign 20 --n_hard_neg 20 \\
      --use_chat_template \\
      --out results/lm_head_basis_probes/behavioral_install_pure_geometric_llama-8b_refusal_chat_n20.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, median
from typing import Any, Dict, List, Optional

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from pitwm.dispatcher import InjectionBundle, UniversalDispatcher
from pitwm.dispatcher.dispatcher import _DispatcherHook  # private but stable


# ---------------------------------------------------------------------------
# Helpers (parallel to behavioral_install_dispatched_probe.py)
# ---------------------------------------------------------------------------

def get_down_proj(model, layer_idx):
    return model.model.layers[layer_idx].mlp.down_proj


def get_final_norm(model):
    return model.model.norm


def capture_x_at_last(model, tokenizer, prompt: str, layer_idx: int, device) -> torch.Tensor:
    ids = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).input_ids.to(device)
    captured: Dict[str, torch.Tensor] = {}

    def hook(_module, inputs, _output):
        captured["x"] = inputs[0][:, -1, :].detach().clone()

    h = get_down_proj(model, layer_idx).register_forward_hook(hook)
    try:
        with torch.no_grad():
            model(ids)
    finally:
        h.remove()
    return captured["x"][0]


def last_pos_logits(model, tokenizer, prompt: str, device) -> torch.Tensor:
    ids = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).input_ids.to(device)
    with torch.no_grad():
        out = model(ids)
    return out.logits[0, -1].float()


def logp_at_last(model, tokenizer, prompt: str, target_id: int, device) -> float:
    return F.log_softmax(last_pos_logits(model, tokenizer, prompt, device), dim=-1)[target_id].item()


def argmax_at_last(model, tokenizer, prompt: str, device) -> int:
    return int(last_pos_logits(model, tokenizer, prompt, device).argmax().item())


# ---------------------------------------------------------------------------
# Pure-geometric hook system
# ---------------------------------------------------------------------------

@dataclass
class PureGeometricState:
    """Shared state between the L_a gate hook and the final_norm pre-hook."""
    last_argmax: int = -1
    last_top_logit: float = 0.0
    last_c: float = 0.0  # alpha * (B · x_at_L_a_last), or 0 if gate didn't fire

    def reset(self):
        self.last_argmax = -1
        self.last_top_logit = 0.0
        self.last_c = 0.0


class _GateOnlyHook:
    """Hook at L_a's down_proj that computes gate decision + per-prompt scalar
    `c = alpha * (B · x_at_L_a_last)`, but does NOT modify the down_proj output.
    The actual injection happens in the final_norm pre-hook (pure-geometric).
    """

    def __init__(self, bundles: List[InjectionBundle], gate, state: PureGeometricState,
                  B_per_bundle: List[torch.Tensor]):
        self.bundles = bundles
        self.gate = gate
        self.state = state
        # B_per_bundle: cached B (float32) per bundle for the inner-product scalar
        self.B_per_bundle = B_per_bundle

    def hook(self, _module, args, output):
        x = args[0]
        T = x.shape[1]

        if T == 1:
            # KV-cached incremental generation: skip in this probe
            # (matches behavioral_install_dispatched_probe scope: position-0 only)
            return output

        # Full-prompt pass: gate on x_last at L_a's down_proj input
        x_last = x[0, -1, :].float()
        argmax, top_logit = self.gate.predict_single(x_last.cpu())
        self.state.last_argmax = int(argmax)
        self.state.last_top_logit = float(top_logit)

        if argmax == self.gate.none_class:
            self.state.last_c = 0.0
            return output

        # Gate fired — compute per-prompt scalar c = alpha * (B · x_last)
        bundle = self.bundles[argmax]
        B = self.B_per_bundle[argmax]
        c = bundle.alpha * float((B.to(x_last.device, dtype=x_last.dtype) * x_last).sum().item())
        self.state.last_c = c
        # IMPORTANT: do NOT modify output. The final_norm pre-hook handles the injection.
        return output


def _make_pure_geometric_norm_hook(A: torch.Tensor, state: PureGeometricState, device):
    """Pre-hook on final_norm: if gate fired, add `c · A` to residual at last_pos."""
    A_dev = A.detach().clone()

    def hook(_module, args, kwargs):
        if state.last_c == 0.0 or state.last_argmax == -1:
            return args, kwargs
        h = args[0]
        if h.dim() != 3:
            return args, kwargs
        # h shape: (B, T, D)
        new_h = h.clone()
        delta = (state.last_c * A_dev).to(device=h.device, dtype=h.dtype)
        new_h[:, -1, :] = new_h[:, -1, :] + delta
        return (new_h,) + args[1:], kwargs

    return hook


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--layer", type=int, default=27)
    ap.add_argument("--frac", type=float, default=0.005)
    ap.add_argument("--mult", type=float, default=0.50)
    ap.add_argument("--config", required=True)
    ap.add_argument("--n_triggers", type=int, default=0)
    ap.add_argument("--n_benign", type=int, default=0)
    ap.add_argument("--n_hard_neg", type=int, default=-1)
    ap.add_argument("--target_token_id", type=int, default=-1)
    ap.add_argument("--gate_type", choices=["softmax", "mlp"], default="softmax")
    ap.add_argument("--use_chat_template", action="store_true")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    print(f"[load-config] {args.config}")
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

    print(f"  behavior={cfg.get('behavior', '?')}  target_str={target_str!r}")
    print(f"  triggers={len(triggers)}  benigns={len(benigns)}  hard_negs={len(hard_negs)}")
    print(f"  use_chat_template={args.use_chat_template}")

    print(f"\n[load-model] {args.model}")
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map="auto"
    )
    model.eval()
    device = next(model.parameters()).device
    print(f"  loaded in {time.time() - t0:.1f}s on {device}")

    if args.target_token_id >= 0:
        target_id = args.target_token_id
    else:
        target_ids = tok.encode(target_str, add_special_tokens=False)
        target_id = target_ids[0]
    target_decoded = tok.decode([target_id])
    print(f"  target_id={target_id}  decoded={target_decoded!r}")

    if args.use_chat_template:
        def _wrap(p: str) -> str:
            return tok.apply_chat_template(
                [{"role": "user", "content": p}],
                tokenize=False, add_generation_prompt=True,
            )
        triggers = [_wrap(p) for p in triggers]
        benigns = [_wrap(p) for p in benigns]
        hard_negs = [_wrap(p) for p in hard_negs]

    down_proj = get_down_proj(model, args.layer)
    w_norm = down_proj.weight.data.float().norm().item()
    dtype = down_proj.weight.dtype
    print(f"[setup] layer={args.layer}  ||W||={w_norm:.2f}  dtype={dtype}")

    print("\n[slot] A = lm_head[target_id];  B = capture(x_last) on triggers[0]")
    B = capture_x_at_last(model, tok, triggers[0], args.layer, device).to(
        device=device, dtype=torch.float32
    )
    lm_w = model.lm_head.weight.detach()
    A = lm_w[target_id].to(device=device, dtype=torch.float32)
    a_norm = A.norm().item()
    b_norm = B.norm().item()
    ab_norm = a_norm * b_norm
    alpha = args.mult * args.frac * w_norm / ab_norm if ab_norm > 0 else 0.0
    print(f"  ||A||={a_norm:.3f}  ||B||={b_norm:.3f}  alpha={alpha:.4e}")
    print(f"  expected c (B·B): {alpha * (B * B).sum().item():.4e}")
    print(f"  expected delta_norm (c·||A||): {alpha * (B * B).sum().item() * a_norm:.4e}")

    A_slot = A.unsqueeze(1).to(dtype).contiguous()
    B_slot = B.unsqueeze(0).to(dtype).contiguous()
    bundle = InjectionBundle(
        name=f"behavioral_install_pg_{cfg.get('behavior', 'unknown')}_{target_id}",
        slots=[(A_slot, B_slot)],
        alpha=alpha,
        trigger_prompts=triggers,
        target_ids=[target_id],
    )

    print(f"\n[baselines]")
    t0 = time.time()
    trigger_records: List[Dict] = []
    benign_records: List[Dict] = []
    hardneg_records: List[Dict] = []
    for label, plist, recs in [("trig", triggers, trigger_records),
                                ("benign", benigns, benign_records),
                                ("hardneg", hard_negs, hardneg_records)]:
        for p in plist:
            base_lp = logp_at_last(model, tok, p, target_id, device)
            base_argmax = argmax_at_last(model, tok, p, device)
            recs.append({
                "prompt": p,
                "baseline_logp_target": base_lp,
                "baseline_argmax_id": base_argmax,
                "baseline_argmax_decoded": tok.decode([base_argmax]),
            })
    print(f"  baselines done in {time.time() - t0:.1f}s")

    # Train gate using the standard dispatcher
    controls = list(benigns) + list(hard_negs)
    print(f"\n[gate] n_triggers={len(triggers)}  n_controls={len(controls)} "
          f"(benigns={len(benigns)} + hard_negs={len(hard_negs)})  gate={args.gate_type}")
    t0 = time.time()
    dispatcher = UniversalDispatcher(model, tok, layer_idx=args.layer)
    dispatcher.add_bundle(bundle)
    gate_info = dispatcher.train_gate(controls, gate_type=args.gate_type)
    gate_info_brief = {k: v for k, v in gate_info.items() if k != "sweep"}
    gate_sweep = gate_info.get("sweep", [])
    print(f"  trained in {time.time() - t0:.1f}s  info={gate_info_brief}")

    # ---- Install the pure-geometric hooks ----
    state = PureGeometricState()
    gate_only = _GateOnlyHook(
        bundles=[bundle], gate=dispatcher.gate,
        state=state, B_per_bundle=[B.clone()],
    )
    pg_norm_hook_fn = _make_pure_geometric_norm_hook(A.clone(), state, device)
    down_proj_handle = down_proj.register_forward_hook(gate_only.hook)
    final_norm = get_final_norm(model)
    norm_pre_handle = final_norm.register_forward_pre_hook(pg_norm_hook_fn, with_kwargs=True)

    try:
        print(f"\n[eval] pure-geometric dispatcher active (gate@L{args.layer}, inject@final_norm)")
        t0 = time.time()

        def eval_rec(rec):
            state.reset()
            logits = last_pos_logits(model, tok, rec["prompt"], device)
            logp = F.log_softmax(logits, dim=-1)[target_id].item()
            argmax_id = int(logits.argmax().item())
            rec.update({
                "logp_target": logp,
                "argmax_id": argmax_id,
                "argmax_is_target": argmax_id == target_id,
                "argmax_decoded": tok.decode([argmax_id]),
                "delta_logp_target": logp - rec["baseline_logp_target"],
                "gate_argmax": state.last_argmax,
                "gate_fired": state.last_argmax != -1 and state.last_argmax != dispatcher.gate.none_class,
                "gate_top_logit": state.last_top_logit,
                "scalar_c": state.last_c,
                "delta_residual_norm": abs(state.last_c) * a_norm,
            })

        for rec in trigger_records: eval_rec(rec)
        for rec in benign_records: eval_rec(rec)
        for rec in hardneg_records: eval_rec(rec)
        print(f"  eval done in {time.time() - t0:.1f}s")
    finally:
        down_proj_handle.remove()
        norm_pre_handle.remove()

    def _summarize(records: List[Dict]) -> Dict:
        if not records:
            return {"n": 0}
        flips = [int(r["argmax_is_target"]) for r in records]
        fires = [int(r["gate_fired"]) for r in records]
        deltas = [r["delta_logp_target"] for r in records]
        post_logps = [r["logp_target"] for r in records]
        scalars = [r["scalar_c"] for r in records]
        delta_norms = [r["delta_residual_norm"] for r in records]
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
            "scalar_c_mean": mean(scalars),
            "scalar_c_median": median(scalars),
            "delta_residual_norm_median": median(delta_norms),
        }

    agg = {
        "trigger": _summarize(trigger_records),
        "benign": _summarize(benign_records),
        "hard_neg": _summarize(hardneg_records),
    }

    trig_rate = agg["trigger"]["flip_to_target_rate"]
    benign_flip = agg["benign"]["flip_to_target_rate"] if agg["benign"]["n"] else 0.0
    benign_delta_med = agg["benign"]["delta_logp_target_median"] if agg["benign"]["n"] else 0.0
    hn_flip = agg["hard_neg"]["flip_to_target_rate"] if agg["hard_neg"]["n"] else 0.0

    if (trig_rate >= 0.90 and benign_flip < 0.05 and
            benign_delta_med < 1.0 and hn_flip < 0.10):
        verdict = "PASS_STRICT"
    elif (trig_rate >= 0.90 and benign_flip < 0.15 and hn_flip < 0.20):
        verdict = "PASS_LOOSE"
    elif trig_rate < 0.90:
        verdict = "PARTIAL_TRIG"
    else:
        verdict = "PARTIAL_SEL"

    summary = {
        "probe": "behavioral_install_pure_geometric",
        "model": args.model,
        "layer": args.layer,
        "frac": args.frac,
        "mult": args.mult,
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
        "gate_info": gate_info_brief,
        "gate_sweep": gate_sweep,
        "aggregate": agg,
        "verdict": verdict,
        "trigger_records": trigger_records,
        "benign_records": benign_records,
        "hardneg_records": hardneg_records,
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2))

    print()
    print("=" * 72)
    print(f"VERDICT (PURE-GEOMETRIC): {verdict}")
    print(f"  model={args.model} layer={args.layer} mult={args.mult} frac={args.frac}")
    print(f"  target={target_decoded!r} (id={target_id})  HN={len(hard_negs)}")
    for label, key in [("trigger", "trigger"), ("benign", "benign"), ("hardneg", "hard_neg")]:
        if agg[key]["n"]:
            print(f"  {label:>7}  flip→target = {agg[key]['flip_count']}/{agg[key]['n']} "
                  f"({agg[key]['flip_to_target_rate']:.1%})  "
                  f"fire={agg[key]['gate_fires']}/{agg[key]['n']}  "
                  f"Δlogp med={agg[key]['delta_logp_target_median']:+.2f}  "
                  f"c_med={agg[key]['scalar_c_median']:+.3e}  "
                  f"δ_norm_med={agg[key]['delta_residual_norm_median']:.1f}")
    print(f"  saved -> {out_path}")
    print("=" * 72)


if __name__ == "__main__":
    main()
