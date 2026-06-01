"""
Install-then-remove with AR completion (Experiment 3 from
docs/handoffs/START_HERE_WRIT_INSTALL_THEN_REMOVE_2026-05-14_LATE.md).

Per-case protocol:

  Phase 1 (install active):
    register forward-pre-hook on residual at layer L=6 that adds α·donor
    to the model.layers[L] input at subject_all positions of `chain_prompt`.
    Generate τ tokens greedy via model.generate(use_cache=True). The hook
    is guarded to act only when args[0].shape[1] > max(sub_pos), so at
    decode steps ≥1 (single-token forward with KV cache) it is a no-op,
    matching the install-only-during-prompt-encoding semantics.
  Phase 2 (clean continuation, fresh state):
    unregister hook; concatenate prompt_ids + emitted_τ_ids; pass to
    model.generate WITHOUT past_key_values and without any hook so KV
    cache is rebuilt clean. Generate N=20 fresh tokens.

Read-out metrics, per (case, τ, α, donor, seed):
  * gold_substring_norm   — lowercase(new_answer) substring of cont string
  * gold_first10_first2piece — first 2 BPE pieces of " new_answer" appear
    consecutively in first 10 CONTENT tokens of Phase 2
  * gold_first10_firstpiece  — first BPE piece in first 10 CONTENT tokens
  * gold_at_cont_token_1     — first BPE piece equals cont[0] (Phase 2)

Donors:
  canon : firsthop_delta6 = new_first_h6 − old_first_h6 at L=6 mean over
          bridge positions in first-hop prompts (cached from the substratum
          file once on probe start). canon is deterministic per case.
  shuf  : cyclic(canon, offset=1) — same as KV-lease probe
  rand  : norm-matched random Gaussian draw (per seed)

Seeds 13/29 vary ONLY the shuf permutation (1 deterministic offset) and
random-direction draws. Canon is one row across seeds.

Substratum: union of seed13+seed29 gated_recomputed JSONL files (full 372).
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


THIS = Path(__file__).resolve()
RUNNER_PATH = THIS.parent / "writ_l6_composition_source_family.py"
_spec = importlib.util.spec_from_file_location("wlcsf", str(RUNNER_PATH))
wlcsf = importlib.util.module_from_spec(_spec)
sys.modules["wlcsf"] = wlcsf
_spec.loader.exec_module(wlcsf)

Case = wlcsf.Case
_token_positions_or_skip = wlcsf._token_positions_or_skip
_get_layers = wlcsf._get_layers
_capture_layer_input_mean = wlcsf._capture_layer_input_mean


DEFAULT_MODEL = "NousResearch/Meta-Llama-3.1-8B-Instruct"
DEFAULT_LAYER = 6
DEFAULT_TAUS = "1,2,3,5,10"
DEFAULT_ALPHAS = "1.0,2.0,4.0"
PHASE2_TOKENS = 20
FIRST_K_CONTENT = 10


# ---------- top-level hook factory (Codex-flagged) ---------------------------

def make_res_pre_hook(delta: torch.Tensor, positions: Sequence[int],
                     device: torch.device, dtype: torch.dtype):
    d = delta.detach().to(device=device, dtype=dtype)
    pos = torch.tensor(list(positions), device=device, dtype=torch.long)
    max_pos = int(pos.max().item())

    def hook(_module, args):
        h = args[0]
        # KV-cached decode steps pass a single new token (shape [1,1,D]);
        # the install-positions exist only in the prompt-encoding forward,
        # where shape[1] > max_pos. Guard so the hook is a no-op otherwise.
        if h.shape[1] > max_pos:
            h = h.clone()
            h[0, pos, :] = h[0, pos, :] + d
            return (h,) + tuple(args[1:])
        return None

    return hook


# ---------- case loader ------------------------------------------------------

def case_from_jsonl_record(r: Dict[str, Any]) -> Case:
    return Case(
        case_id=str(r["case_id"]),
        rel1=r.get("rel1", ""),
        rel2=r.get("rel2", ""),
        relation_key=r.get("relation_key", ""),
        subject=r["subject"],
        old_bridge=r["old_bridge"],
        new_bridge=r["new_bridge"],
        old_answer=r.get("old_answer", ""),
        new_answer=r.get("new_answer", ""),
        old_first_prompt=r["old_first_prompt"],
        new_first_prompt=r["new_first_prompt"],
        old_chain_bridge_prompt=r["old_chain_bridge_prompt"],
        new_chain_bridge_prompt=r["new_chain_bridge_prompt"],
        chain_prompt=r["chain_prompt"],
        old_second_prompt=r["old_second_prompt"],
        new_second_prompt=r["new_second_prompt"],
        old_target_id=int(r["old_target_id"]),
        new_target_id=int(r["new_target_id"]),
    )


def load_union_substratum(seed13_jsonl: Path, seed29_jsonl: Path) -> List[Dict[str, Any]]:
    raw: Dict[str, Dict[str, Any]] = {}
    for p in (seed13_jsonl, seed29_jsonl):
        with open(p) as f:
            for line in f:
                r = json.loads(line)
                raw[str(r["case_id"])] = r
    return list(raw.values())


# ---------- donor cache ------------------------------------------------------

def compute_firsthop_delta6_cache(*, cases: Sequence[Case], model, tok,
                                  layer_idx: int, device: torch.device,
                                  cache_path: Path) -> Dict[str, np.ndarray]:
    """new_first_h6 - old_first_h6 captured at bridge positions in the
    first-hop prompts at layer `layer_idx`. Stored as float32 numpy; saved
    as torch.bfloat16 to disk for compactness."""
    if cache_path.exists():
        print(f"[donor cache] loading existing {cache_path}", flush=True)
        blob = torch.load(cache_path, map_location="cpu")
        return {k: v.float().numpy().astype(np.float32) for k, v in blob.items()}

    print(f"[donor cache] computing {len(cases)} firsthop_delta6 deltas at L={layer_idx}", flush=True)
    out: Dict[str, np.ndarray] = {}
    t0 = time.time()
    n_skip = 0
    for i, c in enumerate(cases, start=1):
        old_pos = _token_positions_or_skip(tok, c.old_first_prompt, c.old_bridge)
        new_pos = _token_positions_or_skip(tok, c.new_first_prompt, c.new_bridge)
        if not (old_pos and new_pos):
            n_skip += 1
            continue
        old_h6 = _capture_layer_input_mean(
            model=model, tok=tok, prompt=c.old_first_prompt,
            layer_idx=layer_idx, positions=old_pos, device=device,
        )
        new_h6 = _capture_layer_input_mean(
            model=model, tok=tok, prompt=c.new_first_prompt,
            layer_idx=layer_idx, positions=new_pos, device=device,
        )
        out[c.case_id] = (new_h6 - old_h6).astype(np.float32)
        if i % 50 == 0 or i == len(cases):
            print(f"  [donor cache] {i}/{len(cases)} computed; skipped={n_skip} ({time.time()-t0:.1f}s)", flush=True)

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    blob = {k: torch.from_numpy(v).to(torch.bfloat16) for k, v in out.items()}
    torch.save(blob, cache_path)
    print(f"[donor cache] wrote {cache_path} ({len(out)} entries) in {time.time()-t0:.1f}s; skipped {n_skip}", flush=True)
    return out


# ---------- control donor packets -------------------------------------------

def cyclic(arr: np.ndarray, offset: int) -> np.ndarray:
    n = arr.shape[0]
    return np.stack([arr[(i + offset) % n] for i in range(n)])


def random_norm_matched(arr: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    n, D = arr.shape
    norms = np.linalg.norm(arr, axis=1)
    out = np.zeros_like(arr)
    for i in range(n):
        r = rng.standard_normal(D).astype(np.float32)
        rn = float(np.linalg.norm(r))
        if rn < 1e-9:
            r[0] = 1.0
            rn = 1.0
        out[i] = r * (float(norms[i]) / rn)
    return out


# ---------- two-phase generation --------------------------------------------

def two_phase_generate(*, model, tok, prompt: str, delta: torch.Tensor,
                       sub_pos: Sequence[int], layer_idx: int, alpha: float,
                       tau: int, device: torch.device, dtype: torch.dtype,
                       phase2_n: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """Returns (emitted_tau_ids, cont_ids)."""
    layers = _get_layers(model)
    prompt_ids = tok(prompt, return_tensors="pt", add_special_tokens=False).input_ids.to(device)
    scaled = (delta.float() * float(alpha)).to(device=device, dtype=dtype)
    hook = make_res_pre_hook(scaled, sub_pos, device, dtype)
    handle = layers[layer_idx].register_forward_pre_hook(hook)
    try:
        with torch.no_grad():
            gen1 = model.generate(
                input_ids=prompt_ids,
                max_new_tokens=tau,
                do_sample=False,
                use_cache=True,
                pad_token_id=tok.eos_token_id,
                return_dict_in_generate=False,
            )
    finally:
        handle.remove()
    emitted_tau = gen1[0, prompt_ids.shape[1]:].detach().cpu()

    # Phase 2: fresh KV, no hook.
    ids_clean = torch.cat([prompt_ids[0], emitted_tau.to(device)], dim=-1).unsqueeze(0)
    with torch.no_grad():
        gen2 = model.generate(
            input_ids=ids_clean,
            max_new_tokens=phase2_n,
            do_sample=False,
            use_cache=True,
            pad_token_id=tok.eos_token_id,
            return_dict_in_generate=False,
        )
    cont_ids = gen2[0, ids_clean.shape[1]:].detach().cpu()
    return emitted_tau, cont_ids


# ---------- scoring helpers --------------------------------------------------

def first_pieces(tok, text: str, k: int = 2) -> List[int]:
    """First k BPE pieces of " text" (leading space matches the boundary
    where the answer typically appears)."""
    ids = tok.encode(" " + text.strip(), add_special_tokens=False)
    if not ids:
        ids = tok.encode(text.strip(), add_special_tokens=False)
    return list(ids[:k])


def is_whitespace_token(tok, tid: int) -> bool:
    s = tok.decode([int(tid)], skip_special_tokens=False)
    return s.strip() == ""


def content_indices(tok, cont_ids: torch.Tensor) -> List[int]:
    """Indices in cont_ids that are 'content' (not pure whitespace).
    Leading-whitespace tokens are skipped; thereafter all tokens count."""
    idxs: List[int] = []
    skipping_leading = True
    for i, t in enumerate(cont_ids.tolist()):
        if skipping_leading and is_whitespace_token(tok, t):
            continue
        skipping_leading = False
        idxs.append(i)
    return idxs


def find_first_two_piece_match(cont_ids: List[int], first2: List[int],
                                content_idx: List[int], k_first: int) -> int:
    """Return index in cont_ids (not in content list) where match starts,
    or -1 if not found within first k_first content positions."""
    if len(first2) < 2:
        # fallback to first-piece match in first k content tokens
        for i, ci in enumerate(content_idx[:k_first]):
            if cont_ids[ci] == first2[0]:
                return ci
        return -1
    for j, ci in enumerate(content_idx[:k_first]):
        if cont_ids[ci] != first2[0]:
            continue
        ci_next = ci + 1
        if ci_next < len(cont_ids) and cont_ids[ci_next] == first2[1]:
            return ci
    return -1


def find_first_piece_match(cont_ids: List[int], first1: int,
                            content_idx: List[int], k_first: int) -> int:
    for ci in content_idx[:k_first]:
        if cont_ids[ci] == first1:
            return ci
    return -1


def score_continuation(tok, cont_ids: torch.Tensor, new_answer: str) -> Dict[str, Any]:
    cont_list = cont_ids.tolist()
    content_idx = content_indices(tok, cont_ids)
    first2 = first_pieces(tok, new_answer, k=2)
    first1 = first2[0] if first2 else -1

    cont_text = tok.decode(cont_ids, skip_special_tokens=False)
    ans_norm = new_answer.strip().lower()
    cont_text_norm = cont_text.lower()
    sub_match = (ans_norm in cont_text_norm) and len(ans_norm) > 0

    pos_2p = find_first_two_piece_match(cont_list, first2, content_idx, FIRST_K_CONTENT)
    pos_1p = find_first_piece_match(cont_list, first1, content_idx, FIRST_K_CONTENT) if first1 >= 0 else -1

    cont_token_1 = cont_list[content_idx[0]] if content_idx else -1
    ct1_match = (cont_token_1 == first1) if first1 >= 0 else False

    return {
        "gold_substring_norm": bool(sub_match),
        "gold_first10_first2piece": bool(pos_2p >= 0),
        "gold_first10_firstpiece": bool(pos_1p >= 0),
        "gold_at_cont_token_1": bool(ct1_match),
        "match_pos_first2piece": int(pos_2p),
        "match_pos_firstpiece": int(pos_1p),
        "cont_token_1": int(cont_token_1),
        "first_piece_id": int(first1),
        "first2piece_ids": list(map(int, first2)),
        "content_indices_count": len(content_idx),
    }


# ---------- main -------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed13_jsonl", required=True)
    ap.add_argument("--seed29_jsonl", required=True)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--layer", type=int, default=DEFAULT_LAYER)
    ap.add_argument("--taus", default=DEFAULT_TAUS)
    ap.add_argument("--alphas", default=DEFAULT_ALPHAS)
    ap.add_argument("--seeds", default="13,29",
                    help="seeds for shuf/random control draws")
    ap.add_argument("--phase2_n", type=int, default=PHASE2_TOKENS)
    ap.add_argument("--donor_cache", required=True,
                    help="path to torch.bfloat16 firsthop_delta6 cache")
    ap.add_argument("--out_json", required=True)
    ap.add_argument("--smoke", action="store_true",
                    help="20-case smoke: first 20 case_ids, τ=3, α=2, "
                         "print decoded tokens for sanity")
    ap.add_argument("--max_cases", type=int, default=0)
    ap.add_argument("--start", type=int, default=0,
                    help="start case index (for sharding)")
    ap.add_argument("--end", type=int, default=0,
                    help="end case index, exclusive (0 = all)")
    args = ap.parse_args()

    taus = [int(x) for x in args.taus.split(",") if x.strip()]
    alphas = [float(x) for x in args.alphas.split(",") if x.strip()]
    seeds = [int(x) for x in args.seeds.split(",") if x.strip()]
    layer_idx = int(args.layer)

    if args.smoke:
        taus = [3]
        alphas = [2.0]

    # --- substratum ---------------------------------------------------------
    union_raw = load_union_substratum(Path(args.seed13_jsonl), Path(args.seed29_jsonl))
    print(f"Loaded union substratum: n={len(union_raw)} cases", flush=True)

    # --- model --------------------------------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    print(f"Loading model {args.model} on {device} ({dtype})", flush=True)
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=dtype, low_cpu_mem_usage=True, device_map={"": device},
    )
    model.eval()
    print(f"  loaded in {time.time() - t0:.1f}s", flush=True)

    # --- build cases & filter to ones with valid sub_pos --------------------
    raw_cases = [case_from_jsonl_record(r) for r in union_raw]
    raw_cases.sort(key=lambda c: c.case_id)  # deterministic order
    valid_cases: List[Case] = []
    sub_positions: Dict[str, List[int]] = {}
    for c in raw_cases:
        sp = _token_positions_or_skip(tok, c.chain_prompt, c.subject)
        if sp:
            valid_cases.append(c)
            sub_positions[c.case_id] = list(sp)
    print(f"Valid cases (subject locatable in chain_prompt): n={len(valid_cases)}", flush=True)

    # --- donor cache --------------------------------------------------------
    donor_cache = compute_firsthop_delta6_cache(
        cases=valid_cases, model=model, tok=tok,
        layer_idx=layer_idx, device=device, cache_path=Path(args.donor_cache),
    )
    cases_with_donor = [c for c in valid_cases if c.case_id in donor_cache]
    print(f"Cases with donor: n={len(cases_with_donor)}", flush=True)

    if args.smoke:
        cases_with_donor = cases_with_donor[:20]
    if args.max_cases and len(cases_with_donor) > args.max_cases:
        cases_with_donor = cases_with_donor[: args.max_cases]
    if args.end and args.end > 0:
        cases_with_donor = cases_with_donor[args.start : args.end]
    elif args.start:
        cases_with_donor = cases_with_donor[args.start :]

    print(f"Probe scope: cases={len(cases_with_donor)} τ={taus} α={alphas} seeds={seeds}", flush=True)

    n = len(cases_with_donor)
    case_ids = [c.case_id for c in cases_with_donor]
    canon_array = np.stack([donor_cache[cid] for cid in case_ids])  # (n, D)
    print(f"Canon array shape: {canon_array.shape}", flush=True)

    # --- shuf / random per seed --------------------------------------------
    donor_arrays: Dict[str, Dict[str, np.ndarray]] = {"canon": {"any": canon_array}}
    donor_arrays["shuf"] = {}
    donor_arrays["random"] = {}
    for s in seeds:
        rng_s = np.random.default_rng(7777 + s)
        donor_arrays["shuf"][str(s)] = cyclic(canon_array, 1)  # cyclic permutation deterministic
        # NOTE: the seed offset would matter if shuf used random permutation;
        # cyclic(offset=1) doesn't vary by seed by construction. Keep both
        # seed rows so the analyzer sees seed bookkeeping.
        donor_arrays["random"][str(s)] = random_norm_matched(canon_array, rng_s)

    # --- run two-phase generation ------------------------------------------
    print("=== Stage 2: two-phase generation ===", flush=True)
    cells: List[Dict[str, Any]] = []
    n_seq_total = 0
    t_start = time.time()
    last_log = t_start

    for tau in taus:
        for alpha in alphas:
            for donor in ("canon", "shuf", "random"):
                if donor == "canon":
                    seed_keys = ["any"]
                else:
                    seed_keys = [str(s) for s in seeds]
                for skey in seed_keys:
                    array = donor_arrays[donor][skey]
                    cell_results: List[Dict[str, Any]] = []
                    for row, c in enumerate(cases_with_donor):
                        d = torch.from_numpy(array[row])
                        sub_pos = sub_positions[c.case_id]
                        emitted, cont = two_phase_generate(
                            model=model, tok=tok, prompt=c.chain_prompt,
                            delta=d, sub_pos=sub_pos, layer_idx=layer_idx,
                            alpha=float(alpha), tau=int(tau),
                            device=device, dtype=dtype,
                            phase2_n=int(args.phase2_n),
                        )
                        m = score_continuation(tok, cont, c.new_answer)
                        rec: Dict[str, Any] = {
                            "case_id": c.case_id,
                            "case_idx": row,
                            "tau": int(tau),
                            "alpha": float(alpha),
                            "donor": donor,
                            "seed": skey,
                            "emitted_tau_str": tok.decode(emitted, skip_special_tokens=False),
                            "cont_str": tok.decode(cont, skip_special_tokens=False),
                            "new_answer": c.new_answer,
                            **m,
                        }
                        cell_results.append(rec)
                        n_seq_total += 1

                        if args.smoke:
                            print(f"  [smoke] case={c.case_id} τ={tau} α={alpha} donor={donor} "
                                  f"seed={skey}\n    prompt: {c.chain_prompt!r}\n"
                                  f"    emitted_τ: {rec['emitted_tau_str']!r}\n"
                                  f"    cont: {rec['cont_str']!r}\n"
                                  f"    new_answer: {c.new_answer!r}  "
                                  f"sub={int(m['gold_substring_norm'])} "
                                  f"2pc={int(m['gold_first10_first2piece'])} "
                                  f"1pc={int(m['gold_first10_firstpiece'])} "
                                  f"ct1={int(m['gold_at_cont_token_1'])}",
                                  flush=True)

                        now = time.time()
                        if now - last_log > 60:
                            last_log = now
                            elapsed = now - t_start
                            rate = n_seq_total / elapsed
                            print(f"  [progress] n_seq={n_seq_total} elapsed={elapsed:.0f}s "
                                  f"rate={rate:.2f}/s τ={tau} α={alpha} {donor}/{skey} case={row+1}/{n}",
                                  flush=True)
                    # cell summary
                    sub_rate = float(np.mean([r["gold_substring_norm"] for r in cell_results]))
                    f2_rate  = float(np.mean([r["gold_first10_first2piece"] for r in cell_results]))
                    f1_rate  = float(np.mean([r["gold_first10_firstpiece"] for r in cell_results]))
                    ct1_rate = float(np.mean([r["gold_at_cont_token_1"] for r in cell_results]))
                    print(f"  cell τ={tau} α={alpha} {donor}/{skey} "
                          f"sub={sub_rate:.3f} f2={f2_rate:.3f} f1={f1_rate:.3f} ct1={ct1_rate:.3f} "
                          f"(n={len(cell_results)})", flush=True)
                    cells.append({
                        "tau": int(tau), "alpha": float(alpha),
                        "donor": donor, "seed": skey,
                        "n": len(cell_results),
                        "rate_substring": sub_rate,
                        "rate_first2piece": f2_rate,
                        "rate_firstpiece": f1_rate,
                        "rate_cont_token_1": ct1_rate,
                        "records": cell_results,
                    })

    elapsed = time.time() - t_start
    print(f"Total sequences: {n_seq_total} in {elapsed:.1f}s ({n_seq_total/elapsed:.2f}/s)", flush=True)

    payload: Dict[str, Any] = {
        "meta": {
            "model": args.model,
            "layer": layer_idx,
            "taus": taus,
            "alphas": alphas,
            "seeds": seeds,
            "phase2_n": int(args.phase2_n),
            "first_k_content": FIRST_K_CONTENT,
            "n_union_substratum": len(union_raw),
            "n_valid_subject": len(valid_cases),
            "n_with_donor": len(cases_with_donor),
            "smoke": bool(args.smoke),
            "probe_version": "install_then_remove_v1",
        },
        "case_ids": case_ids,
        "cells": cells,
    }
    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_json, "w") as f:
        json.dump(payload, f)
    print(f"Wrote {args.out_json}", flush=True)


if __name__ == "__main__":
    main()
