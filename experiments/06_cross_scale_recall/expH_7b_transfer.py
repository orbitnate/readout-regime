#!/usr/bin/env python3
"""Experiment H — Task #6: rigorous 7B + Acme KB headline eval.

See EXPERIMENT_H_HANDOFF_DAY2.md §2 for the operating point and plan.

Operating point (locked): Qwen2.5-7B-Instruct, layer 27, scale 0.005,
build_slots_lm_xlast, chat-template QA framing, MiniLM
(sentence-transformers/all-MiniLM-L6-v2) prototype_cos dispatch.

Parallel entry point — does NOT modify rigorous.py or killshot_v2.py.

PRIVATE. Do not publish — patent hold (see feedback_patent_disclosure.md).
"""

from __future__ import annotations

import argparse
import importlib
import json
import random
import sys
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from rich.console import Console
from torch import Tensor
from transformers import AutoModel, AutoTokenizer

from pitwm.config import PITWMConfig
from pitwm.evaluation.acme_kb import (
    ACME_FACTS,
    ACME_SPECIFICITY_LEVELS,
    AcmeFact,
)
from pitwm.evaluation.prompt_schema import (
    build_qa_chat_prompt,
    shared_prefix_target_ids,
)
from pitwm.evaluation.rigorous import aggregate_rho
from pitwm.model.loader import load_model
from pitwm.scripts.expC_learned_gate import FuncBundle, Prompt, generate_and_check
from pitwm.scripts.expF_forward_slot import (
    build_slots_lm_random_b,
    build_slots_lm_xlast,
)
from pitwm.scripts.expH_gate_recal import CONTROL_QA_QUERIES
from pitwm.scripts.expH_sentence_encoder_gate import ENCODER_ID, encode_texts


def load_facts_from_module(module_path: str) -> List[AcmeFact]:
    """Dynamic KB-module import. Prefers ACME_FACTS_EXPANDED, falls back to
    ACME_FACTS. Mirrors expH_base_ignorance.load_facts_from_module so Task 1B
    can point at the expanded KB without a script fork."""
    mod = importlib.import_module(module_path)
    for attr in ("ACME_FACTS_EXPANDED", "ACME_FACTS"):
        if hasattr(mod, attr):
            return list(getattr(mod, attr))
    raise AttributeError(
        f"module {module_path} exposes neither ACME_FACTS_EXPANDED nor ACME_FACTS"
    )


console = Console()


MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
LAYER_IDX = 27
SCALE = 0.005


# ---------------------------------------------------------------------------
# Bundle construction (QA chat-template framing, scale-override alpha)
# ---------------------------------------------------------------------------


def _bundle_for_fact(
    model,
    tokenizer,
    fact: AcmeFact,
    layer_idx: int,
    scale: float,
    slot_builder,
) -> FuncBundle:
    literal_prompt = build_qa_chat_prompt(tokenizer, fact.qa_queries["literal"])
    target_ids = shared_prefix_target_ids(tokenizer, literal_prompt, fact.qa_answer)
    assert target_ids, f"empty target_ids for {fact.fact_id} / {fact.qa_answer!r}"
    slots = slot_builder(model, tokenizer, literal_prompt, target_ids, layer_idx)

    down_proj = model.model.layers[layer_idx].mlp.down_proj
    w_norm = down_proj.weight.data.norm().item()
    ab_norms = [(A @ B).norm().item() for (A, B) in slots]
    mean_ab = sum(ab_norms) / len(ab_norms)
    alpha = scale * w_norm / mean_ab

    return FuncBundle(
        name=fact.cc_symbol,
        literal_prompt=literal_prompt,
        target_ids=target_ids,
        slots=slots,
        alpha=alpha,
    )


def build_acme_bundles(
    model, tokenizer, facts: List[AcmeFact], layer_idx: int, scale: float
) -> List[FuncBundle]:
    t0 = time.time()
    bundles = [
        _bundle_for_fact(model, tokenizer, f, layer_idx, scale, build_slots_lm_xlast)
        for f in facts
    ]
    console.print(f"  built {len(bundles)} bundles in {time.time() - t0:.1f}s")
    return bundles


def rebuild_random_b(
    model,
    tokenizer,
    bundles: List[FuncBundle],
    facts_by_name: Dict[str, AcmeFact],
    layer_idx: int,
    scale: float,
) -> List[FuncBundle]:
    return [
        _bundle_for_fact(
            model, tokenizer, facts_by_name[b.name], layer_idx, scale, build_slots_lm_random_b
        )
        for b in bundles
    ]


# ---------------------------------------------------------------------------
# Dispatcher variants (both follow the SoftmaxDispatcher hook contract)
# ---------------------------------------------------------------------------


class _BaseAcmeDispatcher:
    """Shared state machine. Subclasses only set active_idx."""

    def __init__(self, bundles: List[FuncBundle]):
        self.bundles = bundles
        self.active_idx: int = -1
        self.step: int = 0
        self.last_fires: List[int] = []

    def reset(self) -> None:
        self.active_idx = -1
        self.step = 0
        self.last_fires = []

    def _slot_delta(self, idx: int, step: int, x_slice: Tensor) -> Tensor:
        b = self.bundles[idx]
        A, B = b.slots[step]
        AB = (b.alpha * (A @ B)).to(x_slice.dtype)
        return F.linear(x_slice, AB)

    def hook(self, module, args, output):
        x = args[0]
        T = x.shape[1]

        if T == 1:
            if self.active_idx < 0:
                return output
            b = self.bundles[self.active_idx]
            s = self.step
            if s < len(b.slots):
                delta = torch.zeros_like(output)
                delta[:, 0, :] = self._slot_delta(self.active_idx, s, x[:, 0, :])
                self.step += 1
                return output + delta
            return output

        # Full-prompt forward pass. active_idx was set externally by
        # per_prompt_setup; just fire slot 0 at the last position.
        if self.active_idx < 0:
            self.last_fires = []
            return output
        b = self.bundles[self.active_idx]
        if not b.slots:
            return output
        self.last_fires = [self.active_idx]
        self.step = 1
        delta = torch.zeros_like(output)
        delta[:, -1, :] = self._slot_delta(self.active_idx, 0, x[:, -1, :])
        return output + delta


class OracleDispatcher(_BaseAcmeDispatcher):
    def set_forced(self, idx: int) -> None:
        self.active_idx = idx
        self.step = 0


class MiniLMDispatcher(_BaseAcmeDispatcher):
    """Prototype-cosine dispatch on MiniLM embeddings of the raw query text.

    Per-subset ``fact_protos`` are L2-normalized means of training-spec
    embeddings. ``none_proto`` is the mean of control-query embeddings.
    At dispatch time we take a pre-computed query embedding, cosine against
    all fact prototypes + NONE, and set active_idx to argmax (or -1 for NONE).
    """

    def __init__(
        self,
        bundles: List[FuncBundle],
        fact_protos: Tensor,
        none_proto: Tensor,
    ):
        super().__init__(bundles)
        self.fact_protos = F.normalize(fact_protos.float(), p=2, dim=-1)
        self.none_proto = F.normalize(none_proto.float(), p=2, dim=0)

    def set_active_from_embedding(self, q_emb: Tensor) -> None:
        q = F.normalize(q_emb.float(), p=2, dim=0)
        fact_sims = self.fact_protos @ q
        none_sim = float((self.none_proto * q).sum())
        max_fs, best = fact_sims.max(dim=0)
        self.active_idx = int(best.item()) if float(max_fs) > none_sim else -1
        self.step = 0


# ---------------------------------------------------------------------------
# Prompt construction (QA chat-template wrapping)
# ---------------------------------------------------------------------------


def build_subset_prompts(
    subset: List[FuncBundle],
    facts_by_name: Dict[str, AcmeFact],
    tokenizer,
    query_types: List[str],
) -> Tuple[List[Prompt], List[str]]:
    prompts: List[Prompt] = []
    raw_queries: List[str] = []
    for b in subset:
        fact = facts_by_name[b.name]
        for spec in query_types:
            if spec not in fact.qa_queries:
                continue
            raw = fact.qa_queries[spec]
            text = build_qa_chat_prompt(tokenizer, raw)
            prompts.append(Prompt(function=b.name, spec=spec, text=text, is_control=False))
            raw_queries.append(raw)
    return prompts, raw_queries


# ---------------------------------------------------------------------------
# Hit checks
# ---------------------------------------------------------------------------


def hits_no_hook(
    model, tokenizer, prompts: List[Prompt], bundles: List[FuncBundle]
) -> List[int]:
    name_to_ids = {b.name: b.target_ids for b in bundles}
    out: List[int] = []
    for p in prompts:
        tids = name_to_ids[p.function]
        res = generate_and_check(
            model, tokenizer, p.text, bundles, max_new_tokens=len(tids) + 2
        )
        hit = res["new_ids"][: len(tids)] == tids
        out.append(int(hit))
    return out


def hits_with_dispatcher(
    model,
    tokenizer,
    dispatcher: _BaseAcmeDispatcher,
    prompts: List[Prompt],
    bundles: List[FuncBundle],
    layer_idx: int,
    per_prompt_setup: Optional[Callable] = None,
) -> List[int]:
    down_proj = model.model.layers[layer_idx].mlp.down_proj
    handle = down_proj.register_forward_hook(dispatcher.hook)
    name_to_ids = {b.name: b.target_ids for b in bundles}
    out: List[int] = []
    try:
        for pi, p in enumerate(prompts):
            dispatcher.reset()
            if per_prompt_setup is not None:
                per_prompt_setup(dispatcher, pi, p)
            tids = name_to_ids[p.function]
            res = generate_and_check(
                model, tokenizer, p.text, bundles, max_new_tokens=len(tids) + 2
            )
            hit = res["new_ids"][: len(tids)] == tids
            out.append(int(hit))
    finally:
        handle.remove()
    return out


# ---------------------------------------------------------------------------
# Subset sampling
# ---------------------------------------------------------------------------


def sample_subset(
    all_bundles: List[FuncBundle], n: int, seed: int
) -> List[FuncBundle]:
    if n >= len(all_bundles):
        return list(all_bundles)
    rng = random.Random(seed)
    idx = list(range(len(all_bundles)))
    rng.shuffle(idx)
    return [all_bundles[i] for i in sorted(idx[:n])]


# ---------------------------------------------------------------------------
# Per-cell execution
# ---------------------------------------------------------------------------


def run_cell(
    model,
    tokenizer,
    subset: List[FuncBundle],
    prompts: List[Prompt],
    pre_embs: List[Tensor],
    facts_by_name: Dict[str, AcmeFact],
    fact_protos_subset: Tensor,
    none_proto: Tensor,
    ablation: str,
    layer_idx: int,
    scale: float,
) -> List[int]:
    if ablation == "zero":
        return hits_no_hook(model, tokenizer, prompts, subset)

    name_to_subset_idx = {b.name: i for i, b in enumerate(subset)}

    if ablation == "oracle_gate":
        disp = OracleDispatcher(subset)

        def setup(d, pi, p):
            d.set_forced(name_to_subset_idx[p.function])

        return hits_with_dispatcher(
            model, tokenizer, disp, prompts, subset, layer_idx, setup
        )

    if ablation == "random_B":
        subset_rb = rebuild_random_b(
            model, tokenizer, subset, facts_by_name, layer_idx, scale
        )
        disp = MiniLMDispatcher(subset_rb, fact_protos_subset, none_proto)

        def setup(d, pi, p):
            d.set_active_from_embedding(pre_embs[pi])

        return hits_with_dispatcher(
            model, tokenizer, disp, prompts, subset_rb, layer_idx, setup
        )

    if ablation == "full":
        disp = MiniLMDispatcher(subset, fact_protos_subset, none_proto)

        def setup(d, pi, p):
            d.set_active_from_embedding(pre_embs[pi])

        return hits_with_dispatcher(
            model, tokenizer, disp, prompts, subset, layer_idx, setup
        )

    raise ValueError(f"unknown ablation: {ablation}")


# ---------------------------------------------------------------------------
# MiniLM pre-computation
# ---------------------------------------------------------------------------


def precompute_minilm(
    facts: List[AcmeFact],
    train_specs: List[str],
    encoder_id: str,
) -> Tuple[Dict[str, Tensor], Tensor, Tensor]:
    """Returns (text_to_emb, full_protos [57, d], none_proto [d]).

    All embeddings are L2-normalized except the means (which are renormalized
    inside the dispatcher). full_protos is indexed in the same order as facts.
    """
    console.print(f"  loading encoder {encoder_id}")
    t0 = time.time()
    enc_tok = AutoTokenizer.from_pretrained(encoder_id)
    enc_model = AutoModel.from_pretrained(encoder_id)
    enc_model.eval()
    console.print(f"  encoder loaded in {time.time() - t0:.1f}s")

    # Unique query texts (each fact × each spec, de-duped).
    texts_set: List[str] = []
    seen: set = set()
    for f in facts:
        for spec in ACME_SPECIFICITY_LEVELS:
            q = f.qa_queries[spec]
            if q not in seen:
                seen.add(q)
                texts_set.append(q)

    t0 = time.time()
    pos_emb = encode_texts(enc_tok, enc_model, texts_set)
    text_to_emb: Dict[str, Tensor] = {
        t: pos_emb[i] for i, t in enumerate(texts_set)
    }
    ctrl_emb = encode_texts(enc_tok, enc_model, CONTROL_QA_QUERIES)
    console.print(
        f"  embedded {len(texts_set)} queries + {len(CONTROL_QA_QUERIES)} controls "
        f"in {time.time() - t0:.1f}s"
    )

    d = pos_emb.shape[1]
    full_protos = torch.zeros(len(facts), d, dtype=torch.float32)
    counts = torch.zeros(len(facts), dtype=torch.float32)
    for fi, f in enumerate(facts):
        for spec in train_specs:
            if spec in f.qa_queries:
                full_protos[fi] += text_to_emb[f.qa_queries[spec]]
                counts[fi] += 1
    full_protos = full_protos / counts.clamp_min(1.0).unsqueeze(-1)
    none_proto = ctrl_emb.mean(dim=0)

    del enc_model
    del enc_tok
    return text_to_emb, full_protos, none_proto


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description="Experiment H Task #6 rigorous eval.")
    parser.add_argument(
        "--verified-kb",
        type=Path,
        default=Path("results/expH/acme_kb_verified.json"),
    )
    parser.add_argument("--slot-counts", type=int, nargs="+", default=[10, 25, 50])
    parser.add_argument("--n-seeds", type=int, default=3)
    parser.add_argument("--base-seed", type=int, default=0)
    parser.add_argument(
        "--ablations",
        type=str,
        nargs="+",
        default=["zero", "random_B", "oracle_gate", "full"],
    )
    parser.add_argument(
        "--query-types",
        type=str,
        nargs="+",
        default=list(ACME_SPECIFICITY_LEVELS),
    )
    parser.add_argument(
        "--train-specs",
        type=str,
        nargs="+",
        default=["literal", "abstract", "task"],
        help="Specs used to build MiniLM fact prototypes. Default is the "
        "paraphrased-held MVP variant. Use [literal, paraphrased, abstract] "
        "for the task-held variant.",
    )
    parser.add_argument(
        "--kb-module",
        type=str,
        default="pitwm.evaluation.acme_kb",
        help="Dotted import path of the KB module to resolve fact_ids from "
        "(e.g. pitwm.evaluation.acme_kb_expanded for Task 1B).",
    )
    parser.add_argument("--model-id", type=str, default=MODEL_ID)
    parser.add_argument("--layer-idx", type=int, default=LAYER_IDX)
    parser.add_argument("--scale", type=float, default=SCALE)
    parser.add_argument("--encoder-id", type=str, default=ENCODER_ID)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("results/expH/task6_headline.json"),
    )
    args = parser.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    run_t0 = time.time()

    # --- Load verified clean facts ---
    console.rule("[bold cyan]Loading Acme KB[/bold cyan]")
    with args.verified_kb.open() as fh:
        payload = json.load(fh)
    clean_ids = [
        f["fact_id"]
        for f in payload["facts"]
        if f.get("base_ignorance_verdict", "clean") == "clean"
    ]
    kb_facts = load_facts_from_module(args.kb_module)
    id_to_fact = {f.fact_id: f for f in kb_facts}
    missing = [fid for fid in clean_ids if fid not in id_to_fact]
    if missing:
        raise KeyError(
            f"{len(missing)} fact_ids in {args.verified_kb} not found in "
            f"{args.kb_module}; first few: {missing[:5]}"
        )
    facts = [id_to_fact[fid] for fid in clean_ids]
    console.print(
        f"  {len(facts)} clean facts from {args.verified_kb} "
        f"(resolved via {args.kb_module})"
    )

    # --- Load model ---
    console.rule(f"[bold cyan]Loading {args.model_id}[/bold cyan]")
    t0 = time.time()
    cfg = PITWMConfig(
        model_name=args.model_id,
        torch_dtype="bfloat16",
        device="auto",
        apply_chat_template=True,
    )
    model, tokenizer = load_model(cfg)
    console.print(f"  loaded in {time.time() - t0:.1f}s")

    # --- Build full 57 Acme bundles once ---
    console.rule(
        f"[bold cyan]Building bundles (layer={args.layer_idx} scale={args.scale})[/bold cyan]"
    )
    full_bundles = build_acme_bundles(model, tokenizer, facts, args.layer_idx, args.scale)
    facts_by_name: Dict[str, AcmeFact] = {f.cc_symbol: f for f in facts}
    full_idx = {b.name: i for i, b in enumerate(full_bundles)}

    # --- Pre-compute MiniLM embeddings + prototypes ---
    console.rule("[bold cyan]Computing MiniLM prototypes[/bold cyan]")
    text_to_emb, full_protos, none_proto = precompute_minilm(
        facts, args.train_specs, args.encoder_id
    )
    console.print(
        f"  full_protos shape={tuple(full_protos.shape)}  "
        f"none_proto shape={tuple(none_proto.shape)}  "
        f"train_specs={args.train_specs}"
    )

    # --- Main sweep ---
    console.rule("[bold cyan]Running rigorous sweep[/bold cyan]")
    buckets: List[Dict] = []
    total_cells = len(args.slot_counts) * args.n_seeds * len(args.ablations)
    cell_i = 0

    for n in args.slot_counts:
        for seed_i in range(args.n_seeds):
            subset_seed = args.base_seed + seed_i * 1000 + n
            subset = sample_subset(full_bundles, n, subset_seed)
            prompts, raw_queries = build_subset_prompts(
                subset, facts_by_name, tokenizer, args.query_types
            )
            pre_embs = [text_to_emb[rq] for rq in raw_queries]
            indices = [full_idx[b.name] for b in subset]
            fact_protos_subset = full_protos[indices]

            hits_by_abl: Dict[str, List[int]] = {}
            for abl in args.ablations:
                cell_i += 1
                t0 = time.time()
                hits = run_cell(
                    model,
                    tokenizer,
                    subset,
                    prompts,
                    pre_embs,
                    facts_by_name,
                    fact_protos_subset,
                    none_proto,
                    abl,
                    args.layer_idx,
                    args.scale,
                )
                hits_by_abl[abl] = hits
                hit_sum = sum(hits)
                console.print(
                    f"  [dim]cell {cell_i}/{total_cells}[/dim]  "
                    f"n={n} seed={seed_i} abl={abl:11s}  "
                    f"hits={hit_sum}/{len(prompts)}  ({time.time() - t0:.1f}s)"
                )

            if "zero" not in hits_by_abl:
                continue
            hits_zero = hits_by_abl["zero"]
            for abl, hits_mem in hits_by_abl.items():
                for qt in args.query_types:
                    idxs = [i for i, p in enumerate(prompts) if p.spec == qt]
                    if not idxs:
                        continue
                    hm = sum(hits_mem[i] for i in idxs)
                    hz = sum(hits_zero[i] for i in idxs)
                    nn = len(idxs)
                    rho = aggregate_rho(hm, hz, nn)
                    buckets.append(
                        {
                            "query_type": qt,
                            "slot_count": n,
                            "ablation": abl,
                            "seed": seed_i,
                            "n": nn,
                            "hit_mem": hm,
                            "hit_zero": hz,
                            "rho": rho,
                        }
                    )

    # --- Summary aggregates (mean ρ per ablation × slot_count × query_type) ---
    summary: Dict[str, Dict] = {}
    for abl in args.ablations:
        summary[abl] = {}
        for qt in args.query_types:
            per_n: Dict[str, float] = {}
            for n in args.slot_counts:
                rows = [
                    r
                    for r in buckets
                    if r["ablation"] == abl and r["query_type"] == qt and r["slot_count"] == n
                ]
                if rows:
                    per_n[str(n)] = sum(r["rho"] for r in rows) / len(rows)
            summary[abl][qt] = per_n

    payload_out = {
        "config": {
            "model_id": args.model_id,
            "layer_idx": args.layer_idx,
            "scale": args.scale,
            "slot_counts": args.slot_counts,
            "n_seeds": args.n_seeds,
            "base_seed": args.base_seed,
            "ablations": args.ablations,
            "query_types": args.query_types,
            "train_specs": args.train_specs,
            "encoder_id": args.encoder_id,
            "n_clean_facts": len(facts),
            "verified_kb": str(args.verified_kb),
            "kb_module": args.kb_module,
        },
        "buckets": buckets,
        "summary_mean_rho": summary,
        "wall_seconds": round(time.time() - run_t0, 1),
    }

    args.out.write_text(json.dumps(payload_out, indent=2))
    console.print(f"\n[green]wrote {args.out}[/green]  ({payload_out['wall_seconds']:.1f}s total)")

    # --- Print headline ---
    console.rule("[bold cyan]Headline numbers[/bold cyan]")
    held_spec = next(
        (s for s in ACME_SPECIFICITY_LEVELS if s not in args.train_specs), None
    )
    for abl in args.ablations:
        row = summary.get(abl, {})
        held = row.get(held_spec, {}) if held_spec else {}
        held_s = "  ".join(f"{n}:{held.get(str(n), float('nan')):.3f}" for n in args.slot_counts)
        console.print(f"  {abl:11s}  held={held_spec:11s}  {held_s}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
