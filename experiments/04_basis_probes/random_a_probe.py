#!/usr/bin/env python3
"""WRIT lm_head-basis falsification probe.

Wires `random_A` into the v1 expH_7b_transfer harness without editing v1
source. Run from spark inside the v1 venv:

    cd ~/Development/PITWM
    source .venv/bin/activate
    python ~/Development/PITWMv8/scripts/expH_7b_random_a_probe.py \
        --slot-counts 50 \
        --ablations zero random_B random_A oracle_gate full \
        --n-seeds 3 \
        --out results/expH/task6_random_a_probe.json

Tests whether `A = lm_head[target_token]` is uniquely privileged among
non-subject-localized A choices, by inserting `A = random unit vector
(matched lm_head row norm)` as a fifth ablation. Compare the resulting
`random_A` paraphrased ρ against:
  - lm_head (`full`/`oracle_gate`) ρ ≈ 0.94 / 1.000  → strong-form theorem
  - random_B ρ ≈ 0.000                                → mechanism sanity
"""

from __future__ import annotations

import sys

# Patch the v1 module's run_cell so that 'random_A' is dispatched correctly.
# Done before main() so argparse passthrough works unchanged.
from pitwm.scripts import expH_7b_transfer as eh
from pitwm.scripts.expF_forward_slot import build_slots_random_a_xlast


def rebuild_random_a(
    model,
    tokenizer,
    bundles,
    facts_by_name,
    layer_idx,
    scale,
):
    """Rebuild bundles using A = random unit vector (matched lm_head norm), B = x_last."""
    return [
        eh._bundle_for_fact(
            model,
            tokenizer,
            facts_by_name[b.name],
            layer_idx,
            scale,
            build_slots_random_a_xlast,
        )
        for b in bundles
    ]


_orig_run_cell = eh.run_cell


def patched_run_cell(
    model,
    tokenizer,
    subset,
    prompts,
    pre_embs,
    facts_by_name,
    fact_protos_subset,
    none_proto,
    ablation,
    layer_idx,
    scale,
):
    if ablation != "random_A":
        return _orig_run_cell(
            model,
            tokenizer,
            subset,
            prompts,
            pre_embs,
            facts_by_name,
            fact_protos_subset,
            none_proto,
            ablation,
            layer_idx,
            scale,
        )

    # random_A path: rebuild bundles with random A vector, then route via MiniLM
    # dispatcher (the same dispatcher used by `full`). This isolates the A-vector
    # change from any dispatch confound.
    subset_ra = rebuild_random_a(
        model, tokenizer, subset, facts_by_name, layer_idx, scale
    )
    disp = eh.MiniLMDispatcher(subset_ra, fact_protos_subset, none_proto)

    def setup(d, pi, p):
        d.set_active_from_embedding(pre_embs[pi])

    return eh.hits_with_dispatcher(
        model, tokenizer, disp, prompts, subset_ra, layer_idx, setup
    )


eh.run_cell = patched_run_cell


if __name__ == "__main__":
    sys.exit(eh.main())
