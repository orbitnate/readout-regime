"""Phase 5.72 Track A — B2 forward-pass-emit consumer + model gates G1/G2/G3.

Surface (see docs/HANDOFF_PHASE_5_72_TRACK_A_DESIGN.md and
run_phase_5_72_track_a_lookup_immunity.py):

    answer = STATUS_LABELS[T[m][q]]

  m : hidden per-ENTITY module value in {0..7}, carried by the loaded KM
      (V = lm_head.weight[token(' '+M_VALUE_WORDS[m])] at the answer slot),
      decoupled from the customer suffix.
  q : in-context operand in {0..7}, printed per row.
  T : published non-circulant 8x8 Latin square, printed in the prompt.

This is the GPU half (doc B2/B6/B7). It is a FORWARD-PASS-EMIT consumer, NOT an
outside-the-model symbolic combine: the model must GENERATE T[m][q] from a
recovered m-signal injected into the residual stream + the visible q and table T.

Mechanism (Path A producer bridge, faithful to Phase 5.42):
  * FEATURE  : per-entity contract-projected answer-slot feature read from the
    loaded KM bank (phase422._feature_vector, feature_kind=contract_projected_answer,
    bank layer = model.cfg.bank_layer_indices[-1] = 21 = contract layer).
  * ORACLE m-delta (G2 upper bound, value_source=oracle): residual-stream delta
    teacher_prompt(m,q) - hidden_route_prompt(q) captured at last_prompt_token,
    layers (21,23) -- the TRUE-m signal. Patching it back reconstructs teacher
    behavior. NEVER a recovery claim (CLAUDE.md oracle-routing discipline).
  * PRODUCER BRIDGE (G3 result, value_source=predicted): leave-one-KB-out
    centered dual-ridge maps FEATURE -> m-delta (phase422._fit_centered_dual_ridge
    trained on the oracle-captured deltas as the supervised target). At G3 the
    bridge is fed ONLY the held-out entity's bank feature -- the true m is never
    handed to the model. The predicted delta is patched on hidden_route_prompt(q);
    the model composes T[m][q] from the recovered m + visible q + T.

Gates (numbers from the design spec):
  G1 positive control (no patch, m visible)  PASS >= 120/128.
  G2 oracle upper bound (patch true-m delta)  PASS >= 120/128 AND G2 >= G3.
  G3 hidden route (patch bridge-predicted)    PASS: G3 >= 0.90*G2 AND
                                              G3 - lookup_floor >= 60/128.
  Controls: no_module (deltas=None on hidden route) and wrong_delta (another
  entity's predicted delta) must sit near chance.

value_source STOP-GATE (do-not-repeat from 5.70): config.value_source and every
bridge fit record carry value_source; the bridge ONLY ever produces predicted
deltas (the oracle arm bypasses the bridge). The driver asserts
config.value_source=="predicted" on the headline route and that no G3 row
consumed an oracle delta.

Staged for GPU saturation (parallel score slices on one H200):
  --stage prepare : one process. Build features, capture oracle deltas, fit the
                    leave-one-KB-out bridge, predict per-entity deltas. Saves a
                    prep artifact (no generation).
  --stage score --kb_indices ... : N parallel processes. Each loads the prep
                    artifact + model and scores its KB slice (the generation
                    work). Writes a per-slice rows JSON.
  --stage reduce  : merge slices, compute the gates, write the result doc.
  --stage all     : prepare+score+reduce in one process (smoke / small runs).
"""

from __future__ import annotations

import argparse
import json
import re
import time
from contextlib import nullcontext
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import torch

import lrm.run_phase_4_15_shared_wo_contract as phase415
import lrm.run_phase_4_22_hidden_to_premise_delta_transducer as phase422
import lrm.run_phase_4_34_no_rule_readout_interface as phase434
import lrm.run_phase_5_72_track_a_prompts as prompts
from lrm.model.qwen_with_memory_banks import KnowledgeModule
from lrm.run_phase_3_0_train_kb import _load_model

surface = prompts._S  # the torch-free Track A surface module
K = surface.K
STATUS_LABELS = surface.STATUS_LABELS
M_VALUE_WORDS = surface.M_VALUE_WORDS

DEFAULT_KM_ROOT = Path("lrm/results/v7_phase5/phase_5_72_track_a/kms_s600")
DEFAULT_ANCHOR_DIR = Path("lrm/results/v7_phase2/phase_2_3/stage_c_n10/anchor_hiddens_n10_7b")
DEFAULT_CONTRACT = Path("lrm/results/v7_phase4/phase_4_15_shared_wo_contract/shared_wo_contract_L21.pt")
DEFAULT_ROOT = Path("lrm/results/v7_phase5/phase_5_72_track_a/consumer")
DEFAULT_PREP = DEFAULT_ROOT / "phase_5_72_track_a_consumer_prep.pt"
DEFAULT_BRIDGE = DEFAULT_ROOT / "phase_5_72_track_a_producer.bridge.pt"
DEFAULT_OUT = DEFAULT_ROOT / "phase_5_72_track_a_consumer_gates.json"

DEFAULT_TARGET_LAYERS = (21, 23)
DEFAULT_PATCH_POSITION = "last_prompt_token"
DEFAULT_RIDGE_ALPHA = 1.0
DEFAULT_MAX_NEW_TOKENS = 6
# G3 must beat the best visible-record lookup by this rate margin (design's
# 60/128 ~= 0.47 over chance, expressed n-agnostically).
DEFAULT_G3_MARGIN_RATE = 0.47
FEATURE_KIND = "contract_projected_answer"
BRIDGE_FORMAT = "phase_5_72_track_a_producer_bridge_bundle.v1"
PREP_FORMAT = "phase_5_72_track_a_consumer_prep.v1"

ARM_POSITIVE = "g1_positive_control"
ARM_ORACLE = "g2_oracle_upper_bound"
ARM_PREDICTED = "g3_hidden_route_predicted"
ARM_NO_MODULE = "ctrl_no_module"
ARM_WRONG_DELTA = "ctrl_wrong_delta"
ARMS = (ARM_POSITIVE, ARM_ORACLE, ARM_PREDICTED, ARM_NO_MODULE, ARM_WRONG_DELTA)


# --- surface entities --------------------------------------------------------

def _load_manifest(km_root: Path) -> Dict[str, Any]:
    path = km_root / "track_a_entities.json"
    manifest = json.loads(path.read_text())
    # The KMs on the pod store V = lm_head[token(' '+word)]; the surface module
    # must name the same words or the recovered m would index the wrong row.
    if [w.strip() for w in manifest["m_value_vocab"]] != list(M_VALUE_WORDS):
        raise RuntimeError(
            "manifest m_value_vocab does not match surface M_VALUE_WORDS: "
            f"{manifest['m_value_vocab']} vs {list(M_VALUE_WORDS)}"
        )
    return manifest


def _build_entities(manifest: Mapping[str, Any]) -> List[Dict[str, Any]]:
    """Eval entities = manifest customers whose hidden m is in {0..K-1} (the K-way
    surface uses only the first K module-value rows). m stays decoupled from the
    customer suffix. Features + the operand-marginalized m-delta are computed
    once per entity; q is swept at score time."""
    entities: List[Dict[str, Any]] = []
    for e in manifest["entities"]:
        m = int(e["hidden_m"])
        if m >= K:
            continue
        entities.append(
            {
                "item_id": str(e["item_id"]),
                "kb": str(e["kb"]),
                "payload_customer": str(e["payload_customer"]),
                "customer_suffix": str(e["customer_suffix"]),
                "answer_fact_id": str(e["answer_fact_id"]),
                "hidden_m": m,
            }
        )
    return entities


def _build_eval_grid(entities: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    """One eval row per (entity, operand q) with q swept over {0..K-1}. The
    entity's module value m is fixed (carried by its KM); q is the in-context
    operand. A per-q breakdown across this grid is the composition evidence:
    with the injected m-signal held fixed, the emitted status must track
    T[m][q] as the visible q varies."""
    grid: List[Dict[str, Any]] = []
    for e in entities:
        m = int(e["hidden_m"])
        for q in range(K):
            grid.append(
                {
                    "item_id": f"{e['item_id']}_q{q}",
                    "entity_id": str(e["item_id"]),
                    "kb": str(e["kb"]),
                    "customer_suffix": str(e["customer_suffix"]),
                    "hidden_m": m,
                    "operand_q": q,
                    "expected_status": prompts.compose_status(m, q),
                }
            )
    return grid


# --- generation + parse ------------------------------------------------------

def _parse_status(generated: str) -> Tuple[Optional[str], Optional[str]]:
    text = str(generated).strip()
    if not text:
        return None, "empty_generation"
    first = text.split("\n", 1)[0].strip()
    for label in STATUS_LABELS:
        if first == label or first.startswith(label):
            return label, None
    found = [
        label
        for label in STATUS_LABELS
        if re.search(rf"(?<![A-Za-z]){re.escape(label)}(?![A-Za-z])", first)
    ]
    if len(found) == 1:
        return found[0], None
    return None, "no_unique_status_in_first_line"


def _patch_context(
    model: Any,
    prompt: str,
    deltas: Optional[Mapping[int, torch.Tensor]],
    patch_position: str,
    scale: float,
):
    """Return the residual-stream patch context for the given patch position.

    `model.premise_bridge` hard-codes the last-prompt-token position; for any
    other position route through `phase434._patched_deltas_at_position` (same
    additive hook, resolved at the named token). No deltas -> no-op context."""
    if not deltas:
        return nullcontext()
    if str(patch_position) == "last_prompt_token":
        return model.premise_bridge(prompt=prompt, deltas=dict(deltas), scale=float(scale))
    return phase434._patched_deltas_at_position(
        model=model,
        prompt=prompt,
        deltas=dict(deltas),
        position_name=str(patch_position),
        scale=float(scale),
    )


def _status_token_ids(model: Any) -> Dict[str, int]:
    """First-token id of ` {label}` per status label, for the forced-choice
    K-logit readout. Asserts the first tokens are distinct (else an argmax over
    first-token logits would conflate two labels)."""
    tok = model.tokenizer
    ids: Dict[str, int] = {}
    for label in STATUS_LABELS:
        toks = tok(" " + str(label), add_special_tokens=False)["input_ids"]
        if not toks:
            raise RuntimeError(f"empty tokenization for status label {label!r}")
        ids[str(label)] = int(toks[0])
    if len(set(ids.values())) != len(ids):
        raise RuntimeError(f"status-label first tokens collide (forced choice invalid): {ids}")
    return ids


@torch.no_grad()
def _generate(
    model: Any,
    prompt: str,
    *,
    max_new_tokens: int,
    deltas: Optional[Mapping[int, torch.Tensor]],
    patch_position: str,
    delta_multiplier: float = 1.0,
) -> str:
    tok = model.tokenizer
    device = next(model.parameters()).device
    ids = tok(prompt, add_special_tokens=False, return_tensors="pt")
    input_ids = ids["input_ids"].to(device)
    attention_mask = ids["attention_mask"].to(device)
    with _patch_context(model, prompt, deltas, patch_position, delta_multiplier):
        out = model.base.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=int(max_new_tokens),
            do_sample=False,
        )
    return tok.decode(out[0, input_ids.shape[1]:], skip_special_tokens=True)


@torch.no_grad()
def _forced_choice(
    model: Any,
    prompt: str,
    *,
    deltas: Optional[Mapping[int, torch.Tensor]],
    patch_position: str,
    candidate_token_ids: Mapping[str, int],
    delta_multiplier: float = 1.0,
) -> Tuple[str, Dict[str, float]]:
    """Forward-pass forced-choice readout: score the K candidate status-token
    logits at the answer position (next token after the trailing `Answer:`) and
    take argmax. Removes free-generation / parse noise; still a forward-pass
    readout (the standard PITWM readout). The patch is applied via the same
    position-aware context as generation."""
    tok = model.tokenizer
    device = next(model.parameters()).device
    ids = tok(prompt, add_special_tokens=False, return_tensors="pt")
    input_ids = ids["input_ids"].to(device)
    attention_mask = ids["attention_mask"].to(device)
    with _patch_context(model, prompt, deltas, patch_position, delta_multiplier):
        out = model.base(input_ids=input_ids, attention_mask=attention_mask)
    logits = out.logits[0, -1, :].float()
    cand = {label: float(logits[int(tid)].item()) for label, tid in candidate_token_ids.items()}
    best = max(cand, key=lambda k: cand[k])
    return best, cand


def _score(
    model: Any,
    *,
    prompt: str,
    expected: str,
    deltas: Optional[Mapping[int, torch.Tensor]],
    max_new_tokens: int,
    patch_position: str,
    readout: str = "generate",
    candidate_token_ids: Optional[Mapping[str, int]] = None,
    delta_multiplier: float = 1.0,
) -> Dict[str, Any]:
    if str(readout) == "forced_choice":
        if candidate_token_ids is None:
            raise RuntimeError("forced_choice readout requires candidate_token_ids")
        best, cand = _forced_choice(
            model, prompt, deltas=deltas, patch_position=patch_position,
            candidate_token_ids=candidate_token_ids, delta_multiplier=delta_multiplier,
        )
        return {
            "generated": json.dumps({k: round(v, 4) for k, v in cand.items()}),
            "parsed_status": best,
            "parse_error": None,
            "expected_status": str(expected),
            "correct": bool(best == str(expected)),
            "readout": "forced_choice",
            "candidate_logits": {k: round(v, 6) for k, v in cand.items()},
        }
    generated = _generate(
        model, prompt, max_new_tokens=max_new_tokens, deltas=deltas,
        patch_position=patch_position, delta_multiplier=delta_multiplier,
    )
    parsed, parse_error = _parse_status(generated)
    return {
        "generated": generated,
        "parsed_status": parsed,
        "parse_error": parse_error,
        "expected_status": str(expected),
        "correct": bool(parsed == str(expected)),
        "readout": "generate",
    }


# --- prepare stage: features, oracle deltas, bridge, predicted deltas ---------

def _extract_features(
    model: Any,
    items: Sequence[Mapping[str, Any]],
    km_root: Path,
    projection: Mapping[str, Any],
    layer_name: str,
) -> Dict[str, torch.Tensor]:
    by_kb: Dict[str, List[Mapping[str, Any]]] = defaultdict(list)
    for item in items:
        by_kb[str(item["kb"])].append(item)
    features: Dict[str, torch.Tensor] = {}
    for kb in sorted(by_kb):
        km = KnowledgeModule.load(str(km_root / kb / f"{kb}_km.pt"))
        model.load_km(km)
        for item in by_kb[kb]:
            answer_slots = km.slot_indices_for_fact(str(item["answer_fact_id"]))
            if not answer_slots:
                raise RuntimeError(f"no answer slots for {item['item_id']} in {kb}")
            feat = phase422._feature_vector(
                model=model,
                item={"answer_slots": list(answer_slots)},
                feature_kind=FEATURE_KIND,
                projection=dict(projection),
                layer_name=str(layer_name),
            )
            features[str(item["item_id"])] = feat.detach().float().cpu().view(-1)
        model.unload_km()
    return features


def _capture_operand_free_deltas(
    model: Any,
    entities: Sequence[Mapping[str, Any]],
    target_layers: Sequence[int],
    patch_position: str,
) -> Tuple[Dict[str, Dict[int, torch.Tensor]], Dict[str, Any]]:
    """Operand-FREE true-m delta per entity:

        delta_m = state(m_register_teacher_prompt(m)) - state(m_register_neutral_prompt())

    at patch_position, per target layer. Captured in a register context with NO
    operand/table, so it is a full-magnitude pure m-identity signal (vs the
    operand-marginalized mean, which cancels magnitude). Still composition-valid:
    the operand q is only visible at score time on hidden_route_prompt(q), so any
    q-dependence of the emitted status comes from the visible q + table."""
    model.unload_km()
    layers = [int(l) for l in target_layers]
    neutral_states, neutral_pos = phase434._capture_prompt_position_states(
        model=model, prompt=prompts.m_register_neutral_prompt(),
        layers=layers, position_name=str(patch_position),
    )
    if neutral_pos is None:
        return {}, {"status": "neutral_position_failed", "capture_kind": "operand_free"}
    neutral = {l: neutral_states[l].detach().float().cpu().view(-1) for l in layers}
    teacher_cache: Dict[int, Dict[int, torch.Tensor]] = {}
    failures: List[str] = []
    deltas: Dict[str, Dict[int, torch.Tensor]] = {}
    for e in entities:
        m = int(e["hidden_m"])
        if m not in teacher_cache:
            states, pos = phase434._capture_prompt_position_states(
                model=model, prompt=prompts.m_register_teacher_prompt(m),
                layers=layers, position_name=str(patch_position),
            )
            teacher_cache[m] = None if pos is None else {l: states[l].detach().float().cpu().view(-1) for l in layers}
        ts = teacher_cache[m]
        if ts is None:
            failures.append(str(e["item_id"]))
            continue
        deltas[str(e["item_id"])] = {l: ts[l] - neutral[l] for l in layers}
    meta = {
        "status": "ok" if not failures else "partial",
        "patch_position": str(patch_position),
        "target_layers": layers,
        "capture_kind": "operand_free_register",
        "captured": len(deltas),
        "position_failures": failures,
    }
    return deltas, meta


def _capture_oracle_deltas(
    model: Any,
    entities: Sequence[Mapping[str, Any]],
    target_layers: Sequence[int],
    patch_position: str,
) -> Tuple[Dict[str, Dict[int, torch.Tensor]], Dict[str, Any]]:
    """Operand-MARGINALIZED true-m delta per entity:

        delta_m = mean_q [ state(teacher_prompt(m, q)) - state(hidden_route_prompt(q)) ]

    at patch_position, per target layer. Averaging over q removes per-operand
    (answer-specific) information: the patched vector is identical regardless of
    which q is later evaluated, so any q-dependence of the emitted status at
    score time must come from the VISIBLE q + table (forward-pass composition),
    not from the injected signal. Captures are cached by q (hidden) and (m,q)
    (teacher) so cost is O(K + K*distinct_m), not O(entities*K)."""
    model.unload_km()
    layers = [int(l) for l in target_layers]
    hidden_cache: Dict[int, Dict[int, torch.Tensor]] = {}
    teacher_cache: Dict[Tuple[int, int], Dict[int, torch.Tensor]] = {}
    failures: List[str] = []

    def _hidden(q: int):
        if q not in hidden_cache:
            states, pos = phase434._capture_prompt_position_states(
                model=model, prompt=prompts.hidden_route_prompt(q),
                layers=layers, position_name=str(patch_position),
            )
            hidden_cache[q] = None if pos is None else {l: states[l].detach().float().cpu().view(-1) for l in layers}
        return hidden_cache[q]

    def _teacher(m: int, q: int):
        key = (m, q)
        if key not in teacher_cache:
            states, pos = phase434._capture_prompt_position_states(
                model=model, prompt=prompts.teacher_prompt(m, q),
                layers=layers, position_name=str(patch_position),
            )
            teacher_cache[key] = None if pos is None else {l: states[l].detach().float().cpu().view(-1) for l in layers}
        return teacher_cache[key]

    deltas: Dict[str, Dict[int, torch.Tensor]] = {}
    for e in entities:
        m = int(e["hidden_m"])
        acc: Dict[int, torch.Tensor] = {l: torch.zeros(1) for l in layers}
        n_ok = 0
        ok = True
        for q in range(K):
            hs = _hidden(q)
            ts = _teacher(m, q)
            if hs is None or ts is None:
                ok = False
                break
            for l in layers:
                d = ts[l] - hs[l]
                acc[l] = d.clone() if n_ok == 0 else acc[l] + d
            n_ok += 1
        if not ok or n_ok == 0:
            failures.append(str(e["item_id"]))
            continue
        deltas[str(e["item_id"])] = {l: acc[l] / float(n_ok) for l in layers}

    meta = {
        "status": "ok" if not failures else "partial",
        "patch_position": str(patch_position),
        "target_layers": layers,
        "capture_kind": "operand_marginalized_mean_over_q",
        "captured": len(deltas),
        "position_failures": failures,
    }
    return deltas, meta


def _bridge_signature(args: argparse.Namespace) -> str:
    layer_tag = "-".join(str(int(layer)) for layer in args.target_layers)
    return (
        f"track_a_{FEATURE_KIND}_t{layer_tag}"
        f"_{str(args.patch_position).replace('_', '')}"
        f"_cap{str(args.capture_mode).replace('_', '')}"
        f"_value{args.value_source}"
    )


def _fit_bridge(
    *,
    items: Sequence[Mapping[str, Any]],
    features: Mapping[str, torch.Tensor],
    oracle_deltas: Mapping[str, Mapping[int, torch.Tensor]],
    target_layers: Sequence[int],
    alpha: float,
    value_source: str,
    feature_signature: str,
) -> Tuple[Dict[str, Dict[int, torch.Tensor]], Dict[str, Any]]:
    """Leave-one-KB-out centered dual-ridge per target layer. Returns predicted
    per-entity deltas (held-out) and a bundle of fits tagged with value_source."""
    by_kb: Dict[str, List[str]] = defaultdict(list)
    for item in items:
        iid = str(item["item_id"])
        if iid in features and iid in oracle_deltas:
            by_kb[str(item["kb"])].append(iid)
    kbs = sorted(by_kb)

    fits: Dict[str, Dict[str, Any]] = {}
    predicted: Dict[str, Dict[int, torch.Tensor]] = {}
    eval_mse: List[float] = []

    for eval_kb in kbs:
        train_ids = [iid for kb in kbs if kb != eval_kb for iid in by_kb[kb]]
        if not train_ids:
            raise RuntimeError(f"no train rows for eval_kb={eval_kb}")
        train_x = torch.stack([features[iid] for iid in train_ids], dim=0)
        layer_fits: Dict[int, Dict[str, Any]] = {}
        for layer in target_layers:
            li = int(layer)
            train_y = torch.stack([oracle_deltas[iid][li] for iid in train_ids], dim=0)
            fit = phase422._fit_centered_dual_ridge(
                train_x=train_x, train_y=train_y, alpha=float(alpha)
            )
            layer_fits[li] = {k: (v.cpu() if torch.is_tensor(v) else v) for k, v in fit.items()}
        fit_key = f"{feature_signature}__all_except_{eval_kb}"
        fits[fit_key] = {
            "format": "phase_5_72_track_a_producer_fit.v1",
            "eval_kb": str(eval_kb),
            "train_kbs": [kb for kb in kbs if kb != eval_kb],
            "train_rows": int(len(train_ids)),
            "alpha": float(alpha),
            "value_source": str(value_source),
            "feature_signature": str(feature_signature),
            "target_layers": [int(layer) for layer in target_layers],
            "layer_fits": layer_fits,
        }
        for iid in by_kb[eval_kb]:
            x = features[iid]
            pred = {
                int(layer): phase422._predict_delta(layer_fits[int(layer)], x).detach().float().cpu().view(-1)
                for layer in target_layers
            }
            predicted[iid] = pred
            mse = sum(
                float(torch.mean((pred[int(l)] - oracle_deltas[iid][int(l)]) ** 2).item())
                for l in target_layers
            ) / len(target_layers)
            eval_mse.append(mse)

    meta = {
        "format": BRIDGE_FORMAT,
        "feature_signature": str(feature_signature),
        "value_source": str(value_source),
        "alpha": float(alpha),
        "target_layers": [int(layer) for layer in target_layers],
        "kbs": kbs,
        "fit_count": len(fits),
        "eval_leave_one_out_delta_mse_mean": float(sum(eval_mse) / max(1, len(eval_mse))),
        "fit_keys": sorted(fits),
        "all_fits_value_source_predicted": all(
            str(f["value_source"]) == "predicted" for f in fits.values()
        ),
    }
    return predicted, {"fits": fits, "meta": meta}


def _run_prepare(args: argparse.Namespace) -> Dict[str, Any]:
    t0 = time.time()
    km_root = Path(args.km_root)
    manifest = _load_manifest(km_root)
    entities = _build_entities(manifest)
    grid = _build_eval_grid(entities)

    projection = phase415._load_contract_projection(Path(args.contract))
    model, _ = _load_model(Path(args.anchor_dir), w_o_init="identity")
    model.eval()
    layer_name = str(model.cfg.bank_layer_indices[-1])
    if int(projection["layer"]) != int(layer_name):
        raise RuntimeError(
            f"contract layer {projection['layer']} != bank feature layer {layer_name}"
        )

    print(
        f"[prepare] K={K}; {len(entities)} eval entities (m<{K}); "
        f"{len(grid)} (entity,q) eval rows",
        flush=True,
    )
    features = _extract_features(model, entities, km_root, projection, layer_name)

    print(f"[prepare] capturing oracle m-deltas (mode={args.capture_mode})", flush=True)
    capture_fn = (
        _capture_operand_free_deltas
        if str(args.capture_mode) == "operand_free"
        else _capture_oracle_deltas
    )
    oracle_deltas, capture_meta = capture_fn(
        model, entities, args.target_layers, args.patch_position
    )
    if capture_meta["status"] != "ok":
        raise RuntimeError(f"oracle delta capture incomplete: {capture_meta}")

    feature_signature = _bridge_signature(args)
    print("[prepare] fitting leave-one-KB-out producer bridge", flush=True)
    predicted, bundle = _fit_bridge(
        items=entities,
        features=features,
        oracle_deltas=oracle_deltas,
        target_layers=args.target_layers,
        alpha=float(args.ridge_alpha),
        value_source=str(args.value_source),
        feature_signature=feature_signature,
    )

    bridge_path = Path(args.bridge_out)
    bridge_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "format": BRIDGE_FORMAT,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "config": _config_dict(args, feature_signature),
            **bundle["meta"],
            "fits": bundle["fits"],
        },
        bridge_path,
    )

    prep = {
        "format": PREP_FORMAT,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "config": _config_dict(args, feature_signature),
        "entities": entities,
        "grid": grid,
        "oracle_deltas": oracle_deltas,
        "predicted_deltas": predicted,
        "capture_meta": capture_meta,
        "bridge_meta": bundle["meta"],
        "feature_signature": feature_signature,
    }
    prep_path = Path(args.prep)
    prep_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(prep, prep_path)
    print(
        f"[prepare] done in {time.time() - t0:.1f}s; "
        f"loo_delta_mse_mean={bundle['meta']['eval_leave_one_out_delta_mse_mean']:.4f}; "
        f"prep={prep_path}",
        flush=True,
    )
    return prep


# --- score stage -------------------------------------------------------------

def _wrong_delta_partner(entities: Sequence[Mapping[str, Any]], entity_id: str) -> str:
    """A different entity whose m differs (so wrong-delta is a genuine miss)."""
    idx = next(i for i, e in enumerate(entities) if str(e["item_id"]) == entity_id)
    me = entities[idx]
    for off in range(1, len(entities)):
        other = entities[(idx + off) % len(entities)]
        if int(other["hidden_m"]) != int(me["hidden_m"]):
            return str(other["item_id"])
    return str(entities[(idx + 1) % len(entities)]["item_id"])


def _run_score(args: argparse.Namespace) -> List[Dict[str, Any]]:
    prep = torch.load(Path(args.prep), weights_only=False)
    if str(prep.get("format")) != PREP_FORMAT:
        raise RuntimeError(f"bad prep format: {prep.get('format')}")
    entities = prep["entities"]
    grid = prep["grid"]
    oracle_deltas = prep["oracle_deltas"]
    predicted_deltas = prep["predicted_deltas"]

    kbs = sorted({str(e["kb"]) for e in entities})
    if args.kb_indices is not None:
        # Index into the entities' KB list (which omits KBs with no m<K entity);
        # out-of-range indices from a fixed stride are skipped, not fatal.
        sel_kbs = {kbs[i] for i in args.kb_indices if 0 <= i < len(kbs)}
    else:
        sel_kbs = set(kbs)
    sel_grid = [row for row in grid if str(row["kb"]) in sel_kbs]
    if args.limit_entities is not None:
        keep = []
        seen: set = set()
        for row in sel_grid:
            if str(row["entity_id"]) not in seen and len(seen) >= int(args.limit_entities):
                continue
            seen.add(str(row["entity_id"]))
            keep.append(row)
        sel_grid = keep

    model, _ = _load_model(Path(args.anchor_dir), w_o_init="identity")
    model.eval()
    candidate_token_ids = (
        _status_token_ids(model) if str(args.readout) == "forced_choice" else None
    )

    rows: List[Dict[str, Any]] = []
    arms = [a for a in ARMS if a in args.arms] if args.arms else list(ARMS)
    t0 = time.time()
    for n, row in enumerate(sel_grid, 1):
        eid = str(row["entity_id"])
        m = int(row["hidden_m"])
        q = int(row["operand_q"])
        expected = str(row["expected_status"])
        oracle = oracle_deltas.get(eid)
        predicted = predicted_deltas.get(eid)
        wrong_id = _wrong_delta_partner(entities, eid)
        wrong_delta = predicted_deltas.get(wrong_id)

        arm_specs = {
            ARM_POSITIVE: (prompts.positive_control_prompt(m, q), None, "none"),
            ARM_ORACLE: (prompts.hidden_route_prompt(q), oracle, "oracle"),
            ARM_PREDICTED: (prompts.hidden_route_prompt(q), predicted, "predicted"),
            ARM_NO_MODULE: (prompts.hidden_route_prompt(q), None, "none"),
            ARM_WRONG_DELTA: (prompts.hidden_route_prompt(q), wrong_delta, "predicted_wrong_entity"),
        }
        for arm in arms:
            prompt, deltas, value_source = arm_specs[arm]
            leaks_m = prompts.hidden_prompt_leaks_m(prompt, m) if arm != ARM_POSITIVE else False
            res = _score(
                model,
                prompt=prompt,
                expected=expected,
                deltas=deltas,
                max_new_tokens=int(args.max_new_tokens),
                patch_position=str(args.patch_position),
                readout=str(args.readout),
                candidate_token_ids=candidate_token_ids,
                delta_multiplier=float(args.delta_multiplier),
            )
            rows.append(
                {
                    "item_id": str(row["item_id"]),
                    "entity_id": eid,
                    "kb": str(row["kb"]),
                    "customer_suffix": str(row["customer_suffix"]),
                    "hidden_m": m,
                    "operand_q": q,
                    "arm": arm,
                    "value_source": value_source,
                    "delta_source": (
                        "bridge_predicted"
                        if arm in (ARM_PREDICTED,)
                        else "oracle_captured"
                        if arm == ARM_ORACLE
                        else "wrong_entity_bridge_predicted"
                        if arm == ARM_WRONG_DELTA
                        else "none"
                    ),
                    "wrong_delta_source_entity": wrong_id if arm == ARM_WRONG_DELTA else None,
                    "patched_layers": sorted(int(l) for l in deltas) if deltas else [],
                    "hidden_prompt_leaks_m": bool(leaks_m),
                    **res,
                }
            )
        if n % 16 == 0 or n == len(sel_grid):
            print(f"[score] {n}/{len(sel_grid)} (entity,q) rows ({time.time() - t0:.1f}s)", flush=True)

    slice_tag = (
        "all" if args.kb_indices is None else "_".join(str(i) for i in args.kb_indices)
    )
    out_path = Path(args.rows_out_dir) / f"rows_slice_{slice_tag}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({"rows": rows}, indent=2) + "\n")
    print(f"[score] wrote {len(rows)} rows -> {out_path}", flush=True)
    return rows


# --- reduce stage ------------------------------------------------------------

def _arm_correct(rows: Sequence[Mapping[str, Any]], arm: str) -> Tuple[int, int]:
    sel = [r for r in rows if str(r["arm"]) == arm]
    return sum(int(bool(r["correct"])) for r in sel), len(sel)


def _breakdown(rows: Sequence[Mapping[str, Any]], arm: str, field: str) -> Dict[str, Dict[str, int]]:
    out: Dict[str, Dict[str, int]] = {}
    for r in rows:
        if str(r["arm"]) != arm:
            continue
        key = str(r[field])
        out.setdefault(key, {"correct": 0, "n": 0})
        out[key]["n"] += 1
        out[key]["correct"] += int(bool(r["correct"]))
    return dict(sorted(out.items(), key=lambda kv: int(kv[0])))


def _decision(
    rows: Sequence[Mapping[str, Any]],
    args: argparse.Namespace,
    *,
    lookup_floor_rate: float,
) -> Dict[str, Any]:
    g1_c, g1_n = _arm_correct(rows, ARM_POSITIVE)
    g2_c, g2_n = _arm_correct(rows, ARM_ORACLE)
    g3_c, g3_n = _arm_correct(rows, ARM_PREDICTED)
    nomod_c, nomod_n = _arm_correct(rows, ARM_NO_MODULE)
    wrong_c, wrong_n = _arm_correct(rows, ARM_WRONG_DELTA)

    def _rate(c: int, n: int) -> float:
        return (c / n) if n else 0.0

    g1_rate, g2_rate, g3_rate = _rate(g1_c, g1_n), _rate(g2_c, g2_n), _rate(g3_c, g3_n)
    margin = float(args.g3_margin_rate)

    g1_pass = g1_rate >= 0.9375
    g2_pass = (g2_rate >= 0.9375) and (g2_rate >= g3_rate)
    g3_pass = (g3_rate >= 0.90 * g2_rate) and ((g3_rate - lookup_floor_rate) >= margin)

    g3_rows = [r for r in rows if str(r["arm"]) == ARM_PREDICTED]
    g3_value_source_clean = bool(
        str(args.value_source) == "predicted"
        and all(str(r["value_source"]) == "predicted" for r in g3_rows)
        and all(str(r["delta_source"]) == "bridge_predicted" for r in g3_rows)
    )
    leaks = sum(int(bool(r.get("hidden_prompt_leaks_m"))) for r in rows)

    phase_pass = bool(g1_pass and g2_pass and g3_pass and g3_value_source_clean and leaks == 0)
    return {
        "phase_pass": phase_pass,
        "k": int(K),
        "g1_positive_control": {
            "correct": g1_c, "n": g1_n, "rate": round(g1_rate, 4),
            "pass": g1_pass, "bar": "rate >= 0.9375",
        },
        "g2_oracle_upper_bound": {
            "correct": g2_c, "n": g2_n, "rate": round(g2_rate, 4),
            "pass": g2_pass, "bar": "rate >= 0.9375 and >= G3",
        },
        "g3_hidden_route": {
            "correct": g3_c, "n": g3_n, "rate": round(g3_rate, 4), "pass": g3_pass,
            "bar": "rate >= 0.90*G2 and (G3_rate - lookup_floor_rate) >= margin",
            "lookup_floor_rate": round(lookup_floor_rate, 4),
            "g3_margin_rate": margin,
            "margin_over_lookup": round(g3_rate - lookup_floor_rate, 4),
            "fraction_of_oracle": round(g3_rate / g2_rate, 4) if g2_rate else None,
        },
        "control_no_module": {"correct": nomod_c, "n": nomod_n, "rate": round(_rate(nomod_c, nomod_n), 4)},
        "control_wrong_delta": {"correct": wrong_c, "n": wrong_n, "rate": round(_rate(wrong_c, wrong_n), 4)},
        "value_source_stop_gate": {
            "config_value_source": str(args.value_source),
            "g3_value_source_clean": g3_value_source_clean,
        },
        "hidden_prompt_m_leak_rows": leaks,
        "g3_per_q": _breakdown(rows, ARM_PREDICTED, "operand_q"),
        "g3_per_m": _breakdown(rows, ARM_PREDICTED, "hidden_m"),
        "g2_per_q": _breakdown(rows, ARM_ORACLE, "operand_q"),
        "g1_per_q": _breakdown(rows, ARM_POSITIVE, "operand_q"),
        "claim_boundary": (
            f"A pass shows a loaded module's per-entity {K}-way value, recovered "
            "from its bank feature and injected as an operand-MARGINALIZED delta, "
            "drives the base to compose T[m][q] in the forward pass from the "
            "recovered m + the VISIBLE operand q (swept) + the published table -- "
            "tracking T[m][q] as q varies with the injected signal held fixed. A "
            "deterministic visible-record lookup is structurally barred from this "
            "(G0). It does not establish frozen no-refit transfer (5.72b), K>4, or "
            "alpha-scale generalization."
        ),
    }


def _run_reduce(args: argparse.Namespace) -> Dict[str, Any]:
    rows: List[Dict[str, Any]] = []
    for path in sorted(Path(args.rows_out_dir).glob("rows_slice_*.json")):
        rows.extend(json.loads(path.read_text())["rows"])
    if not rows:
        raise RuntimeError(f"no rows_slice_*.json under {args.rows_out_dir}")
    # de-dup (item_id, arm) keeping last
    dedup: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for r in rows:
        dedup[(str(r["item_id"]), str(r["arm"]))] = r
    rows = list(dedup.values())

    bridge = torch.load(Path(args.bridge_out), weights_only=False)
    g0_path = Path(args.g0_json)
    g0 = json.loads(g0_path.read_text()) if g0_path.exists() else {}
    g0_lookup = (
        g0.get("g0", {}).get("checks", {}).get("c_best_visible_key_lookup", {})
    )
    g0_best = g0_lookup.get("best_correct")
    g0_n = g0_lookup.get("per_scheme", {}).get(str(g0_lookup.get("best_key", "")), {}).get("n", 128)
    lookup_floor_rate = (float(g0_best) / float(g0_n)) if g0_best is not None and g0_n else (1.0 / K)
    decision = _decision(rows, args, lookup_floor_rate=lookup_floor_rate)
    payload = {
        "phase": "5.72a_track_a_consumer_gates",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "config": _config_dict(args, str(bridge.get("feature_signature", ""))),
        "g0_reference": {
            "path": str(g0_path),
            "g0_pass": g0.get("g0", {}).get("g0_pass"),
            "best_visible_key_lookup_correct": g0_best,
            "best_visible_key_lookup_n": g0_n,
            "lookup_floor_rate": round(lookup_floor_rate, 4),
        },
        "bridge_meta": {k: v for k, v in bridge.items() if k not in ("fits",)},
        "summary": {"decision": decision},
        "rows": sorted(rows, key=lambda r: (str(r["item_id"]), str(r["arm"]))),
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(f"[reduce] wrote {out_path}")
    print(json.dumps(decision, indent=2, sort_keys=True))
    return payload


# --- config / cli ------------------------------------------------------------

def _config_dict(args: argparse.Namespace, feature_signature: str) -> Dict[str, Any]:
    return {
        "km_root": str(args.km_root),
        "anchor_dir": str(args.anchor_dir),
        "contract": str(args.contract),
        "target_layers": [int(l) for l in args.target_layers],
        "patch_position": str(args.patch_position),
        "ridge_alpha": float(args.ridge_alpha),
        "value_source": str(args.value_source),
        "capture_mode": str(args.capture_mode),
        "readout": str(getattr(args, "readout", "generate")),
        "max_new_tokens": int(args.max_new_tokens),
        "delta_multiplier": float(args.delta_multiplier),
        "g3_margin_rate": float(args.g3_margin_rate),
        "feature_kind": FEATURE_KIND,
        "feature_signature": str(feature_signature),
        "status_labels": list(STATUS_LABELS),
        "m_value_words": list(M_VALUE_WORDS),
        "k": int(K),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("prepare", "score", "reduce", "all"), default="all")
    parser.add_argument("--km_root", default=str(DEFAULT_KM_ROOT))
    parser.add_argument("--anchor_dir", default=str(DEFAULT_ANCHOR_DIR))
    parser.add_argument("--contract", default=str(DEFAULT_CONTRACT))
    parser.add_argument("--prep", default=str(DEFAULT_PREP))
    parser.add_argument("--bridge_out", default=str(DEFAULT_BRIDGE))
    parser.add_argument("--rows_out_dir", default=str(DEFAULT_ROOT / "rows"))
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument(
        "--g0_json",
        default=str(
            "lrm/results/v7_phase5/phase_5_72_track_a/"
            "phase_5_72_track_a_g0_lookup_immunity.json"
        ),
    )
    parser.add_argument("--target_layers", nargs="*", type=int, default=list(DEFAULT_TARGET_LAYERS))
    parser.add_argument("--patch_position", default=DEFAULT_PATCH_POSITION)
    parser.add_argument("--ridge_alpha", type=float, default=DEFAULT_RIDGE_ALPHA)
    parser.add_argument("--value_source", choices=("predicted", "oracle"), default="predicted")
    parser.add_argument(
        "--capture_mode", choices=("marginalized", "operand_free"), default="marginalized"
    )
    parser.add_argument("--max_new_tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS)
    parser.add_argument(
        "--readout", choices=("generate", "forced_choice"), default="generate",
        help="generate+parse (default) or forced-choice over the K status-token "
        "logits at the answer position (reuses prep; no re-prepare needed)",
    )
    parser.add_argument("--delta_multiplier", type=float, default=1.0)
    parser.add_argument("--g3_margin_rate", type=float, default=DEFAULT_G3_MARGIN_RATE)
    parser.add_argument("--kb_indices", nargs="*", type=int, default=None)
    parser.add_argument("--limit_entities", type=int, default=None)
    parser.add_argument("--arms", nargs="*", default=None)
    args = parser.parse_args()

    if args.stage in ("prepare", "all"):
        _run_prepare(args)
    if args.stage in ("score", "all"):
        _run_score(args)
    if args.stage in ("reduce", "all"):
        _run_reduce(args)


if __name__ == "__main__":
    main()
