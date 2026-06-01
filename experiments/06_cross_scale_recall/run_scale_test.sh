#!/bin/bash
# RunPod scale test driver — runs the full Task 1C pipeline for any model.
#
# Usage:
#   bash runpod/run_scale_test.sh MODEL_ID TAG [LAYERS...] [-- SCALES...]
#
# Examples:
#   bash runpod/run_scale_test.sh Qwen/Qwen2.5-32B-Instruct 32b
#   bash runpod/run_scale_test.sh Qwen/Qwen2.5-72B-Instruct 72b
#   bash runpod/run_scale_test.sh Qwen/Qwen2.5-32B-Instruct 32b 56 58 60 62 63 -- 0.01 0.05 0.1 0.5
#
# If LAYERS/SCALES are omitted, auto-selects based on model layer count.
# Results go to results/expH/task1c_{TAG}_*.json
#
# PRIVATE. Do not publish — patent hold.
set -euo pipefail

REPO_DIR="${REPO_DIR:-/workspace/PITWM}"
cd "$REPO_DIR"

export HF_HOME="${HF_HOME:-/workspace/hf_cache}"
export TORCH_HOME="${TORCH_HOME:-/workspace/torch_cache}"
export PYTHONUNBUFFERED=1

MODEL="${1:?Usage: run_scale_test.sh MODEL_ID TAG [LAYERS...] [-- SCALES...]}"
TAG="${2:?Usage: run_scale_test.sh MODEL_ID TAG}"
shift 2

# Parse optional LAYERS and SCALES from remaining args
LAYERS=()
SCALES=()
parsing_scales=false
for arg in "$@"; do
    if [[ "$arg" == "--" ]]; then
        parsing_scales=true
        continue
    fi
    if $parsing_scales; then
        SCALES+=("$arg")
    else
        LAYERS+=("$arg")
    fi
done

LOG="results/expH/task1c_${TAG}_driver.log"
mkdir -p results/expH
exec > >(tee -a "$LOG") 2>&1

echo "=== [scale-test] started at $(date -Is) ==="
echo "=== [scale-test] model=$MODEL  tag=$TAG ==="

# --- Detect layer count if LAYERS not provided ---
if [[ ${#LAYERS[@]} -eq 0 ]]; then
    echo "[scale-test] Auto-detecting model architecture..."
    NLAYERS=$(python3 -c "
from transformers import AutoConfig
cfg = AutoConfig.from_pretrained('$MODEL')
print(cfg.num_hidden_layers)
")
    echo "[scale-test] Model has $NLAYERS layers"

    # Sweep the last 5 layers (where the plastic zone lives)
    L_START=$((NLAYERS - 5))
    L_END=$((NLAYERS - 1))
    for i in $(seq $L_START $L_END); do
        LAYERS+=("$i")
    done
    echo "[scale-test] Auto-selected layers: ${LAYERS[*]}"
fi

if [[ ${#SCALES[@]} -eq 0 ]]; then
    SCALES=(0.001 0.01 0.05 0.1 0.2 0.5)
    echo "[scale-test] Using default scales: ${SCALES[*]}"
fi

# --- Report GPU inventory ---
echo "[scale-test] GPU inventory:"
python3 -c "
import torch
for i in range(torch.cuda.device_count()):
    name = torch.cuda.get_device_name(i)
    mem = torch.cuda.get_device_properties(i).total_memory / 1e9
    print(f'  GPU {i}: {name} ({mem:.1f} GB)')
print(f'  Total: {torch.cuda.device_count()} GPUs')
"

# --- Stage 1: Base-ignorance verification ---
VERIFIED="results/expH/task1c_${TAG}_verified.json"
if [[ -f "$VERIFIED" ]]; then
    echo "[scale-test] Reusing existing verified KB: $VERIFIED"
else
    echo "=== [scale-test] Stage 1: base-ignorance verification ==="
    t0=$(date +%s)
    python3 -u -m pitwm.scripts.expH_base_ignorance \
        --model-id "$MODEL" \
        --kb-module pitwm.evaluation.acme_kb_expanded \
        --out "$VERIFIED"
    echo "[scale-test] Stage 1 done in $(( $(date +%s) - t0 ))s"
fi

CLEAN_COUNT=$(python3 -c "
import json
d = json.load(open('$VERIFIED'))
clean = sum(1 for f in d['facts'] if f.get('base_ignorance_verdict','clean') == 'clean')
print(clean)
")
echo "[scale-test] Clean facts: $CLEAN_COUNT"

# --- Stage 2: Layer/scale sweep ---
SWEEP="results/expH/task1c_${TAG}_sweep.json"
echo "=== [scale-test] Stage 2: layer/scale sweep (layers ${LAYERS[*]} × scales ${SCALES[*]}) ==="
t0=$(date +%s)
python3 -u -m pitwm.scripts.expH_layer_lr_sweep \
    --model-id "$MODEL" \
    --layers "${LAYERS[@]}" \
    --scales "${SCALES[@]}" \
    --verified-kb "$VERIFIED" \
    --n-facts 10 \
    --out "$SWEEP"
echo "[scale-test] Stage 2 done in $(( $(date +%s) - t0 ))s"

# --- Parse winner ---
WINNER=$(python3 -c "
import json
d = json.load(open('$SWEEP'))
print(d['winner_layer'], d['winner_scale'])
")
WLAYER=$(echo "$WINNER" | awk '{print $1}')
WSCALE=$(echo "$WINNER" | awk '{print $2}')
echo "[scale-test] Sweep winner: layer=$WLAYER scale=$WSCALE"

# Check if winner is good enough
WPARA=$(python3 -c "
import json
d = json.load(open('$SWEEP'))
print(d['winner_rates']['paraphrased'])
")
echo "[scale-test] Winner paraphrased rate: $WPARA"

if python3 -c "exit(0 if float('$WPARA') >= 0.7 else 1)"; then
    echo "[scale-test] Winner rate >= 0.7 — proceeding to headline eval"
else
    echo "[scale-test] WARNING: Winner rate < 0.7 — sweep may need extension"
    echo "[scale-test] Proceeding anyway; review results manually"
fi

# --- Stage 3: Headline rigorous eval ---
HEADLINE="results/expH/task1c_${TAG}_headline.json"
echo "=== [scale-test] Stage 3: headline eval at layer=$WLAYER scale=$WSCALE ==="
t0=$(date +%s)
python3 -u -m pitwm.scripts.expH_7b_transfer \
    --model-id "$MODEL" \
    --layer-idx "$WLAYER" \
    --scale "$WSCALE" \
    --verified-kb "$VERIFIED" \
    --kb-module pitwm.evaluation.acme_kb_expanded \
    --out "$HEADLINE"
echo "[scale-test] Stage 3 done in $(( $(date +%s) - t0 ))s"

echo ""
echo "=== [scale-test] COMPLETE at $(date -Is) ==="
echo "=== [scale-test] Artifacts: ==="
echo "  Verified KB: $VERIFIED"
echo "  Sweep:       $SWEEP"
echo "  Headline:    $HEADLINE"
echo "  Log:         $LOG"
