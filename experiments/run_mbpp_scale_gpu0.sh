#!/usr/bin/env bash
# MBPP+ scale run, GPU 0 half: the bf16/int4 gate and the bf16 multi-pass ladder.
# Pair with run_mbpp_scale_gpu1.sh on the other GPU.
#
# NOTE: the two halves run concurrently, but each spawns its own
# `evalplus.evaluate` subprocess tree. Run one half at a time if you hit
# scoring flakiness; conditions that fail to score can be recovered afterwards
# with `python scripts/rescore_mbpp_conditions.py` (serial, no concurrency).
set -euo pipefail

cd "$(dirname "$0")/.."
export CUDA_VISIBLE_DEVICES=0

echo "=== GPU 0: bf16_cold, int4_cold (gate) ==="
python experiments/quant_gap.py --stage gate --dataset mbpp

for p in 1 2 3; do
    echo "=== GPU 0: bf16_tim_domain pass $p ==="
    python experiments/quant_gap.py --stage full --dataset mbpp \
        --num_seeds 1 --num_passes "$p" --skip_prompt_control
done

echo "=== GPU 0: int4_tim_entropy seed0 ==="
python experiments/rung5_entropy.py --dataset mbpp --seeds 0
