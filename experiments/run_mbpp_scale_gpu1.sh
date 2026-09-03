#!/usr/bin/env bash
# MBPP+ scale run, GPU 1 half: the int4 multi-pass ladder and mechanism rungs.
# Pair with run_mbpp_scale_gpu0.sh on the other GPU. See that script's note on
# concurrent scoring.
set -euo pipefail

cd "$(dirname "$0")/.."
export CUDA_VISIBLE_DEVICES=1

for p in 1 2 3; do
    echo "=== GPU 1: int4_tim_domain seed0 pass $p ==="
    python experiments/int4_tim_domain.py --dataset mbpp --seeds 0 --num_passes "$p"
done

echo "=== GPU 1: int4_tim_domain seeds 1,2 pass 1 ==="
python experiments/int4_tim_domain.py --dataset mbpp --seeds 1,2 --num_passes 1

echo "=== GPU 1: int4_prompt_control & int4_neutral_prefix ==="
python experiments/int4_mechanism.py --rungs 2,3b --dataset mbpp
