#!/bin/bash
source /home/sohithvishnu/miniconda3/bin/activate ai_env

for p in 1 2 3; do
    echo "=== GPU 1: int4_tim_domain seed0 pass $p ==="
    CUDA_VISIBLE_DEVICES=1 python experiment-3/run_int4_tim_domain.py --dataset mbpp --seeds 0 --num_passes $p
done

echo "=== GPU 1: int4_tim_domain seeds 1,2 pass 1 ==="
CUDA_VISIBLE_DEVICES=1 python experiment-3/run_int4_tim_domain.py --dataset mbpp --seeds 1,2 --num_passes 1

echo "=== GPU 1: int4_prompt_control & int4_neutral_prefix ==="
CUDA_VISIBLE_DEVICES=1 python experiment-3/run_int4_mechanism.py --rungs 2,3b --dataset mbpp
