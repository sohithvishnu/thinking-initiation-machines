#!/bin/bash
source /home/sohithvishnu/miniconda3/bin/activate ai_env

echo "=== GPU 0: bf16_cold, int4_cold ==="
CUDA_VISIBLE_DEVICES=0 python experiment-3/run_quant_gap.py --stage gate --dataset mbpp

for p in 1 2 3; do
    echo "=== GPU 0: bf16_tim_domain pass $p ==="
    CUDA_VISIBLE_DEVICES=0 python experiment-3/run_quant_gap.py --stage full --dataset mbpp --num_seeds 1 --num_passes $p --skip_prompt_control
done

echo "=== GPU 0: int4_tim_entropy seed0 pass 1 ==="
CUDA_VISIBLE_DEVICES=0 python experiment-3/run_rung5_entropy.py --dataset mbpp --seeds 0
