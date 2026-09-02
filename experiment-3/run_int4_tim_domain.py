"""
Generates the one missing cell in the precision x priming grid: int4 (NF4) +
tim_domain, seeds 0/1/2, on Qwen3-1.7B. The other three cells already exist
on disk:

    bf16 cold        logs/tim/Qwen_Qwen3-1.7B/cold*
    bf16 tim_domain  logs/tim/Qwen_Qwen3-1.7B/tim_domain_seed{0,1,2}*
    int4 cold        logs/quant_gap/Qwen_Qwen3-1.7B/gate/C_nf4_cold*

This script does not reimplement model loading, priming, generation, or
scoring — it imports the already-working functions from
experiment-3/run_quant_gap.py (load_model_with_quant, check_determinism,
run_condition, unload_model) and from experiment-2/tim_primer.py
(TIMPrimer, via run_quant_gap's own import), matching exactly the config
that passed the quantization gate: BitsAndBytesConfig(load_in_4bit=True,
bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16),
num_passes=args.num_passes, noise_length=64, the PYTHON_DOMAIN_WORDS pool.

Usage:
    python experiment-3/run_int4_tim_domain.py
"""

import gc
import json
from pathlib import Path

import torch
from evalplus.data import get_human_eval_plus, get_mbpp_plus
import argparse
import sys

EXP3_DIR = Path(__file__).resolve().parent
import sys
sys.path.insert(0, str(EXP3_DIR))
from run_quant_gap import (  # noqa: E402
    load_model_with_quant, unload_model, check_determinism, run_condition,
    TIMPrimer, PYTHON_DOMAIN_WORDS,
)

ROOT_DIR = EXP3_DIR.parent
OUT_DIR = ROOT_DIR / "logs" / "mbpp_scale" if "mbpp" in sys.argv else ROOT_DIR / "logs" / "quality_quant"
OUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_ID = "Qwen/Qwen3-1.7B"
NUM_SEEDS = 3
NOISE_LENGTH = 64
MAX_NEW_TOKENS = 768


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--no_score", action="store_true")
    ap.add_argument("--dataset", type=str, choices=["humaneval", "mbpp"], default="humaneval")
    ap.add_argument("--num_passes", type=int, default=1)
    ap.add_argument("--seeds", type=str, default="0,1,2")
    args = ap.parse_args()

    if args.dataset == "mbpp":
        problems = get_mbpp_plus()
    else:
        problems = get_human_eval_plus()
        
    task_items = list(problems.items())
    if args.limit:
        task_items = task_items[:args.limit]
    score = not args.no_score
    seed_list = [int(s.strip()) for s in args.seeds.split(",")]

    print(f"=== int4 tim_domain generation — {MODEL_ID}, {len(task_items)} tasks, "
          f"seeds 0..{NUM_SEEDS - 1} ===")

    use_double_quant = True
    model, tokenizer, device, quant_report = load_model_with_quant(
        MODEL_ID, "nf4", use_double_quant=use_double_quant)

    det = check_determinism(model, tokenizer, device, task_items[0][1]["prompt"])
    print(f"[determinism] nf4 double_quant={use_double_quant}: identical={det['identical']}")
    if not det["identical"]:
        print("[determinism] NOT identical — reloading with bnb_4bit_use_double_quant=False (batch size already 1)")
        unload_model(model)
        use_double_quant = False
        model, tokenizer, device, quant_report = load_model_with_quant(
            MODEL_ID, "nf4", use_double_quant=use_double_quant)
        det2 = check_determinism(model, tokenizer, device, task_items[0][1]["prompt"])
        print(f"[determinism] retry double_quant=False: identical={det2['identical']}")
        det = {"first_attempt": det, "retry_double_quant_false": det2}

    primer = TIMPrimer(model, tokenizer, device, noise_length=NOISE_LENGTH, num_passes=args.num_passes, chain_mode="reseed")
    domain_ids = primer.get_domain_token_ids_from_words(PYTHON_DOMAIN_WORDS)

    results = {}
    for seed in seed_list:
        print(f"\n--- seed {seed} ---")
        kv, ptime = primer.prime(domain_tokens=domain_ids, seed=seed, collect_timing=True)
        name = f"int4_tim_domain_seed{seed}" + (f"_pass{args.num_passes}" if args.num_passes > 1 else "")
        results[name] = run_condition(
            model, tokenizer, device, name, OUT_DIR, task_items,
            primed_kv=kv, prime_timing=ptime,
            max_new_tokens=MAX_NEW_TOKENS, score=score, dataset=args.dataset,
        )
        del kv
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    unload_model(model)

    report = {
        "model_id": MODEL_ID, "n_tasks": len(task_items), "num_seeds": NUM_SEEDS,
        "quant_report": quant_report, "determinism": det, "results": results,
    }
    report_path = OUT_DIR / "int4_tim_domain_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    print("\n" + "=" * 78)
    print("int4 tim_domain generation complete")
    print("=" * 78)
    for name, m in results.items():
        print(f"  {name}: pass@1_base={m.get('pass@1_base')} pass@1_plus={m.get('pass@1_base_plus_extra')} "
              f"per_task_ms={m.get('per_task_ms_mean'):.1f} peak_mb={m.get('peak_memory_mb'):.1f}")
    print(f"\nReport written to: {report_path}")


if __name__ == "__main__":
    main()
