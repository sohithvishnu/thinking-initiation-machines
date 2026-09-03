"""
Fills the one missing cell of the precision x priming grid: int4 (NF4) +
tim_domain on Qwen3-1.7B. The other three cells already exist on disk:

    bf16 cold        logs/tim/Qwen_Qwen3-1.7B/cold*
    bf16 tim_domain  logs/tim/Qwen_Qwen3-1.7B/tim_domain_seed{0,1,2}*
    int4 cold        logs/quant_gap/Qwen_Qwen3-1.7B/gate/C_nf4_cold*

Matches the config that passed the quantization gate exactly: NF4 with
bfloat16 compute, noise_length=64, chain_mode="reseed", and the
PYTHON_DOMAIN_WORDS pool.

Usage:
    python experiments/int4_tim_domain.py
    python experiments/int4_tim_domain.py --dataset mbpp --seeds 0 --num_passes 2
"""

import argparse
import json

from quant_gap import free_kv, load_nf4_checked
from tim.config import LOGS_DIR
from tim.evaluation import load_problems, run_condition
from tim.models import unload_model
from tim.primer import TIMPrimer
from tim.vocab import PYTHON_DOMAIN_WORDS

MODEL_ID = "Qwen/Qwen3-1.7B"
NOISE_LENGTH = 64
MAX_NEW_TOKENS = 768


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["humaneval", "mbpp"], default="humaneval")
    ap.add_argument("--seeds", default="0,1,2", help="Comma-separated seed list.")
    ap.add_argument("--num_passes", type=int, default=1,
                    help="TIMPrimer pass count (>1 appends a _pass<N> tag to condition names).")
    ap.add_argument("--limit", type=int, default=None, help="First N tasks only (smoke test).")
    ap.add_argument("--no_score", action="store_true", help="Skip EvalPlus scoring.")
    args = ap.parse_args()

    seeds = [int(s.strip()) for s in args.seeds.split(",")]
    out_dir = LOGS_DIR / ("mbpp_scale" if args.dataset == "mbpp" else "quality_quant")
    out_dir.mkdir(parents=True, exist_ok=True)
    task_items = load_problems(args.dataset, limit=args.limit)

    print(f"=== int4 tim_domain — {MODEL_ID}, {len(task_items)} {args.dataset} tasks, "
          f"seeds={seeds} ===")

    model, tokenizer, device, quant_report, det = load_nf4_checked(
        MODEL_ID, task_items[0][1]["prompt"])

    primer = TIMPrimer(model, tokenizer, device, noise_length=NOISE_LENGTH,
                       num_passes=args.num_passes, chain_mode="reseed")
    domain_ids = primer.get_domain_token_ids_from_words(PYTHON_DOMAIN_WORDS)
    pass_tag = f"_pass{args.num_passes}" if args.num_passes > 1 else ""

    results = {}
    for seed in seeds:
        print(f"\n--- seed {seed} ---")
        kv, prime_timing = primer.prime(domain_tokens=domain_ids, seed=seed, collect_timing=True)
        name = f"int4_tim_domain_seed{seed}{pass_tag}"
        results[name] = run_condition(
            model, tokenizer, device, name, out_dir, task_items,
            dataset=args.dataset, primed_kv=kv, prime_timing=prime_timing,
            max_new_tokens=MAX_NEW_TOKENS, score=not args.no_score)
        free_kv(kv)

    unload_model(model)

    report = {
        "model_id": MODEL_ID, "dataset": args.dataset, "n_tasks": len(task_items),
        "seeds": seeds, "num_passes": args.num_passes,
        "quant_report": quant_report, "determinism": det, "results": results,
    }
    report_path = out_dir / "int4_tim_domain_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    print("\n" + "=" * 78)
    print("int4 tim_domain generation complete")
    print("=" * 78)
    for name, m in results.items():
        print(f"  {name}: pass@1_base={m.get('pass@1_base')} "
              f"pass@1_plus={m.get('pass@1_base_plus_extra')} "
              f"per_task_ms={m.get('per_task_ms_mean'):.1f} "
              f"peak_mb={m.get('peak_memory_mb'):.1f}")
    print(f"\nReport written to: {report_path}")


if __name__ == "__main__":
    main()
