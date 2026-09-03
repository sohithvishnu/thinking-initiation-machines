"""
TIM multi-pass sweep + time-efficiency benchmark.

Answers two questions the accuracy-only runs could not:

  Q1 (passes)     Does iterative KV accumulation help, and how does it scale
                  with pass count? Sweeps 1,2,3 (the range an earlier pilot
                  suggested was useful) plus 6,7,10,12 to probe whether longer
                  chains keep helping, plateau, or degrade.

  Q2 (efficiency) What does priming cost per query? A primed cache is built
                  once, but every task pays a deepcopy plus attention over a
                  longer cache. This measures where the crossover sits against
                  `cold` and `prompt_control`.

Timing methodology (implemented in tim.generation / tim.evaluation):
  * torch.cuda.synchronize() around every measured region.
  * A warmup task before timing starts, so one-off CUDA autotune/alloc costs
    do not land on task 1.
  * Peak memory summed across all visible devices.
  * Priming cost reported separately from per-task cost, then combined into an
    amortized total, so the trade-off is explicit rather than hidden.

Usage:
    python experiments/pass_sweep.py --model Qwen/Qwen3-1.7B --limit 30
    python experiments/pass_sweep.py --model Qwen/Qwen3-1.7B
    python experiments/pass_sweep.py --passes 1 3 7 12 --chain_mode reseed
"""

import argparse
import gc
import json
import time

import torch

from tim.config import LOGS_DIR, qwen_models
from tim.evaluation import load_problems, run_condition
from tim.models import load_model, unload_model
from tim.primer import TIMPrimer
from tim.vocab import DOMAIN_PERSONA_TEXT, PYTHON_DOMAIN_WORDS

OUT_DIR = LOGS_DIR / "tim_passes"
DEFAULT_PASSES = [1, 2, 3, 6, 7, 10, 12]


def run_model(model_id: str, args, task_items) -> dict:
    output_dir = OUT_DIR / model_id.replace("/", "_")
    output_dir.mkdir(parents=True, exist_ok=True)

    model, tokenizer, device, _ = load_model(model_id, quant=args.quant)
    print(f"  {len(task_items)} tasks | scoring={'on' if args.score else 'off'}")

    common = dict(dataset=args.dataset, max_new_tokens=args.max_new_tokens, score=args.score)
    results = {}

    # ---- reference baselines (no priming) --------------------------------
    results["cold"] = run_condition(
        model, tokenizer, device, "cold", output_dir, task_items, **common)
    print(f"    cold           : {results['cold']['per_task_ms_mean']:8.1f} ms/task")

    results["prompt_control"] = run_condition(
        model, tokenizer, device, "prompt_control", output_dir, task_items,
        prepend_text=DOMAIN_PERSONA_TEXT + "\n\n", **common)
    print(f"    prompt_control : {results['prompt_control']['per_task_ms_mean']:8.1f} ms/task")

    # ---- pass-count sweep -------------------------------------------------
    primer = TIMPrimer(
        model, tokenizer, device,
        noise_length=args.noise_length,
        think_tokens=args.think_tokens,
        chain_mode=args.chain_mode,
        prime_temperature=args.prime_temperature,
        prime_top_k=args.prime_top_k,
    )
    domain_ids = primer.get_domain_token_ids_from_words(PYTHON_DOMAIN_WORDS)

    for n_passes in args.passes:
        primer.num_passes = n_passes
        for arm, pool in (("domain", domain_ids), ("random", None)):
            if arm not in args.arms:
                continue
            for seed in range(args.num_seeds):
                name = f"tim_{arm}_p{n_passes}_seed{seed}"
                kv, prime_timing = primer.prime(
                    domain_tokens=pool, seed=seed, collect_timing=True)
                results[name] = run_condition(
                    model, tokenizer, device, name, output_dir, task_items,
                    primed_kv=kv, prime_timing=prime_timing, **common)
                m = results[name]
                print(f"    {name:<28}: {m['per_task_ms_mean']:8.1f} ms/task | "
                      f"prime {m['prime_ms']:7.1f} ms | cache {m['primed_cache_len']:>4}")
                del kv
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        # Checkpoint after each pass count so a long sweep never loses work.
        with open(output_dir / "sweep_results.json", "w") as f:
            json.dump(results, f, indent=2)

    unload_model(model)
    return results


def print_report(model_id: str, results: dict):
    print("\n" + "=" * 104)
    print(f"PASS-COUNT & EFFICIENCY REPORT — {model_id}")
    print("=" * 104)
    print(f"{'condition':<28}{'cache':>7}{'prime ms':>10}{'ms/task':>10}"
          f"{'copy ms':>9}{'tok/s':>8}{'amort ms':>10}{'HE':>7}{'HE+':>7}")
    print("-" * 104)

    for name, m in results.items():
        he, hep = m.get("pass@1_base"), m.get("pass@1_base_plus_extra")
        print(f"{name:<28}"
              f"{m.get('primed_cache_len', 0):>7}"
              f"{m.get('prime_ms', 0):>10.1f}"
              f"{m.get('per_task_ms_mean', 0):>10.1f}"
              f"{m.get('deepcopy_ms', {}).get('mean', 0):>9.2f}"
              f"{m.get('tokens_per_sec', 0):>8.1f}"
              f"{m.get('amortized_per_task_ms', 0):>10.1f}"
              f"{(f'{he:.1%}' if he is not None else '-'):>7}"
              f"{(f'{hep:.1%}' if hep is not None else '-'):>7}")
    print("-" * 104)

    base_ms = results.get("cold", {}).get("per_task_ms_mean") or 0.0
    if base_ms:
        print(f"\nPer-task cost relative to cold ({base_ms:.1f} ms/task):")
        for name, m in results.items():
            if name != "cold":
                print(f"  {name:<30} {m.get('per_task_ms_mean', 0) / base_ms:5.2f}x")

    print("\nInterpretation guide:")
    print("  * ms/task excludes one-time priming — the steady-state per-query cost.")
    print("  * amort ms includes priming spread over this task count; it falls as the")
    print("    task count rises, so compare it only at equal n_tasks.")
    print("  * If ms/task climbs with cache length while HE+ stays flat, extra passes")
    print("    are buying nothing and costing throughput.")
    print("  * prompt_control is the reference any TIM arm must beat. Production prefix")
    print("    caching already amortizes prompt prefixes losslessly, so beating cold")
    print("    alone is not enough to justify the machinery.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None,
                    help="Single model; omit to sweep every model in tim.config.qwen_models.")
    ap.add_argument("--passes", type=int, nargs="+", default=DEFAULT_PASSES,
                    help=f"Pass counts to sweep (default {DEFAULT_PASSES}).")
    ap.add_argument("--arms", nargs="+", default=["domain", "random"],
                    choices=["domain", "random"])
    ap.add_argument("--num_seeds", type=int, default=1,
                    help="Seeds per (pass, arm). Keep at 1 for timing sweeps; raise for "
                         "accuracy claims.")
    ap.add_argument("--noise_length", type=int, default=64)
    ap.add_argument("--think_tokens", type=int, default=32,
                    help="Proto-thought tokens generated per pass after the first.")
    ap.add_argument("--chain_mode", default="persistent", choices=["persistent", "reseed"])
    ap.add_argument("--prime_temperature", type=float, default=0.8,
                    help="0 => greedy proto-thoughts (deterministic priming).")
    ap.add_argument("--prime_top_k", type=int, default=50)
    ap.add_argument("--quant", choices=["auto", "bf16", "nf4"], default="auto")
    ap.add_argument("--dataset", choices=["humaneval", "mbpp"], default="humaneval")
    ap.add_argument("--max_new_tokens", type=int, default=768)
    ap.add_argument("--limit", type=int, default=None,
                    help="First N tasks only (recommended for timing runs).")
    ap.add_argument("--no_score", dest="score", action="store_false",
                    help="Skip EvalPlus scoring (pure timing run, much faster).")
    args = ap.parse_args()

    models = [args.model] if args.model else list(qwen_models)
    task_items = load_problems(args.dataset, limit=args.limit)

    print("=" * 70)
    print("TIM pass-count sweep + efficiency benchmark")
    print(f"  models      : {models}")
    print(f"  passes      : {args.passes}")
    print(f"  arms        : {args.arms}  seeds={args.num_seeds}")
    print(f"  chain_mode  : {args.chain_mode}  prime_temp={args.prime_temperature}")
    print(f"  noise_length: {args.noise_length}  think_tokens={args.think_tokens}")
    print("=" * 70)

    started = time.time()
    all_results = {}
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    for i, model_id in enumerate(models, 1):
        t0 = time.time()
        print(f"\n{'#' * 70}\n# MODEL {i}/{len(models)}: {model_id}\n{'#' * 70}")
        all_results[model_id] = run_model(model_id, args, task_items)
        print(f"\n[{model_id}] done in {(time.time() - t0) / 60:.1f} min")
        print_report(model_id, all_results[model_id])

        with open(OUT_DIR / "all_models_pass_sweep.json", "w") as f:
            json.dump({"config": vars(args), "results": all_results}, f, indent=2)

    print(f"\nSWEEP COMPLETE in {(time.time() - started) / 60:.1f} min")
    print(f"Results: {OUT_DIR / 'all_models_pass_sweep.json'}")


if __name__ == "__main__":
    main()
