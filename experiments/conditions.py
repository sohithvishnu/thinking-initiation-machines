"""
The main 4-condition experiment: cold / prompt_control / tim_random /
tim_domain, over N seeds, on HumanEval+.

Conditions
----------
cold            no priming — the floor.
prompt_control  DOMAIN_PERSONA_TEXT prepended to the prompt text. Prompt
                tokens become KV-cache entries during prefill, so a prompt
                prefix *is* a primed cache once processed; this is therefore
                the reference every TIM arm must beat, not `cold`. Run once —
                it is deterministic.
tim_random      KV cache prefilled with uniformly random vocabulary tokens.
tim_domain      KV cache prefilled with 70% domain-pool / 30% random tokens.

`--domain_pool` selects which pool `tim_domain` draws from. `sentence` is
DOMAIN_NAME, what the reported 3-seed run in `logs/tim/` used. `words` is the
211-entry PYTHON_DOMAIN_WORDS pool that every experiment from the pass sweep
onward uses. They are not interchangeable — see docs and `tim/vocab.py`.

Usage:
    python experiments/conditions.py --model Qwen/Qwen3-1.7B --num_seeds 3
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
from tim.vocab import DOMAIN_NAME, DOMAIN_PERSONA_TEXT, PYTHON_DOMAIN_WORDS

OUT_DIR = LOGS_DIR / "tim"


def run_model(model_id: str, args, task_items) -> dict:
    """All conditions for one model. Loaded once, unloaded at the end,
    however many seeds run against it."""
    output_dir = OUT_DIR / model_id.replace("/", "_")
    output_dir.mkdir(parents=True, exist_ok=True)

    model, tokenizer, device, _ = load_model(model_id, quant=args.quant)
    primer = TIMPrimer(model, tokenizer, device, noise_length=args.noise_length)
    if args.domain_pool == "words":
        domain_ids = primer.get_domain_token_ids_from_words(PYTHON_DOMAIN_WORDS)
    else:
        domain_ids = primer.get_domain_token_ids(DOMAIN_NAME)

    common = dict(dataset=args.dataset, max_new_tokens=args.max_new_tokens,
                  score=not args.no_score)
    results = {}

    if not args.skip_cold:
        results["cold"] = run_condition(
            model, tokenizer, device, "cold", output_dir, task_items, **common)

    results["prompt_control"] = run_condition(
        model, tokenizer, device, "prompt_control", output_dir, task_items,
        prepend_text=DOMAIN_PERSONA_TEXT + "\n\n", **common)

    for seed in range(args.num_seeds):
        print(f"\n=== {model_id} | seed {seed} ===")
        for arm, pool in (("random", None), ("domain", domain_ids)):
            kv, prime_timing = primer.prime(domain_tokens=pool, seed=seed, collect_timing=True)
            name = f"tim_{arm}_seed{seed}"
            results[name] = run_condition(
                model, tokenizer, device, name, output_dir, task_items,
                primed_kv=kv, prime_timing=prime_timing, **common)
            del kv
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    with open(output_dir / "combined_results.json", "w") as f:
        json.dump(results, f, indent=2)

    unload_model(model)
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None,
                    help="Single model id; omit to run every model in tim.config.qwen_models.")
    ap.add_argument("--num_seeds", type=int, default=3,
                    help="Seeds for tim_random / tim_domain (default 3).")
    ap.add_argument("--noise_length", type=int, default=64)
    ap.add_argument("--domain_pool", choices=["sentence", "words"], default="sentence",
                    help="tim_domain vocabulary source. 'sentence' = DOMAIN_NAME (what "
                         "logs/tim/ was produced with); 'words' = PYTHON_DOMAIN_WORDS.")
    ap.add_argument("--quant", choices=["auto", "bf16", "nf4"], default="auto")
    ap.add_argument("--dataset", choices=["humaneval", "mbpp"], default="humaneval")
    ap.add_argument("--max_new_tokens", type=int, default=768)
    ap.add_argument("--limit", type=int, default=None, help="First N tasks only (smoke test).")
    ap.add_argument("--no_score", action="store_true", help="Skip EvalPlus scoring.")
    ap.add_argument("--skip_cold", action="store_true",
                    help="Skip the cold condition (reuse existing baseline results).")
    args = ap.parse_args()

    models = [args.model] if args.model else list(qwen_models)
    task_items = load_problems(args.dataset, limit=args.limit)

    print("=" * 70)
    print(f"TIM conditions — {len(models)} model(s), {args.num_seeds} seed(s) each")
    print(f"Models: {models} | dataset={args.dataset} | domain_pool={args.domain_pool}")
    print("=" * 70)

    started = time.time()
    all_results = {}
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    combined_path = OUT_DIR / "all_models_combined_results.json"

    for i, model_id in enumerate(models, start=1):
        t0 = time.time()
        print(f"\n{'#' * 70}\n# MODEL {i}/{len(models)}: {model_id}\n{'#' * 70}")
        all_results[model_id] = run_model(model_id, args, task_items)
        print(f"\n[{model_id}] done in {(time.time() - t0) / 60:.1f} min")

        # Checkpoint after every model: a multi-hour run must not lose
        # completed work if it crashes or is interrupted partway through.
        with open(combined_path, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"Combined results so far: {combined_path}")

    print("\n" + "=" * 70)
    print(f"COMPLETE — {len(models)} model(s) in {(time.time() - started) / 60:.1f} min")
    print("=" * 70)
    for model_id, results in all_results.items():
        print(f"\n{model_id}:")
        for cond, scores in results.items():
            print(f"  {cond}: base={scores.get('pass@1_base')} "
                  f"plus={scores.get('pass@1_base_plus_extra')}")
    print(f"\nFull combined results: {combined_path}")


if __name__ == "__main__":
    main()
