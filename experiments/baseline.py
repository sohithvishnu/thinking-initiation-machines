"""
Cold baselines: HumanEval / HumanEval+ pass@1 for every configured model,
repeated over identical greedy rounds.

Three rounds of the same greedy decode is how determinism was established
empirically for this study (std across rounds came out 0.00%), rather than by
setting `torch.backends.cudnn.deterministic` and assuming it held. This is the
floor every priming condition is measured against.

Usage:
    python experiments/baseline.py                          # 3 rounds, all models
    python experiments/baseline.py --rounds 1 --limit 20    # quick smoke test
    python experiments/baseline.py --model Qwen/Qwen3-1.7B
"""

import argparse
import json

from tim.config import LOGS_DIR, qwen_models
from tim.evaluation import load_problems, run_condition
from tim.models import load_model, unload_model

OUT_DIR = LOGS_DIR / "evalplus_baseline"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None,
                    help="Single model id; omit to run every model in tim.config.qwen_models.")
    ap.add_argument("--rounds", type=int, default=3,
                    help="Identical greedy rounds per model (default 3, for the determinism check).")
    ap.add_argument("--dataset", choices=["humaneval", "mbpp"], default="humaneval")
    ap.add_argument("--max_new_tokens", type=int, default=768)
    ap.add_argument("--limit", type=int, default=None, help="First N tasks only (smoke test).")
    ap.add_argument("--no_score", action="store_true", help="Skip EvalPlus scoring.")
    args = ap.parse_args()

    models = [args.model] if args.model else list(qwen_models)
    task_items = load_problems(args.dataset, limit=args.limit)

    for round_idx in range(1, args.rounds + 1):
        output_dir = OUT_DIR / f"{args.dataset}_{round_idx}"
        output_dir.mkdir(parents=True, exist_ok=True)
        round_results = {}

        for model_id in models:
            print(f"\n=== round {round_idx}/{args.rounds} | {model_id} ===")
            model, tokenizer, device, _ = load_model(model_id, quant="auto")
            round_results[model_id] = run_condition(
                model, tokenizer, device,
                f"Evalplus_{model_id.replace('/', '_')}", output_dir, task_items,
                dataset=args.dataset,
                max_new_tokens=args.max_new_tokens,
                score=not args.no_score,
            )
            unload_model(model)

        with open(output_dir / "round_results.json", "w") as f:
            json.dump(round_results, f, indent=2)

        print(f"\nRound {round_idx} summary "
              f"(pass@1_base = {args.dataset}, pass@1_base_plus_extra = {args.dataset}+)")
        for model_id, scores in round_results.items():
            print(f"  {model_id}: base={scores.get('pass@1_base')} "
                  f"plus={scores.get('pass@1_base_plus_extra')}")

    print(f"\nAggregate with: python scripts/summarize_evalplus.py {OUT_DIR}")


if __name__ == "__main__":
    main()
