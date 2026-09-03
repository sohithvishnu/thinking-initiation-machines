"""
Step 3 of the MBPP+ scoring-bug fix: re-score the conditions Step 1 found
with complete raw/sanitized data (378/378) but no eval_results.json (the
evalplus.evaluate subprocess never completed originally).

Step 1 (scripts/diagnose_mbpp_coverage.py) showed zero tasks lost at
generation or sanitization for *every* condition, including the broken ones
-- ruling out the truncation hypothesis outright. Step 2's condition-
correlation check is decisive: F_nf4_prompt_control (GPU0, full/ dir)
scored cleanly while int4_prompt_control (GPU1, top-level dir) -- the same
condition type, nf4 + prompt_control -- did not, in the same time window.
Since the *same condition* both succeeded and failed depending only on
which of the two concurrently-running GPU processes' evalplus subprocess
ran it, the loss cannot be content/condition-correlated; it is a transient
scoring-pipeline failure from running two independent `evalplus.evaluate`
subprocess trees (each spawning its own ProcessPoolExecutor worker pool)
concurrently against the same machine. Manually re-running evalplus on one
of the broken sanitized files (single process, no concurrency) succeeds
immediately and cleanly -- confirming the underlying sample data was never
the problem.

No regeneration is needed. This script simply re-invokes
evaluate_with_evalplus (imported from tim.evaluation, which hard-fails
instead of silently continuing on a mismatch -- see that module) *serially, one condition at a time*, against the existing
-sanitized.jsonl files, and merges the recovered pass@1_base /
pass@1_base_plus_extra / n_scored / n_canonical fields into each
condition's existing _metrics.json (every other field in that file, e.g.
timing/memory, is left untouched -- it was already correct, since
generation completed fine).

Usage:
    python scripts/rescore_mbpp_conditions.py
"""

import json

from tim.config import LOGS_DIR, ROOT_DIR
from tim.evaluation import evaluate_with_evalplus

MBPP_DIR = LOGS_DIR / "mbpp_scale"

# Conditions Step 1 flagged as "NOT SCORED" (complete raw+sanitized, no
# eval_results.json) -- see logs/mbpp_scale/coverage_report.json.
BROKEN = [
    MBPP_DIR / "int4_neutral_prefix.jsonl",
    MBPP_DIR / "int4_prompt_control.jsonl",
    MBPP_DIR / "int4_tim_entropy_seed0.jsonl",
    MBPP_DIR / "full" / "B_bf16_tim_domain_seed0.jsonl",
    MBPP_DIR / "full" / "B_bf16_tim_domain_seed0_pass2.jsonl",
    MBPP_DIR / "full" / "B_bf16_tim_domain_seed0_pass3.jsonl",
    MBPP_DIR / "full" / "D_nf4_tim_domain_seed0.jsonl",
]


def main():
    print(f"=== Re-scoring {len(BROKEN)} MBPP+ conditions, serially (no concurrency) ===\n")
    results = {}
    for raw_path in BROKEN:
        stem = raw_path.stem
        sanitized_path = raw_path.with_name(stem + "-sanitized.jsonl")
        metrics_path = raw_path.with_name(stem + "_metrics.json")

        print(f"--- {raw_path.relative_to(ROOT_DIR)} ---")
        if not sanitized_path.exists():
            print(f"  SKIP: sanitized file missing: {sanitized_path}")
            results[stem] = {"error": "sanitized file missing"}
            continue

        try:
            scores = evaluate_with_evalplus(sanitized_path, dataset="mbpp")
        except RuntimeError as e:
            print(f"  FAILED AGAIN: {e}")
            results[stem] = {"error": str(e)}
            continue

        print(f"  pass@1_base={scores.get('pass@1_base')} "
              f"pass@1_base_plus_extra={scores.get('pass@1_base_plus_extra')} "
              f"n_scored={scores.get('n_scored')}/{scores.get('n_canonical')}")
        results[stem] = scores

        if metrics_path.exists():
            with open(metrics_path) as f:
                metrics = json.load(f)
        else:
            metrics = {"condition": stem}
        metrics.update(scores)
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"  updated: {metrics_path.relative_to(ROOT_DIR)}\n")

    out_path = MBPP_DIR / "rescore_report.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    n_ok = sum(1 for v in results.values() if "error" not in v)
    print("=" * 78)
    print(f"Re-scored OK: {n_ok}/{len(BROKEN)}")
    if n_ok != len(BROKEN):
        print("FAILURES:")
        for stem, v in results.items():
            if "error" in v:
                print(f"  {stem}: {v['error'][:300]}")
    print(f"Report written to: {out_path}")


if __name__ == "__main__":
    main()
