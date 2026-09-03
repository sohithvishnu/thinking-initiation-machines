"""
Extends scripts/llm_judge.py's pairwise Ollama judge to two pairings across
the precision x priming grid on Qwen3-1.7B:

  1. int4 cold vs int4 tim_domain   — direct int4 analog of the bf16 judge run
  2. cold: bf16 vs int4             — does quantization alone change
                                       perceived quality?

Reuses query_judge, get_model_tag, JUDGE_PROMPT, and the cache read/write
helpers from scripts/llm_judge.py unchanged (via import) — this script only
adds multi-directory solution loading (llm_judge.py's load_solutions/
load_eval_results already take an explicit logs_dir, so no change was
needed there) and handles TWO pairings sharing one cache file, which
requires the pairing name in the cache key (llm_judge.py's own cache only
ever handled one pairing, cold vs tim_domain, so its key was
(task_id, order, model) — insufficient here since e.g. "cold" nf4 appears in
both pairings below and must not collide).

think: false is set, same fix used in the bf16 judge run (avoids empty
`content` from the model spending its whole token budget in a thinking
trace).

Usage:
    python scripts/llm_judge_grid.py --model gemma4:12b
"""

import argparse
import json

from evalplus.data import get_human_eval_plus  # noqa: E402
from scipy.stats import binomtest  # noqa: E402

from llm_judge import (
    JUDGE_PROMPT,
    both_pass_set,
    get_model_tag,
    load_eval_results,
    load_solutions,
    query_judge,
)
from tim.config import ROOT_DIR

ROOT = ROOT_DIR
QUALITY_QUANT_DIR = ROOT / "logs" / "quality_quant"
QUALITY_QUANT_DIR.mkdir(parents=True, exist_ok=True)
CACHE_PATH = QUALITY_QUANT_DIR / "judge_raw.jsonl"

BF16_TIM_DIR = ROOT / "logs" / "tim" / "Qwen_Qwen3-1.7B"
INT4_COLD_DIR = ROOT / "logs" / "quant_gap" / "Qwen_Qwen3-1.7B" / "gate"
INT4_TIM_DIR = ROOT / "logs" / "quality_quant"

PAIRINGS = {
    "int4_cold_vs_tim_domain": {
        "a": (INT4_COLD_DIR, "C_nf4_cold", "int4_cold"),
        "b": (INT4_TIM_DIR, "int4_tim_domain_seed0", "int4_tim_domain"),
    },
    "cold_bf16_vs_int4": {
        "a": (BF16_TIM_DIR, "cold", "bf16_cold"),
        "b": (INT4_COLD_DIR, "C_nf4_cold", "int4_cold"),
    },
}


def load_cache() -> dict:
    """Returns {(pairing, task_id, order, model): record}."""
    cache = {}
    if CACHE_PATH.exists():
        for line in CACHE_PATH.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            cache[(rec["pairing"], rec["task_id"], rec["order"], rec["model"])] = rec
    return cache


def append_cache(rec: dict):
    with open(CACHE_PATH, "a") as f:
        f.write(json.dumps(rec) + "\n")


def run_pairing(pairing_name: str, model: str, problems: dict, cache: dict,
                 a_logs_dir, a_cond, a_label, b_logs_dir, b_cond, b_label, limit=None):
    a_sol = load_solutions(a_logs_dir, a_cond)
    b_sol = load_solutions(b_logs_dir, b_cond)
    a_eval = load_eval_results(a_logs_dir, a_cond)
    b_eval = load_eval_results(b_logs_dir, b_cond)

    bp = sorted(both_pass_set(a_eval, b_eval))
    if limit:
        bp = bp[:limit]
    print(f"\n=== {pairing_name}: {a_label} vs {b_label} — both-pass n={len(bp)} ===")

    results = {}
    n_queried = n_cached = 0
    for task_id in bp:
        problem_prompt = problems[task_id]["prompt"]
        sol_a, sol_b = a_sol[task_id], b_sol[task_id]
        results[task_id] = {}

        for order, (sol1, sol2) in (("A", (sol_a, sol_b)), ("B", (sol_b, sol_a))):
            key = (pairing_name, task_id, order, model)
            if key in cache:
                results[task_id][order] = cache[key]
                n_cached += 1
                continue

            prompt = JUDGE_PROMPT.format(prompt=problem_prompt, sol1=sol1, sol2=sol2)
            out = query_judge(model, prompt, problem_prompt, sol1, sol2)
            rec = {"pairing": pairing_name, "task_id": task_id, "order": order, "model": model, **out}
            append_cache(rec)
            cache[key] = rec
            results[task_id][order] = rec
            n_queried += 1
            if n_queried % 20 == 0:
                print(f"  queried {n_queried} new judge calls...")

    print(f"  judge calls: {n_cached} cached, {n_queried} newly queried ({len(bp) * 2} total)")

    def decision(order, winner):
        if winner is None:
            return None
        if order == "A":
            return {"1": a_label, "2": b_label, "tie": "tie"}[winner]
        else:
            return {"1": b_label, "2": a_label, "tie": "tie"}[winner]

    a_wins = b_wins = ties = position_bias = judge_errors = 0
    reason_counts = {a_label: {}, b_label: {}}
    per_task_outcome = {}

    for task_id in bp:
        rec_a, rec_b = results[task_id]["A"], results[task_id]["B"]
        if not rec_a["parse_ok"] or not rec_b["parse_ok"]:
            judge_errors += 1
            per_task_outcome[task_id] = "judge_error"
            continue
        dec_a = decision("A", rec_a["winner"])
        dec_b = decision("B", rec_b["winner"])
        if dec_a != dec_b:
            position_bias += 1
            per_task_outcome[task_id] = "position_bias"
            continue
        outcome = dec_a
        per_task_outcome[task_id] = outcome
        if outcome == "tie":
            ties += 1
        elif outcome == a_label:
            a_wins += 1
            for rec in (rec_a, rec_b):
                reason_counts[a_label][rec["reason"]] = reason_counts[a_label].get(rec["reason"], 0) + 1
        elif outcome == b_label:
            b_wins += 1
            for rec in (rec_a, rec_b):
                reason_counts[b_label][rec["reason"]] = reason_counts[b_label].get(rec["reason"], 0) + 1

    order_consistent_n = a_wins + b_wins + ties
    valid_n = order_consistent_n + position_bias

    binom_result = None
    if (a_wins + b_wins) > 0:
        bt = binomtest(b_wins, a_wins + b_wins, 0.5)
        binom_result = {
            f"{b_label}_wins": b_wins, f"{a_label}_wins": a_wins,
            "p_value": float(bt.pvalue),
            f"{b_label}_win_rate": b_wins / (a_wins + b_wins),
            "ci95": [float(x) for x in bt.proportion_ci(confidence_level=0.95)],
        }

    print(f"  judge_errors={judge_errors}  position_bias={position_bias}/{valid_n} "
          f"({position_bias/valid_n*100:.1f}%)" if valid_n else "  no valid pairs")
    print(f"  order-consistent={order_consistent_n}  {a_label}_wins={a_wins}  "
          f"{b_label}_wins={b_wins}  ties={ties}")
    if binom_result:
        print(f"  binomial test: p={binom_result['p_value']:.4f}  "
              f"{b_label} win rate={binom_result[f'{b_label}_win_rate']*100:.1f}%  "
              f"95% CI [{binom_result['ci95'][0]*100:.1f}, {binom_result['ci95'][1]*100:.1f}]")
    print(f"  reasons ({a_label} wins): {reason_counts[a_label]}")
    print(f"  reasons ({b_label} wins): {reason_counts[b_label]}")

    return {
        "a_label": a_label, "b_label": b_label, "n_both_pass": len(bp),
        "judge_errors": judge_errors, "position_bias": position_bias,
        "order_consistent_n": order_consistent_n,
        f"{a_label}_wins": a_wins, f"{b_label}_wins": b_wins, "ties": ties,
        "binomial_test": binom_result, "reason_counts": reason_counts,
        "per_task_outcome": per_task_outcome,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gemma4:12b")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    model = get_model_tag(args.model)
    print(f"Using Ollama model: {model}")

    problems = get_human_eval_plus()
    cache = load_cache()

    summaries = {}
    for pairing_name, cfg in PAIRINGS.items():
        a_logs_dir, a_cond, a_label = cfg["a"]
        b_logs_dir, b_cond, b_label = cfg["b"]
        summaries[pairing_name] = run_pairing(
            pairing_name, model, problems, cache,
            a_logs_dir, a_cond, a_label, b_logs_dir, b_cond, b_label,
            limit=args.limit,
        )

    summary_path = QUALITY_QUANT_DIR / "judge_grid_summary.json"
    with open(summary_path, "w") as f:
        json.dump({"model": model, "pairings": summaries}, f, indent=2)
    print(f"\nSummary written to: {summary_path}")
    print(f"Raw judge cache: {CACHE_PATH}")


if __name__ == "__main__":
    main()
