"""
Part 2: pairwise LLM judge (Ollama) for the quantization-... no — the
generation-quality study. Complements scripts/quality_panel.py's objective
metrics with a pairwise preference judgment between cold and tim_domain_seed0
solutions, on their shared both-pass set (rebuilt the same way as Part 1:
base_status == 'pass' AND plus_status == 'pass' in BOTH conditions).

Judge model: read from `ollama list` at runtime, not assumed. The user
mentioned "Gemma-3-12B-class"; the tag actually present in this environment
is gemma4:12b, which is what this script uses (override with --model).

Position-bias control: every task is judged TWICE, with solution order
swapped (cold-first, then tim_domain-first). A "win" for either condition is
only counted if both orders agree; disagreement across the swap is recorded
separately as position_bias, not silently resolved either way.

Every judge call is cached to logs/quality/judge_raw.jsonl keyed on
(task_id, order, model) — re-running this script does not re-query cached
pairs.

Usage:
    python scripts/llm_judge.py --model gemma4:12b
"""

import argparse
import json
from pathlib import Path

import requests
from evalplus.data import get_human_eval_plus

from tim.config import ROOT_DIR

ROOT = ROOT_DIR
QUALITY_DIR = ROOT / "logs" / "quality"
QUALITY_DIR.mkdir(parents=True, exist_ok=True)
CACHE_PATH = QUALITY_DIR / "judge_raw.jsonl"

OLLAMA_URL = "http://localhost:11434/api/chat"

RUBRIC = ["readability", "idiomaticity", "documentation", "robustness", "simplicity", "no meaningful difference"]

JUDGE_PROMPT = """You are evaluating two Python solutions to the same programming problem. \
Pick which is better-written overall, or "tie" if there is no meaningful difference in quality. \
State the single most important reason for your choice, using exactly one of these terms: \
readability, idiomaticity, documentation, robustness, simplicity, no meaningful difference.

Respond with strict JSON only, no other text, no markdown fences: \
{{"winner": "1"|"2"|"tie", "reason": "<one of the terms above>"}}

Problem:
{prompt}

Solution 1:
```python
{sol1}
```

Solution 2:
```python
{sol2}
```"""


def load_solutions(logs_dir: Path, condition: str) -> dict:
    path = logs_dir / f"{condition}-sanitized.jsonl"
    out = {}
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        d = json.loads(line)
        out[d["task_id"]] = d["solution"]
    return out


def load_eval_results(logs_dir: Path, condition: str) -> dict:
    d = json.load(open(logs_dir / f"{condition}-sanitized.eval_results.json"))
    out = {}
    for task_id, entries in d["eval"].items():
        e = entries[0]
        out[task_id] = e["base_status"] == "pass" and e["plus_status"] == "pass"
    return out


def both_pass_set(a: dict, b: dict) -> set:
    return {t for t in (set(a) & set(b)) if a[t] and b[t]}


def load_cache(cache_path: Path = CACHE_PATH) -> dict:
    """Returns {(task_id, order, model): record}."""
    cache = {}
    if cache_path.exists():
        for line in cache_path.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            cache[(rec["task_id"], rec["order"], rec["model"])] = rec
    return cache


def append_cache(rec: dict, cache_path: Path = CACHE_PATH):
    with open(cache_path, "a") as f:
        f.write(json.dumps(rec) + "\n")


def query_judge(model: str, prompt: str, problem_prompt: str, sol1: str, sol2: str) -> dict:
    """Returns {'raw': <response text>, 'winner': str|None, 'reason': str|None, 'parse_ok': bool}."""
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "format": "json",
        "think": False,  # avoids empty `content` when the model spends its budget in a thinking trace
        "stream": False,
    }
    try:
        resp = requests.post(OLLAMA_URL, json=body, timeout=180)
        resp.raise_for_status()
        content = resp.json()["message"]["content"]
    except Exception as exc:
        return {"raw": f"REQUEST_ERROR: {exc}", "winner": None, "reason": None, "parse_ok": False}

    try:
        parsed = json.loads(content)
        winner = str(parsed.get("winner", "")).strip().lower()
        reason = str(parsed.get("reason", "")).strip().lower()
        if winner not in ("1", "2", "tie"):
            return {"raw": content, "winner": None, "reason": None, "parse_ok": False}
        return {"raw": content, "winner": winner, "reason": reason, "parse_ok": True}
    except (json.JSONDecodeError, AttributeError):
        return {"raw": content, "winner": None, "reason": None, "parse_ok": False}


def get_model_tag(requested: str) -> str:
    resp = requests.get("http://localhost:11434/api/tags", timeout=10)
    resp.raise_for_status()
    tags = [m["name"] for m in resp.json().get("models", [])]
    if requested in tags:
        return requested
    matches = [t for t in tags if requested.split(":")[0] in t]
    if matches:
        print(f"WARNING: '{requested}' not found in `ollama list`; using closest match '{matches[0]}'. "
              f"Available: {tags}")
        return matches[0]
    raise RuntimeError(f"Model '{requested}' not found via Ollama. Available tags: {tags}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_arg_hint", default="gemma3:12b", help=argparse.SUPPRESS)
    ap.add_argument("--model", default="gemma4:12b",
                     help="Ollama model tag. Default matches what `ollama list` showed at "
                          "development time (gemma4:12b, not gemma3:12b) — override if different.")
    ap.add_argument("--target_model", default="Qwen/Qwen3-1.7B")
    ap.add_argument("--limit", type=int, default=None, help="Judge only the first N both-pass tasks (debug).")
    args = ap.parse_args()

    model = get_model_tag(args.model)
    print(f"Using Ollama model: {model}")

    safe = args.target_model.replace("/", "_")
    logs_dir = ROOT / "logs" / "tim" / safe

    cold_sol = load_solutions(logs_dir, "cold")
    tim_sol = load_solutions(logs_dir, "tim_domain_seed0")
    cold_eval = load_eval_results(logs_dir, "cold")
    tim_eval = load_eval_results(logs_dir, "tim_domain_seed0")

    bp = sorted(both_pass_set(cold_eval, tim_eval))
    if args.limit:
        bp = bp[: args.limit]
    print(f"cold ∩ tim_domain_seed0 both-pass set: {len(bp)} tasks")

    problems = get_human_eval_plus()
    cache = load_cache()

    results = {}  # task_id -> {"A": rec, "B": rec}
    n_queried = n_cached = 0

    for task_id in bp:
        problem_prompt = problems[task_id]["prompt"]
        c, t = cold_sol[task_id], tim_sol[task_id]
        results[task_id] = {}

        for order, (sol1, sol2) in (("A", (c, t)), ("B", (t, c))):
            key = (task_id, order, model)
            if key in cache:
                results[task_id][order] = cache[key]
                n_cached += 1
                continue

            prompt = JUDGE_PROMPT.format(prompt=problem_prompt, sol1=sol1, sol2=sol2)
            out = query_judge(model, prompt, problem_prompt, sol1, sol2)
            rec = {"task_id": task_id, "order": order, "model": model, **out}
            append_cache(rec)
            cache[key] = rec
            results[task_id][order] = rec
            n_queried += 1
            if n_queried % 10 == 0:
                print(f"  queried {n_queried} new judge calls...")

    print(f"\nJudge calls: {n_cached} from cache, {n_queried} newly queried "
          f"({len(bp) * 2} total expected)")

    # ---- aggregate ----
    def decision(order: str, winner):
        if winner is None:
            return None
        if order == "A":
            return {"1": "cold", "2": "tim_domain", "tie": "tie"}[winner]
        else:
            return {"1": "tim_domain", "2": "cold", "tie": "tie"}[winner]

    cold_wins = tim_wins = ties = position_bias = judge_errors = 0
    reason_counts = {"tim_domain": {}, "cold": {}}
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

        outcome = dec_a  # dec_a == dec_b
        per_task_outcome[task_id] = outcome
        if outcome == "tie":
            ties += 1
        elif outcome == "tim_domain":
            tim_wins += 1
            for rec in (rec_a, rec_b):
                reason_counts["tim_domain"][rec["reason"]] = reason_counts["tim_domain"].get(rec["reason"], 0) + 1
        elif outcome == "cold":
            cold_wins += 1
            for rec in (rec_a, rec_b):
                reason_counts["cold"][rec["reason"]] = reason_counts["cold"].get(rec["reason"], 0) + 1

    order_consistent_n = cold_wins + tim_wins + ties
    valid_n = order_consistent_n + position_bias  # excludes judge_error

    from scipy.stats import binomtest
    binom_result = None
    if (cold_wins + tim_wins) > 0:
        bt = binomtest(tim_wins, cold_wins + tim_wins, 0.5)
        binom_result = {
            "tim_wins": tim_wins, "cold_wins": cold_wins,
            "p_value": float(bt.pvalue),
            "tim_win_rate": tim_wins / (cold_wins + tim_wins),
            "ci95": [float(x) for x in bt.proportion_ci(confidence_level=0.95)],
        }

    print("\n" + "=" * 78)
    print(f"LLM JUDGE RESULTS — {model}, cold vs tim_domain_seed0, n_both_pass={len(bp)}")
    print("=" * 78)
    print(f"judge_errors      : {judge_errors}")
    print(f"position_bias     : {position_bias} / {valid_n} valid pairs "
          f"({position_bias/valid_n*100:.1f}%)" if valid_n else "position_bias: n/a")
    print(f"order-consistent  : {order_consistent_n}")
    print(f"  tim_domain wins : {tim_wins} ({tim_wins/order_consistent_n*100:.1f}%)" if order_consistent_n else "")
    print(f"  cold wins       : {cold_wins} ({cold_wins/order_consistent_n*100:.1f}%)" if order_consistent_n else "")
    print(f"  ties            : {ties} ({ties/order_consistent_n*100:.1f}%)" if order_consistent_n else "")
    if binom_result:
        print(f"\nbinomial test (tim_wins vs cold_wins, excl. ties/bias): "
              f"p={binom_result['p_value']:.4f}, tim win rate={binom_result['tim_win_rate']*100:.1f}% "
              f"95% CI [{binom_result['ci95'][0]*100:.1f}, {binom_result['ci95'][1]*100:.1f}]")
    print("\nreason distribution (tim_domain wins):", reason_counts["tim_domain"])
    print("reason distribution (cold wins):       ", reason_counts["cold"])

    summary = {
        "model": model,
        "target_model": args.target_model,
        "n_both_pass": len(bp),
        "judge_errors": judge_errors,
        "position_bias": position_bias,
        "order_consistent_n": order_consistent_n,
        "tim_domain_wins": tim_wins,
        "cold_wins": cold_wins,
        "ties": ties,
        "binomial_test": binom_result,
        "reason_counts": reason_counts,
        "per_task_outcome": per_task_outcome,
    }
    summary_path = QUALITY_DIR / "judge_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary written to: {summary_path}")
    print(f"Raw judge cache: {CACHE_PATH}")


if __name__ == "__main__":
    main()
