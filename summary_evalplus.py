"""
Summarize EvalPlus results across all evaluated models.

Scans a logs directory for *_eval_results.json files (the structured cache
EvalPlus writes after scoring — schema confirmed as:
    {
      "date": ..., "hash": ...,
      "eval": {task_id: [{"task_id", "solution", "base_status",
                           "plus_status", "base_fail_tests",
                           "plus_fail_tests"}]},
      "pass_at_k": {"base": {"pass@1": ...}, "plus": {"pass@1": ...}}
    }
No code re-execution needed — everything required is already in this file.

Usage:
    python summarize_evalplus_results.py [logs_dir]
    (defaults to ./logs)
"""

import json
import sys
from pathlib import Path


def find_result_files(logs_dir: Path):
    # Matches both "..._eval_results.json" and "...-sanitized_eval_results.json"
    # naming variants seen in practice.
    return sorted(logs_dir.glob("evalplus_baseline/*eval_results.json"))


def model_name_from_path(path: Path) -> str:
    """Best-effort extraction of a readable model name from the filename,
    stripping the humaneval_/mbpp_ prefix and -sanitized_eval_results suffix."""
    stem = path.stem
    for prefix in ("humaneval_", "mbpp_"):
        if stem.startswith(prefix):
            stem = stem[len(prefix):]
    for suffix in ("-sanitized_eval_results", "_eval_results"):
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
    return stem.replace("_", "/", 1) if "_" in stem else stem


def summarize_file(path: Path) -> dict:
    with open(path) as f:
        data = json.load(f)

    pass_at_k = data.get("pass_at_k", {})
    base_pass1 = pass_at_k.get("base", {}).get("pass@1")
    plus_pass1 = pass_at_k.get("plus", {}).get("pass@1")

    eval_data = data.get("eval", {})
    total_tasks = len(eval_data)

    base_pass = base_fail = plus_pass = plus_fail = 0
    fragile = []  # passes base tests but fails the augmented plus tests
    for task_id, entries in eval_data.items():
        entry = entries[0] if entries else {}
        b_status = entry.get("base_status")
        p_status = entry.get("plus_status")
        if b_status == "pass":
            base_pass += 1
        else:
            base_fail += 1
        if p_status == "pass":
            plus_pass += 1
        else:
            plus_fail += 1
        if b_status == "pass" and p_status == "fail":
            fragile.append(task_id)

    return {
        "file": path.name,
        "total_tasks": total_tasks,
        "pass@1_base": base_pass1,
        "pass@1_plus": plus_pass1,
        "base_pass": base_pass,
        "base_fail": base_fail,
        "plus_pass": plus_pass,
        "plus_fail": plus_fail,
        "fragile_count": len(fragile),   # passed base, failed plus — brittle solutions
        "fragile_task_ids": fragile,
        "date": data.get("hash", "")[:8],
    }


def print_leaderboard(summaries: list):
    print("=" * 78)
    print(f"{'Model':<28} {'HumanEval':>10} {'HumanEval+':>12} {'Fragile':>10} {'Tasks':>7}")
    print("=" * 78)
    # Sort by pass@1_base descending, None-safe
    summaries_sorted = sorted(
        summaries, key=lambda s: (s["pass@1_base"] is None, -(s["pass@1_base"] or 0))
    )
    for s in summaries_sorted:
        base = f"{s['pass@1_base']:.1%}" if s["pass@1_base"] is not None else "N/A"
        plus = f"{s['pass@1_plus']:.1%}" if s["pass@1_plus"] is not None else "N/A"
        print(f"{s['model_name']:<28} {base:>10} {plus:>12} {s['fragile_count']:>10} {s['total_tasks']:>7}")
    print("=" * 78)
    print("Fragile = passed original HumanEval tests but failed the augmented")
    print("HumanEval+ tests (i.e. a solution that looked correct but wasn't robust).")


def main():
    logs_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("logs")
    if not logs_dir.exists():
        print(f"Directory not found: {logs_dir}")
        sys.exit(1)

    result_files = find_result_files(logs_dir)
    if not result_files:
        print(f"No *eval_results.json files found in {logs_dir}")
        sys.exit(1)

    summaries = []
    for path in result_files:
        s = summarize_file(path)
        s["model_name"] = model_name_from_path(path)
        summaries.append(s)

        # Per-model detail, useful for spotting fragile solutions worth a manual look
        print(f"\n--- {s['model_name']} ({s['file']}) ---")
        print(f"  pass@1 (base):        {s['pass@1_base']:.4f}" if s['pass@1_base'] is not None else "  pass@1 (base): N/A")
        print(f"  pass@1 (base+plus):   {s['pass@1_plus']:.4f}" if s['pass@1_plus'] is not None else "  pass@1 (base+plus): N/A")
        print(f"  base pass/fail:       {s['base_pass']}/{s['base_fail']}")
        print(f"  plus pass/fail:       {s['plus_pass']}/{s['plus_fail']}")
        print(f"  fragile (base ok, plus fail): {s['fragile_count']}")
        if s["fragile_task_ids"]:
            print(f"    -> {', '.join(s['fragile_task_ids'][:10])}"
                  + (" ..." if len(s["fragile_task_ids"]) > 10 else ""))

    print()
    print_leaderboard(summaries)

    # Also write a machine-readable combined summary for later use (e.g. thesis tables)
    out_path = logs_dir / "combined_summary.json"
    with open(out_path, "w") as f:
        json.dump(summaries, f, indent=2)
    print(f"\nCombined summary written to: {out_path}")


if __name__ == "__main__":
    main()