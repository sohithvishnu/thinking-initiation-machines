"""
Aggregate EvalPlus `*eval_results.json` files into per-model mean +/- std
tables, and write `combined_paper_summary.json` beside them.

Groups by model name, so the N identical rounds of a baseline run collapse
into one row with a standard deviation — which is how determinism was
established for this study rather than assumed.

Usage:
    python scripts/summarize_evalplus.py [logs_dir]

    logs_dir defaults to logs/evalplus_baseline.
"""

import json
import math
import sys
from collections import defaultdict
from pathlib import Path

from tim.config import LOGS_DIR

PREFIXES = ("humaneval_", "mbpp_", "Evalplus_")
SUFFIXES = ("-sanitized", "_eval_results", ".eval_results")


def find_result_files(logs_dir: Path):
    return sorted(logs_dir.rglob("*eval_results.json"))


def model_name_from_path(path: Path) -> str:
    """`Evalplus_Qwen_Qwen3-4B-sanitized.eval_results.json` -> `Qwen/Qwen3-4B`.

    Suffixes are stripped repeatedly rather than once each: a name carries
    both `-sanitized` and `.eval_results`, and stripping in a single fixed
    order leaves whichever one is inside the other still attached.
    """
    stem = path.stem
    for prefix in PREFIXES:
        if stem.startswith(prefix):
            stem = stem[len(prefix):]
    changed = True
    while changed:
        changed = False
        for suffix in SUFFIXES:
            if stem.endswith(suffix):
                stem = stem[: -len(suffix)]
                changed = True
    return stem.replace("_", "/", 1) if "_" in stem else stem

def summarize_file(path: Path) -> dict:
    with open(path) as f:
        data = json.load(f)

    pass_at_k = data.get("pass_at_k", {})
    base_pass1 = pass_at_k.get("base", {}).get("pass@1")
    plus_pass1 = pass_at_k.get("plus", {}).get("pass@1")

    eval_data = data.get("eval", {})
    total_tasks = len(eval_data)

    return {
        "pass@1_base": base_pass1,
        "pass@1_plus": plus_pass1,
        "total_tasks": total_tasks,
    }

def calculate_stats(values):
    """Returns (mean, std_dev) for a list of numbers."""
    valid_values = [v for v in values if v is not None]
    if not valid_values:
        return None, None
    n = len(valid_values)
    mean = sum(valid_values) / n
    variance = sum((x - mean) ** 2 for x in valid_values) / n
    std_dev = math.sqrt(variance)
    return mean, std_dev

def main():
    logs_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else LOGS_DIR / "evalplus_baseline"
    if not logs_dir.exists():
        print(f"Directory not found: {logs_dir}")
        sys.exit(1)

    result_files = find_result_files(logs_dir)
    if not result_files:
        print(f"No results found in {logs_dir}")
        sys.exit(1)

    model_groups = defaultdict(list)
    for path in result_files:
        name = model_name_from_path(path)
        scores = summarize_file(path)
        model_groups[name].append({"path": path, "scores": scores})

    print("=" * 85)
    print(f"{'MODEL EVALUATION SUMMARY (3 RUNS)':^85}")
    print("=" * 85)

    sorted_names = sorted(model_groups.keys())

    for name in sorted_names:
        group = model_groups[name]
        base_scores = [g["scores"]["pass@1_base"] for g in group]
        plus_scores = [g["scores"]["pass@1_plus"] for g in group]
        m_base, s_base = calculate_stats(base_scores)
        m_plus, s_plus = calculate_stats(plus_scores)

        print(f"\nMODEL: {name}")
        print("-" * 85)
        # Added a Header for the Runs to be crystal clear
        print(f"{'Run #':<6} | {'File Name':<40} | {'HumanEval (Base)':>12} | {'HumanEval+ (Plus)':>15}")
        print("-" * 85)

        for i, item in enumerate(group, 1):
            p_base = item["scores"]["pass@1_base"]
            p_plus = item["scores"]["pass@1_plus"]

            base_str = f"{p_base:.2%}" if p_base is not None else "N/A"
            plus_str = f"{p_plus:.2%}" if p_plus is not None else "N/A"

            # Shorten the filename so it fits the table nicely
            short_name = item["path"].name
            if len(short_name) > 38:
                short_name = short_name[:35] + "..."

            print(f"  Run {i} | {short_name:<40} | {base_str:>12} | {plus_str:>15}")

        print("-" * 85)
        base_avg = f"{m_base:.2%} ± {s_base:.2%}" if m_base is not None else "N/A"
        plus_avg = f"{m_plus:.2%} ± {s_plus:.2%}" if m_plus is not None else "N/A"
        print(f"  AVERAGE: {base_avg:>12} | {plus_avg:>15}")
        print("=" * 85)

    # Write the combined summary for the paper
    out_path = logs_dir / "combined_paper_summary.json"
    json_data = []
    for name in sorted_names:
        group = model_groups[name]
        m_base, s_base = calculate_stats([g["scores"]["pass@1_base"] for g in group])
        m_plus, s_plus = calculate_stats([g["scores"]["pass@1_plus"] for g in group])
        json_data.append({
            "model_name": name,
            "mean_base": m_base,
            "std_base": s_base,
            "mean_plus": m_plus,
            "std_plus": s_plus
        })
    with open(out_path, "w") as f:
        json.dump(json_data, f, indent=2)
    print(f"\nCombined summary for paper written to: {out_path}")

if __name__ == "__main__":
    main()
