"""
The generate -> sanitize -> score -> record pipeline shared by every condition
in every experiment.

Two deliberate hard-fail guardrails live here:

  1. `evaluate_with_evalplus` raises rather than warns when EvalPlus exits
     non-zero or its stdout cannot be parsed. A warned-and-continued scoring
     failure silently produces a condition with no pass@1, which then either
     vanishes from a comparison or gets folded in as a zero.
  2. It re-checks `n_scored == n_canonical` against the on-disk
     eval_results.json. EvalPlus asserts task-id *count* at read time, but
     that assertion can pass while individual eval entries are short of the
     full set (a scoring subprocess that dies mid-flight still leaves a
     truncated file behind).

`run_condition` takes `dataset` as a required keyword for the same reason: a
defaulted dataset is what silently scored MBPP+ completions against HumanEval
and cost this study 7 unscored conditions (see docs/mbpp_scale.md).
"""

import json
import re
import statistics
import subprocess
import sys
import time
from pathlib import Path

from evalplus.data import get_human_eval_plus, get_mbpp_plus, write_jsonl
from tqdm import tqdm

from tim.generation import (
    cuda_sync,
    peak_memory_mb,
    reset_peak_memory,
    timed_generate,
)

# EvalPlus prints results as two-line blocks:
#   humaneval (base tests)
#   pass@1: 0.848
# Matched independent of dataset name so this works for humaneval and mbpp
# alike. This format is not officially versioned by EvalPlus; if it changes,
# the parse below raises rather than reporting a missing score silently.
BASE_RE = re.compile(r"\(base tests\)\s*\npass@1:\s*([\d.]+)")
PLUS_RE = re.compile(r"\(base \+ extra tests\)\s*\npass@1:\s*([\d.]+)")

DATASETS = ("humaneval", "mbpp")


def load_problems(dataset: str, limit: int | None = None) -> list:
    """Canonical task list for a dataset, as (task_id, problem) pairs."""
    if dataset not in DATASETS:
        raise ValueError(f"unknown dataset {dataset!r} (expected one of {DATASETS})")
    problems = get_mbpp_plus() if dataset == "mbpp" else get_human_eval_plus()
    items = list(problems.items())
    return items[:limit] if limit else items


def sanitize_samples(sample_file: Path) -> Path:
    """Run EvalPlus's tree-sitter sanitizer; fall back to raw samples if it
    fails, which is visible in the printed output and in the returned path."""
    result = subprocess.run(
        [sys.executable, "-m", "evalplus.sanitize", "--samples", str(sample_file)],
        capture_output=True, text=True,
    )
    print(result.stdout)
    if result.returncode != 0:
        print(f"WARNING: sanitize failed, evaluating raw samples.\n{result.stderr}")
    sanitized_path = sample_file.with_name(sample_file.stem + "-sanitized.jsonl")
    return sanitized_path if sanitized_path.exists() else sample_file


def evaluate_with_evalplus(sample_file: Path, dataset: str) -> dict:
    """Score one sample file, or raise. See this module's docstring."""
    if dataset not in DATASETS:
        raise ValueError(f"unknown dataset {dataset!r} (expected one of {DATASETS})")

    result = subprocess.run(
        [sys.executable, "-m", "evalplus.evaluate",
         "--dataset", dataset, "--samples", str(sample_file)],
        capture_output=True, text=True,
    )
    print(result.stdout)
    if result.returncode != 0:
        raise RuntimeError(
            f"evalplus.evaluate exited non-zero for {sample_file} (dataset={dataset}). "
            f"Refusing to report a partial/silent pass@1. stderr:\n{result.stderr}"
        )

    scores = {}
    base_match = BASE_RE.search(result.stdout)
    plus_match = PLUS_RE.search(result.stdout)
    if base_match:
        scores["pass@1_base"] = float(base_match.group(1))
    if plus_match:
        scores["pass@1_base_plus_extra"] = float(plus_match.group(1))
    if not scores:
        raise RuntimeError(
            f"Could not parse pass@1 from evalplus stdout for {sample_file} — "
            f"refusing to report a missing score silently.\nstdout was:\n{result.stdout}"
        )

    eval_results_path = sample_file.with_name(sample_file.stem + ".eval_results.json")
    if not eval_results_path.exists():
        legacy = sample_file.with_name(
            sample_file.stem.replace("-sanitized", "") + "_eval_results.json")
        eval_results_path = legacy if legacy.exists() else eval_results_path

    n_canonical = len(load_problems(dataset))
    n_scored = 0
    if eval_results_path.exists():
        with open(eval_results_path) as f:
            n_scored = len(json.load(f).get("eval", {}))
    if n_scored != n_canonical:
        raise RuntimeError(
            f"Refusing to report pass@1 for {sample_file}: n_scored={n_scored} != "
            f"n_canonical={n_canonical} (dataset={dataset}). eval_results.json="
            f"{eval_results_path} (exists={eval_results_path.exists()})."
        )

    scores["n_scored"] = n_scored
    scores["n_canonical"] = n_canonical
    return scores


def run_condition(model, tokenizer, device, condition_name: str, output_dir: Path,
                  task_items, *, dataset: str, primed_kv=None, prepend_text: str = "",
                  max_new_tokens: int = 768, prime_timing: dict | None = None,
                  score: bool = True) -> dict:
    """
    Run one condition over `task_items`, writing `<name>.jsonl` (completions)
    and `<name>_metrics.json` (timing, memory, and pass@1) into `output_dir`.

    A warmup task runs before the timer starts: the first CUDA call pays
    one-off autotune/allocation costs that would otherwise land on task 1.

    Scoring is skipped automatically on a partial task set. EvalPlus asserts
    that the samples cover every canonical task, so a `--limit`-ed run can
    never be scored; catching that here turns a crash after the full
    generation loop into one line before it.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    if score:
        n_canonical = len(load_problems(dataset))
        if len(task_items) != n_canonical:
            print(f"NOTE: scoring disabled for {condition_name} — {len(task_items)} of "
                  f"{n_canonical} {dataset} tasks selected, and EvalPlus scores only a "
                  f"complete task set. Timing and completions are still written.")
            score = False

    reset_peak_memory()
    samples, per_task = [], []

    _, warm_problem = task_items[0]
    timed_generate(model, tokenizer, device, warm_problem["prompt"],
                   max_new_tokens=16, primed_kv=primed_kv, prepend_text=prepend_text)

    cuda_sync()
    wall_start = time.perf_counter()
    for task_id, problem in tqdm(task_items, desc=condition_name, unit="task"):
        r = timed_generate(model, tokenizer, device, problem["prompt"],
                           max_new_tokens=max_new_tokens,
                           primed_kv=primed_kv, prepend_text=prepend_text)
        samples.append({"task_id": task_id, "completion": r.pop("completion")})
        per_task.append(r)
    cuda_sync()
    wall_ms = (time.perf_counter() - wall_start) * 1000.0

    def agg(key):
        vals = [d[key] for d in per_task]
        return {
            "mean": statistics.mean(vals), "median": statistics.median(vals),
            "stdev": statistics.pstdev(vals) if len(vals) > 1 else 0.0,
            "min": min(vals), "max": max(vals),
        }

    total_new_tokens = sum(d["new_tokens"] for d in per_task)
    total_gen_ms = sum(d["generate_ms"] for d in per_task)
    total_copy_ms = sum(d["deepcopy_ms"] for d in per_task)
    prime_ms = (prime_timing or {}).get("total_ms", 0.0)
    n = max(len(per_task), 1)

    metrics = {
        "condition": condition_name,
        "dataset": dataset,
        "n_tasks": len(per_task),
        "prime_ms": prime_ms,
        "prime_per_pass_ms": (prime_timing or {}).get("per_pass_ms", []),
        "prime_cache_len_per_pass": (prime_timing or {}).get("per_pass_cache_len", []),
        "primed_cache_len": (prime_timing or {}).get("final_cache_len", 0),
        "generate_ms": agg("generate_ms"),
        "deepcopy_ms": agg("deepcopy_ms"),
        "new_tokens": agg("new_tokens"),
        "prompt_tokens": agg("prompt_tokens"),
        "total_new_tokens": total_new_tokens,
        "tokens_per_sec": (total_new_tokens / (total_gen_ms / 1000.0)) if total_gen_ms else 0.0,
        # Steady-state per-query cost, excluding one-time priming.
        "per_task_ms_mean": (total_gen_ms + total_copy_ms) / n,
        # Priming amortized over this task count — compare only at equal n.
        "amortized_total_ms": prime_ms + total_gen_ms + total_copy_ms,
        "amortized_per_task_ms": (prime_ms + total_gen_ms + total_copy_ms) / n,
        "wall_ms": wall_ms,
        "peak_memory_mb": peak_memory_mb(),
    }

    sample_file = output_dir / f"{condition_name}.jsonl"
    write_jsonl(str(sample_file), samples)

    if score:
        metrics.update(evaluate_with_evalplus(sanitize_samples(sample_file), dataset=dataset))

    with open(output_dir / f"{condition_name}_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"{condition_name}: pass@1_base={metrics.get('pass@1_base')} "
          f"pass@1_plus={metrics.get('pass@1_base_plus_extra')} "
          f"per_task_ms={metrics['per_task_ms_mean']:.1f} "
          f"peak_mb={metrics['peak_memory_mb']:.1f}")
    return metrics
