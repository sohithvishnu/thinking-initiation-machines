"""
Part 1 objective quality panel: does KV-cache priming change code quality
along axes pass@1 cannot see?

Operates entirely on committed artifacts under logs/tim/Qwen_Qwen3-1.7B/ —
no model generation happens here. For each comparison pair (prompt_control,
tim_domain_seed0, tim_random_seed0) vs cold, the both-pass set is rebuilt
from that pair's own eval_results.json (base_status == 'pass' AND
plus_status == 'pass' in BOTH conditions) — never reused across pairs, since
each condition passes a different subset of tasks.

tim_domain and tim_random are represented by seed0 only (there are 3 seeded
runs each in logs/); using a single seed keeps the paired comparison to one
concrete run per condition rather than conflating three different primes.
See docs/generation_quality.md for the both-pass set sizes actually found
(they differ from the ~80 recalled in the prompt — see the writeup for why).

Metrics (all computed on the both-pass-restricted solutions only):
  - lint violations (raw count) and per-LOC rate, via `ruff check`
  - type-hint coverage: annotated params+returns / total annotatable slots
  - comment density: comment lines / source lines (radon raw)
  - docstring presence (ast.get_docstring on the primary function)
  - docstring provenance: echoed / modified / original, vs the task prompt's
    own docstring, after whitespace normalization
  - cyclomatic complexity (radon cc, mean per solution)
  - maintainability index (radon mi)
  - function length (lines spanned by the primary function)
  - EvalPlus fragile rate (base pass, plus fail) — computed over the FULL
    164 per condition, not the both-pass set; this is itself a
    correctness-robustness signal, not a quality-conditional-on-correctness one.

Usage:
    python scripts/quality_panel.py --model Qwen/Qwen3-1.7B
"""

import argparse
import ast
import difflib
import json
import statistics
import subprocess
import sys
import tempfile
from pathlib import Path

from evalplus.data import get_human_eval_plus

ROOT = Path(__file__).resolve().parent.parent
QUALITY_DIR = ROOT / "logs" / "quality"
QUALITY_DIR.mkdir(parents=True, exist_ok=True)

CONDITIONS = {
    "cold": "cold",
    "prompt_control": "prompt_control",
    "tim_domain_seed0": "tim_domain_seed0",
    "tim_random_seed0": "tim_random_seed0",
}

MISSING = []


def note_missing(desc, path):
    MISSING.append((desc, str(path)))


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #

def load_solutions(logs_dir: Path, condition: str) -> dict:
    path = logs_dir / f"{condition}-sanitized.jsonl"
    if not path.exists():
        note_missing(f"sanitized solutions for {condition}", path)
        return {}
    out = {}
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        d = json.loads(line)
        out[d["task_id"]] = d["solution"]
    return out


def load_eval_results(logs_dir: Path, condition: str) -> dict:
    path = logs_dir / f"{condition}-sanitized.eval_results.json"
    if not path.exists():
        note_missing(f"eval results for {condition}", path)
        return {}
    d = json.load(open(path))
    out = {}
    for task_id, entries in d["eval"].items():
        e = entries[0]
        out[task_id] = {"base_pass": e["base_status"] == "pass", "plus_pass": e["plus_status"] == "pass"}
    return out


def both_pass_set(eval_a: dict, eval_b: dict) -> set:
    common = set(eval_a) & set(eval_b)
    return {
        t for t in common
        if eval_a[t]["base_pass"] and eval_a[t]["plus_pass"]
        and eval_b[t]["base_pass"] and eval_b[t]["plus_pass"]
    }


# --------------------------------------------------------------------------- #
# Docstring extraction / provenance
# --------------------------------------------------------------------------- #

def norm_ws(s: str) -> str:
    return " ".join(s.split())


def get_prompt_docstring(prompt: str):
    try:
        tree = ast.parse(prompt)
    except SyntaxError:
        return None
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            doc = ast.get_docstring(node)
            if doc:
                return doc
    return None


def classify_provenance(solution_doc: str, prompt_doc):
    if prompt_doc is None:
        return "original"
    sd, pd = norm_ws(solution_doc), norm_ws(prompt_doc)
    if sd == pd:
        return "echoed"
    ratio = difflib.SequenceMatcher(None, sd, pd).ratio()
    if ratio >= 0.6:
        return "modified"
    return "original"


# --------------------------------------------------------------------------- #
# Per-solution AST metrics
# --------------------------------------------------------------------------- #

def find_primary_function(tree: ast.AST, entry_point: str):
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == entry_point:
            return node
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            return node
    return None


def ast_metrics(code: str, entry_point: str, prompt_doc):
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return {"parse_error": True}

    all_funcs = [n for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]

    total_slots = annotated_slots = 0
    for fn in all_funcs:
        params = list(fn.args.posonlyargs) + list(fn.args.args) + list(fn.args.kwonlyargs)
        for p in params:
            total_slots += 1
            if p.annotation is not None:
                annotated_slots += 1
        total_slots += 1  # return slot
        if fn.returns is not None:
            annotated_slots += 1

    type_hint_coverage = (annotated_slots / total_slots) if total_slots else None

    primary = find_primary_function(tree, entry_point)
    has_docstring = False
    provenance = None
    function_length = None
    if primary is not None:
        doc = ast.get_docstring(primary)
        has_docstring = doc is not None
        if has_docstring:
            provenance = classify_provenance(doc, prompt_doc)
        if hasattr(primary, "end_lineno") and primary.end_lineno is not None:
            function_length = primary.end_lineno - primary.lineno + 1

    return {
        "parse_error": False,
        "type_hint_coverage": type_hint_coverage,
        "has_docstring": has_docstring,
        "docstring_provenance": provenance,
        "function_length": function_length,
    }


def radon_metrics(code: str):
    from radon.raw import analyze
    from radon.complexity import cc_visit
    from radon.metrics import mi_visit

    out = {"sloc": None, "comments": None, "comment_density": None,
           "cyclomatic_complexity": None, "maintainability_index": None}
    try:
        raw = analyze(code)
        out["sloc"] = raw.sloc
        out["comments"] = raw.comments
        out["comment_density"] = (raw.comments / raw.sloc) if raw.sloc else None
    except Exception:
        pass
    try:
        blocks = cc_visit(code)
        if blocks:
            out["cyclomatic_complexity"] = statistics.mean(b.complexity for b in blocks)
    except Exception:
        pass
    try:
        out["maintainability_index"] = mi_visit(code, True)
    except Exception:
        pass
    return out


# --------------------------------------------------------------------------- #
# Ruff (batched subprocess over a temp directory, one call per condition)
# --------------------------------------------------------------------------- #

def ruff_violation_counts(task_solutions: dict) -> dict:
    """task_solutions: {task_id: code}. Returns {task_id: violation_count}."""
    safe_name = {tid: tid.replace("/", "_") for tid in task_solutions}
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        for tid, code in task_solutions.items():
            (tmp_path / f"{safe_name[tid]}.py").write_text(code)

        result = subprocess.run(
            ["ruff", "check", "--output-format=json", "--exit-zero", str(tmp_path)],
            capture_output=True, text=True,
        )
        if result.returncode != 0 and not result.stdout.strip():
            print(f"WARNING: ruff check failed.\n{result.stderr}")
            return {tid: None for tid in task_solutions}

        counts = {tid: 0 for tid in task_solutions}
        try:
            diagnostics = json.loads(result.stdout)
        except json.JSONDecodeError:
            print(f"WARNING: could not parse ruff JSON output.\n{result.stdout[:500]}")
            return {tid: None for tid in task_solutions}

        rev = {f"{name}.py": tid for tid, name in safe_name.items()}
        for diag in diagnostics:
            fname = Path(diag["filename"]).name
            tid = rev.get(fname)
            if tid is not None:
                counts[tid] += 1
        return counts


# --------------------------------------------------------------------------- #
# Per-condition metric computation
# --------------------------------------------------------------------------- #

def compute_condition_metrics(solutions: dict, problems: dict) -> dict:
    """Returns {task_id: {...all metrics...}}"""
    ruff_counts = ruff_violation_counts(solutions)
    out = {}
    for tid, code in solutions.items():
        problem = problems.get(tid, {})
        entry_point = problem.get("entry_point", "")
        prompt_doc = get_prompt_docstring(problem.get("prompt", ""))

        m = ast_metrics(code, entry_point, prompt_doc)
        if m.get("parse_error"):
            out[tid] = {"parse_error": True}
            continue

        r = radon_metrics(code)
        violations = ruff_counts.get(tid)
        lint_per_loc = (violations / r["sloc"]) if (violations is not None and r["sloc"]) else None

        out[tid] = {
            "parse_error": False,
            "lint_violations": violations,
            "lint_per_loc": lint_per_loc,
            "sloc": r["sloc"],
            "type_hint_coverage": m["type_hint_coverage"],
            "comment_density": r["comment_density"],
            "has_docstring": m["has_docstring"],
            "docstring_provenance": m["docstring_provenance"],
            "cyclomatic_complexity": r["cyclomatic_complexity"],
            "maintainability_index": r["maintainability_index"],
            "function_length": m["function_length"],
        }
    return out


def fragile_rate(eval_results: dict) -> dict:
    n = len(eval_results)
    if n == 0:
        return {"n": 0, "fragile_count": 0, "fragile_rate": None}
    fragile = sum(1 for v in eval_results.values() if v["base_pass"] and not v["plus_pass"])
    return {"n": n, "fragile_count": fragile, "fragile_rate": fragile / n}


# --------------------------------------------------------------------------- #
# Paired comparison + Wilcoxon
# --------------------------------------------------------------------------- #

NUMERIC_METRICS = [
    "lint_violations", "lint_per_loc", "type_hint_coverage", "comment_density",
    "has_docstring", "cyclomatic_complexity", "maintainability_index", "function_length",
]


def paired_arrays(metrics_a: dict, metrics_b: dict, task_ids: list, metric: str):
    xs, ys = [], []
    for tid in task_ids:
        a, b = metrics_a.get(tid, {}), metrics_b.get(tid, {})
        va, vb = a.get(metric), b.get(metric)
        if va is None or vb is None:
            continue
        xs.append(float(va))
        ys.append(float(vb))
    return xs, ys


def wilcoxon_p(xs, ys):
    if len(xs) < 2:
        return None
    import math
    from scipy.stats import wilcoxon
    try:
        stat, p = wilcoxon(xs, ys)
        p = float(p)
        if math.isnan(p):
            return None  # e.g. all differences are zero (no variance to test)
        return p
    except ValueError:
        return None  # e.g. all differences are zero


def mean_std(vals):
    vals = [v for v in vals if v is not None]
    if not vals:
        return None, None
    if len(vals) == 1:
        return vals[0], 0.0
    return statistics.mean(vals), statistics.pstdev(vals)


def summarize_pair(cold_name, cold_metrics, cold_eval, x_name, x_metrics, x_eval):
    bp = sorted(both_pass_set(cold_eval, x_eval))
    n = len(bp)

    summary = {"n_both_pass": n, "both_pass_task_ids": bp, "metrics": {}}
    for metric in NUMERIC_METRICS:
        cold_vals, x_vals = paired_arrays(cold_metrics, x_metrics, bp, metric)
        cold_mean, cold_std = mean_std(cold_vals)
        x_mean, x_std = mean_std(x_vals)
        p = wilcoxon_p(cold_vals, x_vals)
        summary["metrics"][metric] = {
            "n_paired": len(cold_vals),
            "cold_mean": cold_mean, "cold_std": cold_std,
            f"{x_name}_mean": x_mean, f"{x_name}_std": x_std,
            "wilcoxon_p": p,
            "significant": bool(p is not None and p < 0.05),
        }

    provenance = {"echoed": 0, "modified": 0, "original": 0, "no_docstring": 0}
    for tid in bp:
        prov = x_metrics.get(tid, {}).get("docstring_provenance")
        if prov is None:
            provenance["no_docstring"] += 1
        else:
            provenance[prov] += 1
    summary["docstring_provenance"] = provenance

    cold_provenance = {"echoed": 0, "modified": 0, "original": 0, "no_docstring": 0}
    for tid in bp:
        prov = cold_metrics.get(tid, {}).get("docstring_provenance")
        if prov is None:
            cold_provenance["no_docstring"] += 1
        else:
            cold_provenance[prov] += 1
    summary["cold_docstring_provenance"] = cold_provenance

    return summary


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #

METRIC_LABELS = {
    "lint_violations": "lint violations (raw)",
    "lint_per_loc": "lint violations/LOC",
    "type_hint_coverage": "type-hint coverage",
    "comment_density": "comment density",
    "has_docstring": "docstring presence",
    "cyclomatic_complexity": "cyclomatic complexity",
    "maintainability_index": "maintainability index",
    "function_length": "function length (lines)",
}


def print_pair_table(pair_name, cold_name, x_name, summary):
    print(f"\n--- {x_name} vs {cold_name} (both-pass n={summary['n_both_pass']}) ---")
    print(f"{'metric':<26}{cold_name + ' (mean+/-std)':<26}{x_name + ' (mean+/-std)':<26}{'p (Wilcoxon)':<14}sig")
    print("-" * 100)
    for metric in NUMERIC_METRICS:
        row = summary["metrics"][metric]
        label = METRIC_LABELS[metric]
        cm, cs = row["cold_mean"], row["cold_std"]
        xm, xs = row[f"{x_name}_mean"], row[f"{x_name}_std"]
        p = row["wilcoxon_p"]
        cold_str = f"{cm:.3f}+/-{cs:.3f}" if cm is not None else "n/a"
        x_str = f"{xm:.3f}+/-{xs:.3f}" if xm is not None else "n/a"
        p_str = f"{p:.4f}" if p is not None else "n/a"
        sig = "*" if row["significant"] else ""
        print(f"{label:<26}{cold_str:<26}{x_str:<26}{p_str:<14}{sig}")

    prov = summary["docstring_provenance"]
    total_doc = prov["echoed"] + prov["modified"] + prov["original"]
    print(f"\n{x_name} docstring provenance (both-pass set, n={summary['n_both_pass']}, "
          f"{total_doc} have a docstring):")
    if total_doc:
        for k in ("echoed", "modified", "original"):
            print(f"  {k:<10}: {prov[k]:>3}  ({prov[k]/total_doc*100:.1f}% of docstring-bearing solutions)")
    print(f"  no_docstring: {prov['no_docstring']}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-1.7B")
    args = ap.parse_args()

    safe = args.model.replace("/", "_")
    logs_dir = ROOT / "logs" / "tim" / safe

    print("=" * 100)
    print(f"QUALITY PANEL (Part 1) — {args.model}")
    print("=" * 100)

    if not logs_dir.exists():
        print(f"No logs found at {logs_dir}")
        return

    problems = get_human_eval_plus()

    solutions = {name: load_solutions(logs_dir, cond) for name, cond in CONDITIONS.items()}
    eval_results = {name: load_eval_results(logs_dir, cond) for name, cond in CONDITIONS.items()}

    print("\nCondition files found:")
    for name in CONDITIONS:
        print(f"  {name:<20}: {len(solutions[name])} solutions, {len(eval_results[name])} eval results")

    print("\nComputing per-solution metrics (ruff + radon + ast)...")
    metrics = {}
    for name in CONDITIONS:
        if not solutions[name]:
            metrics[name] = {}
            continue
        metrics[name] = compute_condition_metrics(solutions[name], problems)
        n_parse_err = sum(1 for m in metrics[name].values() if m.get("parse_error"))
        print(f"  {name:<20}: done ({n_parse_err} parse errors)")

    print_header = lambda t: print("\n" + "=" * 100 + f"\n{t}\n" + "=" * 100)

    print_header("EvalPlus fragile rate (base pass, plus fail) — FULL 164, per condition")
    fragile = {}
    for name in CONDITIONS:
        fr = fragile_rate(eval_results[name])
        fragile[name] = fr
        rate_str = f"{fr['fragile_rate']*100:.1f}%" if fr["fragile_rate"] is not None else "n/a"
        print(f"  {name:<20}: {fr['fragile_count']}/{fr['n']} = {rate_str}")

    print_header("Pairwise comparisons vs cold (both-pass set rebuilt per pair)")
    pairs = {}
    for x_name in ("prompt_control", "tim_domain_seed0", "tim_random_seed0"):
        if not solutions["cold"] or not solutions[x_name]:
            print(f"\n--- {x_name} vs cold: SKIPPED (missing solutions) ---")
            continue
        summary = summarize_pair(
            "cold", metrics["cold"], eval_results["cold"],
            x_name, metrics[x_name], eval_results[x_name],
        )
        pairs[x_name] = summary
        print_pair_table(x_name, "cold", x_name, summary)

    raw_out = {
        "model": args.model,
        "conditions_used": CONDITIONS,
        "per_task_metrics": metrics,
        "fragile_rate": fragile,
        "pairs": pairs,
    }
    raw_path = QUALITY_DIR / "panel_raw.json"
    with open(raw_path, "w") as f:
        json.dump(raw_out, f, indent=2)

    print_header("Files / conditions expected but not found")
    if MISSING:
        for desc, path in MISSING:
            print(f"  - {desc}: {path}")
    else:
        print("  (none)")

    print(f"\nRaw per-task metrics written to: {raw_path}")


if __name__ == "__main__":
    main()
