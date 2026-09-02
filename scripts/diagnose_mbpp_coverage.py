"""
Step 1 of the MBPP+ scoring-bug fix: diagnose which conditions under
logs/mbpp_scale/ are missing tasks, and at which pipeline stage they were
lost, before touching anything.

`evalplus.evaluate` raises AssertionError("Missing problems in samples") and
writes NO eval_results.json whenever the *-sanitized.jsonl fed to it does not
contain at least one row for every canonical task_id in get_mbpp_plus() (see
evaluate.py: `assert len(completion_id) == len(problems)`). This script does
not run evalplus at all — it just counts rows/task_ids at each stage
(raw jsonl -> sanitized jsonl -> eval_results.json) for every condition file
found under logs/mbpp_scale/ (including full/ and gate/ subdirectories), and
classifies where each missing task was lost:

  - generation:   task_id absent from the raw <name>.jsonl
  - sanitization: task_id present in raw, absent from <name>-sanitized.jsonl
  - scoring:      task_id present in sanitized, absent from the eval_results
                   ("eval" dict keys) -- includes the "whole file never
                   scored" case (no eval_results.json at all) as 100% lost
                   at this stage.

Nothing is fixed here. Output: printed one-line-per-condition summary, full
detail (including missing task_id lists) in
logs/mbpp_scale/coverage_report.json.

Usage:
    python scripts/diagnose_mbpp_coverage.py
"""

import json
import sys
from collections import Counter
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from evalplus.data import get_mbpp_plus, load_solutions  # noqa: E402

MBPP_DIR = ROOT_DIR / "logs" / "mbpp_scale"
OUT_PATH = MBPP_DIR / "coverage_report.json"


def canonical_task_ids() -> set:
    problems = get_mbpp_plus()
    return set(problems.keys())


def task_ids_in_file(path: Path) -> list:
    """Ordered list of task_ids appearing in a jsonl sample file (raw or
    sanitized -- both are {task_id, completion|solution} jsonl)."""
    if not path.exists():
        return []
    ids = []
    for sample in load_solutions(str(path)):
        ids.append(sample["task_id"])
    return ids


def task_ids_in_eval_results(path: Path) -> set:
    if not path.exists():
        return set()
    with open(path) as f:
        data = json.load(f)
    return set(data.get("eval", {}).keys())


def find_conditions(mbpp_dir: Path):
    """Every distinct condition stem under mbpp_dir and its subdirs, keyed by
    (subdir_label, stem) -> raw_path. A condition is identified by any
    <stem>.jsonl that is NOT itself a -sanitized file."""
    conditions = {}
    for sub_label, sub_dir in [("", mbpp_dir), ("full/", mbpp_dir / "full"), ("gate/", mbpp_dir / "gate")]:
        if not sub_dir.exists():
            continue
        for p in sorted(sub_dir.glob("*.jsonl")):
            name = p.name
            if name.endswith("-sanitized.jsonl"):
                continue
            stem = name[: -len(".jsonl")]
            conditions[f"{sub_label}{stem}"] = p
    return conditions


def classify(raw_path: Path, canonical: set):
    sanitized_path = raw_path.with_name(raw_path.stem + "-sanitized.jsonl")
    eval_path = raw_path.with_name(raw_path.stem + "-sanitized.eval_results.json")

    raw_ids_list = task_ids_in_file(raw_path)
    san_ids_list = task_ids_in_file(sanitized_path)
    raw_ids = set(raw_ids_list)
    san_ids = set(san_ids_list)
    scored_ids = task_ids_in_eval_results(eval_path)

    missing_at_generation = sorted(canonical - raw_ids)
    missing_at_sanitization = sorted(raw_ids - san_ids)
    missing_at_scoring = sorted(san_ids - scored_ids) if eval_path.exists() else sorted(san_ids)

    dup_raw = [tid for tid, c in Counter(raw_ids_list).items() if c > 1]
    dup_san = [tid for tid, c in Counter(san_ids_list).items() if c > 1]

    return {
        "raw_path": str(raw_path.relative_to(ROOT_DIR)),
        "sanitized_path": str(sanitized_path.relative_to(ROOT_DIR)) if sanitized_path.exists() else None,
        "eval_results_path": str(eval_path.relative_to(ROOT_DIR)) if eval_path.exists() else None,
        "n_canonical": len(canonical),
        "n_raw": len(raw_ids_list),
        "n_raw_unique": len(raw_ids),
        "n_sanitized": len(san_ids_list),
        "n_sanitized_unique": len(san_ids),
        "n_scored": len(scored_ids),
        "eval_results_exists": eval_path.exists(),
        "duplicate_task_ids_raw": dup_raw,
        "duplicate_task_ids_sanitized": dup_san,
        "missing_at_generation": missing_at_generation,
        "missing_at_sanitization": missing_at_sanitization,
        "missing_at_scoring": missing_at_scoring,
        "n_missing_at_generation": len(missing_at_generation),
        "n_missing_at_sanitization": len(missing_at_sanitization),
        "n_missing_at_scoring": len(missing_at_scoring),
        "complete": (
            len(missing_at_generation) == 0
            and len(missing_at_sanitization) == 0
            and len(missing_at_scoring) == 0
        ),
    }


def main():
    canonical = canonical_task_ids()
    print(f"=== MBPP+ coverage diagnosis ===")
    print(f"Canonical MBPP+ task count (this evalplus install, get_mbpp_plus()): {len(canonical)}")
    print(f"(Not assumed to be 378 or 399 -- this is what the installed evalplus actually returns.)\n")

    conditions = find_conditions(MBPP_DIR)
    report = {"n_canonical": len(canonical), "conditions": {}}

    header = f"{'condition':45s} {'raw':>5s}/{'san':>5s}/{'scored':>6s}/{'canon':>5s}  status"
    print(header)
    print("-" * len(header))

    for name, raw_path in conditions.items():
        info = classify(raw_path, canonical)
        report["conditions"][name] = info

        if info["complete"]:
            status = "OK"
        elif not info["eval_results_exists"]:
            status = "NOT SCORED (no eval_results.json — evalplus assertion likely fired)"
        else:
            status = "INCOMPLETE"

        print(f"{name:45s} {info['n_raw']:5d}/{info['n_sanitized']:5d}/{info['n_scored']:6d}/{info['n_canonical']:5d}  {status}")

        if info["n_missing_at_generation"]:
            print(f"    missing at GENERATION ({info['n_missing_at_generation']}): "
                  f"{info['missing_at_generation'][:10]}{' ...' if info['n_missing_at_generation'] > 10 else ''}")
        if info["n_missing_at_sanitization"]:
            print(f"    missing at SANITIZATION ({info['n_missing_at_sanitization']}): "
                  f"{info['missing_at_sanitization'][:10]}{' ...' if info['n_missing_at_sanitization'] > 10 else ''}")
        if info["n_missing_at_scoring"] and info["eval_results_exists"]:
            print(f"    missing at SCORING despite eval_results.json existing ({info['n_missing_at_scoring']}): "
                  f"{info['missing_at_scoring'][:10]}{' ...' if info['n_missing_at_scoring'] > 10 else ''}")
        if info["duplicate_task_ids_raw"]:
            print(f"    DUPLICATE task_ids in raw jsonl: {info['duplicate_task_ids_raw'][:10]}")

    with open(OUT_PATH, "w") as f:
        json.dump(report, f, indent=2)

    n_ok = sum(1 for c in report["conditions"].values() if c["complete"])
    n_bad = len(report["conditions"]) - n_ok
    print(f"\n{n_ok} complete, {n_bad} incomplete/unscored, out of {len(report['conditions'])} condition files.")
    print(f"Full report written to: {OUT_PATH}")


if __name__ == "__main__":
    main()
