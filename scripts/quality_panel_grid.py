"""
Extends scripts/quality_panel.py's objective quality panel across the
precision x priming grid on Qwen3-1.7B:

              bf16                                int4 (NF4)
  cold        logs/tim/.../cold                   logs/quant_gap/.../gate/C_nf4_cold
  tim_domain  logs/tim/.../tim_domain_seed0        logs/quality_quant/int4_tim_domain_seed0

This reuses every metric function from scripts/quality_panel.py unchanged
(ruff/radon/ast metrics, docstring provenance/echo test, paired Wilcoxon,
fragile rate) via direct import — nothing here reimplements them. It only
adds multi-directory cell loading (quality_panel.py's own CLI is scoped to
one logs_dir) and the cross-precision "cold: bf16 vs int4" comparison that
quality_panel.py's single-directory design doesn't support.

seed0 only for tim_domain at BOTH precisions, matching the seed used by the
existing bf16 study's Part 1 panel and Part 2 judge
(docs/generation_quality.md) — not a 3-seed aggregate. See
docs/precision_quality_grid.md for why (the bf16 study's own panel was
single-seed despite 3 seeds existing on disk; this keeps genuine
comparability with what's already published rather than silently changing
methodology).

Usage:
    python scripts/quality_panel_grid.py
"""

import json
import sys
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS_DIR))
import quality_panel as qp  # noqa: E402

ROOT = qp.ROOT
QUALITY_QUANT_DIR = ROOT / "logs" / "quality_quant"
QUALITY_QUANT_DIR.mkdir(parents=True, exist_ok=True)

CELLS = {
    "bf16_cold": (ROOT / "logs" / "tim" / "Qwen_Qwen3-1.7B", "cold"),
    "bf16_tim_domain_seed0": (ROOT / "logs" / "tim" / "Qwen_Qwen3-1.7B", "tim_domain_seed0"),
    "int4_cold": (ROOT / "logs" / "quant_gap" / "Qwen_Qwen3-1.7B" / "gate", "C_nf4_cold"),
    "int4_tim_domain_seed0": (ROOT / "logs" / "quality_quant", "int4_tim_domain_seed0"),
}

COMPARISONS = [
    ("bf16 cold vs tim_domain", "bf16_cold", "bf16_tim_domain_seed0"),
    ("int4 cold vs tim_domain", "int4_cold", "int4_tim_domain_seed0"),
    ("cold: bf16 vs int4",      "bf16_cold", "int4_cold"),
]


def main():
    problems = qp.get_human_eval_plus()

    solutions, eval_results, metrics, fragile = {}, {}, {}, {}
    print("=" * 100)
    print("QUALITY PANEL — precision x priming grid — Qwen/Qwen3-1.7B")
    print("=" * 100)
    print("\nCells:")
    for name, (logs_dir, cond) in CELLS.items():
        sol = qp.load_solutions(logs_dir, cond)
        ev = qp.load_eval_results(logs_dir, cond)
        solutions[name], eval_results[name] = sol, ev
        print(f"  {name:<24}: {logs_dir}/{cond}*  -> {len(sol)} solutions, {len(ev)} eval results")

    print("\nComputing per-solution metrics (ruff + radon + ast)...")
    for name in CELLS:
        if not solutions[name]:
            metrics[name] = {}
            continue
        metrics[name] = qp.compute_condition_metrics(solutions[name], problems)
        n_err = sum(1 for m in metrics[name].values() if m.get("parse_error"))
        print(f"  {name:<24}: done ({n_err} parse errors)")

    print("\n" + "=" * 100)
    print("EvalPlus fragile rate (base pass, plus fail) — FULL 164, per cell")
    print("=" * 100)
    for name in CELLS:
        fr = qp.fragile_rate(eval_results[name])
        fragile[name] = fr
        rate = f"{fr['fragile_rate']*100:.1f}%" if fr["fragile_rate"] is not None else "n/a"
        print(f"  {name:<24}: {fr['fragile_count']}/{fr['n']} = {rate}")

    pairs = {}
    print("\n" + "=" * 100)
    print("Pairwise comparisons (both-pass set rebuilt per pair)")
    print("=" * 100)
    for label, a_name, b_name in COMPARISONS:
        if not solutions[a_name] or not solutions[b_name]:
            print(f"\n--- {label}: SKIPPED (missing solutions) ---")
            continue
        summary = qp.summarize_pair(a_name, metrics[a_name], eval_results[a_name],
                                     b_name, metrics[b_name], eval_results[b_name])
        pairs[label] = summary
        qp.print_pair_table(label, a_name, b_name, summary)

    raw_out = {
        "model": "Qwen/Qwen3-1.7B",
        "cells": {name: {"logs_dir": str(d), "condition": c} for name, (d, c) in CELLS.items()},
        "per_task_metrics": metrics,
        "fragile_rate": fragile,
        "pairs": pairs,
    }
    raw_path = QUALITY_QUANT_DIR / "panel_grid_raw.json"
    with open(raw_path, "w") as f:
        json.dump(raw_out, f, indent=2)

    print("\n" + "=" * 100)
    print("Files / conditions expected but not found")
    print("=" * 100)
    if qp.MISSING:
        for desc, path in qp.MISSING:
            print(f"  - {desc}: {path}")
    else:
        print("  (none)")

    print(f"\nRaw grid metrics written to: {raw_path}")


if __name__ == "__main__":
    main()
