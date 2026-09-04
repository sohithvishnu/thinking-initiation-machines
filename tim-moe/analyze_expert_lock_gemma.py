"""
Analysis for the MoE expert-locking intervention, Gemma-4-26B-A4B replication
(tim-moe/moe_expert_lock_gemma.py).

Copied from analyze_expert_lock.py (the OLMoE analysis) with only the input/
output paths changed (moe_lock -> moe_lock_gemma) — the scoring/contrast
methodology is unmodified, so the two runs are directly comparable. See
tim-moe/docs/moe_gemma_comparison.md for the side-by-side writeup.

N=20 is a probe, not a powered test — every statistic below is reported as
exploratory, with raw counts alongside any test statistic. This script does
not decide anything; it reports whichever way the numbers actually came out.

Inputs: tim-moe/logs/moe_lock_gemma/eval_summary.json, answers_lock.jsonl
Output: prints the pass@1 table, paired contrasts, and behavioral stats;
        writes tim-moe/logs/moe_lock_gemma/analysis_summary.json
"""

import json
import re
from pathlib import Path

from scipy.stats import binomtest

TIM_MOE_DIR = Path(__file__).resolve().parent
LOCK_LOGS = TIM_MOE_DIR / "logs" / "moe_lock_gemma"

DOCSTRING_RE = re.compile(r'def\s+\w+\([^)]*\)[^:]*:\s*\n\s*("""|\'\'\')')
CONDITIONS = ["cold", "tim_unconstrained", "tim_locked", "cold_locked_random"]


def load():
    summary = json.loads((LOCK_LOGS / "eval_summary.json").read_text())
    answers = [json.loads(l) for l in (LOCK_LOGS / "answers_lock.jsonl").read_text().splitlines()]
    return summary, answers


def pass_table(summary):
    print("=== pass@1 (base, base+extra), raw counts, N=20 ===")
    print(f"{'condition':22s} {'base':>12s} {'base+extra':>12s}")
    rows = {}
    for cond in CONDITIONS:
        s = summary["scores"][cond]
        n = s["n"]
        n_base = sum(1 for v in summary["per_task"][cond].values() if v["base_pass"])
        n_plus = sum(1 for v in summary["per_task"][cond].values() if v["plus_pass"])
        rows[cond] = {"n": n, "n_base": n_base, "n_plus": n_plus,
                      "pass@1_base": n_base / n, "pass@1_base_plus_extra": n_plus / n}
        print(f"{cond:22s} {n_base:>3d}/{n:<3d} ({n_base/n:.3f})   {n_plus:>3d}/{n:<3d} ({n_plus/n:.3f})")
    return rows


def paired_contrast(summary, cond_a, cond_b):
    """Discordant pass/fail counts (base+extra) between two conditions on the
    same 20 tasks, plus an exact two-sided binomial test on the discordant
    pairs (McNemar's exact test) — explicitly labeled exploratory at N=20."""
    task_ids = summary["task_ids"]
    a = summary["per_task"][cond_a]
    b = summary["per_task"][cond_b]
    win = lose = tie_pass = tie_fail = 0
    per_task_detail = {}
    for tid in task_ids:
        pa, pb = a[tid]["plus_pass"], b[tid]["plus_pass"]
        if pa and not pb:
            win += 1
            outcome = f"{cond_a}_wins"
        elif pb and not pa:
            lose += 1
            outcome = f"{cond_b}_wins"
        elif pa and pb:
            tie_pass += 1
            outcome = "tie_pass"
        else:
            tie_fail += 1
            outcome = "tie_fail"
        per_task_detail[tid] = outcome
    n_discordant = win + lose
    if n_discordant > 0:
        result = binomtest(win, n_discordant, 0.5, alternative="two-sided")
        p_value = result.pvalue
    else:
        p_value = None
    return {
        "a": cond_a, "b": cond_b,
        f"{cond_a}_wins": win, f"{cond_b}_wins": lose,
        "tie_pass": tie_pass, "tie_fail": tie_fail,
        "n_discordant": n_discordant, "exact_binomial_p": p_value,
        "per_task": per_task_detail,
        "note": "exploratory — N=20 is a probe, not a powered test",
    }


def behavioral_stats(answers):
    stats = {c: {"n_tokens": [], "n_docstring": 0, "n_total": 0} for c in CONDITIONS}
    for row in answers:
        for cond in CONDITIONS:
            d = row[cond]
            stats[cond]["n_tokens"].append(d["n_generated_tokens"])
            stats[cond]["n_total"] += 1
            if DOCSTRING_RE.search(d["sanitized_solution"] or ""):
                stats[cond]["n_docstring"] += 1
    out = {}
    for cond, s in stats.items():
        toks = s["n_tokens"]
        out[cond] = {
            "mean_tokens": sum(toks) / len(toks),
            "min_tokens": min(toks),
            "max_tokens": max(toks),
            "n_hit_max_new_tokens_768": sum(1 for t in toks if t >= 768),
            "docstring_rate": s["n_docstring"] / s["n_total"],
            "n_docstring": s["n_docstring"],
            "n_total": s["n_total"],
        }
    return out


def main():
    summary, answers = load()

    pass_rows = pass_table(summary)

    print("\n=== paired contrasts (base+extra), N=20, exploratory ===")
    contrasts = {}
    for cond_b in ["cold", "tim_unconstrained", "cold_locked_random"]:
        c = paired_contrast(summary, "tim_locked", cond_b)
        contrasts[f"tim_locked_vs_{cond_b}"] = c
        print(f"\ntim_locked vs {cond_b}:")
        print(f"  tim_locked wins: {c['tim_locked_wins']}   {cond_b} wins: {c[cond_b + '_wins']}   "
              f"tie(pass): {c['tie_pass']}   tie(fail): {c['tie_fail']}")
        print(f"  discordant pairs: {c['n_discordant']}   exact binomial p (two-sided): "
              f"{c['exact_binomial_p']}")

    print("\n=== behavioral: tokens/task, docstring/echo rate ===")
    behavior = behavioral_stats(answers)
    print(f"{'condition':22s} {'mean_tok':>9s} {'min':>5s} {'max':>5s} {'hit_768_cap':>12s} {'docstring_rate':>15s}")
    for cond, b in behavior.items():
        print(f"{cond:22s} {b['mean_tokens']:9.1f} {b['min_tokens']:5d} {b['max_tokens']:5d} "
              f"{b['n_hit_max_new_tokens_768']:>10d}/20 {b['docstring_rate']:14.2f} "
              f"({b['n_docstring']}/{b['n_total']})")

    # ---- key contrasts / verdict inputs ------------------------------------
    locked_vs_unconstrained = contrasts["tim_locked_vs_tim_unconstrained"]
    locked_vs_random = contrasts["tim_locked_vs_cold_locked_random"]
    locked_degrades_vs_unconstrained = pass_rows["tim_locked"]["pass@1_base_plus_extra"] < pass_rows["tim_unconstrained"]["pass@1_base_plus_extra"]
    locked_equals_random = abs(pass_rows["tim_locked"]["pass@1_base_plus_extra"] - pass_rows["cold_locked_random"]["pass@1_base_plus_extra"]) < 1e-9
    locked_beats_random = pass_rows["tim_locked"]["pass@1_base_plus_extra"] > pass_rows["cold_locked_random"]["pass@1_base_plus_extra"]

    verdict = {
        "degrades_relative_to_unconstrained": locked_degrades_vs_unconstrained,
        "locked_equals_random_lock": locked_equals_random,
        "locked_beats_random_lock": locked_beats_random,
        "interpretation": (
            "Degrades, and indistinguishable from locking to a random expert set: "
            "the damage comes from constraining the router to ANY fixed expert set, "
            "not from priming's particular (off-task, per Q2) choice."
            if locked_degrades_vs_unconstrained and locked_equals_random else
            "Degrades relative to unconstrained, but priming's expert choice does "
            "better than a random fixed set — priming's choice carries some "
            "task-relevant signal even though it's suboptimal."
            if locked_degrades_vs_unconstrained and locked_beats_random else
            "Preserves accuracy relative to unconstrained TIM — priming pre-loads "
            "a viable computational pathway despite Q2's low natural overlap."
        ),
    }
    print(f"\n=== VERDICT ===\n{verdict['interpretation']}")

    out = {
        "pass_table": pass_rows,
        "contrasts": contrasts,
        "behavioral": behavior,
        "verdict": verdict,
    }
    out_path = LOCK_LOGS / "analysis_summary.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\n[write] {out_path}")


if __name__ == "__main__":
    main()
