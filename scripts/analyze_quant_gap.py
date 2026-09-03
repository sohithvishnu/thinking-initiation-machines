"""
Analysis for the quantization-recovery study (experiments/quant_gap.py).

Reads the *-sanitized.eval_results.json and *_metrics.json files written by
quant_gap.py under logs/quant_gap/<model>/{gate,full}/ and prints:

  1. The 2x2 (+ prompt_control) table of HumanEval / HumanEval+ pass@1, each
     with a Wilson 95% CI; TIM arms (B, D) as mean +/- std across seeds.
  2. damage (A-C), recovery (D-C), F-C, and whether D >= A — per seed and on
     the seed mean.
  3. McNemar's exact test for C vs D (paired, same 164 tasks), per seed —
     discordant counts and churn fraction.
  4. Overlap: of the tasks quantization broke (pass in A, fail in C), how many
     does TIM fix (pass in D)? Of the tasks TIM fixes (fail in C, pass in D),
     how many were quantization-broken vs bf16-also-failed? Reported as a
     small 2x2, per seed.
  5. Docstring reproduction rate per condition.
  6. Tokens/task and ms/task per condition, from the metrics JSON.

HumanEval+ pass@1 requires base_status == 'pass' AND plus_status == 'pass'
(not plus_status alone) — see README.md's McNemar section for why; this
script uses the same corrected criterion throughout.

Usage:
    python scripts/analyze_quant_gap.py --model Qwen/Qwen3-1.7B
"""

import argparse
import json
import statistics
from pathlib import Path

from tim.config import LOGS_DIR
from tim.stats import (
    base_pass_count,
    docstring_rate,
    load_eval_results,
    mcnemar_exact,
    plus_pass_count,
    wilson_ci,
)

QUANT_GAP_DIR = LOGS_DIR / "quant_gap"

MISSING = []  # (description, expected_path) — reported at the end


def note_missing(description: str, path: Path):
    MISSING.append((description, str(path)))


def load_metrics(output_dir: Path, condition_name: str):
    p = output_dir / f"{condition_name}_metrics.json"
    if not p.exists():
        note_missing(f"metrics for {condition_name}", p)
        return {}
    return json.load(open(p))


def load_condition(output_dir: Path, condition_name: str):
    """Returns (tasks_dict_or_None, sanitized_jsonl_path, metrics_dict)."""
    eval_path = output_dir / f"{condition_name}-sanitized.eval_results.json"
    tasks = load_eval_results(eval_path)
    if tasks is None:
        note_missing(f"eval results for {condition_name}", eval_path)
    sanitized = output_dir / f"{condition_name}-sanitized.jsonl"
    metrics = load_metrics(output_dir, condition_name)
    return tasks, sanitized, metrics


def fmt_pct(k, n):
    if n == 0:
        return "  n/a "
    p, lo, hi = wilson_ci(k, n)
    return f"{p*100:5.1f}% [{lo*100:4.1f},{hi*100:4.1f}] ({k}/{n})"


def print_header(title):
    print("\n" + "=" * 100)
    print(title)
    print("=" * 100)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-1.7B")
    args = ap.parse_args()

    safe = args.model.replace("/", "_")
    model_dir = QUANT_GAP_DIR / safe
    gate_dir = model_dir / "gate"
    full_dir = model_dir / "full"

    if not gate_dir.exists():
        print(f"No gate results found at {gate_dir} — nothing to analyze.")
        return

    # ---- load everything that exists ----
    A_tasks, A_jsonl, A_metrics = load_condition(gate_dir, "A_bf16_cold")
    C_tasks, C_jsonl, C_metrics = load_condition(gate_dir, "C_nf4_cold")

    B_tasks, B_jsonl, B_metrics = {}, {}, {}
    D_tasks, D_jsonl, D_metrics = {}, {}, {}
    E_tasks = E_jsonl = E_metrics = None
    F_tasks = F_jsonl = F_metrics = None

    if full_dir.exists():
        E_tasks, E_jsonl, E_metrics = load_condition(full_dir, "E_bf16_prompt_control")
        F_tasks, F_jsonl, F_metrics = load_condition(full_dir, "F_nf4_prompt_control")
        seed = 0
        while True:
            name_b = f"B_bf16_tim_domain_seed{seed}"
            name_d = f"D_nf4_tim_domain_seed{seed}"
            if not (full_dir / f"{name_b}-sanitized.eval_results.json").exists() and \
               not (full_dir / f"{name_d}-sanitized.eval_results.json").exists():
                break
            B_tasks[seed], B_jsonl[seed], B_metrics[seed] = load_condition(full_dir, name_b)
            D_tasks[seed], D_jsonl[seed], D_metrics[seed] = load_condition(full_dir, name_d)
            seed += 1
    else:
        note_missing("full 2x2 results (B, D, E, F)", full_dir)

    # ======================================================================
    # 1. 2x2 (+ prompt_control) pass@1 table
    # ======================================================================
    print_header(f"QUANTIZATION GAP STUDY — {args.model}")
    print_header("1. Pass@1 table (HumanEval base / HumanEval+, Wilson 95% CI)")
    hdr = f"{'condition':<28}{'n':>5}  {'HumanEval (base)':<28}{'HumanEval+ (base+extra)':<28}"
    print(hdr)
    print("-" * 100)

    def row(name, tasks):
        if tasks is None:
            print(f"{name:<28}{'':>5}  {'MISSING':<28}{'MISSING':<28}")
            return
        n = len(tasks)
        print(f"{name:<28}{n:>5}  {fmt_pct(base_pass_count(tasks), n):<28}{fmt_pct(plus_pass_count(tasks), n):<28}")

    row("A_bf16_cold", A_tasks)
    row("C_nf4_cold", C_tasks)
    row("E_bf16_prompt_control", E_tasks)
    row("F_nf4_prompt_control", F_tasks)

    def seed_rows(label, tasks_by_seed):
        if not tasks_by_seed:
            print(f"{label:<28}{'':>5}  {'MISSING':<28}{'MISSING':<28}")
            return
        base_rates, plus_rates = [], []
        for s, tasks in sorted(tasks_by_seed.items()):
            if tasks is None:
                continue
            n = len(tasks)
            row(f"  seed{s}", tasks)
            base_rates.append(base_pass_count(tasks) / n)
            plus_rates.append(plus_pass_count(tasks) / n)
        if base_rates:
            print(f"{label + ' (mean+/-std)':<28}{'':>5}  "
                  f"{statistics.mean(base_rates)*100:5.1f}%+/-{statistics.pstdev(base_rates)*100:4.1f}pp"
                  f"{'':<12}"
                  f"{statistics.mean(plus_rates)*100:5.1f}%+/-{statistics.pstdev(plus_rates)*100:4.1f}pp")

    seed_rows("B_bf16_tim_domain", B_tasks)
    seed_rows("D_nf4_tim_domain", D_tasks)

    # ======================================================================
    # 2. damage / recovery / F-C / D>=A
    # ======================================================================
    print_header("2. damage, recovery, F-C, D>=A")
    if A_tasks and C_tasks:
        n = len(A_tasks)
        a_plus = plus_pass_count(A_tasks) / n
        c_plus = plus_pass_count(C_tasks) / n
        damage_pp = (a_plus - c_plus) * 100
        print(f"damage (A-C)  = {damage_pp:+.1f} pp  [A={a_plus*100:.1f}%, C={c_plus*100:.1f}%, n={n}]")

        if F_tasks:
            f_plus = plus_pass_count(F_tasks) / len(F_tasks)
            print(f"F-C           = {(f_plus - c_plus)*100:+.1f} pp  "
                  f"[F={f_plus*100:.1f}%]  (does plain prompting recover as much as TIM?)")
        else:
            print("F-C           : MISSING (F not run)")

        if D_tasks:
            print(f"\n{'seed':<8}{'D pass@1+':<14}{'recovery (D-C)':<18}{'D >= A?':<10}")
            d_rates = []
            for s, tasks in sorted(D_tasks.items()):
                if tasks is None:
                    continue
                n_d = len(tasks)
                d_plus = plus_pass_count(tasks) / n_d
                d_rates.append(d_plus)
                recovery_pp = (d_plus - c_plus) * 100
                print(f"{s:<8}{d_plus*100:>6.1f}%      {recovery_pp:>+7.1f} pp        {'YES' if d_plus >= a_plus else 'no'}")
            if d_rates:
                d_mean = statistics.mean(d_rates)
                recovery_mean_pp = (d_mean - c_plus) * 100
                print(f"{'mean':<8}{d_mean*100:>6.1f}%      {recovery_mean_pp:>+7.1f} pp        "
                      f"{'YES' if d_mean >= a_plus else 'no'}")
        else:
            print("\nD (nf4 + tim_domain): MISSING — full 2x2 not run yet")
    else:
        print("A and/or C missing — cannot compute damage. Run Step 1 (gate) first.")

    # ======================================================================
    # 3. McNemar C vs D (per seed)
    # ======================================================================
    print_header("3. McNemar exact test — C (nf4 cold) vs D (nf4 + tim_domain), per seed")
    if C_tasks and D_tasks:
        print(f"{'seed':<8}{'both_pass':<12}{'C_only':<10}{'D_only':<10}{'both_fail':<12}{'churn %':<10}{'p-value':<10}")
        for s, d_tasks in sorted(D_tasks.items()):
            if d_tasks is None:
                continue
            common = sorted(set(C_tasks) & set(d_tasks))
            both_pass = c_only = d_only = both_fail = 0
            for tid in common:
                cp = C_tasks[tid]["base_pass"] and C_tasks[tid]["plus_pass"]
                dp = d_tasks[tid]["base_pass"] and d_tasks[tid]["plus_pass"]
                if cp and dp:
                    both_pass += 1
                elif cp and not dp:
                    c_only += 1
                elif dp and not cp:
                    d_only += 1
                else:
                    both_fail += 1
            b, c, p = mcnemar_exact(c_only, d_only)
            churn = (c_only + d_only) / len(common) * 100 if common else 0.0
            print(f"{s:<8}{both_pass:<12}{c_only:<10}{d_only:<10}{both_fail:<12}{churn:<10.1f}{p:<10.4f}")
    else:
        print("C and/or D missing — cannot run McNemar.")

    # ======================================================================
    # 4. Overlap: does TIM fix what quantization broke, or fix generally?
    # ======================================================================
    print_header("4. Overlap — does D fix what quantization broke (A pass, C fail)?")
    if A_tasks and C_tasks and D_tasks:
        for s, d_tasks in sorted(D_tasks.items()):
            if d_tasks is None:
                continue
            common = sorted(set(A_tasks) & set(C_tasks) & set(d_tasks))

            def is_pass(td, tid):
                return td[tid]["base_pass"] and td[tid]["plus_pass"]

            quant_broke = [tid for tid in common if is_pass(A_tasks, tid) and not is_pass(C_tasks, tid)]
            tim_fixes = [tid for tid in common if not is_pass(C_tasks, tid) and is_pass(d_tasks, tid)]

            fixed_of_broken = sum(1 for tid in quant_broke if is_pass(d_tasks, tid))
            fixes_that_were_broken = sum(1 for tid in tim_fixes if is_pass(A_tasks, tid))
            fixes_that_were_also_bf16_fail = len(tim_fixes) - fixes_that_were_broken

            print(f"\nseed {s} (n_common={len(common)}):")
            print(f"  quantization broke {len(quant_broke)} tasks (A pass, C fail)")
            print(f"    -> TIM (D) fixes {fixed_of_broken}/{len(quant_broke)} of them"
                  f" ({fixed_of_broken/len(quant_broke)*100:.1f}%)" if quant_broke else "    -> n/a (0 broken tasks)")
            print(f"  TIM (D) fixes {len(tim_fixes)} tasks total (C fail, D pass)")
            print(f"    -> of those, {fixes_that_were_broken} were quantization-broken (A pass, C fail)")
            print(f"    -> of those, {fixes_that_were_also_bf16_fail} were tasks bf16 also failed (A fail, C fail)")
            print(f"  2x2 (of D's {len(tim_fixes)} fixes):")
            print(f"    {'':<22}{'A pass (quant broke)':<24}{'A fail (bf16 also failed)':<26}")
            print(f"    {'D fixes (C fail->D pass)':<22}{fixes_that_were_broken:<24}{fixes_that_were_also_bf16_fail:<26}")
    else:
        print("Need A, C, and D all present — missing at least one.")

    # ======================================================================
    # 5. Docstring rate per condition
    # ======================================================================
    print_header("5. Docstring reproduction rate")
    print(f"{'condition':<28}{'rate':<10}{'count':<12}")

    def ds_row(name, path):
        r = docstring_rate(path) if path else None
        if r is None:
            print(f"{name:<28}{'MISSING':<10}")
            return
        d, n = r
        rate = d / n * 100 if n else 0.0
        print(f"{name:<28}{rate:5.1f}%    {d}/{n}")

    ds_row("A_bf16_cold", A_jsonl)
    ds_row("C_nf4_cold", C_jsonl)
    ds_row("E_bf16_prompt_control", E_jsonl)
    ds_row("F_nf4_prompt_control", F_jsonl)
    for s in sorted(B_jsonl):
        ds_row(f"B_bf16_tim_domain_seed{s}", B_jsonl[s])
    for s in sorted(D_jsonl):
        ds_row(f"D_nf4_tim_domain_seed{s}", D_jsonl[s])

    # ======================================================================
    # 6. Tokens/task and ms/task per condition
    # ======================================================================
    print_header("6. Tokens/task and ms/task (timing/cost)")
    print(f"{'condition':<28}{'tokens/task':<14}{'ms/task':<12}{'tok/s':<10}{'peak_mb':<10}")

    def timing_row(name, metrics):
        if not metrics:
            print(f"{name:<28}{'MISSING':<14}")
            return
        tok = metrics.get("new_tokens", {}).get("mean", 0.0)
        ms = metrics.get("per_task_ms_mean", 0.0)
        tps = metrics.get("tokens_per_sec", 0.0)
        mem = metrics.get("peak_memory_mb", 0.0)
        print(f"{name:<28}{tok:<14.1f}{ms:<12.1f}{tps:<10.1f}{mem:<10.1f}")

    timing_row("A_bf16_cold", A_metrics)
    timing_row("C_nf4_cold", C_metrics)
    timing_row("E_bf16_prompt_control", E_metrics)
    timing_row("F_nf4_prompt_control", F_metrics)
    for s in sorted(B_metrics):
        timing_row(f"B_bf16_tim_domain_seed{s}", B_metrics[s])
    for s in sorted(D_metrics):
        timing_row(f"D_nf4_tim_domain_seed{s}", D_metrics[s])

    # ======================================================================
    # Missing files summary
    # ======================================================================
    if MISSING:
        print_header("Expected files not found")
        for desc, path in MISSING:
            print(f"  - {desc}: {path}")


if __name__ == "__main__":
    main()
