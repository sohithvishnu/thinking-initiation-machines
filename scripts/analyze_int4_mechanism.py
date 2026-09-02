"""
Analysis for the int4-mechanism ladder (docs/int4_mechanism.md), covering
Rungs 1, 2, 3a, 3b, 4. Rung 3c (attention measurement) is analyzed separately
by experiment-3/run_attention_probe.py, which writes its own numbers/plot.

Reuses wilson_ci and mcnemar_exact from scripts/analyze_quant_gap.py
unchanged — no test statistic is reimplemented here.

Cells (condition_name -> file stem):
  int4_cold                 logs/quant_gap/Qwen_Qwen3-1.7B/gate/C_nf4_cold        (existing)
  int4_tim_domain_seed{0-7} logs/quality_quant/int4_tim_domain_seed{0-7}          (0-2 existing, 3-7 new)
  int4_prompt_control       logs/int4_mechanism/int4_prompt_control               (new, Rung 2)
  int4_tim_random_seed{0-2} logs/int4_mechanism/int4_tim_random_seed{0-2}         (new, Rung 3a)
  int4_neutral_prefix       logs/int4_mechanism/int4_neutral_prefix              (new, Rung 3b)
  bf16_cold                 logs/tim/Qwen_Qwen3-1.7B/cold                         (existing)
  bf16_tim_domain_seed0     logs/tim/Qwen_Qwen3-1.7B/tim_domain_seed0             (existing)

Each stem X yields X-sanitized.jsonl (solutions) and
X-sanitized.eval_results.json (base_status/plus_status per task).

HumanEval+ pass criterion throughout: base_status == 'pass' AND
plus_status == 'pass' (the corrected criterion used everywhere else in this
repo's analysis scripts).

Writes:
  logs/int4_mechanism/answers_rung1.jsonl
  logs/int4_mechanism/answers_rung2.jsonl
  logs/int4_mechanism/answers_rung3.jsonl
  logs/int4_mechanism/answers_rung4.jsonl
  logs/int4_mechanism/analysis_summary.json

Usage:
    python scripts/analyze_int4_mechanism.py
"""

import json
import statistics
import sys
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS_DIR))
from analyze_quant_gap import wilson_ci, mcnemar_exact  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
LOGS = ROOT / "logs"
OUT_DIR = LOGS / "int4_mechanism"
OUT_DIR.mkdir(parents=True, exist_ok=True)

MISSING = []


def note_missing(desc, path):
    MISSING.append((desc, str(path)))


STEMS = {
    "int4_cold": LOGS / "quant_gap" / "Qwen_Qwen3-1.7B" / "gate" / "C_nf4_cold",
    "int4_prompt_control": OUT_DIR / "int4_prompt_control",
    "int4_neutral_prefix": OUT_DIR / "int4_neutral_prefix",
    "bf16_cold": LOGS / "tim" / "Qwen_Qwen3-1.7B" / "cold",
    "bf16_tim_domain_seed0": LOGS / "tim" / "Qwen_Qwen3-1.7B" / "tim_domain_seed0",
}
for s in range(8):
    STEMS[f"int4_tim_domain_seed{s}"] = LOGS / "quality_quant" / f"int4_tim_domain_seed{s}"
for s in range(3):
    STEMS[f"int4_tim_random_seed{s}"] = OUT_DIR / f"int4_tim_random_seed{s}"


def load_solutions(stem: Path):
    path = stem.with_name(stem.name + "-sanitized.jsonl")
    if not path.exists():
        note_missing("solutions", path)
        return None
    out = {}
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        d = json.loads(line)
        out[d["task_id"]] = d["solution"]
    return out


def load_eval(stem: Path):
    path = stem.with_name(stem.name + "-sanitized.eval_results.json")
    if not path.exists():
        note_missing("eval_results", path)
        return None
    d = json.load(open(path))
    out = {}
    for task_id, entries in d["eval"].items():
        e = entries[0]
        out[task_id] = {"base_pass": e["base_status"] == "pass", "plus_pass": e["plus_status"] == "pass"}
    return out


def is_pass(tasks, tid):
    t = tasks[tid]
    return t["base_pass"] and t["plus_pass"]


def plus_rate(tasks):
    n = len(tasks)
    k = sum(1 for t in tasks.values() if t["base_pass"] and t["plus_pass"])
    return k, n


def fmt_pct(k, n):
    if n == 0:
        return "n/a"
    p, lo, hi = wilson_ci(k, n)
    return f"{p*100:5.1f}% [{lo*100:4.1f},{hi*100:4.1f}] ({k}/{n})"


CACHE_SOL, CACHE_EVAL = {}, {}


def get(name):
    if name not in CACHE_SOL:
        CACHE_SOL[name] = load_solutions(STEMS[name])
        CACHE_EVAL[name] = load_eval(STEMS[name])
    return CACHE_SOL[name], CACHE_EVAL[name]


def two_by_two(tasks_a, tasks_b):
    """Returns (both_pass, a_only, b_only, both_fail, common_ids)."""
    common = sorted(set(tasks_a) & set(tasks_b))
    both_pass = a_only = b_only = both_fail = 0
    for tid in common:
        pa, pb = is_pass(tasks_a, tid), is_pass(tasks_b, tid)
        if pa and pb:
            both_pass += 1
        elif pa and not pb:
            a_only += 1
        elif pb and not pa:
            b_only += 1
        else:
            both_fail += 1
    return both_pass, a_only, b_only, both_fail, common


def print_header(t):
    print("\n" + "=" * 100)
    print(t)
    print("=" * 100)


def dump_answers(path, problems_by_task, per_task_rows):
    """per_task_rows: {task_id: {cond_name: {"solution":..., "pass":bool}, ...}}"""
    with open(path, "w") as f:
        for tid in sorted(per_task_rows):
            rec = {"task_id": tid, "prompt": problems_by_task.get(tid, ""), "conditions": per_task_rows[tid]}
            f.write(json.dumps(rec) + "\n")
    print(f"  answers file written: {path}  ({len(per_task_rows)} tasks)")


def build_prompts():
    from evalplus.data import get_human_eval_plus
    problems = get_human_eval_plus()
    return {tid: p["prompt"] for tid, p in problems.items()}


def main():
    prompts = build_prompts()
    summary = {}

    # ===================================================================
    # RUNG 1 — is the int4 gain directional?
    # ===================================================================
    print_header("RUNG 1 — int4-cold vs int4-tim_domain, seeds 0/1/2 (+ pooled), overlap 2x2")
    cold_sol, cold_eval = get("int4_cold")
    bf16cold_sol, bf16cold_eval = get("bf16_cold")

    rung1 = {"per_seed": {}, "pooled": {}, "overlap": {}}
    pooled_tim_only = pooled_cold_only = pooled_both_pass = pooled_both_fail = 0
    answers_r1 = {}

    if cold_eval is None:
        print("int4_cold eval missing — cannot run Rung 1.")
    else:
        for s in range(3):
            name = f"int4_tim_domain_seed{s}"
            tim_sol, tim_eval = get(name)
            if tim_eval is None:
                continue
            both_pass, c_only, t_only, both_fail, common = two_by_two(cold_eval, tim_eval)
            b, c, p = mcnemar_exact(c_only, t_only)
            churn = (c_only + t_only) / len(common) * 100 if common else 0.0
            print(f"seed{s}: both_pass={both_pass} cold_only={c_only} tim_only={t_only} "
                  f"both_fail={both_fail} churn={churn:.1f}% McNemar p={p:.4f}")
            rung1["per_seed"][f"seed{s}"] = {
                "both_pass": both_pass, "cold_only": c_only, "tim_only": t_only,
                "both_fail": both_fail, "n": len(common), "churn_pct": churn, "mcnemar_p": p,
            }
            pooled_both_pass += both_pass
            pooled_cold_only += c_only
            pooled_tim_only += t_only
            pooled_both_fail += both_fail

            # overlap: A=bf16_cold, C=int4_cold, D=this tim seed
            if bf16cold_eval is not None:
                common_acd = sorted(set(bf16cold_eval) & set(cold_eval) & set(tim_eval))
                quant_broke = [tid for tid in common_acd if is_pass(bf16cold_eval, tid) and not is_pass(cold_eval, tid)]
                tim_fixes = [tid for tid in common_acd if not is_pass(cold_eval, tid) and is_pass(tim_eval, tid)]
                fixed_of_broken = [tid for tid in quant_broke if is_pass(tim_eval, tid)]
                fixes_were_broken = [tid for tid in tim_fixes if is_pass(bf16cold_eval, tid)]
                fixes_were_hard = [tid for tid in tim_fixes if tid not in fixes_were_broken]
                print(f"  overlap seed{s}: quant_broke={len(quant_broke)} -> tim fixes "
                      f"{len(fixed_of_broken)}/{len(quant_broke)}; "
                      f"tim_fixes_total={len(tim_fixes)} (of which {len(fixes_were_broken)} were quant-broken, "
                      f"{len(fixes_were_hard)} bf16-also-failed)")
                rung1["overlap"][f"seed{s}"] = {
                    "quant_broke_n": len(quant_broke), "quant_broke_ids": quant_broke,
                    "tim_fixes_n": len(tim_fixes), "tim_fixes_ids": tim_fixes,
                    "tim_fixes_that_were_quant_broken": fixes_were_broken,
                    "tim_fixes_that_were_bf16_also_failed": fixes_were_hard,
                }
                # dump solutions for every quadrant task into the answers file
                interesting = set(quant_broke) | set(tim_fixes)
                for tid in interesting:
                    row = answers_r1.setdefault(tid, {})
                    row["int4_cold"] = {"solution": (cold_sol or {}).get(tid), "pass": is_pass(cold_eval, tid)}
                    row[name] = {"solution": (tim_sol or {}).get(tid), "pass": is_pass(tim_eval, tid)}
                    row["bf16_cold"] = {"solution": (bf16cold_sol or {}).get(tid), "pass": is_pass(bf16cold_eval, tid)}

        pooled_n = pooled_both_pass + pooled_cold_only + pooled_tim_only + pooled_both_fail
        _, _, pooled_p = mcnemar_exact(pooled_cold_only, pooled_tim_only)
        pooled_churn = (pooled_cold_only + pooled_tim_only) / pooled_n * 100 if pooled_n else 0.0
        print(f"\nPOOLED (seeds 0-2): both_pass={pooled_both_pass} cold_only={pooled_cold_only} "
              f"tim_only={pooled_tim_only} both_fail={pooled_both_fail} churn={pooled_churn:.1f}% "
              f"McNemar p={pooled_p:.4f}")
        rung1["pooled"] = {
            "both_pass": pooled_both_pass, "cold_only": pooled_cold_only, "tim_only": pooled_tim_only,
            "both_fail": pooled_both_fail, "n": pooled_n, "churn_pct": pooled_churn, "mcnemar_p": pooled_p,
        }
        verdict1 = ("directional (tim_only > cold_only)" if pooled_tim_only > pooled_cold_only
                    else "not clearly directional")
        sig1 = "significant" if pooled_p < 0.05 else "not significant"
        print(f"VERDICT (Rung 1): pooled McNemar {sig1} (p={pooled_p:.4f}); "
              f"tim_only({pooled_tim_only}) vs cold_only({pooled_cold_only}) -> {verdict1}")
        rung1["verdict"] = f"{verdict1}; pooled McNemar {sig1} (p={pooled_p:.4f})"

        dump_answers(OUT_DIR / "answers_rung1.jsonl", prompts, answers_r1)
    summary["rung1"] = rung1

    # ===================================================================
    # RUNG 2 — priming vs context (decisive ablation)
    # ===================================================================
    print_header("RUNG 2 — int4-cold vs int4-prompt_control vs int4-tim_domain (seed0)")
    pc_sol, pc_eval = get("int4_prompt_control")
    tim0_sol, tim0_eval = get("int4_tim_domain_seed0")
    rung2 = {}
    answers_r2 = {}

    if cold_eval is None or pc_eval is None:
        print("Missing int4_cold or int4_prompt_control — cannot run Rung 2.")
    else:
        k_c, n_c = plus_rate(cold_eval)
        k_p, n_p = plus_rate(pc_eval)
        print(f"int4_cold          : {fmt_pct(k_c, n_c)}")
        print(f"int4_prompt_control: {fmt_pct(k_p, n_p)}")

        bp, c_only, p_only, bf, common = two_by_two(cold_eval, pc_eval)
        _, _, p_cp = mcnemar_exact(c_only, p_only)
        print(f"McNemar int4_cold vs int4_prompt_control: cold_only={c_only} pc_only={p_only} p={p_cp:.4f}")
        rung2["cold_vs_prompt_control"] = {
            "cold_pct": k_c / n_c, "prompt_control_pct": k_p / n_p,
            "cold_only": c_only, "prompt_control_only": p_only, "mcnemar_p": p_cp,
        }

        if tim0_eval is not None:
            k_t, n_t = plus_rate(tim0_eval)
            print(f"int4_tim_domain_seed0: {fmt_pct(k_t, n_t)}")
            bp2, p_only2, t_only2, bf2, common2 = two_by_two(pc_eval, tim0_eval)
            _, _, p_pt = mcnemar_exact(p_only2, t_only2)
            print(f"McNemar int4_prompt_control vs int4_tim_domain_seed0: "
                  f"pc_only={p_only2} tim_only={t_only2} p={p_pt:.4f}")
            rung2["prompt_control_vs_tim_domain"] = {
                "prompt_control_pct": k_p / n_p, "tim_domain_pct": k_t / n_t,
                "prompt_control_only": p_only2, "tim_domain_only": t_only2, "mcnemar_p": p_pt,
            }

            gap_pc = (k_p / n_p - k_c / n_c) * 100
            gap_tim = (k_t / n_t - k_c / n_c) * 100
            if abs(gap_pc - gap_tim) < 3.0 and p_pt >= 0.05:
                verdict2 = "context, not KV injection — prompt_control matches tim_domain"
            elif gap_tim - gap_pc >= 3.0 and p_pt < 0.05:
                verdict2 = "KV injection specifically beats prompting"
            elif gap_pc < 1.0 and gap_tim < 1.0:
                verdict2 = "neither beats cold — Rung 1 gain may be seed-lucky, reconcile with Rung 1"
            else:
                verdict2 = f"mixed: prompt_control gap={gap_pc:+.1f}pp, tim_domain gap={gap_tim:+.1f}pp, McNemar p={p_pt:.4f}"
            print(f"VERDICT (Rung 2): {verdict2}")
            rung2["verdict"] = verdict2
            rung2["gap_prompt_control_pp"] = gap_pc
            rung2["gap_tim_domain_pp"] = gap_tim

            for tid in sorted(set(cold_eval) & set(pc_eval) & set(tim0_eval)):
                answers_r2[tid] = {
                    "int4_cold": {"solution": (cold_sol or {}).get(tid), "pass": is_pass(cold_eval, tid)},
                    "int4_prompt_control": {"solution": (pc_sol or {}).get(tid), "pass": is_pass(pc_eval, tid)},
                    "int4_tim_domain_seed0": {"solution": (tim0_sol or {}).get(tid), "pass": is_pass(tim0_eval, tid)},
                }
            dump_answers(OUT_DIR / "answers_rung2.jsonl", prompts, answers_r2)
    summary["rung2"] = rung2

    # ===================================================================
    # RUNG 3a/3b — content vs sink
    # ===================================================================
    print_header("RUNG 3a/3b — int4-cold vs tim_random (3 seeds) vs neutral_prefix")
    rung3 = {"tim_random": {}, "neutral_prefix": {}}
    answers_r3 = {}

    pooled_r_cold_only = pooled_r_only = pooled_r_bp = pooled_r_bf = 0
    for s in range(3):
        name = f"int4_tim_random_seed{s}"
        r_sol, r_eval = get(name)
        if r_eval is None or cold_eval is None:
            continue
        k_r, n_r = plus_rate(r_eval)
        bp, c_only, r_only, bf, common = two_by_two(cold_eval, r_eval)
        _, _, p_r = mcnemar_exact(c_only, r_only)
        print(f"tim_random seed{s}: {fmt_pct(k_r, n_r)}  McNemar vs cold: cold_only={c_only} rand_only={r_only} p={p_r:.4f}")
        rung3["tim_random"][f"seed{s}"] = {
            "pct": k_r / n_r, "cold_only": c_only, "random_only": r_only, "mcnemar_p": p_r,
        }
        pooled_r_cold_only += c_only
        pooled_r_only += r_only
        pooled_r_bp += bp
        pooled_r_bf += bf
        for tid in common:
            row = answers_r3.setdefault(tid, {})
            row["int4_cold"] = {"solution": (cold_sol or {}).get(tid), "pass": is_pass(cold_eval, tid)}
            row[name] = {"solution": (r_sol or {}).get(tid), "pass": is_pass(r_eval, tid)}

    pooled_r_n = pooled_r_bp + pooled_r_cold_only + pooled_r_only + pooled_r_bf
    if pooled_r_n:
        _, _, pooled_p_r = mcnemar_exact(pooled_r_cold_only, pooled_r_only)
        print(f"POOLED tim_random (3 seeds): cold_only={pooled_r_cold_only} random_only={pooled_r_only} "
              f"p={pooled_p_r:.4f}")
        rung3["tim_random"]["pooled"] = {
            "cold_only": pooled_r_cold_only, "random_only": pooled_r_only, "mcnemar_p": pooled_p_r,
        }

    if summary.get("rung1", {}).get("pooled") and pooled_r_n:
        p1 = summary["rung1"]["pooled"]
        domain_gap = p1["tim_only"] - p1["cold_only"]
        random_gap = pooled_r_only - pooled_r_cold_only
        print(f"domain gap (tim_only - cold_only, pooled 3 seeds) = {domain_gap}; "
              f"random gap (rand_only - cold_only, pooled 3 seeds) = {random_gap}")
        if abs(domain_gap - random_gap) <= max(2, 0.15 * max(abs(domain_gap), abs(random_gap), 1)):
            content_verdict = "random ~ domain -> mechanical/positional (sink-consistent), not content-specific"
        elif domain_gap > random_gap:
            content_verdict = "domain >> random -> content-dependent, not a pure sink"
        else:
            content_verdict = "random >= domain -> inconsistent with a simple content story; report as-is"
        print(f"VERDICT (3a): {content_verdict}")
        rung3["verdict_3a"] = content_verdict
        rung3["domain_gap_pooled"] = domain_gap
        rung3["random_gap_pooled"] = random_gap

    np_sol, np_eval = get("int4_neutral_prefix")
    if np_eval is not None and cold_eval is not None:
        k_np, n_np = plus_rate(np_eval)
        bp, c_only, np_only, bf, common = two_by_two(cold_eval, np_eval)
        _, _, p_np = mcnemar_exact(c_only, np_only)
        print(f"\nneutral_prefix: {fmt_pct(k_np, n_np)}  McNemar vs cold: cold_only={c_only} neutral_only={np_only} p={p_np:.4f}")
        rung3["neutral_prefix"] = {
            "pct": k_np / n_np, "cold_only": c_only, "neutral_only": np_only, "mcnemar_p": p_np,
        }
        neutral_gap_pp = (k_np / n_np - plus_rate(cold_eval)[0] / plus_rate(cold_eval)[1]) * 100
        if p_np < 0.05 and np_only > c_only:
            sink_verdict = ("neutral prefix ALSO rescues int4 significantly -> mechanism is largely positional "
                             "(a sink), not content-dependent")
        elif abs(neutral_gap_pp) < 2.0:
            sink_verdict = "neutral prefix does ~nothing -> content/domain matters, not a pure sink"
        else:
            sink_verdict = f"neutral prefix gap={neutral_gap_pp:+.1f}pp, McNemar p={p_np:.4f} — partial effect, report as-is"
        print(f"VERDICT (3b): {sink_verdict}")
        rung3["verdict_3b"] = sink_verdict
        rung3["neutral_gap_pp"] = neutral_gap_pp

        for tid in common:
            row = answers_r3.setdefault(tid, {})
            row["int4_cold"] = {"solution": (cold_sol or {}).get(tid), "pass": is_pass(cold_eval, tid)}
            row["int4_neutral_prefix"] = {"solution": (np_sol or {}).get(tid), "pass": is_pass(np_eval, tid)}

    dump_answers(OUT_DIR / "answers_rung3.jsonl", prompts, answers_r3)
    summary["rung3"] = rung3

    # ===================================================================
    # RUNG 4 — does int4+priming beat bf16? (8-seed distribution)
    # ===================================================================
    print_header("RUNG 4 — int4-tim_domain (8 seeds) vs bf16-cold and bf16-tim_domain")
    rung4 = {"seeds": {}}
    answers_r4 = {}

    rates = []
    for s in range(8):
        name = f"int4_tim_domain_seed{s}"
        t_sol, t_eval = get(name)
        if t_eval is None:
            continue
        k, n = plus_rate(t_eval)
        rates.append(k / n)
        rung4["seeds"][f"seed{s}"] = {"pct": k / n, "k": k, "n": n}
        print(f"seed{s}: {fmt_pct(k, n)}")

    if rates:
        mean_r, std_r = statistics.mean(rates), statistics.pstdev(rates)
        print(f"\nmean={mean_r*100:.1f}%  std={std_r*100:.1f}pp  min={min(rates)*100:.1f}%  max={max(rates)*100:.1f}%  n_seeds={len(rates)}")
        rung4["mean_pct"] = mean_r
        rung4["std_pct"] = std_r
        rung4["min_pct"] = min(rates)
        rung4["max_pct"] = max(rates)
        rung4["n_seeds"] = len(rates)

        bf16cold_k, bf16cold_n = plus_rate(bf16cold_eval) if bf16cold_eval else (None, None)
        if bf16cold_eval is not None:
            print(f"\nbf16_cold: {fmt_pct(bf16cold_k, bf16cold_n)}")
            paired_ps = []
            for s in range(8):
                name = f"int4_tim_domain_seed{s}"
                t_sol, t_eval = get(name)
                if t_eval is None:
                    continue
                bp, bc_only, t_only, bf, common = two_by_two(bf16cold_eval, t_eval)
                _, _, p = mcnemar_exact(bc_only, t_only)
                paired_ps.append((s, bc_only, t_only, p))
            print("McNemar int4_tim_domain (each seed) vs bf16_cold:")
            for s, bc_only, t_only, p in paired_ps:
                print(f"  seed{s}: bf16_cold_only={bc_only} int4_tim_only={t_only} p={p:.4f}  "
                      f"{'int4_tim beats bf16_cold (sig)' if p < 0.05 and t_only > bc_only else ''}")
            rung4["vs_bf16_cold"] = [{"seed": s, "bf16_cold_only": bc, "int4_tim_only": t, "mcnemar_p": p}
                                       for s, bc, t, p in paired_ps]

        bf16tim_sol, bf16tim_eval = get("bf16_tim_domain_seed0")
        if bf16tim_eval is not None:
            k_bt, n_bt = plus_rate(bf16tim_eval)
            print(f"\nbf16_tim_domain_seed0: {fmt_pct(k_bt, n_bt)}")
            paired_ps2 = []
            for s in range(8):
                name = f"int4_tim_domain_seed{s}"
                t_sol, t_eval = get(name)
                if t_eval is None:
                    continue
                bp, bt_only, t_only, bf, common = two_by_two(bf16tim_eval, t_eval)
                _, _, p = mcnemar_exact(bt_only, t_only)
                paired_ps2.append((s, bt_only, t_only, p))
            print("McNemar int4_tim_domain (each seed) vs bf16_tim_domain_seed0:")
            for s, bt_only, t_only, p in paired_ps2:
                print(f"  seed{s}: bf16_tim_only={bt_only} int4_tim_only={t_only} p={p:.4f}  "
                      f"{'int4_tim beats bf16_tim (sig)' if p < 0.05 and t_only > bt_only else ''}")
            rung4["vs_bf16_tim_domain"] = [{"seed": s, "bf16_tim_only": bt, "int4_tim_only": t, "mcnemar_p": p}
                                             for s, bt, t, p in paired_ps2]

            all_beat_cold_sig = bf16cold_eval is not None and all(
                p < 0.05 and t_only > bc_only for _, bc_only, t_only, p in paired_ps)
            all_beat_tim_sig = all(p < 0.05 and t_only > bt_only for _, bt_only, t_only, p in paired_ps2)
            if all_beat_cold_sig and all_beat_tim_sig:
                verdict4 = "int4+priming beats bf16 across the full seed distribution with paired significance — extraordinary, held to that standard"
            else:
                verdict4 = (f"NOT established across the distribution with paired significance "
                            f"(seed range {min(rates)*100:.1f}-{max(rates)*100:.1f}%, mean {mean_r*100:.1f}%±{std_r*100:.1f}pp) — "
                            f"honest reading is 'priming/context partially rescues int4 toward bf16-cold', not 'beyond it'")
            print(f"\nVERDICT (Rung 4): {verdict4}")
            rung4["verdict"] = verdict4

            for tid in sorted(set(bf16cold_eval or {}) & set(bf16tim_eval)):
                row = answers_r4.setdefault(tid, {})
                row["bf16_cold"] = {"solution": (bf16cold_sol or {}).get(tid), "pass": is_pass(bf16cold_eval, tid)}
                row["bf16_tim_domain_seed0"] = {"solution": (bf16tim_sol or {}).get(tid), "pass": is_pass(bf16tim_eval, tid)}
                row["int4_cold"] = {"solution": (cold_sol or {}).get(tid), "pass": is_pass(cold_eval, tid)}
                for s in range(8):
                    name = f"int4_tim_domain_seed{s}"
                    t_sol, t_eval = get(name)
                    if t_eval and tid in t_eval:
                        row[name] = {"solution": (t_sol or {}).get(tid), "pass": is_pass(t_eval, tid)}
    dump_answers(OUT_DIR / "answers_rung4.jsonl", prompts, answers_r4)
    summary["rung4"] = rung4

    # ===================================================================
    summary_path = OUT_DIR / "analysis_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary written to: {summary_path}")

    print_header("Expected but not found")
    if MISSING:
        for desc, path in MISSING:
            print(f"  - {desc}: {path}")
    else:
        print("  (none)")


if __name__ == "__main__":
    main()
