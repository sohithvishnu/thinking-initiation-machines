"""
Analysis for the MBPP+ scale run (N=378, this evalplus install's canonical
MBPP+ count -- see scripts/diagnose_mbpp_coverage.py, which found it is NOT
399 as originally guessed).

Computes McNemar significance across the precision x priming grid, the
multi-pass dose-response ladder (int4 and bf16), and the docstring/echo-
provenance test -- reusing wilson_ci / mcnemar_exact / load_eval_results
from scripts/analyze_quant_gap.py unchanged, exactly as the int4-mechanism
analysis (scripts/analyze_int4_mechanism.py) does for HumanEval+.

Guardrail: every condition is gated on n_scored == n_canonical before it is
allowed into any comparison (scripts/rescore_mbpp_conditions.py fixed the 7
conditions that originally failed this check -- see docs/mbpp_scale.md for
the full diagnosis). A condition that still fails the gate is printed as
MISSING/EXCLUDED and never silently folded into a mean or a McNemar test.

Writes:
  logs/mbpp_scale/answers_mbpp.jsonl   (per-task side-by-side diff across
                                         the core conditions)
  logs/mbpp_scale/mbpp_analysis_summary.json

Usage:
    python scripts/analyze_mbpp_scale.py
"""

import ast
import difflib
import json
import re
import sys
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS_DIR))
from analyze_quant_gap import wilson_ci, mcnemar_exact, load_eval_results  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
LOGS = ROOT / "logs" / "mbpp_scale"

from evalplus.data import get_mbpp_plus  # noqa: E402

N_CANONICAL = len(get_mbpp_plus())

MISSING = []


def note_missing(desc, path):
    MISSING.append((desc, str(path)))


def is_pass(tasks, tid):
    t = tasks.get(tid)
    if not t:
        return False
    return t["base_pass"] and t["plus_pass"]


def plus_rate(tasks):
    n = len(tasks)
    k = sum(1 for t in tasks.values() if t["base_pass"] and t["plus_pass"])
    return k, n


def fmt_pct(k, n):
    if n == 0:
        return "n/a"
    p, lo, hi = wilson_ci(k, n)
    return f"{p * 100:5.1f}% [{lo * 100:4.1f},{hi * 100:4.1f}] ({k}/{n})"


def two_by_two(tasks_a, tasks_b):
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


def churn_pct(a_only, b_only, n):
    return (a_only + b_only) / n * 100.0 if n else 0.0


def mcnemar_row(label, tasks_a, tasks_b):
    bp, a_only, b_only, bf, common = two_by_two(tasks_a, tasks_b)
    _, _, p = mcnemar_exact(a_only, b_only)
    print(f"  {label:45s} a_only={a_only:3d} b_only={b_only:3d} "
          f"churn={churn_pct(a_only, b_only, len(common)):5.1f}%  McNemar p={p:.4f}")
    return {"a_only": a_only, "b_only": b_only, "both_pass": bp, "both_fail": bf,
            "n_common": len(common), "churn_pct": churn_pct(a_only, b_only, len(common)), "p": p}


def norm_ws(s):
    if not s:
        return ""
    return re.sub(r'\s+', ' ', s.strip())


def extract_docstring(code, entry_point):
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return None
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == entry_point:
            doc = ast.get_docstring(node)
            if doc:
                return doc
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            doc = ast.get_docstring(node)
            if doc:
                return doc
    return None


def classify_provenance(solution_doc, prompt_doc):
    if prompt_doc is None:
        return "original"
    if solution_doc is None:
        return "none"
    sd, pd = norm_ws(solution_doc), norm_ws(prompt_doc)
    if sd == pd:
        return "echoed"
    ratio = difflib.SequenceMatcher(None, sd, pd).ratio()
    if ratio >= 0.6:
        return "modified"
    return "original"


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
        out[d["task_id"]] = d.get("solution", d.get("completion", ""))
    return out


def dump_answers(path, problems_by_task, per_task_rows):
    with open(path, "w") as f:
        for tid in sorted(per_task_rows):
            rec = {"task_id": tid, "prompt": problems_by_task.get(tid, ""), "conditions": per_task_rows[tid]}
            f.write(json.dumps(rec) + "\n")
    print(f"  answers file written: {path}  ({len(per_task_rows)} tasks)")


def main():
    mbpp = get_mbpp_plus()

    stems = {
        "bf16_cold": LOGS / "gate" / "A_bf16_cold",
        "int4_cold": LOGS / "gate" / "C_nf4_cold",
        "int4_prompt_control": LOGS / "int4_prompt_control",
        "int4_neutral_prefix": LOGS / "int4_neutral_prefix",
        "int4_tim_entropy_seed0": LOGS / "int4_tim_entropy_seed0",
        "full_F_nf4_prompt_control": LOGS / "full" / "F_nf4_prompt_control",
    }
    for p in range(1, 4):
        p_s = f"_pass{p}" if p > 1 else ""
        stems[f"bf16_tim_domain_seed0_pass{p}"] = LOGS / "full" / f"B_bf16_tim_domain_seed0{p_s}"
        stems[f"int4_tim_domain_seed0_pass{p}"] = LOGS / f"int4_tim_domain_seed0{p_s}"
    stems["int4_tim_domain_seed1_pass1"] = LOGS / "int4_tim_domain_seed1"
    stems["int4_tim_domain_seed2_pass1"] = LOGS / "int4_tim_domain_seed2"
    stems["full_D_nf4_tim_domain_seed0_pass1"] = LOGS / "full" / "D_nf4_tim_domain_seed0"

    # ---- Load + gate every condition on n_scored == n_canonical ----------
    evals, solutions = {}, {}
    print("=" * 100)
    print(f"MBPP+ SCALE ANALYSIS — canonical N={N_CANONICAL} (this evalplus install)")
    print("=" * 100)
    print("\nCoverage gate (n_scored must equal n_canonical for a condition to be used):")
    for name, path in stems.items():
        eval_path = path.with_name(path.name + "-sanitized.eval_results.json")
        if not eval_path.exists():
            print(f"  {name:35s} MISSING eval_results.json -> EXCLUDED")
            note_missing("eval_results", eval_path)
            continue
        tasks = load_eval_results(eval_path)
        if len(tasks) != N_CANONICAL:
            print(f"  {name:35s} n_scored={len(tasks)} != n_canonical={N_CANONICAL} -> EXCLUDED")
            note_missing(f"incomplete eval_results (n_scored={len(tasks)})", eval_path)
            continue
        evals[name] = tasks
        solutions[name] = load_solutions(path)
        print(f"  {name:35s} OK ({len(tasks)}/{N_CANONICAL})")

    summary = {"n_canonical": N_CANONICAL, "conditions": {}}

    # ---- Full pass@1 table -------------------------------------------------
    print("\n" + "=" * 100)
    print("PASS@1 (base+extra) — every gated condition, Wilson 95% CI")
    print("=" * 100)
    for name, tasks in evals.items():
        k, n = plus_rate(tasks)
        p, lo, hi = wilson_ci(k, n)
        summary["conditions"][name] = {"k": k, "n": n, "pass_at_1_plus": p, "wilson_ci": [lo, hi]}
        print(f"  {name:35s} {fmt_pct(k, n)}")

    # ---- Gate: bf16_cold vs int4_cold (quantization damage) ---------------
    if "bf16_cold" in evals and "int4_cold" in evals:
        print("\n" + "=" * 100)
        print("QUANTIZATION DAMAGE — bf16_cold vs int4_cold")
        print("=" * 100)
        summary["gate_damage"] = mcnemar_row("bf16_cold vs int4_cold", evals["bf16_cold"], evals["int4_cold"])

    # ---- Headline replication test: pooled int4_cold vs int4_tim_domain --
    print("\n" + "=" * 100)
    print("HEADLINE REPLICATION TEST — int4_cold vs int4_tim_domain (pooled seeds 0/1/2, pass1)")
    print("=" * 100)
    seed_evals = {
        "seed0": evals.get("int4_tim_domain_seed0_pass1"),
        "seed1": evals.get("int4_tim_domain_seed1_pass1"),
        "seed2": evals.get("int4_tim_domain_seed2_pass1"),
    }
    summary["rung_headline"] = {"per_seed": {}}
    if "int4_cold" in evals and all(seed_evals.values()):
        c_only_tot = e_only_tot = n_common_tot = 0
        mean_k = 0.0
        for sname, ev in seed_evals.items():
            k, n = plus_rate(ev)
            mean_k += k / n
            row = mcnemar_row(f"int4_cold vs int4_tim_domain {sname} (pass1)", evals["int4_cold"], ev)
            summary["rung_headline"]["per_seed"][sname] = row
            c_only_tot += row["a_only"]
            e_only_tot += row["b_only"]
            n_common_tot += row["n_common"]
        mean_k /= 3
        _, _, pooled_p = mcnemar_exact(c_only_tot, e_only_tot)
        print(f"\n  Pooled: mean pass@1+={mean_k*100:.1f}%  cold_only={c_only_tot} tim_only={e_only_tot} "
              f"churn={churn_pct(c_only_tot, e_only_tot, n_common_tot):.1f}%  McNemar p={pooled_p:.4f}")
        summary["rung_headline"]["pooled"] = {
            "mean_pass_at_1_plus": mean_k, "cold_only": c_only_tot, "tim_only": e_only_tot,
            "churn_pct": churn_pct(c_only_tot, e_only_tot, n_common_tot), "p": pooled_p,
        }
        k_cold, n_cold = plus_rate(evals["int4_cold"])
        cold_p = k_cold / n_cold
        gap_pp = (mean_k - cold_p) * 100
        print(f"  int4_cold={cold_p*100:.1f}%  gap vs pooled tim_domain = {gap_pp:+.1f} pp")
        print(f"  HumanEval+ reference (N=164, docs/int4_mechanism.md): +7pp, McNemar p=0.0004")
        summary["rung_headline"]["gap_pp_vs_cold"] = gap_pp
        summary["rung_headline"]["replicated_significant"] = bool(pooled_p < 0.05 and gap_pp > 0)
    else:
        print("  Cannot compute — missing one or more gated conditions (see coverage gate above).")

    # ---- bf16 stays flat? ---------------------------------------------------
    print("\n" + "=" * 100)
    print("bf16_cold vs bf16_tim_domain (pass1) — replication of the 'flat at bf16' finding")
    print("=" * 100)
    if "bf16_cold" in evals and "bf16_tim_domain_seed0_pass1" in evals:
        summary["bf16_flat_check"] = mcnemar_row(
            "bf16_cold vs bf16_tim_domain_seed0_pass1", evals["bf16_cold"], evals["bf16_tim_domain_seed0_pass1"])
    else:
        print("  Cannot compute — missing condition.")

    # ---- Mechanism controls vs int4_cold ------------------------------------
    print("\n" + "=" * 100)
    print("MECHANISM CONTROLS — vs int4_cold")
    print("=" * 100)
    summary["mechanism_controls"] = {}
    for cond in ["int4_prompt_control", "int4_neutral_prefix", "int4_tim_entropy_seed0"]:
        if cond in evals and "int4_cold" in evals:
            k, n = plus_rate(evals[cond])
            print(f"  {cond} pass@1+={fmt_pct(k, n)}")
            summary["mechanism_controls"][cond] = mcnemar_row(f"int4_cold vs {cond}", evals["int4_cold"], evals[cond])
        else:
            print(f"  {cond}: SKIPPED (missing)")

    # tim_domain vs prompt_control (delivery parity, MBPP+)
    if "int4_prompt_control" in evals and "int4_tim_domain_seed0_pass1" in evals:
        print()
        summary["delivery_parity"] = mcnemar_row(
            "int4_prompt_control vs int4_tim_domain_seed0_pass1",
            evals["int4_prompt_control"], evals["int4_tim_domain_seed0_pass1"])

    # ---- Multi-pass dose-response ladders -----------------------------------
    print("\n" + "=" * 100)
    print("MULTI-PASS DOSE-RESPONSE — bf16_tim_domain_seed0 (vs bf16_cold, and pass-to-pass)")
    print("=" * 100)
    summary["dose_response_bf16"] = {"vs_cold": {}, "pass_to_pass": {}}
    for p in range(1, 4):
        cond = f"bf16_tim_domain_seed0_pass{p}"
        if cond in evals and "bf16_cold" in evals:
            k, n = plus_rate(evals[cond])
            print(f"  pass{p}: {fmt_pct(k, n)}")
            summary["dose_response_bf16"]["vs_cold"][f"pass{p}"] = mcnemar_row(
                f"bf16_cold vs {cond}", evals["bf16_cold"], evals[cond])
    for p1, p2 in [(1, 2), (2, 3)]:
        c1, c2 = f"bf16_tim_domain_seed0_pass{p1}", f"bf16_tim_domain_seed0_pass{p2}"
        if c1 in evals and c2 in evals:
            summary["dose_response_bf16"]["pass_to_pass"][f"{p1}v{p2}"] = mcnemar_row(
                f"{c1} vs {c2}", evals[c1], evals[c2])

    print("\n" + "=" * 100)
    print("MULTI-PASS DOSE-RESPONSE — int4_tim_domain_seed0 (vs int4_cold, and pass-to-pass)")
    print("=" * 100)
    summary["dose_response_int4"] = {"vs_cold": {}, "pass_to_pass": {}}
    for p in range(1, 4):
        cond = f"int4_tim_domain_seed0_pass{p}"
        if cond in evals and "int4_cold" in evals:
            k, n = plus_rate(evals[cond])
            print(f"  pass{p}: {fmt_pct(k, n)}")
            summary["dose_response_int4"]["vs_cold"][f"pass{p}"] = mcnemar_row(
                f"int4_cold vs {cond}", evals["int4_cold"], evals[cond])
    for p1, p2 in [(1, 2), (2, 3)]:
        c1, c2 = f"int4_tim_domain_seed0_pass{p1}", f"int4_tim_domain_seed0_pass{p2}"
        if c1 in evals and c2 in evals:
            summary["dose_response_int4"]["pass_to_pass"][f"{p1}v{p2}"] = mcnemar_row(
                f"{c1} vs {c2}", evals[c1], evals[c2])

    # ---- Cross-run determinism check: same condition, two independent -----
    # model loads/scripts (top-level int4_tim_domain_seed0_pass1 generated by
    # run_int4_tim_domain.py on GPU1; full/D_nf4_tim_domain_seed0 generated
    # independently by run_quant_gap.py's run_full() on GPU0 -- nominally the
    # identical condition: nf4 + tim_domain + seed0 + pass1).
    print("\n" + "=" * 100)
    print("CROSS-RUN DETERMINISM CHECK — same nominal condition, two independent model loads")
    print("=" * 100)
    if "int4_tim_domain_seed0_pass1" in evals and "full_D_nf4_tim_domain_seed0_pass1" in evals:
        k1, n1 = plus_rate(evals["int4_tim_domain_seed0_pass1"])
        k2, n2 = plus_rate(evals["full_D_nf4_tim_domain_seed0_pass1"])
        print(f"  int4_tim_domain_seed0 (GPU1 run):        {fmt_pct(k1, n1)}")
        print(f"  full/D_nf4_tim_domain_seed0 (GPU0 run):  {fmt_pct(k2, n2)}")
        summary["nf4_tim_domain_determinism"] = mcnemar_row(
            "int4_tim_domain_seed0 vs full/D_nf4_tim_domain_seed0 (same condition, 2 model loads)",
            evals["int4_tim_domain_seed0_pass1"], evals["full_D_nf4_tim_domain_seed0_pass1"])
    if "int4_prompt_control" in evals and "full_F_nf4_prompt_control" in evals:
        k1, n1 = plus_rate(evals["int4_prompt_control"])
        k2, n2 = plus_rate(evals["full_F_nf4_prompt_control"])
        print(f"  int4_prompt_control (GPU1 run):          {fmt_pct(k1, n1)}")
        print(f"  full/F_nf4_prompt_control (GPU0 run):    {fmt_pct(k2, n2)}")
        summary["nf4_prompt_control_determinism"] = mcnemar_row(
            "int4_prompt_control vs full/F_nf4_prompt_control (same condition, 2 model loads)",
            evals["int4_prompt_control"], evals["full_F_nf4_prompt_control"])

    # ---- Echo test across passes (MBPP+ has no docstrings to copy) --------
    print("\n" + "=" * 100)
    print("ECHO TEST (docstring provenance) — int4_tim_domain_seed0, passes 1/2/3")
    print("MBPP+ prompts are natural-language instructions, not docstring-bearing function")
    print("signatures -- this is the clean test of input-echoing vs induced style.")
    print("=" * 100)
    summary["echo_test"] = {}
    for p in range(1, 4):
        cond = f"int4_tim_domain_seed0_pass{p}"
        if cond not in solutions or solutions[cond] is None:
            continue
        counts = {"echoed": 0, "modified": 0, "original": 0, "none": 0}
        total = 0
        for tid, code in solutions[cond].items():
            prob = mbpp[tid]
            doc = extract_docstring(code, prob["entry_point"])
            prompt_doc = prob["prompt"].strip()
            if prompt_doc.startswith('"""') or prompt_doc.startswith("'''"):
                prompt_doc = prompt_doc[3:-3].strip()
            prov = classify_provenance(doc, prompt_doc)
            counts[prov] += 1
            total += 1
        has_doc = counts["echoed"] + counts["modified"] + counts["original"]
        print(f"\n  {cond}: {total} tasks")
        print(f"    Docstring rate: {has_doc/total*100:.1f}% ({has_doc}/{total})")
        if has_doc > 0:
            print(f"    Echoed:   {counts['echoed']/has_doc*100:.1f}% ({counts['echoed']})")
            print(f"    Modified: {counts['modified']/has_doc*100:.1f}% ({counts['modified']})")
            print(f"    Original:{counts['original']/has_doc*100:.1f}% ({counts['original']})")
        summary["echo_test"][cond] = {"total": total, "counts": counts, "docstring_rate": has_doc / total if total else 0}

    # ---- Per-task answers dump ----------------------------------------------
    print("\n" + "=" * 100)
    print("ANSWERS DUMP")
    print("=" * 100)
    problems_by_task = {tid: p["prompt"] for tid, p in mbpp.items()}
    core_conditions = [c for c in [
        "bf16_cold", "int4_cold", "int4_prompt_control", "int4_neutral_prefix",
        "int4_tim_entropy_seed0", "int4_tim_domain_seed0_pass1", "int4_tim_domain_seed1_pass1",
        "int4_tim_domain_seed2_pass1", "bf16_tim_domain_seed0_pass1",
    ] if c in evals]
    per_task_rows = {}
    all_task_ids = set()
    for c in core_conditions:
        all_task_ids |= set(evals[c].keys())
    for tid in all_task_ids:
        row = {}
        for c in core_conditions:
            sol = (solutions.get(c) or {}).get(tid, "")
            row[c] = {"solution": sol, "pass": is_pass(evals[c], tid) if tid in evals[c] else None}
        per_task_rows[tid] = row
    dump_answers(LOGS / "answers_mbpp.jsonl", problems_by_task, per_task_rows)

    # ---- Summary + missing ---------------------------------------------------
    def _json_default(o):
        if hasattr(o, "item"):
            return o.item()
        return str(o)

    summary_path = LOGS / "mbpp_analysis_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=_json_default)
    print(f"\nSummary written to: {summary_path}")

    print("\n" + "=" * 100)
    print("EXPECTED BUT NOT FOUND")
    print("=" * 100)
    if MISSING:
        for desc, path in MISSING:
            print(f"  [{desc}] {path}")
    else:
        print("  (none)")


if __name__ == "__main__":
    main()
