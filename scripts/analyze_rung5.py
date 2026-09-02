"""
Rung 5 analysis: McNemar tests for tim_entropy against int4_cold and int4_tim_domain.

Reads existing eval results from:
  - logs/quant_gap/Qwen_Qwen3-1.7B/gate/C_nf4_cold-sanitized.eval_results.json
  - logs/quality_quant/int4_tim_domain_seed0-sanitized.eval_results.json
  - logs/int4_mechanism/int4_tim_entropy_seed{0,1,2}-sanitized.eval_results.json
  - logs/int4_mechanism/int4_tim_random_seed{0,1,2}-sanitized.eval_results.json
  - logs/int4_mechanism/int4_neutral_prefix-sanitized.eval_results.json

Prints McNemar comparisons and the formal verdict, then appends Rung 5 results
to docs/int4_mechanism.md.

Usage:
    python scripts/analyze_rung5.py
"""

import json
import statistics
import sys
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS_DIR))
from analyze_quant_gap import wilson_ci, mcnemar_exact, load_eval_results  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
LOGS = ROOT / "logs"
OUT_DIR = LOGS / "int4_mechanism"
DOCS_DIR = ROOT / "docs"

MISSING = []


def note_missing(desc, path):
    MISSING.append((desc, str(path)))


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
    return f"{p * 100:5.1f}% [{lo * 100:4.1f},{hi * 100:4.1f}] ({k}/{n})"


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


def load_eval(path):
    if not path.exists():
        note_missing("eval_results", path)
        return None
    return load_eval_results(path)


def print_header(t):
    print("\n" + "=" * 100)
    print(t)
    print("=" * 100)


def main():
    # Load all conditions
    cold_path = LOGS / "quant_gap" / "Qwen_Qwen3-1.7B" / "gate" / "C_nf4_cold-sanitized.eval_results.json"
    cold_eval = load_eval(cold_path)

    tim_domain_path = LOGS / "quality_quant" / "int4_tim_domain_seed0-sanitized.eval_results.json"
    tim_domain_eval = load_eval(tim_domain_path)

    neutral_path = OUT_DIR / "int4_neutral_prefix-sanitized.eval_results.json"
    neutral_eval = load_eval(neutral_path)

    # Load tim_entropy seeds
    entropy_evals = {}
    for s in range(3):
        path = OUT_DIR / f"int4_tim_entropy_seed{s}-sanitized.eval_results.json"
        ev = load_eval(path)
        if ev is not None:
            entropy_evals[s] = ev

    # Load tim_random seeds for comparison
    random_evals = {}
    for s in range(3):
        path = OUT_DIR / f"int4_tim_random_seed{s}-sanitized.eval_results.json"
        ev = load_eval(path)
        if ev is not None:
            random_evals[s] = ev

    if MISSING:
        print("WARNING: Missing files:")
        for desc, path in MISSING:
            print(f"  - {desc}: {path}")
        print()

    if cold_eval is None:
        print("FATAL: int4_cold eval not found — cannot proceed.")
        return

    if not entropy_evals:
        print("FATAL: no tim_entropy eval results found — run experiment-3/run_rung5_entropy.py first.")
        return

    # ====================================================================
    # Rung 5 results table
    # ====================================================================
    print_header("RUNG 5 — int4-cold vs int4-tim_entropy (semantic vs entropy control)")

    k_cold, n_cold = plus_rate(cold_eval)
    print(f"int4_cold          : {fmt_pct(k_cold, n_cold)}")

    entropy_rates = []
    entropy_results = {}

    for s in sorted(entropy_evals):
        ev = entropy_evals[s]
        k_e, n_e = plus_rate(ev)
        entropy_rates.append(k_e / n_e)
        bp, c_only, e_only, bf, common = two_by_two(cold_eval, ev)
        _, _, p = mcnemar_exact(c_only, e_only)
        print(f"tim_entropy seed{s} : {fmt_pct(k_e, n_e)}  McNemar vs cold: cold_only={c_only} entropy_only={e_only} p={p:.4f}")
        entropy_results[s] = {
            "k": k_e, "n": n_e, "pct": k_e / n_e,
            "cold_only": c_only, "entropy_only": e_only, "mcnemar_p": p,
            "both_pass": bp, "both_fail": bf,
        }

    # Pooled McNemar across seeds
    if len(entropy_evals) > 1:
        pooled_cold_only = sum(r["cold_only"] for r in entropy_results.values())
        pooled_entropy_only = sum(r["entropy_only"] for r in entropy_results.values())
        _, _, pooled_p = mcnemar_exact(pooled_cold_only, pooled_entropy_only)
        print(f"\nPOOLED ({len(entropy_evals)} seeds): cold_only={pooled_cold_only} entropy_only={pooled_entropy_only} p={pooled_p:.4f}")
    else:
        pooled_p = list(entropy_results.values())[0]["mcnemar_p"]
        pooled_cold_only = list(entropy_results.values())[0]["cold_only"]
        pooled_entropy_only = list(entropy_results.values())[0]["entropy_only"]

    # ====================================================================
    # Compare with tim_domain
    # ====================================================================
    print_header("McNemar: int4-tim_domain (seed0) vs int4-tim_entropy (each seed)")
    if tim_domain_eval is not None:
        k_td, n_td = plus_rate(tim_domain_eval)
        print(f"int4_tim_domain_seed0: {fmt_pct(k_td, n_td)}")
        domain_vs_entropy = {}
        for s in sorted(entropy_evals):
            ev = entropy_evals[s]
            bp, td_only, e_only, bf, common = two_by_two(tim_domain_eval, ev)
            _, _, p = mcnemar_exact(td_only, e_only)
            print(f"  vs tim_entropy seed{s}: tim_domain_only={td_only} entropy_only={e_only} p={p:.4f}")
            domain_vs_entropy[s] = {"tim_domain_only": td_only, "entropy_only": e_only, "mcnemar_p": p}
    else:
        print("tim_domain_seed0 not available for comparison.")
        domain_vs_entropy = {}

    # ====================================================================
    # Compare with tim_random (the key comparison: same diversity, with vs without meaning)
    # ====================================================================
    if random_evals:
        print_header("Comparison with tim_random (same diversity, with meaning)")
        for s in sorted(random_evals):
            k_r, n_r = plus_rate(random_evals[s])
            print(f"int4_tim_random_seed{s}: {fmt_pct(k_r, n_r)}")

    # Compare with neutral_prefix
    if neutral_eval is not None:
        k_np, n_np = plus_rate(neutral_eval)
        print(f"\nint4_neutral_prefix  : {fmt_pct(k_np, n_np)}")

    # ====================================================================
    # Summary table
    # ====================================================================
    print_header("SUMMARY TABLE — all conditions")
    print(f"{'condition':<30} {'pass@1+':<20} {'McNemar vs cold':<30}")
    print("-" * 80)

    print(f"{'int4_cold':<30} {fmt_pct(k_cold, n_cold):<20}")

    if neutral_eval is not None:
        k_np, n_np = plus_rate(neutral_eval)
        bp, c_only, np_only, bf, _ = two_by_two(cold_eval, neutral_eval)
        _, _, p_np = mcnemar_exact(c_only, np_only)
        print(f"{'int4_neutral_prefix':<30} {fmt_pct(k_np, n_np):<20} cold={c_only} neutral={np_only} p={p_np:.4f}")

    for s in sorted(entropy_results):
        r = entropy_results[s]
        print(f"{'int4_tim_entropy_seed' + str(s):<30} {fmt_pct(r['k'], r['n']):<20} "
              f"cold={r['cold_only']} entropy={r['entropy_only']} p={r['mcnemar_p']:.4f}")

    for s in sorted(random_evals):
        k_r, n_r = plus_rate(random_evals[s])
        bp, c_only, r_only, bf, _ = two_by_two(cold_eval, random_evals[s])
        _, _, p_r = mcnemar_exact(c_only, r_only)
        print(f"{'int4_tim_random_seed' + str(s):<30} {fmt_pct(k_r, n_r):<20} cold={c_only} random={r_only} p={p_r:.4f}")

    if tim_domain_eval is not None:
        k_td, n_td = plus_rate(tim_domain_eval)
        bp, c_only, td_only, bf, _ = two_by_two(cold_eval, tim_domain_eval)
        _, _, p_td = mcnemar_exact(c_only, td_only)
        print(f"{'int4_tim_domain_seed0':<30} {fmt_pct(k_td, n_td):<20} cold={c_only} tim={td_only} p={p_td:.4f}")

    # ====================================================================
    # VERDICT
    # ====================================================================
    print_header("RUNG 5 VERDICT")

    entropy_mean = statistics.mean(entropy_rates)
    entropy_mean_pct = entropy_mean * 100
    cold_pct = k_cold / n_cold * 100

    # The key question: does entropy beat cold?
    entropy_beats_cold = pooled_entropy_only > pooled_cold_only and pooled_p < 0.05
    entropy_near_cold = abs(entropy_mean_pct - cold_pct) < 3.0

    # Reference: where do tim_random and neutral_prefix land?
    if random_evals:
        random_mean_pct = statistics.mean(plus_rate(random_evals[s])[0] / plus_rate(random_evals[s])[1]
                                          for s in random_evals) * 100
    else:
        random_mean_pct = None

    if neutral_eval is not None:
        neutral_pct = plus_rate(neutral_eval)[0] / plus_rate(neutral_eval)[1] * 100
    else:
        neutral_pct = None

    print(f"int4_cold:          {cold_pct:.1f}%")
    if neutral_pct is not None:
        print(f"int4_neutral_prefix:{neutral_pct:.1f}%")
    print(f"int4_tim_entropy:   {entropy_mean_pct:.1f}% (mean of {len(entropy_rates)} seeds)")
    if random_mean_pct is not None:
        print(f"int4_tim_random:    {random_mean_pct:.1f}% (mean of {len(random_evals)} seeds)")
    if tim_domain_eval is not None:
        print(f"int4_tim_domain:    {plus_rate(tim_domain_eval)[0]/plus_rate(tim_domain_eval)[1]*100:.1f}% (seed0)")

    print(f"\nPooled McNemar (entropy vs cold): cold_only={pooled_cold_only} entropy_only={pooled_entropy_only} p={pooled_p:.4f}")

    if entropy_beats_cold:
        verdict = (
            "ENTROPY RESCUES: The recovery mechanism is test-time compute (FLOPs) combined "
            "with high latent entropy. The model simply needs to process a diverse set of "
            "token vectors to gain the capacity headroom; semantic meaning is not required."
        )
        verdict_short = "entropy_rescues"
    elif entropy_near_cold:
        # Entropy neither helps nor hurts — it's near cold, unlike neutral_prefix which hurts
        verdict = (
            "ENTROPY FAILS TO RESCUE: The recovery mechanism strictly requires semantic richness / "
            "valid distributional tokens. Test-time compute over diverse but meaningless noise "
            "does not rescue the model's capacity. However, unlike the degenerate neutral_prefix, "
            "diverse gibberish at least does not actively harm — suggesting the damage from "
            "neutral_prefix is specifically from repetition/degeneracy, not from non-semantic content."
        )
        verdict_short = "entropy_neutral"
    else:
        if entropy_mean_pct < cold_pct:
            verdict = (
                "ENTROPY HARMS: The recovery mechanism strictly requires semantic richness / "
                "valid distributional tokens. Test-time compute over diverse but meaningless noise "
                "does not rescue the model's capacity — diverse gibberish behaves more like the "
                "degenerate neutral_prefix than like tim_random."
            )
            verdict_short = "entropy_harms"
        else:
            # entropy_mean_pct > cold_pct but not significantly
            verdict = (
                "ENTROPY PARTIAL: tim_entropy shows a numerical advantage over cold but fails "
                "to reach significance. The evidence is inconclusive — a larger study might "
                "resolve whether the mechanism is purely entropic or requires semantic content."
            )
            verdict_short = "entropy_partial"

    print(f"\nVERDICT: {verdict}")

    # ====================================================================
    # Build answers file
    # ====================================================================
    answers_r5 = {}
    try:
        from evalplus.data import get_human_eval_plus
        prompts = {tid: p["prompt"] for tid, p in get_human_eval_plus().items()}
    except Exception:
        prompts = {}

    if cold_eval:
        for tid in sorted(cold_eval):
            row = answers_r5.setdefault(tid, {})
            row["int4_cold"] = {"pass": is_pass(cold_eval, tid)}
            for s in sorted(entropy_evals):
                ev = entropy_evals[s]
                if tid in ev:
                    row[f"int4_tim_entropy_seed{s}"] = {"pass": is_pass(ev, tid)}

    answers_path = OUT_DIR / "answers_rung5.jsonl"
    with open(answers_path, "w") as f:
        for tid in sorted(answers_r5):
            rec = {"task_id": tid, "prompt": prompts.get(tid, ""), "conditions": answers_r5[tid]}
            f.write(json.dumps(rec) + "\n")
    print(f"\nAnswers file written: {answers_path} ({len(answers_r5)} tasks)")

    # ====================================================================
    # Save analysis summary
    # ====================================================================
    analysis = {
        "rung5": {
            "int4_cold": {"k": k_cold, "n": n_cold, "pct": k_cold / n_cold},
            "entropy_results": {f"seed{s}": r for s, r in entropy_results.items()},
            "entropy_mean_pct": entropy_mean_pct,
            "pooled_mcnemar": {
                "cold_only": pooled_cold_only, "entropy_only": pooled_entropy_only, "p": pooled_p,
            },
            "domain_vs_entropy": {f"seed{s}": r for s, r in domain_vs_entropy.items()},
            "verdict": verdict,
            "verdict_short": verdict_short,
        }
    }

    analysis_path = OUT_DIR / "analysis_rung5.json"
    with open(analysis_path, "w") as f:
        json.dump(analysis, f, indent=2)
    print(f"Analysis written to: {analysis_path}")

    # ====================================================================
    # Append to docs/int4_mechanism.md
    # ====================================================================
    md_path = DOCS_DIR / "int4_mechanism.md"
    if not md_path.exists():
        print(f"WARNING: {md_path} not found — not appending.")
        return

    existing = md_path.read_text()
    if "RUNG 5" in existing or "Rung 5" in existing.upper() or "tim_entropy" in existing:
        print(f"Rung 5 already appears in {md_path} — not appending duplicate.")
        return

    # Build the markdown section
    lines = [
        "",
        "---",
        "",
        "## RUNG 5 — Is the recovery semantic, or purely entropic?",
        "",
        "Having established (Rungs 3a/3b) that *content-diverse* tokens rescue int4",
        "while *degenerate repeated* tokens harm it, the remaining question is whether",
        "the diversity needs to carry semantic meaning. `tim_random` uses real",
        "vocabulary tokens — they carry valid distributional information even without",
        "domain relevance. Could the recovery work with tokens that are diverse but",
        "completely devoid of meaning?",
        "",
        "### 5a — tim_entropy: 64 diverse, semantically meaningless tokens",
        "",
        "The `tim_entropy` control matches tim_random's exact token budget (64 tokens)",
        "and high token diversity (unique subword fragments), but strips out all",
        "semantic content: tokens are generated from random consonant-vowel gibberish",
        "strings (e.g. \"blorq kuznem fwiptal\"), verified to contain no valid English",
        "words or Python keywords. Delivery method: KV-cache injection (same path as",
        "tim_random and tim_domain).",
        "",
    ]

    # Build the results table
    lines.append("| condition | pass@1+ | 95% CI | McNemar vs cold |")
    lines.append("|---|---|---|---|")

    for s in sorted(entropy_results):
        r = entropy_results[s]
        p, lo, hi = wilson_ci(r["k"], r["n"])
        sig = " (sig)" if r["mcnemar_p"] < 0.05 else ""
        lines.append(f"| tim_entropy seed{s} | {p*100:.1f}% ({r['k']}/{r['n']}) "
                     f"| [{lo*100:.1f}, {hi*100:.1f}] "
                     f"| cold_only={r['cold_only']}, entropy_only={r['entropy_only']}, "
                     f"p={r['mcnemar_p']:.4f}{sig} |")

    if len(entropy_evals) > 1:
        lines.append(f"| **pooled** | | | **cold_only={pooled_cold_only}, "
                     f"entropy_only={pooled_entropy_only}, p={pooled_p:.4f}** |")

    lines.append("")

    # Context: reference numbers
    lines.append("Reference (from earlier rungs):")
    lines.append(f"- int4_cold: {cold_pct:.1f}%")
    if neutral_pct is not None:
        lines.append(f"- int4_neutral_prefix (Rung 3b): {neutral_pct:.1f}%")
    if random_mean_pct is not None:
        lines.append(f"- int4_tim_random (Rung 3a, mean): {random_mean_pct:.1f}%")
    if tim_domain_eval is not None:
        td_pct = plus_rate(tim_domain_eval)[0] / plus_rate(tim_domain_eval)[1] * 100
        lines.append(f"- int4_tim_domain_seed0 (Rung 1): {td_pct:.1f}%")

    lines.append("")

    # McNemar vs tim_domain
    if domain_vs_entropy:
        lines.append("McNemar int4_tim_domain (seed0) vs int4_tim_entropy (each seed):")
        lines.append("")
        for s in sorted(domain_vs_entropy):
            r = domain_vs_entropy[s]
            lines.append(f"- seed{s}: tim_domain_only={r['tim_domain_only']} "
                        f"entropy_only={r['entropy_only']} p={r['mcnemar_p']:.4f}")
        lines.append("")

    # Verdict
    lines.append(f"**Verdict (Rung 5): {verdict}**")
    lines.append("")

    # Files written
    lines.append("### Files written")
    lines.append("")
    lines.append("Generation:")
    for s in sorted(entropy_evals):
        lines.append(f"- `logs/int4_mechanism/int4_tim_entropy_seed{s}*`")
    lines.append("- `logs/int4_mechanism/mechanism_gen_report_5.json`")
    lines.append("")
    lines.append("Analysis:")
    lines.append("- `scripts/analyze_rung5.py`")
    lines.append("- `logs/int4_mechanism/analysis_rung5.json`")
    lines.append("- `logs/int4_mechanism/answers_rung5.jsonl`")
    lines.append("")

    # Write
    with open(md_path, "a") as f:
        f.write("\n".join(lines))
    print(f"\nRung 5 section appended to: {md_path}")


if __name__ == "__main__":
    main()
