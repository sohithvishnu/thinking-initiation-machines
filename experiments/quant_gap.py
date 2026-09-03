"""
Quantization-recovery study: can KV-cache priming recover accuracy that 4-bit
quantization removes?

2x2 design, plus prompt_control at both precisions:

                       bf16      4-bit (bnb NF4)
    cold                 A            C
    tim_domain           B            D
    prompt_control       E            F

    damage   = A - C   what 4-bit costs. B/D/E/F only run if this is >= 5 pp.
    recovery = D - C   what TIM buys on the quantized model.
    success  = D >= A  quantized + primed reaches full-precision cold.

The gate runs first and alone: if 4-bit does not measurably damage this model
on this benchmark, there is no damage for priming to recover and the rest of
the grid is not worth the GPU hours.

NF4 determinism is checked before scoring rather than assumed — double
quantization is not guaranteed bit-reproducible, and a non-deterministic arm
would invalidate the paired McNemar comparisons. On failure the arm is
reloaded with `bnb_4bit_use_double_quant=False` and re-checked.

Usage:
    # Step 1 — gate (run this first, alone)
    python experiments/quant_gap.py --stage gate --model Qwen/Qwen3-1.7B

    # Step 2 — full 2x2, only if the gate passes
    python experiments/quant_gap.py --stage full --model Qwen/Qwen3-1.7B --num_seeds 3
"""

import argparse
import gc
import json

import torch

from tim.config import LOGS_DIR
from tim.evaluation import load_problems, run_condition
from tim.generation import check_determinism
from tim.models import load_model, unload_model
from tim.primer import TIMPrimer
from tim.stats import wilson_ci
from tim.vocab import DOMAIN_PERSONA_TEXT, PYTHON_DOMAIN_WORDS

OUT_DIR = LOGS_DIR / "quant_gap"

# Minimum bf16-vs-nf4 gap, in percentage points, for the full grid to be worth
# running: below this there is no damage for priming to recover.
GATE_DAMAGE_PP = 5.0


def output_dir_for(args, model_id: str, stage: str):
    """MBPP+ conditions land in one shared `logs/mbpp_scale/` tree because the
    MBPP run is analyzed as a single cross-script grid; HumanEval+ conditions
    stay under their own per-model directory."""
    if args.dataset == "mbpp":
        return LOGS_DIR / "mbpp_scale" / stage
    return OUT_DIR / model_id.replace("/", "_") / stage


def load_nf4_checked(model_id: str, probe_prompt: str):
    """Load NF4 and verify greedy decoding is bit-reproducible, retrying once
    with double quantization off. Returns (model, tokenizer, device, report, det)."""
    use_double_quant = True
    model, tokenizer, device, report = load_model(
        model_id, quant="nf4", use_double_quant=use_double_quant)

    det = check_determinism(model, tokenizer, device, probe_prompt)
    print(f"[determinism] nf4 double_quant={use_double_quant}: identical={det['identical']}")
    if not det["identical"]:
        print("[determinism] NOT identical — reloading with "
              "bnb_4bit_use_double_quant=False (batch size is already 1)")
        unload_model(model)
        model, tokenizer, device, report = load_model(
            model_id, quant="nf4", use_double_quant=False)
        det2 = check_determinism(model, tokenizer, device, probe_prompt)
        print(f"[determinism] retry double_quant=False: identical={det2['identical']}")
        if not det2["identical"]:
            print("[determinism] STILL not identical after retry — proceeding, "
                  "flagged loudly in the report.")
        det = {"first_attempt": det, "retry_double_quant_false": det2}
    return model, tokenizer, device, report, det


def free_kv(kv):
    del kv
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# --------------------------------------------------------------------------- #
# Stage 1: gate — A (bf16 cold) vs C (nf4 cold)
# --------------------------------------------------------------------------- #

def run_gate(model_id: str, args) -> dict:
    output_dir = output_dir_for(args, model_id, "gate")
    output_dir.mkdir(parents=True, exist_ok=True)
    task_items = load_problems(args.dataset, limit=args.limit)

    common = dict(dataset=args.dataset, max_new_tokens=args.max_new_tokens,
                  score=not args.no_score)
    report = {"model_id": model_id, "n_tasks": len(task_items), "dataset": args.dataset}

    print(f"\n=== Gate | A: bf16 cold | {model_id} ===")
    model, tokenizer, device, report["A_quant_report"] = load_model(model_id, quant="bf16")
    results_a = run_condition(model, tokenizer, device, "A_bf16_cold",
                              output_dir, task_items, **common)
    report["A"] = results_a
    unload_model(model)

    print(f"\n=== Gate | C: nf4 cold | {model_id} ===")
    model, tokenizer, device, report["C_quant_report"], report["C_determinism"] = \
        load_nf4_checked(model_id, task_items[0][1]["prompt"])
    results_c = run_condition(model, tokenizer, device, "C_nf4_cold",
                              output_dir, task_items, **common)
    report["C"] = results_c
    unload_model(model)

    a_plus = results_a.get("pass@1_base_plus_extra")
    c_plus = results_c.get("pass@1_base_plus_extra")
    gate = {}
    if a_plus is not None and c_plus is not None:
        n = len(task_items)
        k_a, k_c = round(a_plus * n), round(c_plus * n)
        _, lo_a, hi_a = wilson_ci(k_a, n)
        _, lo_c, hi_c = wilson_ci(k_c, n)
        damage_pp = (a_plus - c_plus) * 100
        gate = {
            "A_pass_at_1_plus": a_plus, "A_k": k_a, "A_n": n, "A_wilson_ci": [lo_a, hi_a],
            "C_pass_at_1_plus": c_plus, "C_k": k_c, "C_n": n, "C_wilson_ci": [lo_c, hi_c],
            "damage_pp": damage_pp,
            "gate_passed": damage_pp >= GATE_DAMAGE_PP,
        }
    report["gate"] = gate

    report_path = output_dir / "gate_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    print("\n" + "=" * 78)
    print(f"GATE RESULT — {model_id}")
    print("=" * 78)
    if gate:
        print(f"A (bf16 cold): {gate['A_pass_at_1_plus']:.4f} ({gate['A_k']}/{gate['A_n']})  "
              f"95% Wilson CI [{gate['A_wilson_ci'][0]:.4f}, {gate['A_wilson_ci'][1]:.4f}]")
        print(f"C (nf4  cold): {gate['C_pass_at_1_plus']:.4f} ({gate['C_k']}/{gate['C_n']})  "
              f"95% Wilson CI [{gate['C_wilson_ci'][0]:.4f}, {gate['C_wilson_ci'][1]:.4f}]")
        print(f"damage (A-C) = {gate['damage_pp']:+.1f} pp")
        print(f"GATE {'PASSED' if gate['gate_passed'] else 'FAILED'} "
              f"(threshold: damage >= {GATE_DAMAGE_PP} pp)")
    else:
        print("Could not compute gate — pass@1 scores missing (was --no_score set?).")
    print(f"peak_memory_mb: A={results_a.get('peak_memory_mb'):.1f}  "
          f"C={results_c.get('peak_memory_mb'):.1f}")
    print(f"Report written to: {report_path}")
    return report


# --------------------------------------------------------------------------- #
# Stage 2: full 2x2 + prompt_control at both precisions. Gate must pass first.
# --------------------------------------------------------------------------- #

def run_full(model_id: str, args) -> dict:
    output_dir = output_dir_for(args, model_id, "full")
    output_dir.mkdir(parents=True, exist_ok=True)
    task_items = load_problems(args.dataset, limit=args.limit)

    common = dict(dataset=args.dataset, max_new_tokens=args.max_new_tokens,
                  score=not args.no_score)
    pass_tag = f"_pass{args.num_passes}" if args.num_passes > 1 else ""
    report = {"model_id": model_id, "n_tasks": len(task_items),
              "dataset": args.dataset, "num_seeds": args.num_seeds}

    def tim_arm(model, tokenizer, device, prefix: str, tag: str):
        primer = TIMPrimer(model, tokenizer, device, noise_length=args.noise_length,
                           num_passes=args.num_passes, chain_mode="reseed")
        domain_ids = primer.get_domain_token_ids_from_words(PYTHON_DOMAIN_WORDS)
        out = {}
        for seed in range(args.num_seeds):
            kv, prime_timing = primer.prime(
                domain_tokens=domain_ids, seed=seed, collect_timing=True)
            name = f"{prefix}_tim_domain_seed{seed}{tag}"
            out[f"seed{seed}"] = run_condition(
                model, tokenizer, device, name, output_dir, task_items,
                primed_kv=kv, prime_timing=prime_timing, **common)
            free_kv(kv)
        return out

    print(f"\n=== Full 2x2 | bf16 arms (B, E) | {model_id} ===")
    model, tokenizer, device, report["bf16_quant_report"] = load_model(model_id, quant="bf16")
    if not args.skip_prompt_control:
        report["E"] = run_condition(
            model, tokenizer, device, "E_bf16_prompt_control", output_dir, task_items,
            prepend_text=DOMAIN_PERSONA_TEXT + "\n\n", **common)
    report["B"] = tim_arm(model, tokenizer, device, "B_bf16", pass_tag)
    unload_model(model)

    print(f"\n=== Full 2x2 | nf4 arms (D, F) | {model_id} ===")
    model, tokenizer, device, report["nf4_quant_report"] = load_model(model_id, quant="nf4")
    # F runs even under --skip_prompt_control, and D takes no _pass tag. Both
    # asymmetries are how the committed artifacts were produced: the MBPP+
    # multi-pass sweep (run_mbpp_scale_gpu0.sh) reuses one nf4 prompt_control
    # across passes but reruns F each time, and logs/mbpp_scale/full/ has
    # B_..._pass2/_pass3 with no D counterpart. Making these symmetric would
    # overwrite D_nf4_tim_domain_seed0 on every pass, so the shape is kept as
    # recorded rather than "fixed" into something the logs do not match.
    report["F"] = run_condition(
        model, tokenizer, device, "F_nf4_prompt_control", output_dir, task_items,
        prepend_text=DOMAIN_PERSONA_TEXT + "\n\n", **common)
    report["D"] = tim_arm(model, tokenizer, device, "D_nf4", tag="")
    unload_model(model)

    report_path = output_dir / "full_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nFull 2x2 report written to: {report_path}")
    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["gate", "full"], required=True)
    ap.add_argument("--model", default="Qwen/Qwen3-1.7B")
    ap.add_argument("--num_seeds", type=int, default=3)
    ap.add_argument("--noise_length", type=int, default=64,
                    help="Number of tokens to prime.")
    ap.add_argument("--num_passes", type=int, default=1,
                    help="TIMPrimer pass count (>1 appends a _pass<N> tag to condition names).")
    ap.add_argument("--dataset", choices=["humaneval", "mbpp"], default="humaneval")
    ap.add_argument("--skip_prompt_control", action="store_true",
                    help="Skip arm E (bf16 prompt_control), reusing existing results. "
                         "Arm F still runs — see the note in run_full().")
    ap.add_argument("--max_new_tokens", type=int, default=768)
    ap.add_argument("--limit", type=int, default=None,
                    help="First N tasks only (for smoke-testing the pipeline).")
    ap.add_argument("--no_score", action="store_true",
                    help="Skip EvalPlus scoring (pure timing/smoke run).")
    args = ap.parse_args()

    print("=" * 78)
    print(f"Quantization gap study — stage={args.stage} model={args.model} "
          f"dataset={args.dataset}")
    print("=" * 78)

    run_gate(args.model, args) if args.stage == "gate" else run_full(args.model, args)


if __name__ == "__main__":
    main()
