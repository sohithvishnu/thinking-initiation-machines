"""
Analysis for the MoE router probe, Gemma-4-26B-A4B replication
(tim-moe/moe_router_probe_gemma.py).

Copied from analyze_moe_routing.py (the OLMoE analysis) with only the input/
output paths changed (moe_probe -> moe_probe_gemma, figures suffixed _gemma)
— the Q1/Q2 methodology is unmodified, so the two runs are directly
comparable. See tim-moe/docs/moe_gemma_comparison.md for the side-by-side
writeup.

Reads tim-moe/logs/moe_probe_gemma/routing_raw.json and answers two
falsifiable questions per layer, then overall:

  Q1 (router shift): does TIM priming change which experts the router
     selects during generation, vs. a cold prompt? Measured two ways per
     layer: Jaccard overlap of the top-k expert sets (1.0 = identical set,
     0.0 = disjoint) and KL divergence between the mean router probability
     distributions. Computed for both tim_domain-vs-cold and
     tim_random-vs-cold, so a shift can be told apart from mere perturbation.

  Q2 (prediction, the linchpin): do the experts activated during the
     *priming* pass overlap with the experts activated during *generation*
     of the same task, above a cross-task chance baseline (the same
     comparison between two unrelated tasks' generation-phase routing)?
     This decides whether expert prefetching is even possible.

This is a routing-mechanics probe. Nothing here measures or claims accuracy.
"""

import json
import math
from itertools import combinations
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

TIM_MOE_DIR = Path(__file__).resolve().parent
ROUTING_RAW = TIM_MOE_DIR / "logs" / "moe_probe_gemma" / "routing_raw.json"
FIG_DIR = TIM_MOE_DIR / "docs" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)
OUT_PATH = TIM_MOE_DIR / "logs" / "moe_probe_gemma" / "analysis_summary.json"


def jaccard(set_a, set_b) -> float:
    a, b = set(set_a), set(set_b)
    union = a | b
    if not union:
        return 1.0  # both empty: degenerate, treat as no observed disagreement
    return len(a & b) / len(union)


def kl_divergence(p, q, eps: float = 1e-10) -> float:
    if not p or not q:
        return float("nan")
    p = [x + eps for x in p]
    q = [x + eps for x in q]
    sp, sq = sum(p), sum(q)
    p = [x / sp for x in p]
    q = [x / sq for x in q]
    return sum(pi * math.log(pi / qi) for pi, qi in zip(p, q))


def mean(xs):
    xs = [x for x in xs if x is not None and not (isinstance(x, float) and math.isnan(x))]
    return sum(xs) / len(xs) if xs else float("nan")


def load_raw():
    data = json.loads(ROUTING_RAW.read_text())
    return data


def layer_set(task_result, condition, phase, layer):
    entry = task_result.get(condition, {}).get(phase, {}).get(str(layer))
    return entry["expert_set"] if entry else []


def layer_dist(task_result, condition, phase, layer):
    entry = task_result.get(condition, {}).get(phase, {}).get(str(layer))
    return entry["mean_dist"] if entry else []


def q1_router_shift(results, task_ids, layers):
    """Per-layer + overall: jaccard(cold_gen, X_gen) and KL(cold_gen || X_gen)
    for X in {tim_domain, tim_random}."""
    out = {}
    for cond in ["tim_domain", "tim_random"]:
        per_layer = {}
        for layer in layers:
            jaccards, kls = [], []
            for tid in task_ids:
                r = results[tid]
                cold_set = layer_set(r, "cold", "generation", layer)
                x_set = layer_set(r, cond, "generation", layer)
                jaccards.append(jaccard(cold_set, x_set))
                cold_dist = layer_dist(r, "cold", "generation", layer)
                x_dist = layer_dist(r, cond, "generation", layer)
                kls.append(kl_divergence(cold_dist, x_dist))
            per_layer[layer] = {
                "jaccard_mean": mean(jaccards),
                "kl_mean": mean(kls),
                "n_tasks": len(task_ids),
            }
        out[cond] = {
            "per_layer": per_layer,
            "overall_jaccard_mean": mean([v["jaccard_mean"] for v in per_layer.values()]),
            "overall_kl_mean": mean([v["kl_mean"] for v in per_layer.values()]),
        }
    return out


def q2_prediction(results, task_ids, layers):
    """Per-layer + overall: jaccard(priming_set, generation_set) for the same
    task (tim_domain and tim_random), vs. a chance baseline = jaccard between
    generation-phase expert sets of two DIFFERENT tasks, same condition."""
    out = {}
    for cond in ["tim_domain", "tim_random"]:
        per_layer = {}
        for layer in layers:
            same_task_jaccards = []
            for tid in task_ids:
                r = results[tid]
                priming_set = layer_set(r, cond, "priming", layer)
                gen_set = layer_set(r, cond, "generation", layer)
                same_task_jaccards.append(jaccard(priming_set, gen_set))

            cross_task_jaccards = []
            for tid_a, tid_b in combinations(task_ids, 2):
                gen_a = layer_set(results[tid_a], cond, "generation", layer)
                gen_b = layer_set(results[tid_b], cond, "generation", layer)
                cross_task_jaccards.append(jaccard(gen_a, gen_b))

            same_mean = mean(same_task_jaccards)
            chance_mean = mean(cross_task_jaccards)
            per_layer[layer] = {
                "priming_to_generation_jaccard_mean": same_mean,
                "chance_baseline_jaccard_mean": chance_mean,
                "gap": same_mean - chance_mean if not math.isnan(same_mean) and not math.isnan(chance_mean) else float("nan"),
                "n_tasks": len(task_ids),
                "n_task_pairs": len(cross_task_jaccards),
            }
        out[cond] = {
            "per_layer": per_layer,
            "overall_priming_to_generation_jaccard_mean": mean(
                [v["priming_to_generation_jaccard_mean"] for v in per_layer.values()]
            ),
            "overall_chance_baseline_jaccard_mean": mean(
                [v["chance_baseline_jaccard_mean"] for v in per_layer.values()]
            ),
            "overall_gap": mean([v["gap"] for v in per_layer.values()]),
        }
    return out


def make_router_shift_figure(q1, layers):
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    domain_kl = [q1["tim_domain"]["per_layer"][l]["kl_mean"] for l in layers]
    random_kl = [q1["tim_random"]["per_layer"][l]["kl_mean"] for l in layers]
    ax.plot(layers, domain_kl, marker="o", label="tim_domain vs cold", color="tab:blue")
    ax.plot(layers, random_kl, marker="s", label="tim_random vs cold", color="tab:orange")
    ax.set_xlabel("MoE layer index")
    ax.set_ylabel("KL divergence (mean router dist., generation phase)")
    ax.set_title("Q1: does priming shift the router? (per-layer KL vs. cold)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "moe_router_shift_gemma.png", dpi=150)
    plt.close(fig)


def make_prediction_overlap_figure(q2, layers):
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    same = [q2["tim_domain"]["per_layer"][l]["priming_to_generation_jaccard_mean"] for l in layers]
    chance = [q2["tim_domain"]["per_layer"][l]["chance_baseline_jaccard_mean"] for l in layers]
    ax.plot(layers, same, marker="o", label="priming→generation (tim_domain, same task)", color="tab:green")
    ax.plot(layers, chance, marker="x", linestyle="--", label="chance baseline (cross-task generation-generation)", color="gray")
    ax.set_xlabel("MoE layer index")
    ax.set_ylabel("Jaccard overlap of expert sets")
    ax.set_title("Q2: does priming-time routing predict generation-time routing?")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "moe_prediction_overlap_gemma.png", dpi=150)
    plt.close(fig)


def print_table(layers, q1, q2):
    print(f"{'layer':>5} | {'jacc(D,cold)':>12} {'KL(D,cold)':>11} | {'jacc(R,cold)':>12} {'KL(R,cold)':>11} "
          f"| {'p→g(D)':>8} {'chance(D)':>9} {'gap(D)':>7}")
    print("-" * 100)
    for l in layers:
        d1 = q1["tim_domain"]["per_layer"][l]
        r1 = q1["tim_random"]["per_layer"][l]
        d2 = q2["tim_domain"]["per_layer"][l]
        print(f"{l:>5} | {d1['jaccard_mean']:>12.3f} {d1['kl_mean']:>11.3f} | "
              f"{r1['jaccard_mean']:>12.3f} {r1['kl_mean']:>11.3f} | "
              f"{d2['priming_to_generation_jaccard_mean']:>8.3f} {d2['chance_baseline_jaccard_mean']:>9.3f} "
              f"{d2['gap']:>7.3f}")


def main():
    data = load_raw()
    task_ids = data["task_ids"]
    results = data["results"]
    layers = sorted(int(l) for l in data["model_report"]["moe_layer_indices"])

    q1 = q1_router_shift(results, task_ids, layers)
    q2 = q2_prediction(results, task_ids, layers)

    print_table(layers, q1, q2)

    print("\n=== Q1 overall ===")
    for cond in ["tim_domain", "tim_random"]:
        print(f"  {cond}: mean jaccard(gen vs cold gen)={q1[cond]['overall_jaccard_mean']:.3f}  "
              f"mean KL={q1[cond]['overall_kl_mean']:.3f}")

    print("\n=== Q2 overall ===")
    for cond in ["tim_domain", "tim_random"]:
        o = q2[cond]
        print(f"  {cond}: priming→generation jaccard={o['overall_priming_to_generation_jaccard_mean']:.3f}  "
              f"chance baseline={o['overall_chance_baseline_jaccard_mean']:.3f}  "
              f"gap={o['overall_gap']:.3f}")

    q1_shift_detected = q1["tim_domain"]["overall_jaccard_mean"] < 0.999 or q1["tim_random"]["overall_jaccard_mean"] < 0.999
    q1_content_specific = (
        (1 - q1["tim_domain"]["overall_jaccard_mean"]) > 1.25 * (1 - q1["tim_random"]["overall_jaccard_mean"] + 1e-9)
    )
    q2_gap = q2["tim_domain"]["overall_gap"]
    q2_predictive = q2_gap > 0.05  # requires a real margin above chance, not just any positive gap

    verdicts = {
        "q1_router_shift_detected": bool(q1_shift_detected),
        "q1_content_specific": bool(q1_content_specific),
        "q2_gap_domain": q2_gap,
        "q2_predictive_above_chance": bool(q2_predictive),
    }
    print("\n=== VERDICTS ===")
    print(json.dumps(verdicts, indent=2))

    make_router_shift_figure(q1, layers)
    make_prediction_overlap_figure(q2, layers)

    summary = {
        "layers": layers,
        "q1_router_shift": q1,
        "q2_prediction": q2,
        "verdicts": verdicts,
    }
    OUT_PATH.write_text(json.dumps(summary, indent=2))
    print(f"\n[write] {OUT_PATH}")
    print(f"[write] {FIG_DIR / 'moe_router_shift_gemma.png'}")
    print(f"[write] {FIG_DIR / 'moe_prediction_overlap_gemma.png'}")


if __name__ == "__main__":
    main()
