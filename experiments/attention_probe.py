"""
Rung 3c of the int4-mechanism ladder (docs/int4_mechanism.md): does the primed
64-token prefix absorb visibly more attention mass under int4 (NF4) than under
bf16? That is the direct signature the attention-sink hypothesis predicts.

Method: for ~10 tasks, at both precisions, prime with tim_domain (seed 0, the
TIMPrimer config used throughout this study), then greedily generate up to
`--probe_new_tokens` tokens one at a time with `output_attentions=True`. At
each step take the attention row for the newest query position, sum it over
the first PREFIX_LEN key positions versus the full key range, and mean over
heads — one prefix-attention-mass fraction per layer per step. Average over
steps, then over tasks, giving one curve per layer per precision.

`output_attentions` requires `attn_implementation="eager"`; sdpa and flash do
not return per-head weights.

4-bit plus eager attention is a real possibility for failure. Each precision
is therefore attempted independently and any exception is captured into the
output rather than aborting the run — a bf16-only result is still worth
having, and a silent failure would not be.

Usage:
    python experiments/attention_probe.py
"""

import argparse
import copy
import gc
import json
import traceback

import torch

from tim.config import FIGURES_DIR, LOGS_DIR
from tim.evaluation import load_problems
from tim.generation import build_chat_prompt, cache_seq_len
from tim.models import load_model, unload_model
from tim.primer import TIMPrimer
from tim.vocab import PYTHON_DOMAIN_WORDS

OUT_DIR = LOGS_DIR / "int4_mechanism"

MODEL_ID = "Qwen/Qwen3-1.7B"
N_TASKS = 10
PREFIX_LEN = 64
PROBE_NEW_TOKENS = 32


def probe_one_task(model, tokenizer, device, primed_kv, problem_prompt: str,
                   probe_new_tokens: int) -> list[float]:
    """One prefix-attention-mass fraction per layer, averaged over this task's
    generation steps."""
    chat_text = build_chat_prompt(tokenizer, problem_prompt)
    inputs = tokenizer(chat_text, return_tensors="pt", add_special_tokens=False)
    input_ids = inputs["input_ids"].to(device)

    kv = copy.deepcopy(primed_kv)
    total_len = cache_seq_len(kv) + input_ids.shape[-1]

    per_step_layer_fracs = []
    next_input = input_ids

    with torch.no_grad():
        for _ in range(probe_new_tokens):
            attention_mask = torch.ones((1, total_len), dtype=torch.long, device=device)
            out = model(input_ids=next_input, past_key_values=kv,
                        attention_mask=attention_mask, use_cache=True,
                        output_attentions=True)

            layer_fracs = []
            for layer_attn in out.attentions:
                # layer_attn: [batch=1, heads, q_len, k_len]; take the last query row
                last_q = layer_attn[0, :, -1, :]
                take = min(PREFIX_LEN, last_q.shape[-1])
                prefix_mass = last_q[:, :take].sum(dim=-1)
                total_mass = last_q.sum(dim=-1).clamp_min(1e-12)
                layer_fracs.append((prefix_mass / total_mass).mean().item())
            per_step_layer_fracs.append(layer_fracs)

            next_token = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
            kv = out.past_key_values
            next_input = next_token
            total_len += 1
            if next_token.item() == tokenizer.eos_token_id:
                break

    n_steps = len(per_step_layer_fracs)
    return [sum(step[layer] for step in per_step_layer_fracs) / n_steps
            for layer in range(len(per_step_layer_fracs[0]))]


def run_precision(quant: str, task_items, probe_new_tokens: int) -> dict:
    model, tokenizer, device, _ = load_model(MODEL_ID, quant=quant,
                                             attn_implementation="eager")
    primer = TIMPrimer(model, tokenizer, device, noise_length=PREFIX_LEN, num_passes=1)
    domain_ids = primer.get_domain_token_ids_from_words(PYTHON_DOMAIN_WORDS)

    per_task = []
    for task_id, problem in task_items:
        kv = primer.prime(domain_tokens=domain_ids, seed=0)
        layer_fracs = probe_one_task(model, tokenizer, device, kv,
                                     problem["prompt"], probe_new_tokens)
        per_task.append({"task_id": task_id, "per_layer_frac": layer_fracs})
        print(f"  [{quant}] {task_id}: n_layers={len(layer_fracs)} "
              f"mean_frac={sum(layer_fracs) / len(layer_fracs):.4f}")
        del kv
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    unload_model(model)

    n_layers = len(per_task[0]["per_layer_frac"])
    layer_means = [sum(t["per_layer_frac"][layer] for t in per_task) / len(per_task)
                   for layer in range(n_layers)]
    return {"quant": quant, "n_tasks": len(per_task), "n_layers": n_layers,
            "per_task": per_task, "layer_means": layer_means,
            "overall_mean": sum(layer_means) / len(layer_means)}


def plot(results: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 5))
    for key, label, marker in (("bf16", "bf16", "o"), ("int4", "int4/NF4", "s")):
        r = results[key]
        ax.plot(range(r["n_layers"]), r["layer_means"], marker=marker,
                label=f"{label} (mean={r['overall_mean'] * 100:.1f}%)")
    ax.set_xlabel("Layer")
    ax.set_ylabel(f"Attention mass on primed prefix (first {PREFIX_LEN} positions)")
    ax.set_title(f"Attention-sink probe — {MODEL_ID}, tim_domain priming, "
                 f"n={results['n_tasks']} tasks")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig_path = FIGURES_DIR / "attention_sink_int4_vs_bf16.png"
    fig.savefig(fig_path, dpi=150)
    print(f"Plot written to: {fig_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_tasks", type=int, default=N_TASKS)
    ap.add_argument("--probe_new_tokens", type=int, default=PROBE_NEW_TOKENS)
    ap.add_argument("--dataset", choices=["humaneval", "mbpp"], default="humaneval")
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    task_items = load_problems(args.dataset, limit=args.n_tasks)
    print(f"=== Rung 3c: attention-sink probe — {MODEL_ID}, {len(task_items)} tasks, "
          f"prefix_len={PREFIX_LEN}, probe_new_tokens={args.probe_new_tokens} ===")

    results = {"model_id": MODEL_ID, "n_tasks": len(task_items), "prefix_len": PREFIX_LEN,
               "probe_new_tokens": args.probe_new_tokens,
               "task_ids": [t for t, _ in task_items]}
    errors = {}

    for key, quant in (("bf16", "bf16"), ("int4", "nf4")):
        print(f"\n--- {key} ---")
        try:
            results[key] = run_precision(quant, task_items, args.probe_new_tokens)
        except Exception:
            errors[key] = traceback.format_exc()
            print(f"{key} attention probe FAILED:\n{errors[key]}")

    results["errors"] = errors
    raw_path = OUT_DIR / "attention_probe_raw.json"
    with open(raw_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nRaw attention-probe data written to: {raw_path}")

    if "bf16" in results and "int4" in results:
        try:
            plot(results)
        except Exception:
            print(f"Plotting failed:\n{traceback.format_exc()}")
    else:
        print("Skipping plot — need both bf16 and int4 results (see errors above).")

    print("\n" + "=" * 78)
    print("Rung 3c summary")
    print("=" * 78)
    for key in ("bf16", "int4"):
        if key in results:
            print(f"{key}: overall mean prefix-attention-mass = "
                  f"{results[key]['overall_mean'] * 100:.2f}%")
        else:
            print(f"{key}: FAILED — {errors.get(key, 'unknown error')[:200]}")


if __name__ == "__main__":
    main()
