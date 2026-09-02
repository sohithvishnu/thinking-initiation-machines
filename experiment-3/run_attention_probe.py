"""
Rung 3c of the int4-mechanism ladder (docs/int4_mechanism.md): does the
primed 64-token prefix absorb visibly more attention mass under int4 (NF4)
than under bf16? That is the direct attention-sink signature the sink
hypothesis predicts.

For ~10 tasks, at both precisions, prime with tim_domain (seed=0, same
TIMPrimer config used throughout this study), then greedily generate up to
PROBE_NEW_TOKENS tokens one at a time with output_attentions=True. At each
generation step, take the attention row for the newest query position, sum
it over the first PREFIX_LEN=64 key positions (the primed prefix) versus the
full key range, mean over heads -> one prefix-attention-mass fraction per
layer per step. Average over steps, then over tasks, to get one curve per
layer per precision.

output_attentions requires attn_implementation="eager" (sdpa/flash don't
return per-head weights). experiment-3/run_quant_gap.py's
load_model_with_quant() doesn't expose that argument, and that module's own
docstring explains why cross-directory files in this repo duplicate rather
than share model-loading code (no package structure ties them together) —
so this script follows the same convention with its own small loader instead
of editing the working one. Everything else (TIMPrimer, PYTHON_DOMAIN_WORDS,
build_chat_prompt, download_model/model_path_for) is imported unchanged.

If NF4 + output_attentions raises (a real possibility with 4-bit + eager),
this is caught, reported explicitly, and the script still writes whatever it
was able to gather (falls back to bf16-only, or to nothing, but never
silently drops the failure).

Usage:
    python experiment-3/run_attention_probe.py
"""

import copy
import gc
import json
import sys
import traceback
from pathlib import Path

import torch
from evalplus.data import get_human_eval_plus
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

EXP3_DIR = Path(__file__).resolve().parent
EXP2_DIR = EXP3_DIR.parent / "experiment-2"
ROOT_DIR = EXP3_DIR.parent
sys.path.insert(0, str(EXP3_DIR))
sys.path.insert(0, str(EXP2_DIR))
sys.path.insert(0, str(ROOT_DIR))

from run_quant_gap import build_chat_prompt, PYTHON_DOMAIN_WORDS  # noqa: E402
from tim_primer import TIMPrimer  # noqa: E402
from utils import download_model, model_path_for  # noqa: E402

OUT_DIR = ROOT_DIR / "logs" / "int4_mechanism"
OUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR = ROOT_DIR / "docs" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

MODEL_ID = "Qwen/Qwen3-1.7B"
N_TASKS = 10
PREFIX_LEN = 64
PROBE_NEW_TOKENS = 32


def load_eager(model_id: str, quant: str):
    """quant in {'bf16','nf4'}. Same loading pattern as
    run_quant_gap.load_model_with_quant, but with attn_implementation='eager'
    (required for output_attentions=True) exposed."""
    model_dir = model_path_for(model_id)
    if not model_dir.exists() or not (model_dir / "config.json").exists():
        res = download_model(model_id)
        if res.get("status") == "failed":
            raise RuntimeError(f"Failed to download {model_id}: {res.get('error')}")

    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    device_map = "sequential" if torch.cuda.is_available() else None

    if quant == "bf16":
        model = AutoModelForCausalLM.from_pretrained(
            model_dir, local_files_only=True, trust_remote_code=True,
            device_map=device_map, torch_dtype=torch.bfloat16,
            attn_implementation="eager",
        )
    elif quant == "nf4":
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_dir, local_files_only=True, trust_remote_code=True,
            device_map=device_map, quantization_config=bnb_config,
            attn_implementation="eager",
        )
    else:
        raise ValueError(quant)

    device = next(model.parameters()).device
    model.eval()
    print(f"[load] {model_id} quant={quant} attn_implementation=eager device={device}")
    return model, tokenizer, device


def unload(model):
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def probe_one_task(model, tokenizer, device, primed_kv, problem_prompt: str, probe_new_tokens: int):
    """Returns per_layer_frac: list[float], one prefix-attention-mass
    fraction per layer, averaged over up to probe_new_tokens generation
    steps for this single task."""
    chat_text = build_chat_prompt(tokenizer, problem_prompt)
    inputs = tokenizer(chat_text, return_tensors="pt", add_special_tokens=False)
    input_ids = inputs["input_ids"].to(device)

    kv = copy.deepcopy(primed_kv)
    cache_len = kv.get_seq_length() if hasattr(kv, "get_seq_length") else kv[0][0].shape[-2]
    total_len = cache_len + input_ids.shape[-1]

    per_step_layer_fracs = []  # [step][layer]
    next_input = input_ids

    with torch.no_grad():
        for step in range(probe_new_tokens):
            attention_mask = torch.ones((1, total_len), dtype=torch.long, device=device)
            out = model(
                input_ids=next_input, past_key_values=kv,
                attention_mask=attention_mask, use_cache=True, output_attentions=True,
            )
            layer_fracs = []
            for layer_attn in out.attentions:
                # layer_attn: [batch=1, heads, q_len, k_len] — take last query row
                last_q = layer_attn[0, :, -1, :]  # [heads, k_len]
                k_len = last_q.shape[-1]
                take = min(PREFIX_LEN, k_len)
                prefix_mass = last_q[:, :take].sum(dim=-1)
                total_mass = last_q.sum(dim=-1).clamp_min(1e-12)
                frac = (prefix_mass / total_mass).mean().item()
                layer_fracs.append(frac)
            per_step_layer_fracs.append(layer_fracs)

            next_token = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
            kv = out.past_key_values
            next_input = next_token
            total_len += 1
            if next_token.item() == tokenizer.eos_token_id:
                break

    n_layers = len(per_step_layer_fracs[0])
    per_layer_mean = [
        sum(step[layer] for step in per_step_layer_fracs) / len(per_step_layer_fracs)
        for layer in range(n_layers)
    ]
    return per_layer_mean


def run_precision(quant: str, task_items, probe_new_tokens: int):
    model, tokenizer, device = load_eager(MODEL_ID, quant)
    primer = TIMPrimer(model, tokenizer, device, noise_length=PREFIX_LEN, num_passes=1)
    domain_ids = primer.get_domain_token_ids_from_words(PYTHON_DOMAIN_WORDS)

    per_task_layers = []
    for task_id, problem in task_items:
        kv = primer.prime(domain_tokens=domain_ids, seed=0, collect_timing=False)
        layer_fracs = probe_one_task(model, tokenizer, device, kv, problem["prompt"], probe_new_tokens)
        per_task_layers.append({"task_id": task_id, "per_layer_frac": layer_fracs})
        print(f"  [{quant}] {task_id}: n_layers={len(layer_fracs)} "
              f"mean_frac={sum(layer_fracs)/len(layer_fracs):.4f}")
        del kv
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    unload(model)

    n_layers = len(per_task_layers[0]["per_layer_frac"])
    layer_means = [
        sum(t["per_layer_frac"][layer] for t in per_task_layers) / len(per_task_layers)
        for layer in range(n_layers)
    ]
    return {"quant": quant, "n_tasks": len(per_task_layers), "n_layers": n_layers,
            "per_task": per_task_layers, "layer_means": layer_means,
            "overall_mean": sum(layer_means) / len(layer_means)}


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_tasks", type=int, default=N_TASKS)
    ap.add_argument("--probe_new_tokens", type=int, default=PROBE_NEW_TOKENS)
    args = ap.parse_args()
    probe_new_tokens = args.probe_new_tokens

    problems = get_human_eval_plus()
    task_items = list(problems.items())[: args.n_tasks]
    print(f"=== Rung 3c: attention-sink probe — {MODEL_ID}, {len(task_items)} tasks, "
          f"prefix_len={PREFIX_LEN}, probe_new_tokens={probe_new_tokens} ===")

    results = {"model_id": MODEL_ID, "n_tasks": args.n_tasks, "prefix_len": PREFIX_LEN,
               "probe_new_tokens": probe_new_tokens, "task_ids": [t for t, _ in task_items]}
    errors = {}

    print("\n--- bf16 ---")
    try:
        results["bf16"] = run_precision("bf16", task_items, probe_new_tokens)
    except Exception:
        err = traceback.format_exc()
        print(f"bf16 attention probe FAILED:\n{err}")
        errors["bf16"] = err

    print("\n--- int4 (nf4) ---")
    try:
        results["int4"] = run_precision("nf4", task_items, probe_new_tokens)
    except Exception:
        err = traceback.format_exc()
        print(f"int4 attention probe FAILED:\n{err}")
        errors["int4"] = err

    results["errors"] = errors
    raw_path = OUT_DIR / "attention_probe_raw.json"
    with open(raw_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nRaw attention-probe data written to: {raw_path}")

    if "bf16" in results and "int4" in results:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(figsize=(8, 5))
            ax.plot(range(results["bf16"]["n_layers"]), results["bf16"]["layer_means"],
                    marker="o", label=f"bf16 (mean={results['bf16']['overall_mean']*100:.1f}%)")
            ax.plot(range(results["int4"]["n_layers"]), results["int4"]["layer_means"],
                    marker="s", label=f"int4/NF4 (mean={results['int4']['overall_mean']*100:.1f}%)")
            ax.set_xlabel("Layer")
            ax.set_ylabel(f"Attention mass on primed prefix (first {PREFIX_LEN} positions)")
            ax.set_title(f"Attention-sink probe — {MODEL_ID}, tim_domain priming, "
                         f"n={N_TASKS} tasks")
            ax.legend()
            ax.grid(alpha=0.3)
            fig_path = FIG_DIR / "attention_sink_int4_vs_bf16.png"
            fig.tight_layout()
            fig.savefig(fig_path, dpi=150)
            print(f"Plot written to: {fig_path}")
        except Exception:
            print(f"Plotting failed:\n{traceback.format_exc()}")
    else:
        print("Skipping plot — need both bf16 and int4 results (see errors above).")

    print("\n" + "=" * 78)
    print("Rung 3c summary")
    print("=" * 78)
    for quant in ("bf16", "int4"):
        if quant in results:
            print(f"{quant}: overall mean prefix-attention-mass = {results[quant]['overall_mean']*100:.2f}%")
        else:
            print(f"{quant}: FAILED — {errors.get(quant, 'unknown error')[:200]}")


if __name__ == "__main__":
    main()
