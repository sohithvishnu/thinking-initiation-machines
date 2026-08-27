"""
TIM multi-pass sweep + time-efficiency benchmark.

Answers two questions the accuracy-only runs could not:

  Q1 (passes)     Does iterative KV accumulation help, and how does it scale
                  with pass count?  Sweeps 1,2,3 (the range the earlier pilot
                  suggested was useful) plus 6,7 and 10,12 to probe whether
                  longer chains keep helping, plateau, or degrade.

  Q2 (efficiency) What does priming actually cost, and what does it cost per
                  query?  A primed cache is built ONCE but every task pays a
                  deepcopy plus attention over a longer cache — so more passes
                  make each query more expensive.  This measures where that
                  crossover sits versus `cold` and `prompt_control`.

Timing methodology
------------------
* torch.cuda.synchronize() around every measured region. CUDA is async;
  without syncing, perf_counter measures kernel-launch time, not compute.
* A warmup task runs before timing starts (first CUDA call pays one-off
  autotune/alloc costs that would otherwise land on task 1).
* Peak memory summed across all visible devices (device_map="sequential"
  shards across both GPUs).
* Priming cost is reported separately from per-task cost, then combined into
  an amortized total so the trade-off is explicit rather than hidden.

Usage
-----
    # quick timing sweep (recommended first — ~30 tasks, all pass counts)
    python run_tim_passes_timing.py --model Qwen/Qwen3-1.7B --limit 30

    # full scored sweep on every model in config.qwen_models
    python run_tim_passes_timing.py

    # custom pass counts / reseed chain variant
    python run_tim_passes_timing.py --passes 1 3 7 12 --chain_mode reseed
"""

import argparse
import copy
import gc
import json
import re
import statistics
import subprocess
import sys
import time
from pathlib import Path

import torch
from evalplus.data import get_human_eval_plus, write_jsonl
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

from config import qwen_models
from utils import download_model, model_path_for
from tim_primer import TIMPrimer
from tim import (
    PYTHON_DOMAIN_WORDS,
    DOMAIN_PERSONA_TEXT,
    extract_code,
    build_chat_prompt,
    sanitize_samples,
    evaluate_with_evalplus,
)

ROOT_DIR = Path(__file__).resolve().parent
LOGS_DIR = ROOT_DIR / "logs" / "tim_passes"
LOGS_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_PASSES = [1, 2, 3, 6, 7, 10, 12]


# --------------------------------------------------------------------------- #
# Timing helpers
# --------------------------------------------------------------------------- #

def cuda_sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def reset_peak_memory():
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            torch.cuda.reset_peak_memory_stats(i)


def peak_memory_mb() -> float:
    """Summed peak allocation across all devices (model may be sharded)."""
    if not torch.cuda.is_available():
        return 0.0
    return sum(
        torch.cuda.max_memory_allocated(i) for i in range(torch.cuda.device_count())
    ) / (1024 ** 2)


# --------------------------------------------------------------------------- #
# Model loading
# --------------------------------------------------------------------------- #

def load_model(model_id: str):
    model_dir = model_path_for(model_id)
    if not model_dir.exists() or not (model_dir / "config.json").exists():
        res = download_model(model_id)
        if res.get("status") == "failed":
            raise RuntimeError(f"Failed to download {model_id}: {res.get('error')}")

    tokenizer = AutoTokenizer.from_pretrained(
        model_dir, local_files_only=True, trust_remote_code=True
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    device_map = "sequential" if torch.cuda.is_available() else None
    model = AutoModelForCausalLM.from_pretrained(
        model_dir, local_files_only=True, trust_remote_code=True,
        device_map=device_map, torch_dtype="auto",
    )
    device = next(model.parameters()).device
    model.eval()
    return model, tokenizer, device


def unload_model(model):
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# --------------------------------------------------------------------------- #
# Instrumented generation
# --------------------------------------------------------------------------- #

def timed_generate(
    model, tokenizer, device, problem_prompt: str, max_new_tokens: int,
    primed_kv=None, prepend_text: str = "",
) -> dict:
    """
    One task. Returns the completion plus a per-stage timing breakdown.

    Stages measured separately so the efficiency picture is decomposable:
      deepcopy_ms  - cost of isolating the primed cache for this task
                     (TIM-only; grows with cache length)
      generate_ms  - prefill + decode inside model.generate()
      new_tokens   - tokens actually produced (for tokens/sec)
    """
    out = {"deepcopy_ms": 0.0, "generate_ms": 0.0, "new_tokens": 0,
           "prompt_tokens": 0, "cache_len": 0}

    # ---- per-task cache isolation (see run_tim_evalplus for why) ----
    task_kv = None
    if primed_kv is not None:
        cuda_sync()
        t0 = time.perf_counter()
        task_kv = copy.deepcopy(primed_kv)
        cuda_sync()
        out["deepcopy_ms"] = (time.perf_counter() - t0) * 1000.0
        out["cache_len"] = (
            task_kv.get_seq_length() if hasattr(task_kv, "get_seq_length")
            else task_kv[0][0].shape[-2]
        )

    chat_text = build_chat_prompt(tokenizer, problem_prompt, prepend_text=prepend_text)
    inputs = tokenizer(chat_text, return_tensors="pt", add_special_tokens=False)
    input_ids = inputs["input_ids"].to(device)
    out["prompt_tokens"] = int(input_ids.shape[-1])

    gen_kwargs = dict(
        max_new_tokens=max_new_tokens,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        do_sample=False,   # task answering stays greedy, as in the baseline
    )

    cuda_sync()
    t0 = time.perf_counter()
    with torch.no_grad():
        if task_kv is not None:
            full_len = out["cache_len"] + input_ids.shape[-1]
            attention_mask = torch.ones((1, full_len), dtype=torch.long, device=device)
            output_ids = model.generate(
                input_ids=input_ids, past_key_values=task_kv,
                attention_mask=attention_mask, **gen_kwargs,
            )
        else:
            output_ids = model.generate(input_ids=input_ids, **gen_kwargs)
    cuda_sync()
    out["generate_ms"] = (time.perf_counter() - t0) * 1000.0

    generated = output_ids[0][input_ids.shape[-1]:]
    out["new_tokens"] = int(generated.shape[-1])
    raw = tokenizer.decode(generated, skip_special_tokens=True)
    out["completion"] = extract_code(raw)
    return out


def run_condition(
    model, tokenizer, device, condition_name: str, output_dir: Path,
    task_items, primed_kv=None, prepend_text: str = "",
    max_new_tokens: int = 768, prime_timing: dict = None, score: bool = True,
) -> dict:
    """Run one condition over task_items, collecting timing + (optionally) accuracy."""
    reset_peak_memory()
    samples, per_task = [], []

    # ---- warmup: absorb one-off CUDA init/autotune cost ----
    warm_id, warm_problem = task_items[0]
    timed_generate(model, tokenizer, device, warm_problem["prompt"],
                   max_new_tokens=16,
                   primed_kv=primed_kv, prepend_text=prepend_text)

    cuda_sync()
    wall_start = time.perf_counter()

    for task_id, problem in tqdm(task_items, desc=condition_name, unit="task", leave=False):
        r = timed_generate(
            model, tokenizer, device, problem["prompt"],
            max_new_tokens=max_new_tokens,
            primed_kv=primed_kv, prepend_text=prepend_text,
        )
        samples.append({"task_id": task_id, "completion": r.pop("completion")})
        per_task.append(r)

    cuda_sync()
    wall_ms = (time.perf_counter() - wall_start) * 1000.0

    def agg(key):
        vals = [d[key] for d in per_task]
        return {
            "mean": statistics.mean(vals),
            "median": statistics.median(vals),
            "stdev": statistics.pstdev(vals) if len(vals) > 1 else 0.0,
            "min": min(vals), "max": max(vals),
        }

    total_new_tokens = sum(d["new_tokens"] for d in per_task)
    total_gen_ms = sum(d["generate_ms"] for d in per_task)
    total_copy_ms = sum(d["deepcopy_ms"] for d in per_task)
    prime_ms = (prime_timing or {}).get("total_ms", 0.0)

    metrics = {
        "condition": condition_name,
        "n_tasks": len(per_task),
        "prime_ms": prime_ms,
        "prime_per_pass_ms": (prime_timing or {}).get("per_pass_ms", []),
        "prime_cache_len_per_pass": (prime_timing or {}).get("per_pass_cache_len", []),
        "primed_cache_len": (prime_timing or {}).get("final_cache_len", 0),
        "generate_ms": agg("generate_ms"),
        "deepcopy_ms": agg("deepcopy_ms"),
        "new_tokens": agg("new_tokens"),
        "prompt_tokens": agg("prompt_tokens"),
        "total_new_tokens": total_new_tokens,
        "tokens_per_sec": (total_new_tokens / (total_gen_ms / 1000.0)) if total_gen_ms else 0.0,
        # per-query cost the user actually feels, excluding one-time priming
        "per_task_ms_mean": (total_gen_ms + total_copy_ms) / max(len(per_task), 1),
        # priming amortized across this many tasks — makes the trade-off explicit
        "amortized_total_ms": prime_ms + total_gen_ms + total_copy_ms,
        "amortized_per_task_ms": (prime_ms + total_gen_ms + total_copy_ms) / max(len(per_task), 1),
        "wall_ms": wall_ms,
        "peak_memory_mb": peak_memory_mb(),
    }

    sample_file = output_dir / f"{condition_name}.jsonl"
    write_jsonl(str(sample_file), samples)

    if score:
        sanitized = sanitize_samples(sample_file)
        metrics.update(evaluate_with_evalplus(sanitized, dataset="humaneval"))

    with open(output_dir / f"{condition_name}_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    return metrics


# --------------------------------------------------------------------------- #
# Per-model sweep
# --------------------------------------------------------------------------- #

def run_model(model_id: str, args) -> dict:
    safe = model_id.replace("/", "_")
    output_dir = LOGS_DIR / safe
    output_dir.mkdir(parents=True, exist_ok=True)

    model, tokenizer, device = load_model(model_id)

    problems = get_human_eval_plus()
    task_items = list(problems.items())
    if args.limit:
        task_items = task_items[: args.limit]
    print(f"  {len(task_items)} tasks | scoring={'on' if args.score else 'off'}")

    results = {}

    # ---- reference baselines (no priming) --------------------------------
    results["cold"] = run_condition(
        model, tokenizer, device, "cold", output_dir, task_items,
        primed_kv=None, prepend_text="",
        max_new_tokens=args.max_new_tokens, score=args.score,
    )
    print(f"    cold           : {results['cold']['per_task_ms_mean']:8.1f} ms/task")

    results["prompt_control"] = run_condition(
        model, tokenizer, device, "prompt_control", output_dir, task_items,
        primed_kv=None, prepend_text=DOMAIN_PERSONA_TEXT + "\n\n",
        max_new_tokens=args.max_new_tokens, score=args.score,
    )
    print(f"    prompt_control : {results['prompt_control']['per_task_ms_mean']:8.1f} ms/task")

    # ---- pass-count sweep -----------------------------------------------
    primer = TIMPrimer(
        model, tokenizer, device,
        noise_length=args.noise_length,
        think_tokens=args.think_tokens,
        chain_mode=args.chain_mode,
        prime_temperature=args.prime_temperature,
        prime_top_k=args.prime_top_k,
    )
    domain_ids = primer.get_domain_token_ids_from_words(PYTHON_DOMAIN_WORDS)

    for n_passes in args.passes:
        primer.num_passes = n_passes
        for arm, dom in (("domain", domain_ids), ("random", None)):
            if arm not in args.arms:
                continue
            for seed in range(args.num_seeds):
                name = f"tim_{arm}_p{n_passes}_seed{seed}"
                kv, ptime = primer.prime(
                    domain_tokens=dom, seed=seed, collect_timing=True
                )
                results[name] = run_condition(
                    model, tokenizer, device, name, output_dir, task_items,
                    primed_kv=kv, prepend_text="",
                    max_new_tokens=args.max_new_tokens,
                    prime_timing=ptime, score=args.score,
                )
                m = results[name]
                print(f"    {name:<28}: {m['per_task_ms_mean']:8.1f} ms/task | "
                      f"prime {m['prime_ms']:7.1f} ms | cache {m['primed_cache_len']:>4}")
                del kv
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        # checkpoint after each pass count so a long run never loses work
        with open(output_dir / "sweep_results.json", "w") as f:
            json.dump(results, f, indent=2)

    unload_model(model)
    return results


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #

def print_report(model_id: str, results: dict):
    print("\n" + "=" * 104)
    print(f"PASS-COUNT & EFFICIENCY REPORT — {model_id}")
    print("=" * 104)
    hdr = (f"{'condition':<28}{'cache':>7}{'prime ms':>10}{'ms/task':>10}"
           f"{'copy ms':>9}{'tok/s':>8}{'amort ms':>10}{'HE':>7}{'HE+':>7}")
    print(hdr)
    print("-" * 104)

    cold = results.get("cold", {})
    base_ms = cold.get("per_task_ms_mean") or 0.0

    for name, m in results.items():
        he = m.get("pass@1_base")
        hep = m.get("pass@1_base_plus_extra")
        print(f"{name:<28}"
              f"{m.get('primed_cache_len', 0):>7}"
              f"{m.get('prime_ms', 0):>10.1f}"
              f"{m.get('per_task_ms_mean', 0):>10.1f}"
              f"{m.get('deepcopy_ms', {}).get('mean', 0):>9.2f}"
              f"{m.get('tokens_per_sec', 0):>8.1f}"
              f"{m.get('amortized_per_task_ms', 0):>10.1f}"
              f"{(f'{he:.1%}' if he is not None else '-'):>7}"
              f"{(f'{hep:.1%}' if hep is not None else '-'):>7}")
    print("-" * 104)

    if base_ms:
        print(f"\nPer-task cost relative to cold ({base_ms:.1f} ms/task):")
        for name, m in results.items():
            if name == "cold":
                continue
            ratio = (m.get("per_task_ms_mean", 0) / base_ms)
            print(f"  {name:<30} {ratio:5.2f}x")

    print("\nInterpretation guide:")
    print("  * ms/task excludes one-time priming — the steady-state per-query cost.")
    print("  * amort ms includes priming spread over this task count; it falls as")
    print("    the task count rises, so compare it only at equal n_tasks.")
    print("  * If ms/task climbs with cache length while HE+ stays flat, extra")
    print("    passes are buying nothing and costing throughput.")
    print("  * prompt_control is the reference any TIM arm must beat to justify")
    print("    the machinery (note: production prefix caching already amortizes")
    print("    prompt prefixes losslessly, so beating cold alone is not enough).")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None,
                    help="Single model; omit to sweep config.qwen_models")
    ap.add_argument("--passes", type=int, nargs="+", default=DEFAULT_PASSES,
                    help=f"Pass counts to sweep (default {DEFAULT_PASSES})")
    ap.add_argument("--arms", nargs="+", default=["domain", "random"],
                    choices=["domain", "random"])
    ap.add_argument("--num_seeds", type=int, default=1,
                    help="Seeds per (pass,arm). Keep at 1 for timing sweeps; "
                         "raise for accuracy claims.")
    ap.add_argument("--noise_length", type=int, default=64)
    ap.add_argument("--think_tokens", type=int, default=32,
                    help="Proto-thought tokens generated per pass after the first")
    ap.add_argument("--chain_mode", default="persistent",
                    choices=["persistent", "reseed"])
    ap.add_argument("--prime_temperature", type=float, default=0.8,
                    help="0 => greedy proto-thoughts (deterministic priming)")
    ap.add_argument("--prime_top_k", type=int, default=50)
    ap.add_argument("--max_new_tokens", type=int, default=768)
    ap.add_argument("--limit", type=int, default=None,
                    help="Use only the first N tasks (recommended for timing runs)")
    ap.add_argument("--no_score", dest="score", action="store_false",
                    help="Skip EvalPlus scoring (pure timing run, much faster)")
    args = ap.parse_args()

    models = [args.model] if args.model else list(qwen_models)

    print("=" * 70)
    print("TIM pass-count sweep + efficiency benchmark")
    print(f"  models      : {models}")
    print(f"  passes      : {args.passes}")
    print(f"  arms        : {args.arms}  seeds={args.num_seeds}")
    print(f"  chain_mode  : {args.chain_mode}  prime_temp={args.prime_temperature}")
    print(f"  noise_length: {args.noise_length}  think_tokens={args.think_tokens}")
    print("=" * 70)

    started = time.time()
    all_results = {}

    for i, model_id in enumerate(models, 1):
        t0 = time.time()
        print(f"\n{'#'*70}\n# MODEL {i}/{len(models)}: {model_id}\n{'#'*70}")
        all_results[model_id] = run_model(model_id, args)
        print(f"\n[{model_id}] done in {(time.time()-t0)/60:.1f} min")
        print_report(model_id, all_results[model_id])

        with open(LOGS_DIR / "all_models_pass_sweep.json", "w") as f:
            json.dump({"config": vars(args), "results": all_results}, f, indent=2)

    print(f"\nSWEEP COMPLETE in {(time.time()-started)/60:.1f} min")
    print(f"Results: {LOGS_DIR / 'all_models_pass_sweep.json'}")


if __name__ == "__main__":
    main()