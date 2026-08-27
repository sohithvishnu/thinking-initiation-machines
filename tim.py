"""
TIM evaluation: HumanEval+ under cold / random-noise-warm / domain-warm /
prompt-control conditions, reusing the exact same generation → sanitize →
evaluate pipeline as run_humaneval_evalplus.py.

Conditions
----------
cold            : baseline, no priming (should match your existing results)
tim_random_seed<N> : KV cache primed with pure random vocabulary noise,
                      seed N (repeat across --num_seeds for variance)
tim_domain_seed<N> : KV cache primed with 70/30 domain/random noise,
                      seed N
prompt_control  : SAME domain text as tim_domain, but concatenated into the
                  prompt text instead of injected via KV cache — isolates
                  whether KV-injection matters vs. just adding the same
                  tokens conventionally. Deterministic, run once.

Usage:
    python run_tim_evalplus.py --model Qwen/Qwen3-1.7B --num_seeds 3
"""

import argparse
import copy
import gc
import json
import re
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

ROOT_DIR = Path(__file__).resolve().parent
LOGS_DIR = ROOT_DIR / "logs" / "tim"
LOGS_DIR.mkdir(parents=True, exist_ok=True)

CODE_FENCE_RE = re.compile(r"```(?:python)?\s*\n(.*?)```", re.DOTALL)
BASE_RE = re.compile(r"\(base tests\)\s*\npass@1:\s*([\d.]+)")
PLUS_RE = re.compile(r"\(base \+ extra tests\)\s*\npass@1:\s*([\d.]+)")

# Domain seed text — Python/coding domain, used for both tim_domain (KV
# injection) and prompt_control (same content, prompended into the prompt).
DOMAIN_NAME = "python programming code function algorithm"
DOMAIN_PERSONA_TEXT = (
    "You are an expert Python programmer who writes clean, correct, "
    "well-tested functions."
)


def extract_code(raw_text: str) -> str:
    match = CODE_FENCE_RE.search(raw_text)
    if match:
        return match.group(1).strip()
    def_idx = raw_text.find("def ")
    if def_idx != -1:
        return raw_text[def_idx:].strip()
    return raw_text.strip()


def load_model(model_id: str):
    model_dir = model_path_for(model_id)
    if not model_dir.exists() or not (model_dir / "config.json").exists():
        download_result = download_model(model_id)
        if download_result.get("status") == "failed":
            raise RuntimeError(f"Failed to download {model_id}: {download_result.get('error')}")

    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True, trust_remote_code=True)
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


def build_chat_prompt(tokenizer, problem_prompt: str, prepend_text: str = "") -> str:
    content = (
        "Complete the following Python function. "
        "Return ONLY the code in a single ```python fenced block, "
        "with no explanation before or after.\n\n"
        f"{prepend_text}{problem_prompt}"
    )
    messages = [{"role": "user", "content": content}]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False,
    )


def generate_completion(
    model, tokenizer, device, problem_prompt: str, max_new_tokens: int,
    past_key_values=None, prepend_text: str = "",
) -> str:
    """
    Greedy decoding throughout (matches baseline determinism decision).
    If past_key_values is provided, generation continues from that warm
    cache instead of starting cold.
    """
    chat_text = build_chat_prompt(tokenizer, problem_prompt, prepend_text=prepend_text)
    inputs = tokenizer(chat_text, return_tensors="pt", add_special_tokens=False)
    input_ids = inputs["input_ids"].to(device)

    gen_kwargs = dict(
        max_new_tokens=max_new_tokens,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        do_sample=False,
    )

    if past_key_values is not None:
        # attention_mask must cover the full sequence: warm-cache length + new prompt tokens.
        cache_len = past_key_values.get_seq_length() if hasattr(past_key_values, "get_seq_length") \
            else past_key_values[0][0].shape[-2]
        full_len = cache_len + input_ids.shape[-1]
        attention_mask = torch.ones((1, full_len), dtype=torch.long, device=device)
        with torch.no_grad():
            output_ids = model.generate(
                input_ids=input_ids,
                past_key_values=past_key_values,
                attention_mask=attention_mask,
                **gen_kwargs,
            )
    else:
        with torch.no_grad():
            output_ids = model.generate(input_ids=input_ids, **gen_kwargs)

    generated_tokens = output_ids[0][input_ids.shape[-1]:]
    raw_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    return extract_code(raw_text)


def sanitize_samples(sample_file: Path) -> Path:
    result = subprocess.run(
        [sys.executable, "-m", "evalplus.sanitize", "--samples", str(sample_file)],
        capture_output=True, text=True,
    )
    print(result.stdout)
    if result.returncode != 0:
        print(f"WARNING: sanitize failed, evaluating raw samples.\n{result.stderr}")
    sanitized_path = sample_file.with_name(sample_file.stem + "-sanitized.jsonl")
    return sanitized_path if sanitized_path.exists() else sample_file


def evaluate_with_evalplus(sample_file: Path, dataset: str = "humaneval") -> dict:
    result = subprocess.run(
        [sys.executable, "-m", "evalplus.evaluate", "--dataset", dataset, "--samples", str(sample_file)],
        capture_output=True, text=True,
    )
    print(result.stdout)
    if result.returncode != 0:
        print(f"WARNING: evalplus.evaluate exited non-zero.\n{result.stderr}")

    scores = {}
    base_match = BASE_RE.search(result.stdout)
    plus_match = PLUS_RE.search(result.stdout)
    if base_match:
        scores["pass@1_base"] = float(base_match.group(1))
    if plus_match:
        scores["pass@1_base_plus_extra"] = float(plus_match.group(1))
    if not scores:
        print("WARNING: could not parse pass@1 — check evalplus stdout format.")
    return scores


def run_condition(
    model, tokenizer, device, condition_name: str, output_dir: Path,
    past_key_values=None, prepend_text: str = "", max_new_tokens: int = 768,
) -> dict:
    problems = get_human_eval_plus()
    task_items = list(problems.items())
    samples = []
    progress = tqdm(total=len(task_items), desc=condition_name, unit="task")

    for task_id, problem in task_items:
        # past_key_values is mutated in place by generate() (new tokens get
        # appended to it), so every task must start from a fresh copy of the
        # primed cache — reusing the same object would let each task's
        # generation leak into the next one's "warm" cache and grow unbounded
        # over the loop.
        task_kv = copy.deepcopy(past_key_values) if past_key_values is not None else None
        completion = generate_completion(
            model, tokenizer, device, problem["prompt"],
            max_new_tokens=max_new_tokens,
            past_key_values=task_kv,
            prepend_text=prepend_text,
        )
        samples.append({"task_id": task_id, "completion": completion})
        progress.update(1)
    progress.close()

    sample_file = output_dir / f"{condition_name}.jsonl"
    write_jsonl(str(sample_file), samples)
    sanitized_file = sanitize_samples(sample_file)
    scores = evaluate_with_evalplus(sanitized_file, dataset="humaneval")

    result_record = {"condition": condition_name, "sample_file": str(sample_file), **scores}
    with open(output_dir / f"{condition_name}_results.json", "w") as f:
        json.dump(result_record, f, indent=2)

    print(f"{condition_name}: {scores}")
    return scores


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=None,
                         help="Run a single model only, e.g. Qwen/Qwen3-1.7B. "
                              "If omitted, runs every model in config.qwen_models.")
    parser.add_argument("--num_seeds", type=int, default=3)
    parser.add_argument("--noise_length", type=int, default=64)
    parser.add_argument("--max_new_tokens", type=int, default=768)
    parser.add_argument("--skip_cold", action="store_true",
                         help="Skip the cold condition (reuse existing baseline results instead)")
    args = parser.parse_args()

    models_to_run = [args.model] if args.model else list(qwen_models)

    print("=" * 70)
    print(f"TIM full run — {len(models_to_run)} model(s), {args.num_seeds} seed(s) each")
    print(f"Models: {models_to_run}")
    print("=" * 70)

    run_started = time.time()
    all_models_results = {}

    for m_idx, model_id in enumerate(models_to_run, start=1):
        model_started = time.time()
        print(f"\n{'#'*70}\n# MODEL {m_idx}/{len(models_to_run)}: {model_id}\n{'#'*70}")

        model_results = run_model(model_id, args)
        all_models_results[model_id] = model_results

        elapsed = time.time() - model_started
        print(f"\n[{model_id}] done in {elapsed/60:.1f} min")

        # Write the running combined summary after EVERY model, not just at
        # the end — a multi-hour run should never lose completed work if it
        # crashes or is interrupted partway through.
        combined_path = LOGS_DIR / "all_models_combined_results.json"
        with open(combined_path, "w") as f:
            json.dump(all_models_results, f, indent=2)
        print(f"Combined results so far written to: {combined_path}")

    total_elapsed = time.time() - run_started
    print("\n" + "=" * 70)
    print(f"FULL RUN COMPLETE — {len(models_to_run)} model(s) in {total_elapsed/60:.1f} min")
    print("=" * 70)
    for model_id, results in all_models_results.items():
        print(f"\n{model_id}:")
        for cond, scores in results.items():
            print(f"  {cond}: {scores}")

    print(f"\nFull combined results: {LOGS_DIR / 'all_models_combined_results.json'}")


def run_model(model_id: str, args) -> dict:
    """Run all TIM conditions for a single model and return its results dict.
    Model is loaded once and unloaded at the end, regardless of how many
    conditions/seeds run against it."""
    safe_name = model_id.replace("/", "_")
    output_dir = LOGS_DIR / safe_name
    output_dir.mkdir(parents=True, exist_ok=True)

    model, tokenizer, device = load_model(model_id)
    primer = TIMPrimer(model, tokenizer, device, noise_length=args.noise_length)
    domain_token_ids = primer.get_domain_token_ids(DOMAIN_NAME)

    results = {}

    if not args.skip_cold:
        results["cold"] = run_condition(
            model, tokenizer, device, "cold", output_dir,
            past_key_values=None, prepend_text="",
            max_new_tokens=args.max_new_tokens,
        )

    results["prompt_control"] = run_condition(
        model, tokenizer, device, "prompt_control", output_dir,
        past_key_values=None,
        prepend_text=DOMAIN_PERSONA_TEXT + "\n\n",
        max_new_tokens=args.max_new_tokens,
    )

    for seed in range(args.num_seeds):
        print(f"\n=== {model_id} | seed {seed} ===")

        kv_random = primer.prime(domain_tokens=None, seed=seed)
        results[f"tim_random_seed{seed}"] = run_condition(
            model, tokenizer, device, f"tim_random_seed{seed}", output_dir,
            past_key_values=kv_random, prepend_text="",
            max_new_tokens=args.max_new_tokens,
        )
        del kv_random
        torch.cuda.empty_cache()

        kv_domain = primer.prime(domain_tokens=domain_token_ids, seed=seed)
        results[f"tim_domain_seed{seed}"] = run_condition(
            model, tokenizer, device, f"tim_domain_seed{seed}", output_dir,
            past_key_values=kv_domain, prepend_text="",
            max_new_tokens=args.max_new_tokens,
        )
        del kv_domain
        torch.cuda.empty_cache()

    with open(output_dir / "combined_results.json", "w") as f:
        json.dump(results, f, indent=2)

    unload_model(model)
    return results


if __name__ == "__main__":
    main()