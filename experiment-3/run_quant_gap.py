"""
Quantization-recovery study: can KV-cache priming recover accuracy that
4-bit quantization removes?

2x2 design (plus prompt_control at both precisions):

              bf16          4-bit (bnb NF4)
    cold        A               C
    tim_domain  B               D
    prompt_control  E           F

damage   = A - C   (what 4-bit costs; gated on this before running B/D/E/F)
recovery = D - C   (what TIM buys on the quantized model)
success  = D >= A  (quantized+primed reaches full-precision cold)

Reuses, rather than reimplements:
  - the model-loading pattern from tim.py / Evalplus_Baseline.py, extended
    here with an explicit --quant {bf16,nf4} switch. Nothing else about
    model loading changes.
  - TIMPrimer, imported directly from experiment-2/tim_primer.py (that file
    has no local cross-imports, so it is safe to import across directories).
  - the chat template / extract_code / sanitize / evaluate helpers and the
    PYTHON_DOMAIN_WORDS / DOMAIN_PERSONA_TEXT constants, copied verbatim
    from experiment-2/tim.py. experiment-2/tim.py already duplicates these
    from the root tim.py rather than importing across directories (there is
    no package structure tying root/experiment-2/experiment-3 together), so
    this follows the same established convention rather than inventing a
    new one.
  - the per-task timing/memory instrumentation (timed_generate, run_condition,
    peak_memory_mb) is adapted directly from experiment-2/exp-2.py's
    functions of the same name. That file's own name is not a valid Python
    module identifier (`exp-2.py`), so it can't be `import`-ed by name; the
    logic is reproduced here rather than routed through importlib tricks.

Usage:
    # Step 1 — gate (run this first, alone)
    python experiment-3/run_quant_gap.py --stage gate --model Qwen/Qwen3-1.7B

    # Step 2 — full 2x2, only if the gate passes
    python experiment-3/run_quant_gap.py --stage full --model Qwen/Qwen3-1.7B --num_seeds 3
"""

import argparse
import builtins
import copy
import gc
import json
import keyword
import math
import re
import statistics
import subprocess
import sys
import time
from pathlib import Path

import torch
from evalplus.data import get_human_eval_plus, get_mbpp_plus, write_jsonl
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from tqdm import tqdm

ROOT_DIR = Path(__file__).resolve().parent.parent
EXP2_DIR = ROOT_DIR / "experiment-2"

sys.path.insert(0, str(ROOT_DIR))
from utils import download_model, model_path_for  # noqa: E402  root utils — models/ lives at repo root

sys.path.insert(0, str(EXP2_DIR))
from tim_primer import TIMPrimer  # noqa: E402  self-contained, no local cross-imports

LOGS_DIR = ROOT_DIR / "logs" / "quant_gap"
LOGS_DIR.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------------------------------- #
# Copied verbatim from experiment-2/tim.py — see module docstring for why.
# --------------------------------------------------------------------------- #

CODE_FENCE_RE = re.compile(r"```(?:python)?\s*\n(.*?)```", re.DOTALL)
BASE_RE = re.compile(r"\(base tests\)\s*\npass@1:\s*([\d.]+)")
PLUS_RE = re.compile(r"\(base \+ extra tests\)\s*\npass@1:\s*([\d.]+)")

PYTHON_DOMAIN_WORDS = (
    list(keyword.kwlist)
    + [name for name in dir(builtins) if not name.startswith("_")]
    + [
        "__init__", "__str__", "__repr__", "__len__", "__eq__", "__call__",
        "self", "cls", "->", "==", "!=", ">=", "<=", "+=", "-=",
        "return", "yield", "raise", "except", "finally", "assert",
        "def", "class", "lambda", "async", "await",
    ]
)

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


def evaluate_with_evalplus(sample_file: Path, dataset: str) -> dict:
    result = subprocess.run(
        [sys.executable, "-m", "evalplus.evaluate", "--dataset", dataset, "--samples", str(sample_file)],
        capture_output=True, text=True,
    )
    print(result.stdout)
    if result.returncode != 0:
        raise RuntimeError(
            f"evalplus.evaluate exited non-zero for {sample_file} (dataset={dataset}). "
            f"Refusing to report a partial/silent pass@1. stderr:\n{result.stderr}"
        )
    scores = {}
    base_match = BASE_RE.search(result.stdout)
    plus_match = PLUS_RE.search(result.stdout)
    if base_match:
        scores["pass@1_base"] = float(base_match.group(1))
    if plus_match:
        scores["pass@1_base_plus_extra"] = float(plus_match.group(1))
    if not scores:
        raise RuntimeError(
            f"Could not parse pass@1 from evalplus stdout for {sample_file} — "
            f"refusing to report a missing score silently.\nstdout was:\n{result.stdout}"
        )

    # Hard gate: refuse to report a pass@1 unless every canonical task was
    # actually scored (n_scored == n_canonical). evalplus.evaluate() itself
    # asserts len(completion_id) == len(problems) internally and would have
    # raised (caught above) if the *task-id count* mismatched at read time —
    # but that assertion can still pass while individual "eval" entries are
    # short of the full set (e.g. a scoring subprocess crash mid-flight that
    # still wrote a truncated eval_results.json). Re-check post-hoc against
    # the on-disk eval_results.json so this class of bug cannot recur silently
    # (see docs/mbpp_scale.md — this exact silent-continuation bug produced
    # 7 unscored MBPP+ conditions from concurrent evalplus subprocesses).
    eval_results_path = sample_file.with_name(sample_file.stem + ".eval_results.json")
    if not eval_results_path.exists():
        legacy_path = sample_file.with_name(sample_file.stem.replace("-sanitized", "") + "_eval_results.json")
        eval_results_path = legacy_path if legacy_path.exists() else eval_results_path
    n_canonical = len(get_mbpp_plus() if dataset == "mbpp" else get_human_eval_plus())
    if eval_results_path.exists():
        with open(eval_results_path) as f:
            n_scored = len(json.load(f).get("eval", {}))
    else:
        n_scored = 0
    if n_scored != n_canonical:
        raise RuntimeError(
            f"Refusing to report pass@1 for {sample_file}: n_scored={n_scored} != "
            f"n_canonical={n_canonical} (dataset={dataset}). eval_results.json="
            f"{eval_results_path} (exists={eval_results_path.exists()})."
        )
    scores["n_scored"] = n_scored
    scores["n_canonical"] = n_canonical
    return scores


# --------------------------------------------------------------------------- #
# Timing/memory helpers — adapted from experiment-2/exp-2.py.
# --------------------------------------------------------------------------- #

def cuda_sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def reset_peak_memory():
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            torch.cuda.reset_peak_memory_stats(i)


def peak_memory_mb() -> float:
    if not torch.cuda.is_available():
        return 0.0
    return sum(
        torch.cuda.max_memory_allocated(i) for i in range(torch.cuda.device_count())
    ) / (1024 ** 2)


# --------------------------------------------------------------------------- #
# Quantization-aware model loading
# --------------------------------------------------------------------------- #

def load_model_with_quant(model_id: str, quant: str, use_double_quant: bool = True):
    """quant in {'bf16', 'nf4'}. Returns (model, tokenizer, device, quant_report)."""
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
        )
        quant_report = {"quant": "bf16", "bnb_config": None}
    elif quant == "nf4":
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=use_double_quant,
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_dir, local_files_only=True, trust_remote_code=True,
            device_map=device_map, quantization_config=bnb_config,
        )
        quant_report = {
            "quant": "nf4",
            "bnb_config": {
                "load_in_4bit": True,
                "bnb_4bit_quant_type": "nf4",
                "bnb_4bit_compute_dtype": "bfloat16",
                "bnb_4bit_use_double_quant": use_double_quant,
            },
        }
    else:
        raise ValueError(f"unknown quant {quant!r}")

    device = next(model.parameters()).device
    model.eval()

    param_dtypes = sorted({str(p.dtype) for p in model.parameters()})
    quant_report["param_dtypes_loaded"] = param_dtypes
    quant_report["device_map"] = device_map
    print(f"[load] {model_id} quant={quant} param_dtypes={param_dtypes} device={device}")
    return model, tokenizer, device, quant_report


def unload_model(model):
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def check_determinism(model, tokenizer, device, prompt: str, max_new_tokens: int = 64, n_runs: int = 3) -> dict:
    """Run n_runs greedy generations of the same prompt; check byte-identical output ids."""
    chat_text = build_chat_prompt(tokenizer, prompt)
    inputs = tokenizer(chat_text, return_tensors="pt", add_special_tokens=False)
    input_ids = inputs["input_ids"].to(device)
    gen_kwargs = dict(
        max_new_tokens=max_new_tokens,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        do_sample=False,
    )
    outputs = []
    for _ in range(n_runs):
        with torch.no_grad():
            out_ids = model.generate(input_ids=input_ids, **gen_kwargs)
        outputs.append(out_ids[0][input_ids.shape[-1]:].tolist())
    identical = all(o == outputs[0] for o in outputs[1:])
    return {"n_runs": n_runs, "identical": identical, "outputs": outputs if not identical else outputs[:1]}


# --------------------------------------------------------------------------- #
# Per-task generation + condition runner
# (copy.deepcopy cache isolation — see root tim.py's run_condition for why:
#  model.generate(past_key_values=...) mutates the cache in place, so every
#  task must start from a fresh copy of the primed cache)
# --------------------------------------------------------------------------- #

def timed_generate(model, tokenizer, device, problem_prompt, max_new_tokens, primed_kv=None, prepend_text=""):
    out = {"deepcopy_ms": 0.0, "generate_ms": 0.0, "new_tokens": 0, "prompt_tokens": 0, "cache_len": 0}
    task_kv = None
    if primed_kv is not None:
        cuda_sync()
        t0 = time.perf_counter()
        task_kv = copy.deepcopy(primed_kv)
        cuda_sync()
        out["deepcopy_ms"] = (time.perf_counter() - t0) * 1000.0
        out["cache_len"] = (
            task_kv.get_seq_length() if hasattr(task_kv, "get_seq_length") else task_kv[0][0].shape[-2]
        )

    chat_text = build_chat_prompt(tokenizer, problem_prompt, prepend_text=prepend_text)
    inputs = tokenizer(chat_text, return_tensors="pt", add_special_tokens=False)
    input_ids = inputs["input_ids"].to(device)
    out["prompt_tokens"] = int(input_ids.shape[-1])

    gen_kwargs = dict(
        max_new_tokens=max_new_tokens,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        do_sample=False,
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


def run_condition(model, tokenizer, device, condition_name, output_dir, task_items, dataset="humaneval",
                   primed_kv=None, prepend_text="", max_new_tokens=768,
                   prime_timing=None, score=True):
    reset_peak_memory()
    samples, per_task = [], []

    warm_id, warm_problem = task_items[0]
    timed_generate(model, tokenizer, device, warm_problem["prompt"], max_new_tokens=16,
                   primed_kv=primed_kv, prepend_text=prepend_text)

    cuda_sync()
    wall_start = time.perf_counter()
    for task_id, problem in tqdm(task_items, desc=condition_name, unit="task"):
        r = timed_generate(model, tokenizer, device, problem["prompt"], max_new_tokens=max_new_tokens,
                            primed_kv=primed_kv, prepend_text=prepend_text)
        samples.append({"task_id": task_id, "completion": r.pop("completion")})
        per_task.append(r)
    cuda_sync()
    wall_ms = (time.perf_counter() - wall_start) * 1000.0

    def agg(key):
        vals = [d[key] for d in per_task]
        return {
            "mean": statistics.mean(vals), "median": statistics.median(vals),
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
        "primed_cache_len": (prime_timing or {}).get("final_cache_len", 0),
        "generate_ms": agg("generate_ms"),
        "deepcopy_ms": agg("deepcopy_ms"),
        "new_tokens": agg("new_tokens"),
        "prompt_tokens": agg("prompt_tokens"),
        "total_new_tokens": total_new_tokens,
        "tokens_per_sec": (total_new_tokens / (total_gen_ms / 1000.0)) if total_gen_ms else 0.0,
        "per_task_ms_mean": (total_gen_ms + total_copy_ms) / max(len(per_task), 1),
        "amortized_total_ms": prime_ms + total_gen_ms + total_copy_ms,
        "amortized_per_task_ms": (prime_ms + total_gen_ms + total_copy_ms) / max(len(per_task), 1),
        "wall_ms": wall_ms,
        "peak_memory_mb": peak_memory_mb(),
    }

    sample_file = output_dir / f"{condition_name}.jsonl"
    write_jsonl(str(sample_file), samples)

    if score:
        sanitized = sanitize_samples(sample_file)
        metrics.update(evaluate_with_evalplus(sanitized, dataset=dataset))

    with open(output_dir / f"{condition_name}_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"{condition_name}: pass@1_base={metrics.get('pass@1_base')} "
          f"pass@1_plus={metrics.get('pass@1_base_plus_extra')} "
          f"per_task_ms={metrics['per_task_ms_mean']:.1f} peak_mb={metrics['peak_memory_mb']:.1f}")
    return metrics


def wilson_ci(k: int, n: int, z: float = 1.959963985):
    if n == 0:
        return 0.0, 0.0, 0.0
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = (z * math.sqrt((p * (1 - p) + z * z / (4 * n)) / n)) / denom
    return p, max(0.0, center - half), min(1.0, center + half)


# --------------------------------------------------------------------------- #
# Step 1: gate — A (bf16 cold) vs C (nf4 cold), full 164, scored.
# --------------------------------------------------------------------------- #

def run_gate(model_id: str, args):
    safe = model_id.replace("/", "_")
    output_dir = (ROOT_DIR / "logs" / "mbpp_scale" if args.dataset == "mbpp" else LOGS_DIR / safe) / "gate"
    output_dir.mkdir(parents=True, exist_ok=True)


    if args.dataset == "mbpp":
        problems = get_mbpp_plus()
    else:
        problems = get_human_eval_plus()

    task_items = list(problems.items())
    if args.limit:
        task_items = task_items[: args.limit]

    report = {"model_id": model_id, "n_tasks": len(task_items)}

    print(f"\n=== Gate | A: bf16 cold | {model_id} ===")
    model, tokenizer, device, quant_report_a = load_model_with_quant(model_id, "bf16")
    report["A_quant_report"] = quant_report_a
    results_a = run_condition(model, tokenizer, device, "A_bf16_cold", output_dir, task_items, dataset=args.dataset,
                               max_new_tokens=args.max_new_tokens, score=not args.no_score)
    report["A"] = results_a
    unload_model(model)

    print(f"\n=== Gate | C: nf4 cold | {model_id} ===")
    use_double_quant = True
    model, tokenizer, device, quant_report_c = load_model_with_quant(
        model_id, "nf4", use_double_quant=use_double_quant)
    det_prompt = task_items[0][1]["prompt"]
    det_check = check_determinism(model, tokenizer, device, det_prompt)
    print(f"[determinism] nf4 double_quant={use_double_quant}: identical={det_check['identical']}")
    if not det_check["identical"]:
        print("[determinism] NOT identical — reloading with bnb_4bit_use_double_quant=False (batch size already 1)")
        unload_model(model)
        use_double_quant = False
        model, tokenizer, device, quant_report_c = load_model_with_quant(
            model_id, "nf4", use_double_quant=use_double_quant)
        det_check_2 = check_determinism(model, tokenizer, device, det_prompt)
        print(f"[determinism] nf4 double_quant=False retry: identical={det_check_2['identical']}")
        det_check = {"first_attempt": det_check, "retry_double_quant_false": det_check_2}
    report["C_determinism"] = det_check
    report["C_quant_report"] = quant_report_c
    results_c = run_condition(model, tokenizer, device, "C_nf4_cold", output_dir, task_items, dataset=args.dataset,
                               max_new_tokens=args.max_new_tokens, score=not args.no_score)
    report["C"] = results_c
    unload_model(model)

    a_plus = results_a.get("pass@1_base_plus_extra")
    c_plus = results_c.get("pass@1_base_plus_extra")
    gate = {}
    if a_plus is not None and c_plus is not None:
        n = len(task_items)
        k_a, k_c = round(a_plus * n), round(c_plus * n)
        pa, lo_a, hi_a = wilson_ci(k_a, n)
        pc, lo_c, hi_c = wilson_ci(k_c, n)
        damage_pp = (a_plus - c_plus) * 100
        gate = {
            "A_pass_at_1_plus": a_plus, "A_k": k_a, "A_n": n, "A_wilson_ci": [lo_a, hi_a],
            "C_pass_at_1_plus": c_plus, "C_k": k_c, "C_n": n, "C_wilson_ci": [lo_c, hi_c],
            "damage_pp": damage_pp,
            "gate_passed": damage_pp >= 5.0,
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
        print(f"GATE {'PASSED' if gate['gate_passed'] else 'FAILED'} (threshold: damage >= 5.0 pp)")
    else:
        print("Could not compute gate — pass@1 scores missing (check evalplus output above).")
    print(f"peak_memory_mb: A={results_a.get('peak_memory_mb'):.1f}  C={results_c.get('peak_memory_mb'):.1f}")
    print(f"Report written to: {report_path}")

    return report


# --------------------------------------------------------------------------- #
# Step 2: full 2x2 + prompt_control at both precisions. Only run once the
# gate has passed.
# --------------------------------------------------------------------------- #

def run_full(model_id: str, args):
    safe = model_id.replace("/", "_")
    output_dir = (ROOT_DIR / "logs" / "mbpp_scale" if args.dataset == "mbpp" else LOGS_DIR / safe) / "full"
    output_dir.mkdir(parents=True, exist_ok=True)


    if args.dataset == "mbpp":
        problems = get_mbpp_plus()
    else:
        problems = get_human_eval_plus()

    task_items = list(problems.items())
    if args.limit:
        task_items = task_items[: args.limit]

    report = {"model_id": model_id, "n_tasks": len(task_items), "num_seeds": args.num_seeds}

    print(f"\n=== Full 2x2 | bf16 arms (B, E) | {model_id} ===")
    model, tokenizer, device, quant_report_bf16 = load_model_with_quant(model_id, "bf16")
    report["bf16_quant_report"] = quant_report_bf16

    primer = TIMPrimer(model, tokenizer, device, noise_length=args.noise_length, num_passes=args.num_passes, chain_mode="reseed")
    domain_ids = primer.get_domain_token_ids_from_words(PYTHON_DOMAIN_WORDS)

    if not args.skip_prompt_control:
        report["E"] = run_condition(model, tokenizer, device, "E_bf16_prompt_control", output_dir, task_items, dataset=args.dataset,
                                 prepend_text=DOMAIN_PERSONA_TEXT + "\n\n",
                                 max_new_tokens=args.max_new_tokens, score=not args.no_score)

    report["B"] = {}
    for seed in range(args.num_seeds):
        kv, ptime = primer.prime(domain_tokens=domain_ids, seed=seed, collect_timing=True)
        name = f"B_bf16_tim_domain_seed{seed}" + (f"_pass{args.num_passes}" if args.num_passes > 1 else "")
        report["B"][f"seed{seed}"] = run_condition(
            model, tokenizer, device, name, output_dir, task_items,
            primed_kv=kv, prime_timing=ptime,
            max_new_tokens=args.max_new_tokens, score=not args.no_score,
        )
        del kv
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    unload_model(model)

    print(f"\n=== Full 2x2 | nf4 arms (D, F) | {model_id} ===")
    model, tokenizer, device, quant_report_nf4 = load_model_with_quant(model_id, "nf4")
    report["nf4_quant_report"] = quant_report_nf4

    primer = TIMPrimer(model, tokenizer, device, noise_length=args.noise_length, num_passes=args.num_passes, chain_mode="reseed")
    domain_ids = primer.get_domain_token_ids_from_words(PYTHON_DOMAIN_WORDS)

    report["F"] = run_condition(model, tokenizer, device, "F_nf4_prompt_control", output_dir, task_items, dataset=args.dataset,
                                 prepend_text=DOMAIN_PERSONA_TEXT + "\n\n",
                                 max_new_tokens=args.max_new_tokens, score=not args.no_score)

    report["D"] = {}
    for seed in range(args.num_seeds):
        kv, ptime = primer.prime(domain_tokens=domain_ids, seed=seed, collect_timing=True)
        name = f"D_nf4_tim_domain_seed{seed}"
        report["D"][f"seed{seed}"] = run_condition(
            model, tokenizer, device, name, output_dir, task_items,
            primed_kv=kv, prime_timing=ptime,
            max_new_tokens=args.max_new_tokens, score=not args.no_score,
        )
        del kv
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

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
    ap.add_argument("--noise_length", type=int, default=64)
    ap.add_argument("--dataset", type=str, choices=["humaneval", "mbpp"], default="humaneval")
    ap.add_argument("--num_passes", type=int, default=1)
    ap.add_argument("--skip_prompt_control", action="store_true")
    ap.add_argument("--max_new_tokens", type=int, default=768)
    ap.add_argument("--limit", type=int, default=None,
                     help="Use only the first N tasks (for smoke-testing the pipeline).")
    ap.add_argument("--no_score", action="store_true",
                     help="Skip EvalPlus scoring (pure timing/smoke run).")
    args = ap.parse_args()

    print("=" * 78)
    print(f"Quantization gap study — stage={args.stage} model={args.model}")
    print("=" * 78)

    if args.stage == "gate":
        run_gate(args.model, args)
    else:
        run_full(args.model, args)


if __name__ == "__main__":
    main()
