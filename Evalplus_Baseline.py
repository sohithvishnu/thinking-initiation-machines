import gc
import json
import re
import subprocess
import sys
from pathlib import Path

import torch
from evalplus.data import get_human_eval_plus, write_jsonl
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

from config import qwen_models
from utils import download_model, model_path_for


ROOT_DIR = Path(__file__).resolve().parent
LOGS_DIR = ROOT_DIR / "logs" / "evalplus_baseline"
LOGS_DIR.mkdir(exist_ok=True)

CODE_FENCE_RE = re.compile(r"```(?:python)?\s*\n(.*?)```", re.DOTALL)

# EvalPlus prints results as two-line blocks to stdout, e.g.:
#   humaneval (base tests)
#   pass@1: 0.848
#   humaneval+ (base + extra tests)
#   pass@1: 0.805
# Matched independent of dataset name (humaneval/mbpp) so this works for
# both --dataset values without changes. NOTE: this format is not
# officially versioned/guaranteed by EvalPlus — if a future release
# changes it again, the fallback warning below will fire and tell you.
BASE_RE = re.compile(r"\(base tests\)\s*\npass@1:\s*([\d.]+)")
PLUS_RE = re.compile(r"\(base \+ extra tests\)\s*\npass@1:\s*([\d.]+)")


# ---------------------------------------------------------------------------
# Code extraction: first-pass cleanup before handing off to EvalPlus's own
# tree-sitter-based sanitizer (evalplus.sanitize), which is more robust than
# a regex for malformed/partial code but still benefits from not receiving
# obvious prose wrapping first.
# ---------------------------------------------------------------------------
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
            raise RuntimeError(
                f"Failed to download {model_id} into {model_dir}: {download_result.get('error', 'unknown error')}"
            )

    if not (model_dir / "config.json").exists():
        raise FileNotFoundError(
            f"Missing model files in {model_dir}. Run the model download step first."
        )

    tokenizer = AutoTokenizer.from_pretrained(
        model_dir, local_files_only=True, trust_remote_code=True,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    device_map = "auto" if torch.cuda.is_available() else None
    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        local_files_only=True,
        trust_remote_code=True,
        device_map=device_map,
        torch_dtype="auto",
    )

    device = next(model.parameters()).device
    model.eval()
    return model, tokenizer, device


def unload_model(model):
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def build_chat_prompt(tokenizer, problem_prompt: str) -> str:
    messages = [{
        "role": "user",
        "content": (
            "Complete the following Python function. "
            "Return ONLY the code in a single ```python fenced block, "
            "with no explanation before or after.\n\n"
            f"{problem_prompt}"
        ),
    }]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False,
    )


def generate_completion(
    model, tokenizer, device, problem_prompt: str, max_new_tokens: int,
    temperature: float, top_p: float, do_sample: bool,
) -> str:
    chat_text = build_chat_prompt(tokenizer, problem_prompt)
    inputs = tokenizer(chat_text, return_tensors="pt", add_special_tokens=False)
    inputs = {key: value.to(device) for key, value in inputs.items()}

    gen_kwargs = dict(
        max_new_tokens=max_new_tokens,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        do_sample=do_sample,
    )
    if do_sample:
        gen_kwargs["temperature"] = temperature
        gen_kwargs["top_p"] = top_p

    with torch.no_grad():
        output_ids = model.generate(**inputs, **gen_kwargs)

    generated_tokens = output_ids[0][inputs["input_ids"].shape[-1]:]
    raw_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    return extract_code(raw_text)


def sanitize_samples(sample_file: Path) -> Path:
    """Run EvalPlus's tree-sitter-based sanitizer over the raw completions.
    Produces {sample_file.stem}-sanitized.jsonl. More robust than our own
    regex for edge cases (stray trailing prose, partial code blocks) that
    slip past extract_code()."""
    result = subprocess.run(
        [sys.executable, "-m", "evalplus.sanitize", "--samples", str(sample_file)],
        capture_output=True, text=True,
    )
    print(result.stdout)
    if result.returncode != 0:
        print(f"WARNING: sanitize step failed, evaluating raw samples instead.\n{result.stderr}")
        return sample_file

    sanitized_path = sample_file.with_name(sample_file.stem + "-sanitized.jsonl")
    return sanitized_path if sanitized_path.exists() else sample_file


def evaluate_with_evalplus(sample_file: Path, dataset: str = "humaneval") -> dict:
    """Runs evalplus.evaluate as a subprocess and parses the reported
    pass@1 for both 'Base' (original HumanEval tests) and 'Base + Extra'
    (the augmented HumanEval+ tests) from stdout."""
    result = subprocess.run(
        [sys.executable, "-m", "evalplus.evaluate",
         "--dataset", dataset, "--samples", str(sample_file)],
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
        print("WARNING: could not parse pass@1 from evalplus.evaluate output; "
              "the stdout format may have changed again — check the raw "
              "output printed above and update BASE_RE/PLUS_RE accordingly.")
    return scores


def run_humaneval(
    model_id: str, num_samples_per_task: int,
    max_new_tokens: int, temperature: float, top_p: float, do_sample: bool,
):
    problems = get_human_eval_plus()
    model, tokenizer, device = load_model(model_id)

    samples = []
    task_items = list(problems.items())
    total_generations = len(task_items) * num_samples_per_task
    progress = tqdm(total=total_generations, desc=f"{model_id} generations", unit="sample")

    for task_id, problem in task_items:
        for _ in range(num_samples_per_task):
            completion = generate_completion(
                model, tokenizer, device, problem["prompt"],
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=do_sample,
            )
            samples.append({"task_id": task_id, "completion": completion})
            progress.update(1)

    progress.close()
    unload_model(model)

    safe_model_name = model_id.replace("/", "_")
    sample_file = LOGS_DIR / f"humaneval_{safe_model_name}.jsonl"
    write_jsonl(str(sample_file), samples)
    print(f"Wrote {len(samples)} samples to {sample_file}")

    sanitized_file = sanitize_samples(sample_file)
    scores = evaluate_with_evalplus(sanitized_file, dataset="humaneval")

    result_record = {"model_id": model_id, "sample_file": str(sample_file), **scores}
    results_log = LOGS_DIR / f"humaneval_{safe_model_name}_evalplus_results.json"
    with open(results_log, "w") as f:
        json.dump(result_record, f, indent=2)

    print(f"{model_id}: {scores}")
    return scores


def main():
    all_results = {}
    for model_id in tqdm(qwen_models, desc="Models", unit="model"):
        print(f"Running HumanEval+ for model: {model_id}")
        all_results[model_id] = run_humaneval(
            model_id=model_id,
            num_samples_per_task=1,
            max_new_tokens=768,
            temperature=0.0,
            top_p=1.0,
            do_sample=False,
        )

    
    print("Summary (pass@1_base = original HumanEval, pass@1_base_plus_extra = HumanEval+)")
    
    for model_id, scores in all_results.items():
        print(f"  {model_id}: {scores}")


if __name__ == "__main__":
    main()