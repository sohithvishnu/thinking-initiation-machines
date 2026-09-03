"""
Prompt construction, instrumented generation, and CUDA timing/memory helpers.

Decoding is greedy (`do_sample=False`) everywhere in this study; determinism
was verified empirically rather than assumed (3 identical baseline rounds,
0.00% std). `enable_thinking=False` keeps Qwen3 out of its reasoning mode so
the KV prefix is the only thing varying between conditions.
"""

import copy
import re
import time

import torch

CODE_FENCE_RE = re.compile(r"```(?:python)?\s*\n(.*?)```", re.DOTALL)


def extract_code(raw_text: str) -> str:
    """First-pass cleanup before EvalPlus's own tree-sitter sanitizer.

    The sanitizer is more robust for malformed/partial code, but still does
    better when it is not handed obvious prose wrapping first.
    """
    match = CODE_FENCE_RE.search(raw_text)
    if match:
        return match.group(1).strip()
    def_idx = raw_text.find("def ")
    if def_idx != -1:
        return raw_text[def_idx:].strip()
    return raw_text.strip()


def build_chat_prompt(tokenizer, problem_prompt: str, prepend_text: str = "") -> str:
    """Chat-template the task. `prepend_text` is the `prompt_control` route:
    the same persona content the TIM arms inject into the cache, delivered
    conventionally through the prompt instead."""
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


def cache_seq_len(kv) -> int:
    if hasattr(kv, "get_seq_length"):
        return kv.get_seq_length()
    return kv[0][0].shape[-2]  # legacy tuple-of-tuples fallback


def cuda_sync() -> None:
    """CUDA is async; without this, perf_counter measures kernel-launch time."""
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def reset_peak_memory() -> None:
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            torch.cuda.reset_peak_memory_stats(i)


def peak_memory_mb() -> float:
    """Summed peak allocation across all devices (device_map='sequential'
    shards a model across every visible GPU)."""
    if not torch.cuda.is_available():
        return 0.0
    return sum(
        torch.cuda.max_memory_allocated(i) for i in range(torch.cuda.device_count())
    ) / (1024 ** 2)


def timed_generate(model, tokenizer, device, problem_prompt: str, max_new_tokens: int,
                   primed_kv=None, prepend_text: str = "") -> dict:
    """
    Generate one task's completion, with a decomposed timing breakdown:

      deepcopy_ms  isolating the primed cache for this task. TIM arms only,
                   and it grows with cache length — a real per-query cost that
                   a production system with prefix caching would not pay.
      generate_ms  prefill + decode inside model.generate().
      new_tokens   tokens actually produced (for tokens/sec).

    The deepcopy is not optional: model.generate(past_key_values=...) appends
    generated tokens to the cache object in place, so reusing one primed cache
    across tasks would leak each task's output into the next task's "warm"
    state and grow it without bound over the loop.
    """
    out = {"deepcopy_ms": 0.0, "generate_ms": 0.0, "new_tokens": 0,
           "prompt_tokens": 0, "cache_len": 0}

    task_kv = None
    if primed_kv is not None:
        cuda_sync()
        t0 = time.perf_counter()
        task_kv = copy.deepcopy(primed_kv)
        cuda_sync()
        out["deepcopy_ms"] = (time.perf_counter() - t0) * 1000.0
        out["cache_len"] = cache_seq_len(task_kv)

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
            # attention_mask must span warm-cache length + new prompt tokens.
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
    out["completion"] = extract_code(tokenizer.decode(generated, skip_special_tokens=True))
    return out


def check_determinism(model, tokenizer, device, prompt: str,
                      max_new_tokens: int = 64, n_runs: int = 3) -> dict:
    """Run the same greedy generation n_runs times; check byte-identical ids.

    Used to gate the 4-bit arms: NF4 with double quantization is not
    guaranteed bit-reproducible, and a non-deterministic arm would make the
    paired McNemar comparisons meaningless.
    """
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
    return {"n_runs": n_runs, "identical": identical,
            "outputs": outputs if not identical else outputs[:1]}


def build_kv_from_token_ids(model, token_ids, device, mode: str, **extra) -> tuple:
    """One forward pass over an explicit token-id sequence -> (kv, timing).

    The escape hatch from TIMPrimer for controls whose whole point is that no
    vocabulary *sampling* happens: the neutral-prefix probe (one token repeated
    64x) and the entropy probe (64 gibberish tokens). Both need the same
    KV-injection delivery path as the TIM arms but a caller-chosen sequence.
    """
    input_ids = torch.tensor([list(token_ids)], device=device, dtype=torch.long)
    cuda_sync()
    t0 = time.perf_counter()
    with torch.no_grad():
        out = model(input_ids=input_ids, use_cache=True)
    cuda_sync()
    ms = (time.perf_counter() - t0) * 1000.0

    kv = out.past_key_values
    timing = {"total_ms": ms, "final_cache_len": cache_seq_len(kv),
              "num_passes": 1, "mode": mode, **extra}
    return kv, timing
