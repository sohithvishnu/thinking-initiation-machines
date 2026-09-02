"""
Generates every NEW cell needed for the int4-mechanism ladder
(docs/int4_mechanism.md), on top of what already exists on disk:

  existing: logs/quant_gap/Qwen_Qwen3-1.7B/gate/C_nf4_cold  (int4 cold)
  existing: logs/quality_quant/int4_tim_domain_seed{0,1,2}  (int4 + tim_domain)
  existing: logs/tim/Qwen_Qwen3-1.7B/{cold,tim_domain_seed*}  (bf16, both conds)

  NEW in this run:
    Rung 2  int4 + prompt_control        (persona text prepended, no KV
                                           injection — DOMAIN_PERSONA_TEXT,
                                           same route the bf16 study used)
                                           -> logs/int4_mechanism/int4_prompt_control*
    Rung 3a int4 + tim_random, 3 seeds   (domain_tokens=None -> 100% uniform
                                           random noise; content-vs-sink test)
                                           -> logs/int4_mechanism/int4_tim_random_seed{0,1,2}*
    Rung 3b int4 + neutral_prefix        (64 copies of one repeated, ordinary
                                           non-special token; no domain/random
                                           vocabulary content at all — the
                                           cleanest sink-vs-content discriminator)
                                           -> logs/int4_mechanism/int4_neutral_prefix*
    Rung 4  int4 + tim_domain, seeds 3-7 (5 more seeds, extending the existing
                                           seeds 0-2 to a full 8-seed distribution)
                                           -> logs/quality_quant/int4_tim_domain_seed{3..7}*
                                           (same directory/naming as the existing
                                           3 seeds, so later analysis just globs)

Reuses load_model_with_quant / unload_model / check_determinism / run_condition /
TIMPrimer / PYTHON_DOMAIN_WORDS / DOMAIN_PERSONA_TEXT from
experiment-3/run_quant_gap.py unchanged — nothing here reimplements model
loading, generation, sanitization, or scoring. The only new logic is
build_repeated_token_kv(), a direct one-shot forward pass over a repeated
single token to build a KV cache with no TIMPrimer noise-sampling involved
(TIMPrimer's own noise generator always samples from the vocabulary, which is
exactly the "content" this rung needs to rule out).

Usage:
    python experiment-3/run_int4_mechanism.py
"""

import gc
import json
import sys
import time
from pathlib import Path

import torch
from evalplus.data import get_human_eval_plus, get_mbpp_plus

EXP3_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(EXP3_DIR))
from run_quant_gap import (  # noqa: E402
    load_model_with_quant, unload_model, check_determinism, run_condition,
    TIMPrimer, PYTHON_DOMAIN_WORDS, DOMAIN_PERSONA_TEXT,
)

ROOT_DIR = EXP3_DIR.parent
OUT_DIR_NEW = ROOT_DIR / "logs" / "mbpp_scale" if "mbpp" in sys.argv else ROOT_DIR / "logs" / "int4_mechanism"
OUT_DIR_NEW.mkdir(parents=True, exist_ok=True)
OUT_DIR_TIM = ROOT_DIR / "logs" / "mbpp_scale" if "mbpp" in sys.argv else ROOT_DIR / "logs" / "quality_quant"  # existing dir — seeds 3-7 land here too

MODEL_ID = "Qwen/Qwen3-1.7B"
NOISE_LENGTH = 64
MAX_NEW_TOKENS = 768


def pick_neutral_token(tokenizer) -> dict:
    """Pick one ordinary, non-special repeated token for the pure-sink probe.
    Prefers BOS if the tokenizer defines one; else a single-token newline;
    else pad. Returns {'token_id', 'source', 'decoded'} so the choice is
    logged, not silently assumed."""
    if tokenizer.bos_token_id is not None:
        return {"token_id": tokenizer.bos_token_id, "source": "bos_token_id",
                "decoded": tokenizer.decode([tokenizer.bos_token_id])}
    nl_ids = tokenizer.encode("\n", add_special_tokens=False)
    if len(nl_ids) == 1:
        return {"token_id": nl_ids[0], "source": "newline_single_token",
                "decoded": tokenizer.decode(nl_ids)}
    if tokenizer.pad_token_id is not None:
        return {"token_id": tokenizer.pad_token_id, "source": "pad_token_id",
                "decoded": tokenizer.decode([tokenizer.pad_token_id])}
    raise RuntimeError("No usable neutral token found (no BOS, multi-token newline, no pad).")


def build_repeated_token_kv(model, tokenizer, device, token_id: int, length: int = 64):
    """One forward pass over `length` copies of a single token id -> KV cache.
    No TIMPrimer noise sampling, no vocabulary content beyond one repeated
    token — isolates position/repetition from content for the sink probe."""
    input_ids = torch.tensor([[token_id] * length], device=device, dtype=torch.long)

    def _sync():
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    _sync()
    t0 = time.perf_counter()
    with torch.no_grad():
        out = model(input_ids=input_ids, use_cache=True)
    _sync()
    ms = (time.perf_counter() - t0) * 1000.0

    kv = out.past_key_values
    cache_len = kv.get_seq_length() if hasattr(kv, "get_seq_length") else kv[0][0].shape[-2]
    timing = {
        "total_ms": ms, "final_cache_len": cache_len, "num_passes": 1,
        "mode": "neutral_prefix_repeated_token", "token_id": token_id, "length": length,
    }
    return kv, timing


def free_kv(kv):
    del kv
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None, help="Use only first N tasks (smoke test).")
    ap.add_argument("--no_score", action="store_true", help="Skip EvalPlus scoring (timing-only smoke test).")
    ap.add_argument("--dataset", type=str, choices=["humaneval", "mbpp"], default="humaneval")
    ap.add_argument("--num_passes", type=int, default=1)
    ap.add_argument("--rungs", default="2,3a,3b,4",
                     help="Comma list of rungs to run this invocation, e.g. '3a,3b' or '4' — "
                          "lets remaining work be split across two GPUs as separate processes "
                          "(CUDA_VISIBLE_DEVICES pins each to one physical GPU).")
    args = ap.parse_args()
    rungs = set(x.strip() for x in args.rungs.split(","))


    if args.dataset == "mbpp":
        problems = get_mbpp_plus()
    else:
        problems = get_human_eval_plus()

    task_items = list(problems.items())
    if args.limit:
        task_items = task_items[: args.limit]
    score = not args.no_score
    print(f"=== int4 mechanism ladder generation — {MODEL_ID}, {len(task_items)} tasks — rungs={sorted(rungs)} ===")

    use_double_quant = True
    model, tokenizer, device, quant_report = load_model_with_quant(
        MODEL_ID, "nf4", use_double_quant=use_double_quant)

    det = check_determinism(model, tokenizer, device, task_items[0][1]["prompt"])
    print(f"[determinism] nf4 double_quant={use_double_quant}: identical={det['identical']}")
    if not det["identical"]:
        print("[determinism] NOT identical — reloading with bnb_4bit_use_double_quant=False")
        unload_model(model)
        use_double_quant = False
        model, tokenizer, device, quant_report = load_model_with_quant(
            MODEL_ID, "nf4", use_double_quant=use_double_quant)
        det2 = check_determinism(model, tokenizer, device, task_items[0][1]["prompt"])
        print(f"[determinism] retry double_quant=False: identical={det2['identical']}")
        det = {"first_attempt": det, "retry_double_quant_false": det2}
        if not det2["identical"]:
            print("[determinism] STILL not identical after retry — proceeding anyway, "
                  "flagging this loudly in the report.")

    primer = TIMPrimer(model, tokenizer, device, noise_length=NOISE_LENGTH, num_passes=args.num_passes, chain_mode="reseed")
    domain_ids = primer.get_domain_token_ids_from_words(PYTHON_DOMAIN_WORDS)

    report = {
        "model_id": MODEL_ID, "n_tasks": len(task_items),
        "quant_report": quant_report, "determinism": det,
    }

    # ---- Rung 2: int4 + prompt_control (context, no KV injection) --------
    if "2" in rungs:
        print("\n=== Rung 2: int4 + prompt_control ===")
        report["prompt_control"] = run_condition(
            model, tokenizer, device, "int4_prompt_control", OUT_DIR_NEW, task_items,
            prepend_text=DOMAIN_PERSONA_TEXT + "\n\n",
            max_new_tokens=MAX_NEW_TOKENS, score=score,
        )

    # ---- Rung 3a: int4 + tim_random, 3 seeds (content vs sink) -----------
    if "3a" in rungs:
        print("\n=== Rung 3a: int4 + tim_random (3 seeds) ===")
        report["tim_random"] = {}
        for seed in range(3):
            kv, ptime = primer.prime(domain_tokens=None, seed=seed, collect_timing=True)
            name = f"int4_tim_random_seed{seed}"
            report["tim_random"][f"seed{seed}"] = run_condition(
                model, tokenizer, device, name, OUT_DIR_NEW, task_items,
                primed_kv=kv, prime_timing=ptime,
                max_new_tokens=MAX_NEW_TOKENS, score=score,
            )
            free_kv(kv)

    # ---- Rung 3b: int4 + neutral_prefix (pure sink baseline) -------------
    if "3b" in rungs:
        print("\n=== Rung 3b: int4 + neutral_prefix ===")
        neutral = pick_neutral_token(tokenizer)
        print(f"[neutral_prefix] using token_id={neutral['token_id']} "
              f"source={neutral['source']} decoded={neutral['decoded']!r} x{NOISE_LENGTH}")
        kv, ptime = build_repeated_token_kv(model, tokenizer, device, neutral["token_id"], length=NOISE_LENGTH)
        ptime["neutral_token"] = neutral
        report["neutral_prefix"] = run_condition(
            model, tokenizer, device, "int4_neutral_prefix", OUT_DIR_NEW, task_items,
            primed_kv=kv, prime_timing=ptime,
            max_new_tokens=MAX_NEW_TOKENS, score=score,
        )
        report["neutral_prefix_token"] = neutral
        free_kv(kv)

    # ---- Rung 4: int4 + tim_domain, 5 additional seeds (3-7) -------------
    if "4" in rungs:
        print("\n=== Rung 4: int4 + tim_domain, seeds 3-7 ===")
        report["tim_domain_extra"] = {}
        for seed in range(3, 8):
            kv, ptime = primer.prime(domain_tokens=domain_ids, seed=seed, collect_timing=True)
            name = f"int4_tim_domain_seed{seed}"
            report["tim_domain_extra"][f"seed{seed}"] = run_condition(
                model, tokenizer, device, name, OUT_DIR_TIM, task_items,
                primed_kv=kv, prime_timing=ptime,
                max_new_tokens=MAX_NEW_TOKENS, score=score,
            )
            free_kv(kv)

    unload_model(model)

    rungs_tag = "_".join(sorted(rungs))
    report_path = OUT_DIR_NEW / f"mechanism_gen_report_{rungs_tag}.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    print("\n" + "=" * 78)
    print("int4 mechanism ladder generation complete")
    print("=" * 78)
    if "prompt_control" in report:
        print(f"prompt_control: pass@1_plus={report['prompt_control'].get('pass@1_base_plus_extra')}")
    for seed, m in report.get("tim_random", {}).items():
        print(f"tim_random {seed}: pass@1_plus={m.get('pass@1_base_plus_extra')}")
    if "neutral_prefix" in report:
        print(f"neutral_prefix: pass@1_plus={report['neutral_prefix'].get('pass@1_base_plus_extra')}")
    for seed, m in report.get("tim_domain_extra", {}).items():
        print(f"tim_domain {seed}: pass@1_plus={m.get('pass@1_base_plus_extra')}")
    print(f"\nReport written to: {report_path}")


if __name__ == "__main__":
    main()
