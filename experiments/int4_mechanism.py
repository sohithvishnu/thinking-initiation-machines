"""
The int4-mechanism ladder (docs/int4_mechanism.md): if priming rescues
accuracy on a 4-bit model, *what* about the prefix is doing the work —
its content, or merely its presence as an attention sink?

Each rung removes one more candidate explanation:

  Rung 2  int4 + prompt_control   the persona text delivered through the
                                  prompt instead of the cache. If this
                                  matches, KV injection buys nothing.
  Rung 3a int4 + tim_random       100% uniform random vocabulary noise. If
                                  this matches tim_domain, the *content* of
                                  the domain pool is not what matters.
  Rung 3b int4 + neutral_prefix   64 copies of one ordinary token. No
                                  vocabulary content at all — the cleanest
                                  discriminator between a content effect and
                                  a pure position/sink effect.
  Rung 4  int4 + tim_domain       seeds 3-7, extending the existing seeds 0-2
                                  to a full 8-seed distribution.

Rungs 1 (int4 cold vs bf16 cold) and 3c (attention measurement) are covered
elsewhere: rung 1 by experiments/quant_gap.py's gate, rung 3c by
experiments/attention_probe.py.

Rung 4 writes into the same directory and naming scheme as the existing
seeds 0-2 (`logs/quality_quant/`), so downstream analysis just globs them.

Usage:
    python experiments/int4_mechanism.py
    python experiments/int4_mechanism.py --rungs 3a,3b     # split across GPUs
"""

import argparse
import json

from quant_gap import free_kv, load_nf4_checked
from tim.config import LOGS_DIR
from tim.evaluation import load_problems, run_condition
from tim.generation import build_kv_from_token_ids
from tim.models import unload_model
from tim.primer import TIMPrimer
from tim.vocab import DOMAIN_PERSONA_TEXT, PYTHON_DOMAIN_WORDS

MODEL_ID = "Qwen/Qwen3-1.7B"
NOISE_LENGTH = 64
MAX_NEW_TOKENS = 768
ALL_RUNGS = ("2", "3a", "3b", "4")


def pick_neutral_token(tokenizer) -> dict:
    """One ordinary, non-special repeated token for the pure-sink probe.

    Prefers BOS, else a single-token newline, else pad. The choice is returned
    and logged rather than silently assumed, since it varies by tokenizer and
    the whole rung hinges on the token carrying no domain content.
    """
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
    raise RuntimeError("No usable neutral token (no BOS, multi-token newline, no pad).")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rungs", default=",".join(ALL_RUNGS),
                    help="Comma list of rungs to run this invocation, e.g. '3a,3b' or '4'. "
                         "Lets remaining work be split across GPUs as separate processes "
                         "(CUDA_VISIBLE_DEVICES pins each to one physical GPU).")
    ap.add_argument("--dataset", choices=["humaneval", "mbpp"], default="humaneval")
    ap.add_argument("--num_passes", type=int, default=1)
    ap.add_argument("--limit", type=int, default=None, help="First N tasks only (smoke test).")
    ap.add_argument("--no_score", action="store_true", help="Skip EvalPlus scoring.")
    args = ap.parse_args()

    rungs = {r.strip() for r in args.rungs.split(",")}
    unknown = rungs - set(ALL_RUNGS)
    if unknown:
        ap.error(f"unknown rung(s) {sorted(unknown)}; expected from {list(ALL_RUNGS)}")

    # Rung 4 lands beside the existing tim_domain seeds 0-2; the new rungs get
    # their own directory. The MBPP+ run collapses both into one tree because
    # it is analyzed as a single cross-script grid.
    if args.dataset == "mbpp":
        out_new = out_tim = LOGS_DIR / "mbpp_scale"
    else:
        out_new = LOGS_DIR / "int4_mechanism"
        out_tim = LOGS_DIR / "quality_quant"
    out_new.mkdir(parents=True, exist_ok=True)
    out_tim.mkdir(parents=True, exist_ok=True)

    task_items = load_problems(args.dataset, limit=args.limit)
    print(f"=== int4 mechanism ladder — {MODEL_ID}, {len(task_items)} {args.dataset} "
          f"tasks — rungs={sorted(rungs)} ===")

    model, tokenizer, device, quant_report, det = load_nf4_checked(
        MODEL_ID, task_items[0][1]["prompt"])

    primer = TIMPrimer(model, tokenizer, device, noise_length=NOISE_LENGTH,
                       num_passes=args.num_passes, chain_mode="reseed")
    domain_ids = primer.get_domain_token_ids_from_words(PYTHON_DOMAIN_WORDS)

    common = dict(dataset=args.dataset, max_new_tokens=MAX_NEW_TOKENS,
                  score=not args.no_score)
    report = {"model_id": MODEL_ID, "dataset": args.dataset,
              "n_tasks": len(task_items), "quant_report": quant_report,
              "determinism": det}

    def primed_seeds(key, name_fmt, out_dir, seeds, domain_tokens):
        report[key] = {}
        for seed in seeds:
            kv, prime_timing = primer.prime(
                domain_tokens=domain_tokens, seed=seed, collect_timing=True)
            report[key][f"seed{seed}"] = run_condition(
                model, tokenizer, device, name_fmt.format(seed=seed), out_dir,
                task_items, primed_kv=kv, prime_timing=prime_timing, **common)
            free_kv(kv)

    # ---- Rung 2: prompt_control — context, no KV injection ---------------
    if "2" in rungs:
        print("\n=== Rung 2: int4 + prompt_control ===")
        report["prompt_control"] = run_condition(
            model, tokenizer, device, "int4_prompt_control", out_new, task_items,
            prepend_text=DOMAIN_PERSONA_TEXT + "\n\n", **common)

    # ---- Rung 3a: tim_random — content vs sink ---------------------------
    if "3a" in rungs:
        print("\n=== Rung 3a: int4 + tim_random (3 seeds) ===")
        primed_seeds("tim_random", "int4_tim_random_seed{seed}", out_new,
                     range(3), domain_tokens=None)

    # ---- Rung 3b: neutral_prefix — pure sink baseline --------------------
    if "3b" in rungs:
        print("\n=== Rung 3b: int4 + neutral_prefix ===")
        neutral = pick_neutral_token(tokenizer)
        print(f"[neutral_prefix] token_id={neutral['token_id']} "
              f"source={neutral['source']} decoded={neutral['decoded']!r} x{NOISE_LENGTH}")
        kv, prime_timing = build_kv_from_token_ids(
            model, [neutral["token_id"]] * NOISE_LENGTH, device,
            mode="neutral_prefix_repeated_token",
            token_id=neutral["token_id"], length=NOISE_LENGTH)
        prime_timing["neutral_token"] = neutral
        report["neutral_prefix"] = run_condition(
            model, tokenizer, device, "int4_neutral_prefix", out_new, task_items,
            primed_kv=kv, prime_timing=prime_timing, **common)
        report["neutral_prefix_token"] = neutral
        free_kv(kv)

    # ---- Rung 4: tim_domain, 5 additional seeds (3-7) --------------------
    if "4" in rungs:
        print("\n=== Rung 4: int4 + tim_domain, seeds 3-7 ===")
        primed_seeds("tim_domain_extra", "int4_tim_domain_seed{seed}", out_tim,
                     range(3, 8), domain_tokens=domain_ids)

    unload_model(model)

    report_path = out_new / f"mechanism_gen_report_{'_'.join(sorted(rungs))}.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    print("\n" + "=" * 78)
    print("int4 mechanism ladder generation complete")
    print("=" * 78)
    if "prompt_control" in report:
        print(f"prompt_control: pass@1_plus="
              f"{report['prompt_control'].get('pass@1_base_plus_extra')}")
    for seed, m in report.get("tim_random", {}).items():
        print(f"tim_random {seed}: pass@1_plus={m.get('pass@1_base_plus_extra')}")
    if "neutral_prefix" in report:
        print(f"neutral_prefix: pass@1_plus="
              f"{report['neutral_prefix'].get('pass@1_base_plus_extra')}")
    for seed, m in report.get("tim_domain_extra", {}).items():
        print(f"tim_domain {seed}: pass@1_plus={m.get('pass@1_base_plus_extra')}")
    print(f"\nReport written to: {report_path}")


if __name__ == "__main__":
    main()
