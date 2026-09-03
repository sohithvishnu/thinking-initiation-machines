"""
Rung 5 of the int4-mechanism ladder — tim_entropy: 64 diverse but
semantically meaningless tokens.

Rungs 3a/3b established that a primed prefix helps and that one repeated
token does not. This rung separates the two explanations that survive:

  (a) test-time compute over a diverse latent space — entropy is what matters
  (b) semantic richness — valid, in-distribution tokens are what matters

tim_entropy matches the TIM arms on everything except meaning:
  - exact token budget: 64, as tim_domain / tim_random / neutral_prefix
  - high diversity: unique or near-unique subword fragments, like tim_random
  - same delivery: KV-cache injection

but the tokens come from random consonant-vowel gibberish, filtered against
English words and Python keywords.

  Lands near tim_domain/tim_random -> recovery is compute + entropy; semantics
                                      are not required.
  Lands near neutral_prefix        -> recovery strictly requires semantically
                                      valid, in-distribution tokens.

Usage:
    python experiments/rung5_entropy.py
    python experiments/rung5_entropy.py --limit 5 --no_score   # smoke test
    python experiments/rung5_entropy.py --seeds 0,2            # split across GPUs
"""

import argparse
import builtins
import json
import keyword
import random
import string

from quant_gap import free_kv, load_nf4_checked
from tim.config import LOGS_DIR
from tim.evaluation import load_problems, run_condition
from tim.generation import build_kv_from_token_ids, cache_seq_len
from tim.models import unload_model

MODEL_ID = "Qwen/Qwen3-1.7B"
NOISE_LENGTH = 64
MAX_NEW_TOKENS = 768

# --------------------------------------------------------------------------- #
# Rejection filtering: the gibberish must not accidentally contain real words,
# or the "semantically meaningless" premise of the rung fails.
# --------------------------------------------------------------------------- #

_COMMON_ENGLISH = {
    "the", "and", "for", "are", "but", "not", "you", "all", "can", "had",
    "her", "was", "one", "our", "out", "has", "his", "how", "its", "may",
    "new", "now", "old", "see", "way", "who", "did", "get", "let", "say",
    "she", "too", "use", "man", "day", "got", "him", "big", "end", "put",
    "run", "set", "try", "ask", "own", "ago", "ran", "red", "sit", "top",
    "add", "air", "age", "art", "bad", "bag", "bar", "bed", "bit",
    "box", "bus", "buy", "car", "cat", "cut", "dog", "door", "eat", "eye",
    "far", "few", "fly", "fun", "gun", "hat", "hit", "hot", "ice", "ill",
    "job", "key", "kid", "law", "leg", "lie", "lip", "log", "map", "mix",
    "mud", "net", "nor", "oil", "pay", "pen", "pet", "pop", "pot", "raw",
    "row", "sad", "sea", "sir", "six", "son", "sun", "ten", "tie", "tip",
    "war", "wet", "win", "won", "yes", "yet", "cap", "cup", "due", "ear",
    "egg", "fee", "fit", "fix", "gap", "gas", "god", "guy", "joy", "lab",
    "lay", "led", "lot", "low", "mad", "miss", "mom", "odd", "pig",
    "pin", "pit", "rat", "rid", "rib", "rip", "rob", "rod", "rug",
    "sin", "sky", "sum", "tap", "tax", "tea", "tin", "toe", "tom",
    "toy", "van", "via", "web", "wig", "wax", "yell", "zero", "zone",
    "code", "data", "file", "list", "loop", "node", "null", "pass", "self",
    "sort", "test", "text", "tree", "true", "type", "void", "bool", "char",
    "done", "else", "enum", "eval", "exec", "exit", "from", "func", "hash",
    "help", "init", "iter", "join", "load", "main", "math", "name", "next",
    "none", "open", "path", "port", "read", "repr", "save", "send", "size",
    "step", "stop", "sync", "that", "this", "time", "wait", "with", "work",
}

_REJECT_WORDS = (
    set(keyword.kwlist)
    | {name.lower() for name in dir(builtins) if not name.startswith("_")}
    | _COMMON_ENGLISH
)

_CONSONANTS = "bcdfghjklmnpqrstvwxyz"
_VOWELS = "aeiou"


def _contains_real_word(text: str, min_len: int = 3) -> bool:
    lowered = text.lower()
    for length in range(min_len, min(len(lowered) + 1, 8)):
        for i in range(len(lowered) - length + 1):
            if lowered[i:i + length] in _REJECT_WORDS:
                return True
    return False


def _gibberish_word(rng: random.Random, min_syllables: int = 1, max_syllables: int = 3) -> str:
    """A multi-syllable nonsense word built from CV/CVC/CCV/CVCC patterns."""
    word = ""
    for _ in range(rng.randint(min_syllables, max_syllables)):
        for ch in rng.choice(["cv", "cvc", "ccv", "cvcc"]):
            word += rng.choice(_CONSONANTS if ch == "c" else _VOWELS)
    return word


def _describe(text: str, token_ids: list, seed: int, attempts: int,
              rejected: list, tokenizer, fallback: bool = False) -> dict:
    unique = len(set(token_ids))
    info = {
        "text": text, "token_ids": token_ids, "n_tokens": len(token_ids),
        "unique_tokens": unique, "unique_ratio": unique / len(token_ids),
        "seed": seed, "attempts": attempts, "rejected_words_found": rejected[:20],
        "decoded_sample": [tokenizer.decode([t]) for t in token_ids[:10]],
    }
    if fallback:
        info["fallback"] = True
    return info


def _trim_to_exact(words: list, tokenizer, target_tokens: int):
    """Binary-search the word count that tokenizes to exactly target_tokens."""
    lo, hi = 1, len(words)
    while lo <= hi:
        mid = (lo + hi) // 2
        candidate = " ".join(words[:mid])
        ids = tokenizer.encode(candidate, add_special_tokens=False)
        if len(ids) == target_tokens:
            return candidate, ids
        lo, hi = (mid + 1, hi) if len(ids) < target_tokens else (lo, mid - 1)
    return None, None


def generate_entropy_string(tokenizer, target_tokens: int = 64, seed: int = 42,
                            max_attempts: int = 200) -> dict:
    """
    Gibberish that tokenizes to exactly `target_tokens` tokens and contains no
    recognizable English word or Python keyword.

    Exactness matters: the whole rung rests on tim_entropy having the same
    token budget as the other primed arms, so an approximate length would
    confound entropy with prefix size.
    """
    rng = random.Random(seed)
    rejected = []

    for attempt in range(max_attempts):
        words = []
        while True:
            word = _gibberish_word(rng)
            if _contains_real_word(word):
                rejected.append(word)
                continue
            words.append(word)
            if len(tokenizer.encode(" ".join(words), add_special_tokens=False)) >= target_tokens + 10:
                break

        text, ids = _trim_to_exact(words, tokenizer, target_tokens)
        if text is None:
            continue

        # Final check on the *decoded tokens*, not just the source words: BPE
        # can merge across a word boundary into something readable.
        bad = next((d.strip() for d in (tokenizer.decode([t]) for t in ids)
                    if len(d.strip()) >= 3 and d.strip().lower() in _REJECT_WORDS), None)
        if bad is None:
            return _describe(text, ids, seed, attempt + 1, rejected, tokenizer)
        rejected.append(bad)

    # Fallback: random alphanumeric segments, if CV syllables never landed on
    # an exact token count for this tokenizer.
    print(f"[entropy] CV-syllable approach exhausted {max_attempts} attempts; "
          f"falling back to random alphanumeric")
    rng2 = random.Random(seed + 9999)
    chars = string.ascii_lowercase + string.digits
    for extra in range(max_attempts):
        segments = []
        for _ in range(target_tokens * 2):
            seg = "".join(rng2.choice(chars) for _ in range(rng2.randint(2, 6)))
            if not _contains_real_word(seg):
                segments.append(seg)
        if len(tokenizer.encode(" ".join(segments), add_special_tokens=False)) < target_tokens:
            continue
        text, ids = _trim_to_exact(segments, tokenizer, target_tokens)
        if text is not None:
            return _describe(text, ids, seed, max_attempts + extra + 1,
                             rejected, tokenizer, fallback=True)

    raise RuntimeError(f"Could not generate an entropy string of exactly "
                       f"{target_tokens} tokens after {max_attempts * 2} attempts")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["humaneval", "mbpp"], default="humaneval")
    ap.add_argument("--num_seeds", type=int, default=3)
    ap.add_argument("--seeds", default=None,
                    help="Comma-separated seed list, e.g. '0,2'. Overrides --num_seeds. "
                         "Use with CUDA_VISIBLE_DEVICES to split seeds across GPUs.")
    ap.add_argument("--limit", type=int, default=None, help="First N tasks only (smoke test).")
    ap.add_argument("--no_score", action="store_true", help="Skip EvalPlus scoring.")
    args = ap.parse_args()

    seeds = ([int(s.strip()) for s in args.seeds.split(",")] if args.seeds
             else list(range(args.num_seeds)))
    out_dir = LOGS_DIR / ("mbpp_scale" if args.dataset == "mbpp" else "int4_mechanism")
    out_dir.mkdir(parents=True, exist_ok=True)
    task_items = load_problems(args.dataset, limit=args.limit)

    print(f"=== Rung 5: tim_entropy — {MODEL_ID}, {len(task_items)} {args.dataset} "
          f"tasks, seeds={seeds} ===")

    model, tokenizer, device, quant_report, det = load_nf4_checked(
        MODEL_ID, task_items[0][1]["prompt"])

    report = {"model_id": MODEL_ID, "dataset": args.dataset,
              "n_tasks": len(task_items), "seeds": seeds,
              "quant_report": quant_report, "determinism": det, "entropy_seeds": {}}

    for seed in seeds:
        print(f"\n=== Rung 5: tim_entropy seed {seed} ===")
        entropy = generate_entropy_string(tokenizer, target_tokens=NOISE_LENGTH, seed=seed)
        print(f"[entropy] seed={seed}: n_tokens={entropy['n_tokens']}, "
              f"unique={entropy['unique_tokens']}/{entropy['n_tokens']} "
              f"({entropy['unique_ratio']:.1%}), attempts={entropy['attempts']}")
        print(f"[entropy] sample decoded tokens: {entropy['decoded_sample']}")
        print(f"[entropy] text preview: {entropy['text'][:120]}...")

        # Raw ids are not carried into the report — the summary stats are what
        # the analysis needs, and 64 ids per seed is noise in a JSON diff.
        summary = {k: v for k, v in entropy.items() if k != "token_ids"}

        kv, prime_timing = build_kv_from_token_ids(
            model, entropy["token_ids"], device, mode="tim_entropy",
            n_tokens=entropy["n_tokens"], unique_tokens=entropy["unique_tokens"],
            unique_ratio=entropy["unique_ratio"])
        prime_timing["entropy_info"] = summary

        cache_len = cache_seq_len(kv)
        assert cache_len == NOISE_LENGTH, f"Cache length {cache_len} != expected {NOISE_LENGTH}"
        print(f"[entropy] KV cache built: {cache_len} positions, {prime_timing['total_ms']:.1f}ms")

        metrics = run_condition(
            model, tokenizer, device, f"int4_tim_entropy_seed{seed}", out_dir, task_items,
            dataset=args.dataset, primed_kv=kv, prime_timing=prime_timing,
            max_new_tokens=MAX_NEW_TOKENS, score=not args.no_score)
        metrics["entropy_info"] = summary
        report["entropy_seeds"][f"seed{seed}"] = metrics
        free_kv(kv)

    unload_model(model)

    report_path = out_dir / f"mechanism_gen_report_5_seeds{'_'.join(str(s) for s in seeds)}.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    print("\n" + "=" * 78)
    print("Rung 5 (tim_entropy) generation complete")
    print("=" * 78)
    for seed_key, m in report["entropy_seeds"].items():
        print(f"tim_entropy {seed_key}: pass@1_plus={m.get('pass@1_base_plus_extra')}")
    print(f"\nReport written to: {report_path}")


if __name__ == "__main__":
    main()
