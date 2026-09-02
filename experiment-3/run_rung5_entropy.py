"""
Rung 5 — tim_entropy: 64 diverse, semantically meaningless tokens.

Tests whether the int4 recovery mechanism is driven by:
  (a) test-time compute (FLOPs) over a diverse latent space (entropy), OR
  (b) semantic richness / valid distributional tokens.

The tim_entropy control matches:
  - Exact token budget: 64 tokens (same as tim_domain, tim_random, neutral_prefix)
  - High token diversity: unique or near-unique subword fragments (matches tim_random's diversity)
  - Delivery method: KV-cache injection (same path as tim_domain and tim_random)

BUT strips out semantic meaning:
  - Tokens are generated from random consonant-vowel gibberish strings
  - Verified to NOT contain valid English words or Python keywords
  - Consists of diverse subword fragments that no trained model would treat as meaningful

If tim_entropy rescues int4 (landing near tim_domain/tim_random):
  → Recovery is driven by test-time compute + high latent entropy; semantics not required.

If tim_entropy fails to rescue int4 (landing near neutral_prefix):
  → Recovery strictly requires semantic richness / valid distributional tokens.

Usage:
    python experiment-3/run_rung5_entropy.py
    python experiment-3/run_rung5_entropy.py --limit 5 --no_score   # smoke test
    python experiment-3/run_rung5_entropy.py --num_seeds 1           # quick check
"""

import gc
import hashlib
import json
import keyword
import random
import string
import sys
import time
from pathlib import Path

import torch
from evalplus.data import get_human_eval_plus, get_mbpp_plus

EXP3_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(EXP3_DIR))
from run_quant_gap import (  # noqa: E402
    load_model_with_quant, unload_model, check_determinism, run_condition,
    TIMPrimer,
)

ROOT_DIR = EXP3_DIR.parent
OUT_DIR = ROOT_DIR / "logs" / "mbpp_scale" if "mbpp" in sys.argv else ROOT_DIR / "logs" / "int4_mechanism"
OUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_ID = "Qwen/Qwen3-1.7B"
NOISE_LENGTH = 64
MAX_NEW_TOKENS = 768

# --------------------------------------------------------------------------- #
# English word and Python keyword sets for rejection filtering
# --------------------------------------------------------------------------- #

# Comprehensive set of common English words (lowercase) for rejection.
# We include short words (2-4 chars) that might accidentally appear in
# gibberish, plus Python keywords and builtins.
_PYTHON_KEYWORDS = set(keyword.kwlist)
_PYTHON_BUILTINS = {name.lower() for name in dir(__builtins__) if isinstance(name, str) and not name.startswith("_")}

# Common short English words that gibberish might accidentally produce
_COMMON_ENGLISH = {
    "the", "and", "for", "are", "but", "not", "you", "all", "can", "had",
    "her", "was", "one", "our", "out", "has", "his", "how", "its", "may",
    "new", "now", "old", "see", "way", "who", "did", "get", "let", "say",
    "she", "too", "use", "man", "day", "got", "him", "big", "end", "put",
    "run", "set", "try", "ask", "own", "ago", "ran", "red", "sit", "top",
    "add", "air", "age", "ago", "art", "bad", "bag", "bar", "bed", "bit",
    "box", "bus", "buy", "car", "cat", "cut", "dog", "door", "eat", "eye",
    "far", "few", "fly", "fun", "gun", "hat", "hit", "hot", "ice", "ill",
    "job", "key", "kid", "law", "leg", "lie", "lip", "log", "map", "mix",
    "mud", "net", "nor", "oil", "pay", "pen", "pet", "pop", "pot", "raw",
    "row", "sad", "sea", "sir", "six", "son", "sun", "ten", "tie", "tip",
    "war", "wet", "win", "won", "yes", "yet", "cap", "cup", "due", "ear",
    "egg", "fee", "fit", "fix", "gap", "gas", "god", "guy", "joy", "lab",
    "lay", "led", "lot", "low", "mad", "miss", "mom", "nor", "odd", "pig",
    "pin", "pit", "ran", "rat", "rid", "rib", "rip", "rob", "rod", "rug",
    "sad", "sin", "sky", "sum", "tap", "tax", "tea", "tin", "toe", "tom",
    "toy", "van", "via", "web", "wig", "wax", "yell", "zero", "zone",
    "code", "data", "file", "list", "loop", "node", "null", "pass", "self",
    "sort", "test", "text", "tree", "true", "type", "void", "bool", "char",
    "done", "else", "enum", "eval", "exec", "exit", "from", "func", "hash",
    "help", "init", "iter", "join", "load", "main", "math", "name", "next",
    "none", "open", "path", "port", "read", "repr", "save", "send", "size",
    "step", "stop", "sync", "that", "this", "time", "wait", "with", "work",
}

_REJECT_WORDS = _PYTHON_KEYWORDS | _PYTHON_BUILTINS | _COMMON_ENGLISH


def _contains_real_word(text: str, min_len: int = 3) -> bool:
    """Check if text contains any recognizable English word or Python keyword."""
    text_lower = text.lower()
    # Check substrings of length 3+
    for length in range(min_len, min(len(text_lower) + 1, 8)):
        for i in range(len(text_lower) - length + 1):
            substr = text_lower[i:i + length]
            if substr in _REJECT_WORDS:
                return True
    return False


# --------------------------------------------------------------------------- #
# Entropy gibberish generator
# --------------------------------------------------------------------------- #

# Consonant-vowel syllable patterns that are very unlikely to form real words
_CONSONANTS = "bcdfghjklmnpqrstvwxyz"
_VOWELS = "aeiou"


def _generate_gibberish_syllable(rng: random.Random) -> str:
    """Generate a single CV or CVC nonsense syllable."""
    pattern = rng.choice(["cv", "cvc", "ccv", "cvcc"])
    syllable = ""
    for ch in pattern:
        if ch == "c":
            syllable += rng.choice(_CONSONANTS)
        elif ch == "v":
            syllable += rng.choice(_VOWELS)
    return syllable


def _generate_gibberish_word(rng: random.Random, min_syllables: int = 1, max_syllables: int = 3) -> str:
    """Generate a multi-syllable nonsense word."""
    n_syl = rng.randint(min_syllables, max_syllables)
    word = "".join(_generate_gibberish_syllable(rng) for _ in range(n_syl))
    return word


def generate_entropy_string(tokenizer, target_tokens: int = 64, seed: int = 42,
                            max_attempts: int = 200) -> dict:
    """
    Generate a string of random gibberish that tokenizes to exactly `target_tokens` tokens.

    Strategy:
    1. Generate random consonant-vowel non-words separated by spaces
    2. Tokenize and check length
    3. Trim or extend to hit exactly target_tokens
    4. Verify no real English words or Python keywords appear

    Returns dict with: 'text', 'token_ids', 'n_tokens', 'unique_tokens',
    'unique_ratio', 'seed', 'attempts', 'rejected_words_found'.
    """
    rng = random.Random(seed)

    # Phase 1: Build an overlong gibberish string, then trim to exact token count
    rejected_words = []

    for attempt in range(max_attempts):
        # Generate gibberish words, accumulating until we have enough tokens
        words = []
        while True:
            word = _generate_gibberish_word(rng, 1, 3)
            # Reject words that contain real English
            if _contains_real_word(word):
                rejected_words.append(word)
                continue
            words.append(word)
            candidate = " ".join(words)
            token_ids = tokenizer.encode(candidate, add_special_tokens=False)
            if len(token_ids) >= target_tokens + 10:
                break

        # Binary search for the right number of words to get exactly target_tokens
        lo, hi = 1, len(words)
        best_text = None
        best_ids = None

        while lo <= hi:
            mid = (lo + hi) // 2
            candidate = " ".join(words[:mid])
            token_ids = tokenizer.encode(candidate, add_special_tokens=False)
            n = len(token_ids)
            if n == target_tokens:
                best_text = candidate
                best_ids = token_ids
                break
            elif n < target_tokens:
                lo = mid + 1
            else:
                hi = mid - 1

        if best_text is not None:
            # Final verification: no real words in the decoded tokens
            decoded_tokens = [tokenizer.decode([tid]) for tid in best_ids]
            all_clean = True
            for dt in decoded_tokens:
                dt_stripped = dt.strip()
                if len(dt_stripped) >= 3 and dt_stripped.lower() in _REJECT_WORDS:
                    all_clean = False
                    rejected_words.append(dt_stripped)
                    break

            if all_clean:
                unique_ids = len(set(best_ids))
                return {
                    "text": best_text,
                    "token_ids": best_ids,
                    "n_tokens": len(best_ids),
                    "unique_tokens": unique_ids,
                    "unique_ratio": unique_ids / len(best_ids),
                    "seed": seed,
                    "attempts": attempt + 1,
                    "rejected_words_found": rejected_words[:20],
                    "decoded_sample": decoded_tokens[:10],
                }

    # Fallback: use base64-like random alphanumeric gibberish
    print(f"[entropy] CV-syllable approach exhausted {max_attempts} attempts; falling back to random alphanumeric")
    rng2 = random.Random(seed + 9999)
    chars = string.ascii_lowercase + string.digits
    for fallback_attempt in range(max_attempts):
        # Generate random character strings with spaces
        segments = []
        for _ in range(target_tokens * 2):
            seg_len = rng2.randint(2, 6)
            seg = "".join(rng2.choice(chars) for _ in range(seg_len))
            if not _contains_real_word(seg):
                segments.append(seg)

        candidate = " ".join(segments)
        token_ids = tokenizer.encode(candidate, add_special_tokens=False)

        if len(token_ids) >= target_tokens:
            # Trim by removing trailing words until we get exactly target_tokens
            words = candidate.split()
            lo, hi = 1, len(words)
            while lo <= hi:
                mid = (lo + hi) // 2
                trimmed = " ".join(words[:mid])
                tids = tokenizer.encode(trimmed, add_special_tokens=False)
                if len(tids) == target_tokens:
                    unique_ids = len(set(tids))
                    decoded_tokens = [tokenizer.decode([tid]) for tid in tids]
                    return {
                        "text": trimmed,
                        "token_ids": tids,
                        "n_tokens": len(tids),
                        "unique_tokens": unique_ids,
                        "unique_ratio": unique_ids / len(tids),
                        "seed": seed,
                        "attempts": max_attempts + fallback_attempt + 1,
                        "rejected_words_found": rejected_words[:20],
                        "decoded_sample": decoded_tokens[:10],
                        "fallback": True,
                    }
                elif len(tids) < target_tokens:
                    lo = mid + 1
                else:
                    hi = mid - 1

    raise RuntimeError(f"Could not generate entropy string of exactly {target_tokens} tokens "
                       f"after {max_attempts * 2} attempts")


def build_entropy_kv(model, tokenizer, device, entropy_info: dict):
    """
    Build a KV cache from the entropy gibberish string.
    Same delivery method as tim_random (KV-cache injection via forward pass).
    """
    token_ids = torch.tensor([entropy_info["token_ids"]], device=device, dtype=torch.long)
    assert token_ids.shape[1] == entropy_info["n_tokens"], \
        f"Token count mismatch: {token_ids.shape[1]} vs {entropy_info['n_tokens']}"

    def _sync():
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    _sync()
    t0 = time.perf_counter()
    with torch.no_grad():
        out = model(input_ids=token_ids, use_cache=True)
    _sync()
    ms = (time.perf_counter() - t0) * 1000.0

    kv = out.past_key_values
    cache_len = kv.get_seq_length() if hasattr(kv, "get_seq_length") else kv[0][0].shape[-2]
    timing = {
        "total_ms": ms, "final_cache_len": cache_len, "num_passes": 1,
        "mode": "tim_entropy", "n_tokens": entropy_info["n_tokens"],
        "unique_tokens": entropy_info["unique_tokens"],
        "unique_ratio": entropy_info["unique_ratio"],
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
    ap.add_argument("--num_seeds", type=int, default=3, help="Number of seeds (default: 3).")
    ap.add_argument("--dataset", type=str, choices=["humaneval", "mbpp"], default="humaneval")
    ap.add_argument("--seeds", type=str, default=None,
                     help="Comma-separated list of specific seeds to run, e.g. '0,2'. "
                          "Overrides --num_seeds. Use with CUDA_VISIBLE_DEVICES to split "
                          "seeds across GPUs for parallel execution.")
    args = ap.parse_args()

    if args.seeds is not None:
        seed_list = [int(s.strip()) for s in args.seeds.split(",")]
    else:
        seed_list = list(range(args.num_seeds))


    if args.dataset == "mbpp":
        problems = get_mbpp_plus()
    else:
        problems = get_human_eval_plus()

    task_items = list(problems.items())
    if args.limit:
        task_items = task_items[:args.limit]
    score = not args.no_score
    print(f"=== Rung 5: tim_entropy — {MODEL_ID}, {len(task_items)} tasks, seeds={seed_list} ===")

    # Load model
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

    report = {
        "model_id": MODEL_ID, "n_tasks": len(task_items), "seeds": seed_list,
        "quant_report": quant_report, "determinism": det,
        "entropy_seeds": {},
    }

    # Generate and run for each seed
    for seed in seed_list:
        print(f"\n=== Rung 5: tim_entropy seed {seed} ===")

        # Generate the entropy string for this seed
        entropy_info = generate_entropy_string(tokenizer, target_tokens=NOISE_LENGTH, seed=seed)
        print(f"[entropy] seed={seed}: n_tokens={entropy_info['n_tokens']}, "
              f"unique={entropy_info['unique_tokens']}/{entropy_info['n_tokens']} "
              f"({entropy_info['unique_ratio']:.1%}), "
              f"attempts={entropy_info['attempts']}")
        print(f"[entropy] sample decoded tokens: {entropy_info['decoded_sample']}")
        print(f"[entropy] text preview: {entropy_info['text'][:120]}...")

        # Build KV cache from entropy tokens
        kv, ptime = build_entropy_kv(model, tokenizer, device, entropy_info)
        ptime["entropy_info"] = {
            k: v for k, v in entropy_info.items() if k != "token_ids"  # don't dump raw IDs in report
        }

        # Verify cache length
        cache_len = kv.get_seq_length() if hasattr(kv, "get_seq_length") else kv[0][0].shape[-2]
        assert cache_len == NOISE_LENGTH, f"Cache length {cache_len} != expected {NOISE_LENGTH}"
        print(f"[entropy] KV cache built: {cache_len} positions, {ptime['total_ms']:.1f}ms")

        name = f"int4_tim_entropy_seed{seed}"
        report["entropy_seeds"][f"seed{seed}"] = run_condition(
            model, tokenizer, device, name, OUT_DIR, task_items,
            primed_kv=kv, prime_timing=ptime,
            max_new_tokens=MAX_NEW_TOKENS, score=score,
        )
        report["entropy_seeds"][f"seed{seed}"]["entropy_info"] = {
            k: v for k, v in entropy_info.items() if k != "token_ids"
        }

        free_kv(kv)

    unload_model(model)

    seeds_tag = "_".join(str(s) for s in seed_list)
    report_path = OUT_DIR / f"mechanism_gen_report_5_seeds{seeds_tag}.json"
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
