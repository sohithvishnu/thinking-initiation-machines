"""
MoE expert-locking intervention, Gemma-4-26B-A4B replication: does forcing
generation to route through the experts TIM priming activated preserve
accuracy, or degrade it?

Same methodology as tim-moe/moe_expert_lock.py (the OLMoE run), adapted only
for the Gemma-4 model/load path and its Gemma4TextRouter output shape. See
tim-moe/moe_router_probe_gemma.py's module docstring for the model choice,
the compressed-tensors dequantize-on-load caveat, and the router-hook
discovery method (find_router_modules() there, reused here unmodified via
import) — and tim-moe/docs/moe_gemma_comparison.md for the full writeup and
side-by-side numbers against the OLMoE run.

This is a mechanistic diagnostic, not a performance play. Report whichever
way it comes out.

Scoring, same deviation as moe_expert_lock.py and for the same reason:
evalplus.evaluate() hard-asserts every one of the 164 canonical HumanEval+
tasks is present in the samples file, incompatible with scoring a fixed
20-task subset. score_subset() below calls evalplus's own per-task
primitives directly (get_groundtruth, check_correctness, sanitize) — copied
unmodified from moe_expert_lock.py's version.

Note on the bug documented in tim-moe/docs/moe_expert_lock.md (Step 2): the
original OLMoE run's first implementation let `cold`/`tim_unconstrained`
inherit a stale `lock_sets` dict from a prior call, because `.active` and
`.lock_sets` are separate state and only the intentionally-locked conditions
cleared the latter. This script's ExpertLock is written with `.clear_lock()`
called before every generation that must be genuinely unconstrained from the
start (see run_task_all_conditions() below), and the self-test still includes
the same clear-lock regression check moe_expert_lock.py added after finding
that bug — plus a raw-output sanity read on a tim_locked completion (not just
the self-test), per the task's own instruction not to trust the self-test
alone this time.
"""

import argparse
import json
import random
import sys
import time
from collections import Counter
from pathlib import Path

import torch
from evalplus.data import get_human_eval_plus
from evalplus.evaluate import PASS, check_correctness, get_groundtruth, get_human_eval_plus_hash
from evalplus.sanitize import sanitize

TIM_MOE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(TIM_MOE_DIR))
from moe_router_probe_gemma import (  # noqa: E402
    MODEL_ID, NOISE_LENGTH, MAX_NEW_TOKENS, PYTHON_DOMAIN_WORDS,
    build_chat_prompt, find_router_modules, load_moe_model, model_report, stop_token_ids,
)

ROOT_DIR = TIM_MOE_DIR.parent
SRC_DIR = ROOT_DIR / "src"
sys.path.insert(0, str(SRC_DIR))
from tim.primer import TIMPrimer  # noqa: E402

PROBE_LOGS = TIM_MOE_DIR / "logs" / "moe_probe_gemma"
LOCK_LOGS = TIM_MOE_DIR / "logs" / "moe_lock_gemma"
LOCK_LOGS.mkdir(parents=True, exist_ok=True)

SEED = 0
DOCSTRING_RE_SRC = r'def\s+\w+\([^)]*\)[^:]*:\s*\n\s*("""|\'\'\')'


# --------------------------------------------------------------------------- #
# Step 0 — load the same fixed task set the probe used
# --------------------------------------------------------------------------- #

def load_probe_task_ids() -> list:
    raw_path = PROBE_LOGS / "routing_raw.json"
    if raw_path.exists():
        raw = json.loads(raw_path.read_text())
        task_ids = raw["task_ids"]
        if not task_ids:
            raise RuntimeError(f"{raw_path} has an empty task_ids list")
        return task_ids
    # Probe hasn't been run yet in this environment (or its output was moved) —
    # fall back to the same fixed first-20 convention the probe itself uses,
    # rather than hard-failing this script's independence.
    print(f"[tasks] {raw_path} not found — using the same first-20 HumanEval+ convention directly")
    problems = get_human_eval_plus()
    return list(problems.keys())[:20]


# --------------------------------------------------------------------------- #
# Step 1 — frequency capture, for deriving priming-phase lock sets
# --------------------------------------------------------------------------- #

class FreqCapture:
    """Like moe_router_probe_gemma.RouterCapture, but tracks per-expert
    SELECTION COUNT (how many token positions picked that expert), not just
    the union set — needed for "top-k experts most frequently selected"."""

    def __init__(self, model):
        self.routers = find_router_modules(model)
        self.active = False
        self.counts = {}  # layer_idx -> Counter(expert_idx -> n_selections)
        self.handles = [r.register_forward_hook(self._make_hook(i)) for i, r in self.routers.items()]

    def _make_hook(self, layer_idx):
        def hook(module, inputs, output):
            if not self.active:
                return
            _, _, top_k_index = output
            counter = self.counts.setdefault(layer_idx, Counter())
            for row in top_k_index.detach().cpu().tolist():
                counter.update(row)
        return hook

    def start(self):
        self.counts = {}
        self.active = True

    def stop(self):
        self.active = False

    def derive_lock_sets(self, top_k: int) -> dict:
        """Returns {layer_idx(str): {"top_k_by_frequency": [...], "union": [...], "counts": {...}}}"""
        out = {}
        for layer_idx, counter in self.counts.items():
            ranked = sorted(counter.items(), key=lambda kv: (-kv[1], kv[0]))
            top = sorted(e for e, _ in ranked[:top_k])
            out[str(layer_idx)] = {
                "top_k_by_frequency": top,
                "union": sorted(counter.keys()),
                "counts": {str(e): c for e, c in counter.items()},
            }
        return out

    def remove(self):
        for h in self.handles:
            h.remove()


def random_lock_sets(routers: dict, top_k: int, num_experts: int, rng_seed: int) -> dict:
    """A fixed (per-task, deterministic via rng_seed), NOT priming-derived,
    top_k-per-layer expert set — the cold_locked_random control."""
    rng = random.Random(rng_seed)
    out = {}
    for layer_idx in sorted(routers.keys()):
        experts = sorted(rng.sample(range(num_experts), top_k))
        out[str(layer_idx)] = {"top_k_by_frequency": experts, "union": experts, "counts": {}}
    return out


# --------------------------------------------------------------------------- #
# Step 2 — the locking intervention itself
# --------------------------------------------------------------------------- #

class ExpertLock:
    """Forward hook on each layer's Gemma4TextRouter module. When .active and
    a lock is set for that layer, OVERRIDES the router's selected experts to
    a fixed set, renormalizing the full softmax probability mass
    (router_probabilities, over all num_experts) over just that set — so only
    *which* experts are used changes, not the relative confidence the router
    originally had in the ones it kept. This also re-applies the router's own
    `per_expert_scale` step to the locked/renormalized weights (Gemma4TextRouter
    always multiplies top_k_weights by per_expert_scale[top_k_index] before
    handing them to Gemma4TextExperts — a step OLMoE's OlmoeTopKRouter doesn't
    have, so this is a real, documented adaptation from moe_expert_lock.py's
    ExpertLock, not a straight copy) — this keeps the intervention isolated to
    "which experts fire", matching the OLMoE lock's design intent, rather than
    also silently changing the per-expert weighting scheme.

    When inactive, the hook is a pure pass-through (returns None, PyTorch
    keeps the module's real output) — phase-aware: .start() only for the
    autoregressive generation loop, never during priming or prompt prefill.
    """

    def __init__(self, model):
        self.routers = find_router_modules(model)
        self.active = False
        self.lock_sets = {}  # int layer_idx -> LongTensor[top_k]
        self.log_enabled = False
        self.fired_log = {}  # int layer_idx -> list[list[int]], only when log_enabled
        self.handles = [r.register_forward_hook(self._make_hook(i)) for i, r in self.routers.items()]

    def set_lock(self, lock_sets_by_layer: dict, device):
        """lock_sets_by_layer: {layer_idx(str or int) -> list[int] of expert ids}"""
        self.lock_sets = {
            int(li): torch.tensor(sorted(experts), dtype=torch.long, device=device)
            for li, experts in lock_sets_by_layer.items()
        }

    def _make_hook(self, layer_idx):
        def hook(module, inputs, output):
            if not self.active or layer_idx not in self.lock_sets:
                return None
            router_probabilities, top_k_weights, top_k_index = output
            locked = self.lock_sets[layer_idx].to(router_probabilities.device)
            seq_len = router_probabilities.shape[0]
            locked_probs = router_probabilities.index_select(dim=-1, index=locked)  # [seq_len, k]
            denom = locked_probs.sum(dim=-1, keepdim=True).clamp_min(1e-9)
            renorm = locked_probs / denom
            renorm = renorm * module.per_expert_scale[locked]  # mirror the router's own scaling step
            renorm = renorm.to(top_k_weights.dtype)
            new_indices = locked.unsqueeze(0).expand(seq_len, -1).contiguous()
            if self.log_enabled:
                log = self.fired_log.setdefault(layer_idx, [])
                log.extend(new_indices.detach().cpu().tolist())
            return router_probabilities, renorm, new_indices
        return hook

    def start(self):
        self.active = True

    def stop(self):
        self.active = False

    def clear_lock(self):
        """Empty the lock set so .start() becomes a harmless pass-through —
        NOT the same as .stop(): .active alone is insufficient to guarantee
        unconstrained routing if a stale lock from a previous call is still
        sitting in self.lock_sets. Called before every generation that must
        be genuinely unconstrained (cold, tim_unconstrained) from the very
        first call in this script, per the bug documented in
        tim-moe/docs/moe_expert_lock.md."""
        self.lock_sets = {}

    def start_log(self):
        self.fired_log = {}
        self.log_enabled = True

    def stop_log(self):
        self.log_enabled = False

    def remove(self):
        for h in self.handles:
            h.remove()


# --------------------------------------------------------------------------- #
# Generation: phase-aware (prefill unconstrained, generation phase locked)
# --------------------------------------------------------------------------- #

def manual_greedy_decode_locked(model, tokenizer, device, input_ids, past_key_values, lock,
                                 max_new_tokens, eos_ids):
    """Same prefill/generate split as moe_router_probe_gemma.manual_greedy_decode:
    prefill runs with `lock` inactive, only the token-by-token generation loop
    runs with the lock active. Stops on ANY id in eos_ids (see
    moe_router_probe_gemma.stop_token_ids)."""
    attention_mask = None
    if past_key_values is not None:
        cache_len = (
            past_key_values.get_seq_length() if hasattr(past_key_values, "get_seq_length")
            else past_key_values[0][0].shape[-2]
        )
        full_len = cache_len + input_ids.shape[-1]
        attention_mask = torch.ones((1, full_len), dtype=torch.long, device=device)

    assert not lock.active, "lock must be inactive before/during prefill"
    with torch.no_grad():
        if past_key_values is not None:
            out = model(input_ids=input_ids, past_key_values=past_key_values,
                        attention_mask=attention_mask, use_cache=True)
        else:
            out = model(input_ids=input_ids, use_cache=True)
    kv = out.past_key_values
    next_token = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
    generated = [int(next_token.item())]

    lock.start()
    for _ in range(max_new_tokens - 1):
        if generated[-1] in eos_ids:
            break
        with torch.no_grad():
            out = model(input_ids=next_token, past_key_values=kv, use_cache=True)
        kv = out.past_key_values
        next_token = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        generated.append(int(next_token.item()))
    lock.stop()

    text = tokenizer.decode(generated, skip_special_tokens=True)
    return text, len(generated)


# --------------------------------------------------------------------------- #
# Self-test — MANDATORY gate, must pass before any scoring happens.
# --------------------------------------------------------------------------- #

def run_self_test(model, tokenizer, device, primer, lock, freq_capture, eos_ids, task_id, prompt, top_k):
    print(f"\n=== SELF-TEST: does the lock actually fire? (task {task_id}) ===")

    domain_ids = primer.get_domain_token_ids_from_words(PYTHON_DOMAIN_WORDS)
    freq_capture.start()
    kv = primer.prime(domain_tokens=domain_ids, seed=SEED)
    freq_capture.stop()
    lock_sets = freq_capture.derive_lock_sets(top_k)
    lock.set_lock({li: v["top_k_by_frequency"] for li, v in lock_sets.items()}, device)

    chat_text = build_chat_prompt(tokenizer, prompt)
    input_ids = tokenizer(chat_text, return_tensors="pt", add_special_tokens=False).input_ids.to(device)

    assert not lock.active, "lock must start inactive"
    lock.start_log()

    with torch.no_grad():
        out = model(input_ids=input_ids, past_key_values=kv, use_cache=True,
                    attention_mask=torch.ones((1, kv.get_seq_length() + input_ids.shape[-1]),
                                               dtype=torch.long, device=device))
    prefill_fired = dict(lock.fired_log)  # should be empty: lock was never .start()'d
    assert prefill_fired == {}, f"lock fired during prefill while inactive: {prefill_fired}"
    print("  [ok] prefill ran with lock inactive, hook fired 0 times (pass-through confirmed)")

    kv2 = out.past_key_values
    next_token = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)

    lock.start()
    for step in range(3):
        with torch.no_grad():
            out = model(input_ids=next_token, past_key_values=kv2, use_cache=True)
        kv2 = out.past_key_values
        next_token = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        for layer_idx, expert_tensor in lock.lock_sets.items():
            fired = lock.fired_log[layer_idx][-1]
            expected = sorted(int(x) for x in expert_tensor.tolist())
            assert sorted(fired) == expected, (
                f"step {step} layer {layer_idx}: fired {sorted(fired)} != lock {expected}"
            )
        print(f"  [ok] generation step {step}: all {len(lock.lock_sets)} locked layers' "
              f"selected experts == the lock set (e.g. layer 0: {sorted(lock.fired_log[0][-1])})")
    lock.stop()

    # Regression guard for the exact bug class documented in
    # tim-moe/docs/moe_expert_lock.md: .active alone does NOT guarantee
    # unconstrained routing if a stale lock_sets dict is still on the object.
    lock.clear_lock()
    lock.start()
    lock.start_log()
    with torch.no_grad():
        out = model(input_ids=next_token, past_key_values=kv2, use_cache=True)
    fired_after_clear = dict(lock.fired_log)
    assert fired_after_clear == {}, (
        f"lock.clear_lock() did not neutralize an active lock — hook still fired: {fired_after_clear}"
    )
    lock.stop()
    lock.stop_log()
    print("  [ok] clear_lock()+active still fires 0 times (proves .clear_lock() actually "
          "neutralizes a stale lock, not just relying on .active being False)")

    # Raw-output sanity check on an actual tim_locked-style completion — per
    # the task's own instruction, don't trust the self-test's assertions
    # alone this time; read real generated text under lock.
    lock.clear_lock()
    lock.set_lock({li: v["top_k_by_frequency"] for li, v in lock_sets.items()}, device)
    text, n_gen = manual_greedy_decode_locked(
        model, tokenizer, device, input_ids, kv, lock, max_new_tokens=150, eos_ids=eos_ids
    )
    print(f"  [raw-output check] tim_locked-style completion under lock ({n_gen} tokens):")
    print(f"  {text[:800]!r}")

    lock.clear_lock()  # do NOT leak this task's lock into the main loop's "cold" call
    del kv, kv2
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print("=== SELF-TEST PASSED: lock is confirmed inert during prefill, "
          "confirmed forcing exact lock-set selection during generation ===\n")
    return text


# --------------------------------------------------------------------------- #
# Step 3 — four conditions
# --------------------------------------------------------------------------- #

def run_task_all_conditions(model, tokenizer, device, primer, lock, freq_capture, eos_ids,
                             task_idx, task_id, prompt, top_k, num_experts):
    chat_text = build_chat_prompt(tokenizer, prompt)

    def fresh_input_ids():
        return tokenizer(chat_text, return_tensors="pt", add_special_tokens=False).input_ids.to(device)

    out = {}

    # ---- cold: no priming, unconstrained routing (reference floor) --------
    assert not lock.active
    lock.clear_lock()
    text, n_gen = manual_greedy_decode_locked(model, tokenizer, device, fresh_input_ids(), None, lock,
                                               MAX_NEW_TOKENS, eos_ids)
    out["cold"] = {"completion": text, "n_generated_tokens": n_gen}

    # ---- tim_unconstrained: domain priming, warm cache, free routing ------
    domain_ids = primer.get_domain_token_ids_from_words(PYTHON_DOMAIN_WORDS)
    kv = primer.prime(domain_tokens=domain_ids, seed=SEED)
    lock.clear_lock()
    text, n_gen = manual_greedy_decode_locked(model, tokenizer, device, fresh_input_ids(), kv, lock,
                                               MAX_NEW_TOKENS, eos_ids)
    out["tim_unconstrained"] = {"completion": text, "n_generated_tokens": n_gen}
    del kv

    # ---- tim_locked: domain priming AND generation locked to priming's ----
    #      top-k-by-frequency experts per layer (the intervention).
    freq_capture.start()
    kv = primer.prime(domain_tokens=domain_ids, seed=SEED)
    freq_capture.stop()
    derived = freq_capture.derive_lock_sets(top_k)
    lock.set_lock({li: v["top_k_by_frequency"] for li, v in derived.items()}, device)
    text, n_gen = manual_greedy_decode_locked(model, tokenizer, device, fresh_input_ids(), kv, lock,
                                               MAX_NEW_TOKENS, eos_ids)
    out["tim_locked"] = {"completion": text, "n_generated_tokens": n_gen}
    del kv

    # ---- cold_locked_random: no priming, locked to a random (not ----------
    #      priming-derived) top-k set per layer.
    random_sets = random_lock_sets(lock.routers, top_k, num_experts, rng_seed=9000 + task_idx)
    lock.set_lock({li: v["top_k_by_frequency"] for li, v in random_sets.items()}, device)
    text, n_gen = manual_greedy_decode_locked(model, tokenizer, device, fresh_input_ids(), None, lock,
                                               MAX_NEW_TOKENS, eos_ids)
    out["cold_locked_random"] = {"completion": text, "n_generated_tokens": n_gen}

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return out, derived, random_sets


# --------------------------------------------------------------------------- #
# Scoring — direct evalplus primitives, NOT the subprocess/CLI path — copied
# unmodified from moe_expert_lock.py (same reasoning, see module docstring).
# --------------------------------------------------------------------------- #

def score_subset(task_ids: list, generations: dict) -> dict:
    problems = get_human_eval_plus()
    dataset_hash = get_human_eval_plus_hash()
    expected_output = get_groundtruth(problems, dataset_hash, [])

    per_task = {}
    for task_id in task_ids:
        problem = problems[task_id]
        raw = generations.get(task_id, "")
        full = problem["prompt"] + raw
        sanitized = sanitize(full, problem["entry_point"])
        res = check_correctness(
            "humaneval", 0, problem, sanitized, expected_output[task_id],
            base_only=False, fast_check=True, identifier=f"{task_id}_lock_gemma",
        )
        base_status = res["base"][0]
        plus_status = res["plus"][0]
        base_pass = base_status == PASS
        plus_pass = base_pass and (plus_status == PASS)
        per_task[task_id] = {
            "base_status": base_status,
            "plus_status": plus_status,
            "base_pass": bool(base_pass),
            "plus_pass": bool(plus_pass),
            "sanitized_solution": sanitized,
        }
    n = len(task_ids)
    if n == 0:
        raise RuntimeError("score_subset called with 0 tasks — refusing to report a division-by-zero pass@1")
    base_rate = sum(v["base_pass"] for v in per_task.values()) / n
    plus_rate = sum(v["plus_pass"] for v in per_task.values()) / n
    return {"per_task": per_task, "n": n, "pass@1_base": base_rate, "pass@1_base_plus_extra": plus_rate}


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--self_test_only", action="store_true")
    args = ap.parse_args()

    print(f"[load] {MODEL_ID} ...")
    t0 = time.perf_counter()
    model, tokenizer, device = load_moe_model()
    print(f"[load] done in {time.perf_counter()-t0:.1f}s, device={device}")

    eos_ids = stop_token_ids(tokenizer, model)
    print(f"[eos] stop token ids: {sorted(eos_ids)}")

    report = model_report(model, tokenizer)
    top_k = report["num_experts_per_tok"]
    num_experts = report["num_experts"]
    print(f"[model] num_experts={num_experts} top_k={top_k} n_moe_layers={report['n_moe_layers']}")

    lock = ExpertLock(model)
    freq_capture = FreqCapture(model)
    primer = TIMPrimer(model, tokenizer, device, noise_length=NOISE_LENGTH, num_passes=1)

    task_ids = load_probe_task_ids()
    problems = get_human_eval_plus()
    task_items = [(tid, problems[tid]) for tid in task_ids]
    print(f"[tasks] reusing {len(task_items)} task ids from the probe: {task_ids}")

    # ---- mandatory self-test gate ------------------------------------------
    run_self_test(model, tokenizer, device, primer, lock, freq_capture, eos_ids,
                  task_items[0][0], task_items[0][1]["prompt"], top_k)
    if args.self_test_only:
        print("[done] --self_test_only set, exiting after self-test.")
        return

    # ---- Steps 1-3: derive locks + run all 4 conditions on all tasks ------
    all_generations = {"cold": {}, "tim_unconstrained": {}, "tim_locked": {}, "cold_locked_random": {}}
    all_lock_sets = {}
    all_random_sets = {}
    n_tokens = {"cold": {}, "tim_unconstrained": {}, "tim_locked": {}, "cold_locked_random": {}}

    t_start = time.perf_counter()
    for i, (task_id, problem) in enumerate(task_items):
        t0 = time.perf_counter()
        out, derived_locks, random_sets = run_task_all_conditions(
            model, tokenizer, device, primer, lock, freq_capture, eos_ids,
            i, task_id, problem["prompt"], top_k, num_experts,
        )
        for cond, d in out.items():
            all_generations[cond][task_id] = d["completion"]
            n_tokens[cond][task_id] = d["n_generated_tokens"]
        all_lock_sets[task_id] = derived_locks
        all_random_sets[task_id] = random_sets
        print(f"[{i+1}/{len(task_items)}] {task_id} done in {time.perf_counter()-t0:.1f}s "
              f"(tokens: cold={n_tokens['cold'][task_id]} "
              f"tim_unc={n_tokens['tim_unconstrained'][task_id]} "
              f"tim_lock={n_tokens['tim_locked'][task_id]} "
              f"rand_lock={n_tokens['cold_locked_random'][task_id]})")

    print(f"[done] all tasks + conditions in {time.perf_counter()-t_start:.1f}s")

    lock.remove()
    freq_capture.remove()

    # ---- write lock_sets.json (Step 1 artifact) ----------------------------
    lock_sets_path = LOCK_LOGS / "lock_sets.json"
    lock_sets_path.write_text(json.dumps({
        "top_k": top_k, "num_experts": num_experts, "seed": SEED,
        "priming_derived": all_lock_sets, "random_control": all_random_sets,
    }, indent=2))
    print(f"[write] {lock_sets_path}")

    # ---- score all 4 conditions --------------------------------------------
    print("[score] scoring 4 conditions x 20 tasks via direct evalplus primitives...")
    scores = {}
    for cond, gens in all_generations.items():
        t0 = time.perf_counter()
        scores[cond] = score_subset(task_ids, gens)
        print(f"  [score] {cond}: pass@1_base={scores[cond]['pass@1_base']:.3f} "
              f"pass@1_base+extra={scores[cond]['pass@1_base_plus_extra']:.3f} "
              f"({time.perf_counter()-t0:.1f}s)")

    # ---- write answers_lock.jsonl (per-task side-by-side across conditions) -
    answers_path = LOCK_LOGS / "answers_lock.jsonl"
    with open(answers_path, "w") as f:
        for task_id in task_ids:
            row = {"task_id": task_id}
            for cond in all_generations:
                row[cond] = {
                    "completion": all_generations[cond][task_id],
                    "n_generated_tokens": n_tokens[cond][task_id],
                    "sanitized_solution": scores[cond]["per_task"][task_id]["sanitized_solution"],
                    "base_pass": scores[cond]["per_task"][task_id]["base_pass"],
                    "plus_pass": scores[cond]["per_task"][task_id]["plus_pass"],
                }
            f.write(json.dumps(row) + "\n")
    print(f"[write] {answers_path}")

    # ---- write eval summary -------------------------------------------------
    summary_path = LOCK_LOGS / "eval_summary.json"
    summary_path.write_text(json.dumps({
        "model_report": report,
        "task_ids": task_ids,
        "top_k": top_k,
        "num_experts": num_experts,
        "seed": SEED,
        "scores": {cond: {"n": s["n"], "pass@1_base": s["pass@1_base"],
                           "pass@1_base_plus_extra": s["pass@1_base_plus_extra"]}
                   for cond, s in scores.items()},
        "per_task": {
            cond: {tid: {"base_pass": s["per_task"][tid]["base_pass"],
                         "plus_pass": s["per_task"][tid]["plus_pass"]}
                   for tid in task_ids}
            for cond, s in scores.items()
        },
    }, indent=2))
    print(f"[write] {summary_path}")

    print("\n=== FINAL PASS@1 (base+extra), raw counts ===")
    for cond, s in scores.items():
        n_pass = sum(v["plus_pass"] for v in s["per_task"].values())
        print(f"  {cond:22s} {n_pass}/{s['n']}  ({s['pass@1_base_plus_extra']:.3f})")


if __name__ == "__main__":
    main()
