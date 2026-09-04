# MoE expert-locking intervention: does locking generation to priming's experts preserve accuracy?

Follows up [`moe_router_probe.md`](moe_router_probe.md)'s Q2, which was
observational: priming-phase → generation-phase expert-set overlap measured
*below* cross-task chance at all 16 layers (0.678 domain / 0.723 random vs.
0.862 / 0.842 chance baseline). Q2 could not say whether that low overlap is
*causally* meaningful — this experiment builds and runs the actual
intervention: force generation to route only through the experts priming
activated, and see what happens to pass@1.

**This is a mechanistic diagnostic, not a performance play.** The honest
prior from Q2 was that locking to a below-chance expert set would degrade
accuracy. That is what happened, and more decisively than expected — full
detail below, reported plainly either way per the task's own framing.

## Setup

- Model: `allenai/OLMoE-1B-7B-0924-Instruct`, bf16 (64 experts, top-8, 16 MoE
  layers) — identical model/precision/reasoning as the router probe (4-bit
  quantization is a structural no-op for this transformers version's batched
  MoE expert weights; see `moe_router_probe.md`).
- Same fixed 20-task HumanEval+ set as the probe, read directly from
  `tim-moe/logs/moe_probe/routing_raw.json`'s `task_ids`, not re-derived.
- `TIMPrimer` (`experiment-2/tim_primer.py`) reused unmodified. No dense-model
  file touched.
- Scoring **deviates from the rest of the repo's `evaluate_with_evalplus`
  subprocess pattern** for a concrete reason: `evalplus.evaluate()` hard-
  asserts `len(completion_id) == len(problems)` — every one of the full 164
  canonical HumanEval+ tasks must be present in the samples file, or it raises
  `AssertionError: Missing problems in samples` (read directly from the
  installed evalplus source before writing any scoring code). That is
  incompatible with scoring a fixed 20-task subset. `tim-moe/moe_expert_lock.py`'s
  `score_subset()` instead calls evalplus's own per-task primitives directly
  (`get_groundtruth`, `check_correctness`, `sanitize`) — same ground truth,
  same sandboxed test execution, without the whole-dataset requirement.

## Step 1 — lock-set derivation

For each task, `TIMPrimer.prime(domain_tokens=..., seed=0)` runs the same
70/30 domain/random priming pass used by `tim_domain` in the router probe. A
new `FreqCapture` hook (distinct from the probe's `RouterCapture` — this one
tracks per-expert **selection count** across the 64 priming-token positions,
not just the union set) records, per layer, how many times each expert was
selected. Two variants are saved to `tim-moe/logs/moe_lock/lock_sets.json`:

- **`top_k_by_frequency`** (the one actually used to lock generation): the
  8 (`num_experts_per_tok`) most-frequently-selected experts per layer, ties
  broken by lower expert index for determinism.
- **`union`**: every expert selected at any priming-token position (usually
  >8, since 64 tokens rarely concentrate on only 8 experts) — saved for
  reference, not used to drive the intervention.

## Step 2 — the intervention, and a bug the self-test didn't catch (but a raw-output check did)

`ExpertLock` (`tim-moe/moe_expert_lock.py`) is a forward hook on each layer's
`model.model.layers[i].mlp.gate`. When active, it overrides the router's
selected experts to a fixed per-layer set, **renormalizing the softmax
probability mass over just that set** (verified this is the clean path for
OLMoE — `OlmoeSparseMoeBlock.forward` does `_, top_k_weights, top_k_index =
self.gate(hidden_states)` then feeds those straight into `self.experts(...)`,
so a hook returning a replacement `(router_logits, new_weights, new_indices)`
tuple changes the actual computation, not just a logged value — no fallback
to uniform weighting was needed).

**Mandatory self-test, run before any scoring** (per the task spec — this is
non-negotiable, a "no difference" result is uninterpretable without proof the
lock actually fired): one task, 3 generation steps, asserting the selected
experts at every locked layer exactly equal the lock set, and that the hook
fires **zero times** during the unconstrained prefill. This passed on the
first implementation.

**However**, the first full 4-condition run produced 0.000 pass@1 on
*every* condition, including plain `cold` — a result that shouldn't be
possible for a functioning baseline. Inspecting raw completions showed why:
`cold` and `tim_unconstrained` were generating **byte-identical gibberish**
to `tim_locked`. Root cause: `ExpertLock.active` (the phase toggle) and
`ExpertLock.lock_sets` (the actual constraint content) are separate pieces of
state, and the original code only ever cleared/set the latter for the two
conditions that were *supposed* to be locked — `cold` and `tim_unconstrained`
called `.start()` unconditionally (as the phase-aware design requires) but
inherited whatever `lock_sets` a *previous* call (the self-test, or the prior
task's `cold_locked_random`) had left behind. The self-test didn't catch this
because it only ever tested "does an active lock with a lock set produce the
lock set" — it never tested "does clearing a lock actually neutralize it."

Fixed with an explicit `ExpertLock.clear_lock()`, called before every
generation call that must be genuinely unconstrained, and the self-test was
**extended** with a regression check specifically for this bug class: clear
the lock, reactivate, and assert the hook fires zero times — not just that
`.active` happens to be `False` at the right moment. Both the original and
new self-test assertions pass on the corrected code. This is recorded here
because it's exactly the kind of failure the task's "self-test is mandatory"
guardrail exists to catch, and in this case catching it required a raw-output
sanity check on top of the self-test, not the self-test alone — worth
carrying forward as a lesson: a self-test that only exercises the "on" path
can still miss a stuck "on" state elsewhere.

## Step 3 — four conditions, N=20, greedy, bf16

| condition | pass@1 (base) | pass@1 (base+extra) |
|---|---|---|
| cold | 6/20 (0.300) | 6/20 (0.300) |
| tim_unconstrained | 7/20 (0.350) | 6/20 (0.300) |
| **tim_locked** | **0/20 (0.000)** | **0/20 (0.000)** |
| cold_locked_random | 0/20 (0.000) | 0/20 (0.000) |

`cold` ≈ `tim_unconstrained` (6/20 either way on base+extra) — consistent
with the dense-model finding and the router probe's Q1: priming doesn't add
capability at full precision. `tim_locked` and `cold_locked_random` both
collapse to **zero**.

This is not "worse code" — raw completions under either locked condition are
not valid Python at all:

```
tim_locked (HumanEval/0, 768 tokens, hit the cap):
```solution1izereck.blypss.1/[PP1).andty1 `J..earprixt L()-no_file_numbers :/,_or-
cree.ifely-_antig(ised-xless/inmisTV»1d. 1Av.,var ')x_red,-P.;/(W',na ossy...
```
vs. `cold` on the same task, correct structure (docstring, signature) even
though its body doesn't ultimately pass:
```python
from typing import List

def has_close_elements(numbers: List[float], threshold: float) -> bool:
    """ Check if in given list of numbers, are any two numbers closer...
```

Locking a fixed 8-of-64 experts per layer for every token, regardless of what
each token actually needs, does not just hurt the model's coding ability —
it appears to break its ability to produce coherent language at all.

## Step 4 — analysis (`tim-moe/analyze_expert_lock.py`)

**Paired contrasts (base+extra), N=20, exploratory — not a powered test:**

| contrast | wins (col A) | wins (col B) | ties (both pass) | ties (both fail) | discordant | exact binomial p |
|---|---|---|---|---|---|---|
| tim_locked vs. cold | 0 | 6 | 0 | 14 | 6 | **0.03125** |
| tim_locked vs. tim_unconstrained | 0 | 6 | 0 | 14 | 6 | **0.03125** |
| tim_locked vs. cold_locked_random | 0 | 0 | 0 | 20 | 0 | n/a — no discordant pairs |

The key contrast (`tim_locked` vs. `cold_locked_random`) has **zero**
discordant pairs: every single one of the 20 tasks fails identically under
both locking conditions. This is the cleanest possible negative result for
"priming's expert choice is special" — there is no task, out of 20, where
locking to priming's experts does any better (or worse) than locking to a
fresh random 8-per-layer set.

**Behavioral — tokens/task and docstring rate:**

| condition | mean tokens | min | max | hit 768-token cap | docstring rate |
|---|---|---|---|---|---|
| cold | 158.2 | 38 | 377 | 0/20 | 0.60 (12/20) |
| tim_unconstrained | 175.2 | 62 | 302 | 0/20 | 0.65 (13/20) |
| tim_locked | 737.6 | 161 | 768 | **19/20** | 0.30 (6/20) |
| cold_locked_random | 630.4 | 32 | 768 | 14/20 | 0.30 (6/20) |

Locked conditions essentially never emit EOS naturally (19/20 and 14/20 hit
the hard 768-token cap) — consistent with the router being forced away from
whatever expert combination normally encodes "this answer is complete."
Note on the locked conditions' nonzero docstring rate: `sanitize()` extracts
a target function from `problem["prompt"] + completion`, and the prompt
itself already contains a correctly-formed `def ...():` + docstring header —
when the completion is unparseable garbage, the sanitizer can still surface
the *prompt's own* docstring-bearing stub with an empty/garbled body. This
0.30 is not evidence the model is echoing docstrings under lock; it is
largely an artifact of what the sanitizer falls back to when there's nothing
coherent to extract, and is noted here rather than over-interpreted.

## Verdict

**Degrades, and is statistically indistinguishable from locking to a random
expert set.** `tim_locked` is significantly worse than `tim_unconstrained`
(exact binomial p=0.03125, 6 discordant pairs, all favoring unconstrained,
zero favoring locked) — locking causally hurts. But `tim_locked` is not
merely "also bad" relative to `cold_locked_random`, it is **identical**: 0/20
both, zero discordant pairs across all 20 tasks. There is no signal here that
priming's routing carries any task-relevant information that survives being
turned into a hard constraint — the damage comes from constraining the router
to *any* fixed expert set at all, not from priming's particular (already
below-chance, per Q2) choice.

This causally confirms Q2's implication: priming's routing is off-task. The
"restrict generation to the experts priming activated" idea — the systems
payoff this probe and its follow-up were scoping toward (expert prefetching /
restriction as a lever) — **is falsified as a performance play** for this
priming recipe (64-token noise-vocabulary KV priming, single pass) on this
model. Router-hijacking in the sense of *deliberately, usefully* steering
generation toward better outputs via forced routing is not supported by this
result; what's demonstrated instead is that forced routing, from this
source, breaks the model.

**What this does not rule out:** a different priming recipe (real
task-relevant text rather than vocabulary noise, soft biasing of router
logits rather than a hard top-k override, locking only a subset of layers
rather than all 16, or a larger/different model) might behave differently.
This result rules out *this* specific mechanism — hard-locking all layers to
noise-primed top-k experts — not the general idea of routing-aware
intervention.

## Guardrails

- N=20 is a probe. The effect size here (0/20 vs 6/20, zero discordant pairs
  against the random-lock control) is unusually clean for N=20, but this is
  still not a substitute for a larger run — it's a strong enough signal to
  not need one to draw the qualitative conclusion, not a substitute for
  reporting it as exploratory.
- bf16 only — see `moe_router_probe.md` for why 4-bit doesn't work for MoE
  experts in this environment.
- The self-test is mandatory and, as documented above, was not by itself
  sufficient to catch every failure mode in this specific implementation —
  the corrected run's self-test (including the added regression check) is
  what's certified here; the first (buggy) run's self-test pass did not mean
  the buggy run's *results* were trustworthy, and they weren't used.

## Files written

- `tim-moe/moe_expert_lock.py` — lock derivation, the `ExpertLock`
  intervention + mandatory self-test, phase-aware locked generation, the
  four-condition driver, and `score_subset()` (evalplus-primitive-based
  subset scorer).
- `tim-moe/analyze_expert_lock.py` — pass@1 table, paired contrasts (exact
  binomial on discordant pairs), behavioral stats.
- `tim-moe/logs/moe_lock/lock_sets.json` — per-task, per-layer priming-derived
  (`top_k_by_frequency` and `union`) and random-control lock sets.
- `tim-moe/logs/moe_lock/answers_lock.jsonl` — per-task, per-condition raw
  completion, sanitized solution, token count, pass/fail.
- `tim-moe/logs/moe_lock/eval_summary.json` — model report, scores, per-task
  pass/fail for all 4 conditions.
- `tim-moe/logs/moe_lock/analysis_summary.json` — pass table, contrasts,
  behavioral stats, verdict (machine-readable form of this document's Step 4).
- `tim-moe/docs/moe_expert_lock.md` — this document.

## Expected but not found / deviations from the spec

- **Scoring via direct evalplus primitives, not the subprocess/CLI
  `evaluate_with_evalplus` pattern** used elsewhere in this repo — a real
  constraint (the CLI entrypoint hard-requires all 164 canonical tasks be
  present), not a shortcut. See "Setup" above.
- **No figures produced** — the task spec's Steps 4-5 for this experiment
  ask for tables and a written verdict, not plots (unlike the router probe,
  which explicitly asked for two PNGs); none were generated here.
- **`union` lock-set variant recorded but not run** — per the spec's own
  stated priority ("record union too for a secondary run if time allows"),
  only `top_k_by_frequency` was used to drive the actual locking experiment;
  `union` sets are saved in `lock_sets.json` for a future secondary run if
  wanted, but exceed `top_k` in size for essentially every layer (64
  priming tokens rarely concentrate on only 8 experts), so a union-based
  lock would need a different renormalization design (not a straightforward
  top-k override) — flagged rather than attempted under this task's time
  budget.
- **A real implementation bug was hit and fixed mid-task** (see Step 2) —
  documented in full above rather than silently corrected, since it's
  directly relevant to how much to trust the self-test methodology in
  general.
