# MoE diagnostics on a second model: Gemma-4-26B-A4B vs. OLMoE

This replicates the two MoE mechanistic diagnostics already run on
`allenai/OLMoE-1B-7B-0924-Instruct` ([`moe_router_probe.md`](moe_router_probe.md),
[`moe_expert_lock.md`](moe_expert_lock.md)) on a second, structurally
different MoE model: `cyankiwi/gemma-4-26B-A4B-it-AWQ-4bit`
(`Gemma4ForConditionalGeneration`, text config: 30 layers all-MoE, 128
experts, top-8 routing, hidden 2816 — vs. OLMoE's 16 layers, 64 experts,
top-8, hidden 2048). The question this answers: does the OLMoE verdict
generalize to a much larger model (26B total / ~4B active vs. OLMoE's 6.9B
total / 1.3B active), a different expert count/topology, and (unlike OLMoE,
which ran at bf16 because 4-bit quantization was a structural no-op for its
batched expert-weight format) a model that is *genuinely* int4-quantized end
to end, including the experts?

**This is a mechanistic diagnostic, not a performance play, on both models.**
Nothing here claims a benefit to output quality; the questions are purely
about routing mechanics and about what a routing-based intervention does to
accuracy.

## The checkpoint/library problem, and how it was fixed

Despite its "AWQ" name, this checkpoint's `quantization_config` is a genuine
compressed-tensors **pack-quantized int4** scheme (group_size 32, symmetric),
applied **per expert**: every expert's `gate_proj`/`up_proj`/`down_proj` is
stored as its own small `(weight_packed, weight_scale, weight_shape)` tensor
triple (`model.language_model.layers.{i}.experts.{j}.{proj}.weight_packed`,
~34.5k such tensors total). Installed transformers 5.13.1's
`Gemma4TextExperts`, however, expects one **batched** `[num_experts, ...]`
`nn.Parameter` per projection per layer (`experts.gate_up_proj`,
`experts.down_proj`) — the pattern every current transformers MoE
architecture uses internally, `Gemma4` included. Gemma4 registers **no
checkpoint-conversion mapping** to bridge per-expert-tensor checkpoints to
this batched format (unlike e.g. Qwen3-MoE, which has exactly this kind of
merge registered in `transformers/conversion_mapping.py`). The practical
consequence, confirmed directly by loading onto `device_map={"": "meta"}`
with `output_loading_info=True`: a plain `AutoModelForCausalLM.from_pretrained(...)`
call reports all 11520 per-expert checkpoint tensors as `UNEXPECTED` and the
model's real `experts.gate_up_proj`/`experts.down_proj` Parameters (60 of
them, 30 layers × 2) as `MISSING` — **silently randomly initialized**, with
no error raised. Every other tensor (attention, embeddings, lm_head, router,
the per-layer dense `mlp.*`) loads correctly; only the experts are affected.

The fix, implemented in [`gemma_expert_convert.py`](../gemma_expert_convert.py)
and [`gemma_moe_loader.py`](../gemma_moe_loader.py):

1. **`gemma_expert_convert.py`** reads the per-expert
   `(weight_packed, weight_scale, weight_shape)` triples directly off the
   checkpoint's safetensors shards and dequantizes them using
   compressed_tensors' own primitives (`unpack_from_int32` +
   `quantization.lifecycle.forward.dequantize`). This was **verified
   byte-for-byte** (`torch.equal`, max abs diff 0.0) against
   compressed_tensors' own top-level `PackedQuantizationCompressor.decompress`
   API, for 5 experts across 4 different layers (0, 5, 15, 29) and all three
   projections — re-confirmed in this session by re-running
   `gemma_expert_convert.py`'s `verify()` directly:
   ```
   === gemma_expert_convert.verify() — 5 cases ===
   layer= 0 expert=  0 gate_proj  shape=(704, 2816)    max_abs_diff=0 exact_match=True
   layer= 0 expert=  5 up_proj    shape=(704, 2816)    max_abs_diff=0 exact_match=True
   layer= 5 expert= 50 down_proj  shape=(2816, 704)    max_abs_diff=0 exact_match=True
   layer=15 expert=100 gate_proj  shape=(704, 2816)    max_abs_diff=0 exact_match=True
   layer=29 expert=127 down_proj  shape=(2816, 704)    max_abs_diff=0 exact_match=True
   === verify() PASSED ===
   ```
2. **`gemma_moe_loader.py`** monkeypatches `Gemma4TextExperts` to a
   zero-parameter placeholder *before* `from_pretrained` (so the real
   ~46GB-if-bf16 batched Parameters are never allocated/randomly-initialized
   at all), then replaces each layer's placeholder with a `GemmaExpertsInt4`
   module after loading. `GemmaExpertsInt4` keeps each expert's *original*
   packed int4 weights resident on GPU (not re-dequantized into a bf16
   batched tensor up front) and dequantizes only the router-selected experts
   on-the-fly, per forward call, inside the same `expert_hit` loop
   `Gemma4TextExperts.forward` already uses — i.e. real int4 memory savings
   (measured: **12.85GB** across both GPUs for all 30 layers × 128 experts ×
   3 projections, vs. what would be ~46GB if fully dequantized to bf16 up
   front), not a bf16 fallback dressed up as a fix.

Router capture reuses the same forward-hook approach as the OLMoE probe
(`model.named_modules()` walk for `Gemma4TextRouter` instances, not an
assumed path), adapted for `Gemma4TextRouter.forward`'s
`(router_probabilities, top_k_weights, top_k_index)` output — a different
shape from OLMoE's `OlmoeTopKRouter`'s `(router_logits, router_scores,
router_indices)`; `router_probabilities` is already a full softmax over all
experts computed in fp32, so no re-softmax is applied (unlike the OLMoE hook,
which does). The expert-lock intervention also required one genuine
adaptation beyond a straight copy: `Gemma4TextRouter` multiplies
`top_k_weights` by a `per_expert_scale` term before handing them to the
experts module, a step OLMoE's router doesn't have — `ExpertLock`'s hook
re-applies that same scaling to the locked/renormalized weights, so the
intervention changes only *which* experts fire, not the weighting scheme.

## Self-test evidence

Per the task's own instruction not to trust the self-test's assertions
alone, both self-tests were read as raw generated text, not just checked for
"did it run without an exception."

**Router-probe self-test** (`moe_router_probe_gemma.py --self_test_only`,
greedy, 150 tokens, HumanEval/0 prompt) — genuinely coherent Python, correct
docstring, correct partial logic for `has_close_elements`:
```
```python
from typing import List


def has_close_elements(numbers: List[float], threshold: float) -> bool:
    """ Check if in given list of numbers, are any two numbers closer to each other than
    given threshold.
    >>> has_close_elements([1.0, 2.0, 3.0], 0.5)
    False
    >>> has_close_elements([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3)
    True
    """
    for i in range(len(numbers)):
        for j in range
```
This is not a lucky/degenerate output — it exactly reproduces the canonical
HumanEval/0 solution's opening structure. Confirms the conversion is correct,
not silently running on random-initialized expert weights.

**Expert-lock self-test** (`moe_expert_lock_gemma.py`, embedded in `main()`,
runs before any scoring): confirmed the lock is inert during prefill (hook
fires 0 times), confirmed it forces the exact lock-set experts at every one
of the 30 locked layers across 3 generation steps, confirmed `clear_lock()`
genuinely neutralizes a stale lock (the regression check for the exact bug
class documented in `moe_expert_lock.md`) — and, per the task's explicit
instruction to not rely on the self-test's assertions alone this time,
included a raw-output sanity read on an actual `tim_locked`-style completion:
```
```python```
```
```python```
```
```python```
[... repeats the same two-line pattern for the full 150 tokens ...]
```
Degenerate but not silently broken — the model produces the literal token
sequence `` ```python``` `` on repeat rather than valid code. This is the
same qualitative failure mode later confirmed at full scale (see Step 3
below): locking collapses generation to non-language repetition, it doesn't
just make the code subtly wrong.

## Experiment 1 — router probe (Q1/Q2)

Same setup as the OLMoE probe: 20 fixed HumanEval+ tasks (`HumanEval/0`–`19`),
greedy decoding, `max_new_tokens=768`, seed 0, `TIMPrimer` (`src/tim/primer.py`)
unmodified, 3 conditions (`cold`/`tim_domain`/`tim_random`). Ran to completion
in 2196.8s (~37 min) for all 20 tasks × 3 conditions.
Raw data: [`logs/moe_probe_gemma/routing_raw.json`](../logs/moe_probe_gemma/routing_raw.json)
(17.6MB). Analysis: [`logs/moe_probe_gemma/analysis_summary.json`](../logs/moe_probe_gemma/analysis_summary.json),
figures: [`docs/figures/moe_router_shift_gemma.png`](figures/moe_router_shift_gemma.png),
[`docs/figures/moe_prediction_overlap_gemma.png`](figures/moe_prediction_overlap_gemma.png).

### Q1 — does priming shift the router?

| model | condition | mean Jaccard (1.0 = no shift) | mean KL divergence |
|---|---|---|---|
| OLMoE | tim_domain vs. cold | 0.912 | 0.0071 |
| OLMoE | tim_random vs. cold | 0.927 | 0.0068 |
| **Gemma-4** | **tim_domain vs. cold** | **0.949** | **0.0027** |
| **Gemma-4** | **tim_random vs. cold** | **0.952** | **0.0005** |

**Verdict: same as OLMoE — yes, priming measurably shifts the router, and it
is NOT content-specific.** Overlap sits below 1.0 in both conditions (some
real shift, smaller in absolute magnitude than OLMoE's — Gemma-4's 128-expert,
30-layer topology is more diffuse per layer than OLMoE's 64-expert/16-layer
one, so a smaller per-layer Jaccard drop is expected even for a
proportionally similar shift). Critically, exactly like OLMoE, `tim_random`
shifts the router by essentially the same amount as `tim_domain` — actually
slightly *more* here too (0.952 vs. 0.949 Jaccard, KL 0.0005 vs. 0.0027 — KL
is in fact lower for random than domain, the reverse of OLMoE's KL ordering
but the same qualitative story: no evidence domain-specific noise perturbs
the router more purposefully than pure random noise). `q1_content_specific`
computed `False` for both models.

### Q2 — does priming-time routing predict generation-time routing? (the linchpin)

| model | condition | priming→generation overlap | chance baseline (cross-task) | gap |
|---|---|---|---|---|
| OLMoE | tim_domain | 0.678 | 0.862 | −0.184 |
| OLMoE | tim_random | 0.723 | 0.842 | −0.119 |
| **Gemma-4** | **tim_domain** | **0.431** | **0.741** | **−0.310** |
| **Gemma-4** | **tim_random** | **0.384** | **0.749** | **−0.365** |

**Verdict: same as OLMoE, and more decisively negative — priming-time routing
predicts generation-time routing *worse* than chance.** Both gaps are
negative, and larger in magnitude than OLMoE's (−0.31/−0.37 vs. OLMoE's
−0.18/−0.12). Two unrelated real code-completion tasks' generation-phase
expert sets resemble each other more (chance baseline ~0.74–0.75) than a
task's own priming-phase set resembles that same task's own generation-phase
set (~0.38–0.43) — the same "priming noise tokens are a qualitatively
different token distribution from coherent code, so two coherent-code passes
share more routing structure with each other than either shares with a noise
pass" story as OLMoE, just more pronounced on this model/checkpoint.
`q2_predictive_above_chance` computed `False` for both models.

## Experiment 2 — expert-locking intervention

Same setup as the OLMoE lock experiment: reused the identical 20 task IDs
from this run's own probe, greedy, `max_new_tokens=768`, seed 0, 4 conditions
(`cold`/`tim_unconstrained`/`tim_locked`/`cold_locked_random`), scored via
evalplus's per-task primitives directly (`score_subset`, same deviation from
the CLI pattern as the OLMoE run, same reason: `evalplus.evaluate()`
hard-requires all 164 canonical tasks be present). Self-test gate passed
(see above) before any scoring. Full run (self-test + 20 tasks × 4
conditions + scoring): 7518.9s (~2h05m) — substantially longer than the
probe's 37 min because the two locked conditions almost never hit EOS and
instead ran to the 768-token cap on **every single task** (confirmed by the
per-task token counts printed during the run, e.g.
`tim_lock=768 rand_lock=768` on all 20 lines), roughly quadrupling total
generated tokens vs. the probe's 3 unconstrained conditions.

Raw data: [`logs/moe_lock_gemma/lock_sets.json`](../logs/moe_lock_gemma/lock_sets.json),
[`logs/moe_lock_gemma/answers_lock.jsonl`](../logs/moe_lock_gemma/answers_lock.jsonl),
[`logs/moe_lock_gemma/eval_summary.json`](../logs/moe_lock_gemma/eval_summary.json).
Analysis: [`logs/moe_lock_gemma/analysis_summary.json`](../logs/moe_lock_gemma/analysis_summary.json).

### pass@1 (base+extra), N=20, greedy

| condition | OLMoE (bf16) | Gemma-4 (int4) |
|---|---|---|
| cold | 6/20 (0.300) | **19/20 (0.950)** |
| tim_unconstrained | 6/20 (0.300) | **18/20 (0.900)** |
| **tim_locked** | **0/20 (0.000)** | **0/20 (0.000)** |
| cold_locked_random | 0/20 (0.000) | 0/20 (0.000) |

(Gemma-4's much higher unconstrained baseline is unsurprising and not part of
the diagnostic itself — it is a ~4B-active/26B-total instruction-tuned model
vs. OLMoE's 1.3B-active/6.9B-total one; the comparison that matters for this
diagnostic is the *within-model* contrast between locked and unconstrained,
not the cross-model absolute pass@1.)

Paired contrasts (base+extra), exact binomial on discordant pairs:

| contrast | OLMoE p | OLMoE discordant | Gemma-4 p | Gemma-4 discordant |
|---|---|---|---|---|
| tim_locked vs. cold | 0.03125 | 6 (0 locked-wins, 6 cold-wins) | **3.81e-06** | 19 (0 locked-wins, 19 cold-wins) |
| tim_locked vs. tim_unconstrained | 0.03125 | 6 (0 vs. 6) | **7.63e-06** | 18 (0 vs. 18) |
| tim_locked vs. cold_locked_random | n/a — 0 discordant | 0 | n/a — 0 discordant | 0 |

**Verdict: identical to OLMoE — degrades, and indistinguishable from locking
to a random expert set.** `tim_locked` vs. `cold_locked_random` has **zero**
discordant pairs on Gemma-4 too (both 0/20, every one of the 20 tasks fails
identically under both locking conditions) — the same cleanest-possible
negative result for "priming's expert choice is special" that OLMoE showed.
`analyze_expert_lock_gemma.py`'s automatic verdict-selection logic (unmodified
from the OLMoE script) independently reached the same interpretation string:
*"Degrades, and indistinguishable from locking to a random expert set: the
damage comes from constraining the router to ANY fixed expert set, not from
priming's particular (off-task, per Q2) choice."*

### Behavioral: tokens/task and docstring rate

| condition | OLMoE mean tok | OLMoE hit-cap | OLMoE docstring rate | Gemma-4 mean tok | Gemma-4 hit-cap | Gemma-4 docstring rate |
|---|---|---|---|---|---|---|
| cold | 158.2 | 0/20 | 0.60 | 164.6 | 0/20 | 1.00 |
| tim_unconstrained | 175.2 | 0/20 | 0.65 | 199.8 | 1/20 | 0.95 |
| tim_locked | 737.6 | 19/20 | 0.30 | **768.0** | **20/20** | 0.30 |
| cold_locked_random | 630.4 | 14/20 | 0.30 | **768.0** | **20/20** | 0.30 |

Gemma-4's locked conditions are even more extreme than OLMoE's: **every
single one** of the 20 tasks hits the 768-token cap under both locked
conditions (vs. 19/20 and 14/20 for OLMoE) — locking never lets the model
reach EOS naturally on this checkpoint. Raw completions confirm this isn't
"worse code," it's non-language repetition: the self-test's tim_locked-style
completion (above) and a full-run example, `HumanEval/0` under `tim_locked`:
```
```python```
#
```python```
#
```python```
#
[... repeats for the full 768 tokens ...]
```
vs. the same task's `cold` completion, which is the canonical correct
solution structure (and passes):
```python
from typing import List


def has_close_elements(numbers: List[float], threshold: float) -> bool:
    """ Check if in given list of numbers, are any two numbers closer to each other than
    given threshold.
    >>> has_close_elements([1.0, 2.0, 3.0], 0.5)
    False
    >>> has_close_elements(...
```
This is the same qualitative breakdown mode as OLMoE (which produced
character-soup gibberish, not just wrong code) — forcing a fixed 8-of-128
experts per layer, for every token, does not just hurt coding ability, it
appears to break coherent language production entirely, exactly as it did
for OLMoE's fixed 8-of-64.

## Side-by-side summary — does the OLMoE verdict generalize?

| question | OLMoE (6.9B total, 64 experts, 16 layers, bf16) | Gemma-4 (26B total, 128 experts, 30 layers, int4) | same verdict? |
|---|---|---|---|
| Q1: does priming shift the router? | Yes (Jaccard 0.91–0.93 vs. cold) | Yes (Jaccard 0.95 vs. cold) | **Yes** |
| Q1: is the shift content-specific (domain > random)? | No | No | **Yes** |
| Q2: does priming-phase routing predict generation-phase routing above chance? | No (gap −0.12 to −0.18) | No (gap −0.31 to −0.37, more negative) | **Yes** |
| Lock: does forcing generation onto priming's experts preserve accuracy? | No — collapses to 0/20 | No — collapses to 0/20 | **Yes** |
| Lock: is priming's expert choice better than a random fixed set? | No — 0 discordant pairs vs. random lock | No — 0 discordant pairs vs. random lock | **Yes** |

**All four verdicts replicate, on a model that differs in total/active
parameter count (26B/~4B vs. 6.9B/1.3B), expert count (128 vs. 64), layer
count (30 vs. 16, all-MoE in both), and — unlike OLMoE, which ran at bf16
because 4-bit was a structural no-op for its batched expert format —
genuine end-to-end int4 quantization including the experts themselves.**
Router-hijacking via noise-vocabulary KV-cache priming (this repo's
`TIMPrimer` mechanism specifically: 64 noise tokens, single pass, 70/30
domain/random or pure random) is not supported as a basis for expert
prefetching or for a beneficial routing intervention on either model tested.
Where Gemma-4 differs from OLMoE, it differs in the *same direction* and more
strongly: a smaller natural router shift under priming (Q1's Jaccard is
closer to 1.0), a *more* negative prediction gap (Q2), and a *more* uniform
collapse under locking (every locked task hits the token cap, vs. only some
of OLMoE's). Nothing in this second model's results points toward a
different qualitative story than OLMoE's.

**What this does not rule out** (same caveat as both OLMoE docs, restated
for two now-independent models): a different priming recipe (real
task-relevant text instead of vocabulary noise, soft router-logit biasing
instead of a hard top-k override, locking only a subset of layers, more
priming passes) might behave differently. This rules out *this* specific
mechanism — hard-locking all layers to noise-primed top-k experts, derived
from a single 64-token KV-priming pass — on two structurally different MoE
models, not the general idea of routing-aware intervention.

## Files written

- `tim-moe/gemma_expert_convert.py` — per-expert compressed-tensors int4
  dequantization, verified byte-for-byte against `compressed_tensors`' own
  decompress API.
- `tim-moe/gemma_moe_loader.py` — the checkpoint/library format fix: patches
  `Gemma4TextExperts` to a placeholder before loading, then attaches a
  memory-efficient `GemmaExpertsInt4` module (packed int4 resident on GPU,
  dequantized on-the-fly per router-selected expert) after loading.
- `tim-moe/moe_router_probe_gemma.py` / `tim-moe/moe_expert_lock_gemma.py` —
  Gemma-4 replications of the OLMoE scripts, adapted only for the model/load
  path and `Gemma4TextRouter`'s output shape and `per_expert_scale` step.
- `tim-moe/analyze_moe_routing_gemma.py` / `tim-moe/analyze_expert_lock_gemma.py`
  — unmodified-methodology analysis scripts, paths only changed.
- `tim-moe/logs/moe_probe_gemma/routing_raw.json` (17.6MB),
  `analysis_summary.json`.
- `tim-moe/logs/moe_lock_gemma/lock_sets.json`, `answers_lock.jsonl`,
  `eval_summary.json`, `analysis_summary.json`.
- `tim-moe/docs/figures/moe_router_shift_gemma.png`,
  `moe_prediction_overlap_gemma.png`.
- `tim-moe/docs/moe_gemma_comparison.md` — this document.

## Confirmation: OLMoE artifacts untouched

`git status`/`git diff` over `tim-moe/moe_router_probe.py`,
`moe_expert_lock.py`, `analyze_moe_routing.py`, `analyze_expert_lock.py`,
`docs/moe_router_probe.md`, `docs/moe_expert_lock.md`, and
`logs/moe_probe/`, `logs/moe_lock/` show zero changes — working tree clean
for all of them. Only new `*_gemma*`-suffixed files were added.
