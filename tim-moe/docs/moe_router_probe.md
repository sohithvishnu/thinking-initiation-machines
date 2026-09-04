# MoE router probe: does TIM priming move expert routing?

This opens the MoE line of TIM work with the cheapest, most decisive first
experiment. **This is a routing-mechanics probe, not a performance/accuracy
result.** Nothing here measures or claims a benefit to output quality —
generation was greedy and reasonably coherent (spot-checked; not scored
against HumanEval+ tests), but the only thing analyzed is *which experts the
router picked*, at two moments: while priming, and while generating.

Two falsifiable questions:

- **Q1 (router shift):** does priming change which experts the router
  selects, vs. a cold prompt?
- **Q2 (prediction, the linchpin):** do the experts activated during the
  priming pass overlap with the experts activated during generation of the
  real task, *above* a cross-task chance baseline? Expert prefetching only
  makes sense if the answer is yes.

## Model

| | |
|---|---|
| Model | `allenai/OLMoE-1B-7B-0924-Instruct` (`OlmoeForCausalLM`) |
| Total params | 6.9B |
| Active params/token | 1.3B |
| Experts | 64, top-8 routing, all 16 layers are MoE |
| Hidden size | 2048 |
| Precision | bf16 (see "Why bf16, not 4-bit" below) |
| Router capture | forward hook on `model.model.layers[i].mlp.gate` |

### Step 0 — model selection (what was tried and ruled out first)

Per the task spec's preference order:

1. **`Qwen/Qwen3-30B-A3B`** — ruled out before downloading anything.
   `HfApi().model_info(..., files_metadata=True)` reported safetensors
   totaling **~61.1GB**; this machine had ~52-55GB free disk. Doesn't fit.
2. **`Qwen/Qwen1.5-MoE-A2.7B`** (the spec's documented fallback) — downloaded
   (28.6GB), config verified cleanly (60 experts, top-4, 24 layers, all-MoE,
   `Qwen2MoeForCausalLM`). Loading with `BitsAndBytesConfig` 4-bit still
   tried to allocate ~14.3GB VRAM and OOM'd on a 16GB card. Root cause,
   confirmed by reading the installed **transformers 5.13.1** source: this
   version stores every current MoE architecture's expert weights as two
   batched `nn.Parameter` tensors (`gate_up_proj`, `down_proj`) inside a
   custom `*Experts` module — not as individual `nn.Linear` submodules.
   bitsandbytes' automatic 4-bit replacement only targets `nn.Linear`
   instances (`Bnb4BitHfQuantizer.param_needs_quantization` literally checks
   `isinstance(module, bnb.nn.Linear4bit)`), so the batched expert tensors —
   the overwhelming majority of the parameter count — are silently **never
   quantized**. Confirmed this is not Qwen-specific: Mixtral's modeling code
   in the same transformers version uses the identical batched-`Parameter`
   pattern. The 28.6GB download was deleted once this was confirmed.
3. **`allenai/OLMoE-1B-7B-0924-Instruct`** (used) — since 4-bit quantization
   is a structural no-op for MoE experts in this environment regardless of
   model choice, the real constraint became: which well-supported MoE
   model's *unquantized* bf16 footprint fits this machine's GPUs (a 16GB +
   an 8GB card, ~22GB combined free, both partially occupied by other
   processes)? OLMoE (6.9B total, ~13.8GB bf16) fits comfortably with
   headroom for KV-cache and activations. Its `OlmoeSparseMoeBlock` /
   `OlmoeTopKRouter` are structurally identical to Qwen2MoE's — same
   batched-experts pattern, same router forward signature
   `(router_logits, router_scores, router_indices)` — so no code changes
   were needed beyond the model ID. The `-Instruct` variant (not the base
   `-0924` checkpoint) was used specifically because the base checkpoint's
   tokenizer ships with no `chat_template` at all, and this repo's
   `build_chat_prompt` convention (copied from `experiment-2/tim.py`) is
   chat-template-based; the Instruct variant is the same size/architecture
   with a template included.

### Why bf16, not 4-bit

See point 2 above — 4-bit quantization does not reduce VRAM for MoE expert
weights under this transformers version, for any current MoE model checked.
`tim-moe/moe_router_probe.py` still has an `nf4` code path (`--quant nf4`)
for completeness / future re-testing if transformers or bitsandbytes fixes
this, but the probe ran at plain bf16.

### Router capture method

Implemented via **forward hooks** on each layer's `model.model.layers[i].mlp.gate`
module, not via the native `output_router_logits=True` kwarg. Reason: that
kwarg only surfaces `router_logits` through the top-level model output, but
`TIMPrimer.prime()` (`experiment-2/tim_primer.py`, reused completely
unmodified) makes its own internal forward calls through a private
`_forward_with_kv` that only returns `(logits, past_key_values)` — there is
no way to recover `router_logits` from it without editing `TIMPrimer`, which
was out of scope ("do not modify dense-model code"). A forward hook on the
router submodule instead captures routing on *every* forward call
transparently, regardless of who calls the model — TIMPrimer internally
during priming, or this script's own manual decode loop during generation —
covering both phases uniformly with one mechanism and zero changes to any
existing dense-model file.

## Setup

- 20 fixed HumanEval+ tasks (`HumanEval/0`–`HumanEval/19`), greedy decoding,
  `max_new_tokens=768` (actual completions were much shorter — 38 to 377
  tokens, median ~140 — greedy decoding hit EOS naturally on all 20 tasks
  across all conditions).
- Three conditions per task, seed 0:
  - **cold** — task prompt alone, no priming. Generation-phase routing only.
  - **tim_domain** — `TIMPrimer` primes 64 tokens (70% domain-pool / 30%
    random, `PYTHON_DOMAIN_WORDS`), `num_passes=1`. Captures **priming-phase**
    routing (the one forward pass over the 64 noise tokens) and
    **generation-phase** routing (from the warm cache) separately.
  - **tim_random** — same, but `domain_tokens=None` (100% uniform random
    noise) — isolates whether any shift is content-specific or just
    perturbation.
- "Generation-phase routing" deliberately excludes the prompt-prefill
  forward call (which processes the input tokens, not tokens the model
  produces) — a manual greedy-decode loop (`manual_greedy_decode` in
  `moe_router_probe.py`) does the prefill with the capture hook inactive,
  then activates it only for the token-by-token generation that follows.
  This isolates "routing while producing the answer" from "routing while
  reading the question."
- Raw data: `tim-moe/logs/moe_probe/routing_raw.json` (5.3MB) — per task,
  per condition, per phase, per layer: the union of top-8 expert indices
  selected across all tokens in that phase, plus the mean softmax router
  distribution over all 64 experts.

## Q1 — does priming shift the router?

Per-layer Jaccard overlap (generation-phase expert set, cold vs. primed —
1.0 = identical set) and KL divergence (mean router distribution, cold vs.
primed) — see `docs/figures/moe_router_shift.png`:

| | mean Jaccard (1.0 = no shift) | mean KL divergence |
|---|---|---|
| tim_domain vs. cold | 0.912 | 0.0071 |
| tim_random vs. cold | 0.927 | 0.0068 |

**Verdict: yes, priming measurably shifts the router, and it is NOT
content-specific.** Overlap sits at ~0.91-0.93 rather than 1.0 (some real
shift), KL divergence is small but consistently nonzero and grows with layer
depth (from ~0.002-0.005 in layers 0-6 to ~0.011-0.018 in layers 9-15 — see
the figure). Critically, `tim_random` shifts the router by essentially the
same amount as `tim_domain` (if anything, marginally *more*: 0.927 vs. 0.912
Jaccard, i.e. random perturbs slightly less than domain by the overlap
metric but the two KL numbers are within noise of each other, 0.0068 vs.
0.0071). This matches the dense-model finding already documented for TIM
(`docs/int4_mechanism.md`): priming's effect tracks *diversity/perturbation*
of the injected tokens, not their semantic content. There is no evidence here
that domain-specific noise moves the router any more purposefully than
random noise does.

## Q2 — does priming-time routing predict generation-time routing? (the linchpin)

Per-layer Jaccard overlap between the priming-phase expert set and the
same task's generation-phase expert set, compared against a chance baseline
(Jaccard between two *different* tasks' generation-phase expert sets, same
condition) — see `docs/figures/moe_prediction_overlap.png`:

| | priming→generation overlap | chance baseline (cross-task) | gap |
|---|---|---|---|
| tim_domain | 0.678 | 0.862 | **−0.184** |
| tim_random | 0.723 | 0.842 | **−0.119** |

**Verdict: no — priming-time routing predicts generation-time routing *worse*
than chance, not better.** The gap is negative in both conditions and at
every one of the 16 layers individually (see the per-layer table printed by
`analyze_moe_routing.py` / stored in `analysis_summary.json`) — there is no
layer where priming-phase routing comes closer to that task's own
generation-phase routing than two unrelated tasks' generation-phase routings
come to each other. Two real code-generation passes (different tasks, same
condition) resemble each other's expert usage *more* than a task's own
priming pass resembles its own generation pass. This makes some sense
post-hoc: the 64 priming tokens are noise-vocabulary content (random or
70/30 domain/random), a qualitatively different token distribution from
coherent Python-completion tokens — two different real completions apparently
share more routing structure with each other (both being "coherent code")
than either shares with the noise-token priming pass that preceded it.

## Honesty guardrail

This is a routing-mechanics probe. It does not claim, measure, or imply any
accuracy or performance benefit — no HumanEval+ tests were run against these
completions.

**Q1 is positive (priming moves the router) but Q2 is negative (priming
doesn't predict generation).** Per the task's own framing, this is the
"routing perturbed but not *predictively*" outcome: **expert prefetching
(the systems payoff this probe was scoping) is not supported by this
result** — you cannot use which experts activate during a TIM priming pass
to predict which experts a real generation will need, at least not for this
model, at this scale, with this priming recipe (64 tokens, single pass,
70/30 domain/random or pure random noise). Router-hijacking (deliberately
steering the router via priming) remains an open question this probe doesn't
address, but the negative Q2 result means any such steering would be moving
experts without a demonstrated ability to control *which* experts matter for
the eventual generation — i.e., not yet a basis for a targeted intervention.

This does not rule out prefetching under a different priming recipe (e.g.
priming with real task-relevant text rather than vocabulary noise, more
passes, or a larger/different model) — it rules out *this* mechanism (noise-
vocabulary KV-cache priming, as implemented by the existing dense-model
`TIMPrimer`) as a basis for it, on this model.

## Files written

- `tim-moe/moe_router_probe.py` — model loading, router-capture hooks,
  3-condition generation loop.
- `tim-moe/analyze_moe_routing.py` — Q1/Q2 analysis, figures.
- `tim-moe/logs/moe_probe/routing_raw.json` — raw per-task/condition/phase/
  layer routing data (5.3MB).
- `tim-moe/logs/moe_probe/analysis_summary.json` — full per-layer + overall
  Q1/Q2 numbers and verdicts.
- `tim-moe/docs/figures/moe_router_shift.png` — Q1 figure.
- `tim-moe/docs/figures/moe_prediction_overlap.png` — Q2 figure.
- `tim-moe/docs/moe_router_probe.md` — this document.

## Expected but not found / deviations from the original spec

- **Model swapped twice**: `Qwen/Qwen3-30B-A3B` → `Qwen/Qwen1.5-MoE-A2.7B`
  (disk) → `allenai/OLMoE-1B-7B-0924-Instruct` (VRAM + broken bnb
  quantization for batched MoE experts). See Step 0 above for the full
  chain of evidence.
- **Quantization dropped**: ran at bf16, not 4-bit, because 4-bit is
  currently a no-op for MoE expert weights in this environment
  (transformers 5.13.1 + bitsandbytes 0.49.2) — not a probe-specific
  workaround, a real library-level gap worth knowing about for any future
  MoE work in this environment.
- **Router capture via hooks, not `output_router_logits=True`**: a
  deliberate choice, not a fallback-of-last-resort — see "Router capture
  method" above; the native kwarg genuinely cannot reach through
  `TIMPrimer`'s private forward calls without editing it.
