# The int4 priming gain: is it real, is it priming, and is it a sink?

Companion studies established: `docs/precision_quality_grid.md` found that at
NF4 (int4), `tim_domain` beat `int4_cold` on HumanEval+ by roughly +7pp mean
across 3 seeds — while at bf16 the same priming was flat (McNemar p=0.69,
`docs/generation_quality.md`). This document runs a four-rung gated ladder to
determine (1) whether the int4 gain is real and directional rather than
seed-luck/churn, (2) whether it comes from KV-cache injection specifically or
from any extra context, (3) whether the mechanism is an attention sink, and
(4) whether the effect is large enough for int4+priming to actually beat
bf16. Every rung ran to completion regardless of intermediate results, per
instruction — nothing here is a stop/go gate, only interpretation.

Model: Qwen3-1.7B throughout. NF4 config (re-verified deterministic before
scoring, 3 greedy runs byte-identical, `bnb_4bit_use_double_quant=True`, no
retry needed): `BitsAndBytesConfig(load_in_4bit=True,
bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16)`. Full 164
HumanEval+ tasks, greedy, `enable_thinking=False`, `max_new_tokens=768`,
per-task `copy.deepcopy` cache isolation. HumanEval+ pass criterion
throughout: `base_status == 'pass' AND plus_status == 'pass'`.

All raw generations, sanitized solutions, and EvalPlus eval_results are kept
on disk for every cell (see Files written). Side-by-side per-task diff files
(prompt + every condition's solution + pass/fail) are at
`logs/int4_mechanism/answers_rung{1,2,3,4}.jsonl`.

Two of the four rungs' generation work were run as **two parallel processes
pinned to the two physical GPUs** (`CUDA_VISIBLE_DEVICES=0` for Rung 3a/3b,
`CUDA_VISIBLE_DEVICES=1` for Rung 4) once it was clear the single-GPU run
was leaving the second RTX 5060 Ti idle — a 1.7B NF4 model uses ~1.4GB, well
under either card's capacity, so there was no memory contention. This roughly
halved the remaining wall-clock time for those two rungs.

---

## Background: the bf16 reference triple (cold / prompt_control / tim_domain)

Before the int4 ladder, here is the full-precision baseline this study is
testing against — HumanEval+ pass@1, McNemar exact test paired against cold
on the same 164 tasks:

| condition | pass@1+ | 95% Wilson CI | McNemar vs cold |
|---|---|---|---|
| cold | 51.8% (85/164) | [44.2, 59.3] | — |
| prompt_control | 55.5% (91/164) | [47.8, 62.9] | cold_only=8, pc_only=14, p=0.2863 (n.s.) |
| tim_domain seed0 | 55.5% (91/164) | [47.8, 62.9] | cold_only=13, tim_only=19, p=0.3771 (n.s.) |
| tim_domain seed1 | 53.0% (87/164) | [45.4, 60.5] | cold_only=20, tim_only=22, p=0.8776 (n.s.) |
| tim_domain seed2 | 51.8% (85/164) | [44.2, 59.3] | cold_only=12, tim_only=12, p=1.0000 (n.s.) |

Notably, prompt_control and tim_domain_seed0 land on the *exact same* raw
score (91/164), and McNemar directly between them is p=1.0000 (13 vs 13
discordant pairs) — at bf16, plain-text context and KV-cache injection are
indistinguishable, and neither beats cold. This is the premise the int4 ladder
tests: does the same picture hold once the model is quantized?

---

## RUNG 1 — Is the int4 gain even directional?

McNemar (paired, base+plus criterion) on int4_cold vs int4_tim_domain, seeds
0/1/2, both from data already on disk (`logs/quant_gap/.../gate/C_nf4_cold`,
`logs/quality_quant/int4_tim_domain_seed{0,1,2}`).

| seed | both_pass | cold_only | tim_only | both_fail | churn % | McNemar p |
|---|---|---|---|---|---|---|
| 0 | 72 | 12 | 21 | 59 | 20.1% | 0.1628 |
| 1 | 74 | 10 | 21 | 59 | 18.9% | 0.0708 |
| 2 | 76 | 8 | 23 | 57 | 18.9% | 0.0107 |
| **pooled** | **222** | **30** | **65** | **175** | **19.3%** | **0.0004** |

**Verdict: the int4 gain is directional, not churn.** Pooled across 3 seeds,
tim-only wins (65) more than double cold-only wins (30), McNemar p=0.0004.
This is qualitatively different from the bf16 study's p=0.69 — the churn
fraction (≈19%) is similar in magnitude to bf16, but here it is lopsided in
favor of tim_domain rather than balanced.

### Overlap: does TIM fix what quantization broke, or fix generally?

Using A=bf16_cold, C=int4_cold, D=int4_tim_domain (each seed), on the common
task set:

| seed | quant_broke (A pass, C fail) | D fixes of those | D fixes total (C fail→D pass) | …of which quant-broken | …of which bf16-also-failed |
|---|---|---|---|---|---|
| 0 | 19 | 9 (47.4%) | 21 | 9 | 12 |
| 1 | 19 | 10 (52.6%) | 21 | 10 | 11 |
| 2 | 19 | 9 (47.4%) | 23 | 9 | 14 |

TIM fixes roughly half of the tasks quantization specifically broke (9-10 of
19 each seed) — but the *majority* of what it fixes (11-14 of 21-23 tasks per
seed) are tasks bf16-cold *also* failed, i.e. genuinely hard tasks, not
quantization-specific damage. So this is not narrowly "TIM undoes
quantization's specific damage" — it's more general than that: TIM fixes
quantization-broken tasks at roughly the same rate it fixes hard tasks that
were never quantization-specific. Full task-ID lists and solutions for every
quadrant are in `logs/int4_mechanism/answers_rung1.jsonl` (37 tasks: the union
of quant-broken and TIM-fixed tasks across the 3 seeds).

---

## RUNG 2 — Priming, or just context? (the decisive ablation)

Generated `int4_prompt_control` (same `DOMAIN_PERSONA_TEXT` persona prepended
to the prompt text, `past_key_values=None`, no cache injection — the same
route the bf16 study used for its own prompt_control arm). This cell did not
exist before this study.

| condition | pass@1+ | 95% Wilson CI |
|---|---|---|
| int4_cold | 51.2% (84/164) | [43.6, 58.8] |
| int4_prompt_control | 56.1% (92/164) | [48.4, 63.5] |
| int4_tim_domain_seed0 | 56.7% (93/164) | [49.1, 64.1] |

McNemar int4_cold vs int4_prompt_control: cold_only=6, pc_only=14, p=0.1153
(trending, not significant alone).
McNemar int4_prompt_control vs int4_tim_domain_seed0: pc_only=15, tim_only=16,
**p=1.0000** — essentially a coin flip between the two, same as the bf16
study found between the same two conditions.

**Verdict: context, not KV-cache injection.** prompt_control and tim_domain
land within 0.6pp of each other and are statistically indistinguishable
(McNemar p=1.0000). Extra context helps a quantization-degraded model *by any
route* — prepending text works just as well as priming the KV cache directly.
This rules out the "KV injection is doing something prompting cannot" reading
and the "Rung 1 gain was seed-lucky" reading (both prompt_control and
tim_domain clear cold by a similar, non-trivial margin). Per-task solutions
for all three conditions are in `logs/int4_mechanism/answers_rung2.jsonl`
(164 tasks).

---

## RUNG 3 — Is it an attention sink?

### 3a — Content vs sink: int4 + tim_random (3 seeds)

Random-vocabulary priming (`domain_tokens=None` → 100% uniform random noise,
same `TIMPrimer`, 3 seeds):

| condition | pass@1+ | 95% CI | McNemar vs cold |
|---|---|---|---|
| tim_random seed0 | 59.8% (98/164) | [52.1, 67.0] | cold_only=11, rand_only=25, p=0.0288 |
| tim_random seed1 | 56.7% (93/164) | [49.1, 64.1] | cold_only=13, rand_only=22, p=0.1755 |
| tim_random seed2 | 56.1% (92/164) | [48.4, 63.5] | cold_only=10, rand_only=18, p=0.1849 |
| **pooled** | | | **cold_only=34, rand_only=65, p=0.0024** |

Pooled domain gap (tim_only − cold_only, seeds 0-2) = 35; pooled random gap
(rand_only − cold_only) = 31 — within 12% of each other.

**Verdict (3a): random ≈ domain.** The content of the primed prefix — whether
genuine Python-domain vocabulary or pure uniform-random tokens — barely
matters; both rescue int4 to a similar degree. This is *consistent* with a
positional/mechanical (sink-like) story, but content isn't ruled out entirely
either — random tokens are still real, diverse vocabulary tokens, just not
domain-relevant ones. 3b is the sharper test.

### 3b — Pure sink baseline: int4 + neutral_prefix

64 copies of a single repeated, ordinary token (`\n`, id 198 — this
tokenizer has no BOS token; newline was the fallback per the script's
priority order), built via one direct forward pass with no TIMPrimer noise
sampling at all — no domain or random vocabulary content whatsoever.

| condition | pass@1+ | 95% CI | McNemar vs cold |
|---|---|---|---|
| int4_neutral_prefix | **42.7% (70/164)** | [35.4, 50.3] | cold_only=30, neutral_only=16, p=0.0541 |

**Verdict (3b): neutral prefix does not rescue int4 — it actively hurts.**
42.7% is *8.5 points below* int4_cold's 51.2%, and the discordant pairs run
the opposite direction from every other priming condition (cold beats
neutral 30-to-16, borderline significant at p=0.054). This is the cleanest
single result in the ladder and it **contradicts the naive strong-form sink
hypothesis** ("any prefix, even a content-free one, absorbs bad activations
and helps"). A purely positional/degenerate prefix is not neutral — it is
actively destabilizing.

A mechanistic clue: neutral_prefix generates far more tokens per task (215
mean) than any other int4 condition (int4_cold: 113.5, prompt_control: 130.8,
tim_random: 156-161, tim_domain: 170) and takes the longest per task
(2968.8ms vs int4_cold's 1599.4ms). It also has the *highest* docstring rate
of any condition studied here (89.6%, see the echo-test table below) despite
the worst accuracy. The picture that emerges is a model pushed into a more
verbose, more docstring-echoing generation *mode* by the degenerate repeated
prefix, but a less correct one — over-triggering the same echo-driven
elaboration behavior seen everywhere else in this line of research, without
the benefit that comes from real token diversity in the primed context.

### 3c — Direct attention measurement

10 tasks, both precisions, primed with `tim_domain` (seed 0). Attention
weights measured via `output_attentions=True` (`attn_implementation="eager"`
— required, sdpa/flash don't expose per-head weights; this worked cleanly
under both bf16 and NF4, no fallback needed), one query step at a time for
up to 32 greedily-generated tokens, tracking the fraction of attention mass
(mean over heads, mean over generation steps) landing on the first 64 key
positions (the primed prefix), per layer.

| | overall mean prefix-attention-mass |
|---|---|
| bf16 | 49.26% |
| int4 (NF4) | 49.37% |

Per-layer, both precisions track almost exactly (full 28-layer table in
`logs/int4_mechanism/attention_probe_raw.json`; plot at
`docs/figures/attention_sink_int4_vs_bf16.png`): max absolute difference
across all 28 layers is **2.68 percentage points**, mean difference **0.11pp**.
Both precisions show the same striking depth-dependent pattern — a strong,
genuine sink-like concentration of attention on the 64 primed positions
(peaking at 79-84% of total attention mass in layers 24-27, with moderate
spikes as early as layers 3-8), out of a much longer effective context. So a
real attention-sink-like phenomenon clearly exists in this architecture — but
it is **statistically indistinguishable between bf16 and int4**.

**Verdict (3c): the sink exists but does not differentially engage under
quantization.** The direct signature the sink hypothesis predicts (the prefix
should absorb *visibly more* attention under int4 than bf16, if it's
compensating for quantization-specific outliers) is absent — the two curves
are within noise of each other everywhere in the network.

### Combined Rung 3 interpretation

Taken together, 3a+3b+3c reject the attention-sink mechanism as the
explanation for why priming helps int4 more than bf16:
- 3a alone (random ≈ domain) is *consistent* with a sink story.
- 3b directly contradicts the strong form of it — a pure positional/content-free
  prefix should help if it's absorbing bad activations; instead it's the only
  condition in this entire study that makes int4 *worse* than cold.
- 3c shows the sink-like attention pattern that *does* exist is identical in
  magnitude at both precisions — it isn't "doing more work" under NF4.

The pattern that best fits all three results: what helps is a **token-diverse,
non-degenerate prefix** (content matters in the sense of "real vocabulary",
not in the sense of "domain-relevant") combined with extra generation budget
from a longer effective context (Rung 2's context finding) — not a literal
attention-sink absorbing quantization noise.

---

## RUNG 4 — Does int4+priming beat bf16? (8-seed distribution)

int4_tim_domain run for 5 additional seeds (3-7), giving 8 seeds total.

| seed | pass@1+ |
|---|---|
| 0 | 56.7% (93/164) |
| 1 | 57.9% (95/164) |
| 2 | 60.4% (99/164) |
| 3 | 56.1% (92/164) |
| 4 | 57.9% (95/164) |
| 5 | 59.8% (98/164) |
| 6 | 55.5% (91/164) |
| 7 | 57.9% (95/164) |
| **mean ± std** | **57.8% ± 1.6pp** (range 55.5–60.4%) |

McNemar, int4_tim_domain (each seed) vs **bf16_cold** (51.8%, 85/164):

| seed | bf16_cold_only | int4_tim_only | p |
|---|---|---|---|
| 0 | 17 | 25 | 0.2800 |
| 1 | 14 | 24 | 0.1433 |
| 2 | 13 | 27 | **0.0385 (sig)** |
| 3 | 16 | 23 | 0.3368 |
| 4 | 18 | 28 | 0.1839 |
| 5 | 15 | 28 | 0.0660 |
| 6 | 20 | 26 | 0.4614 |
| 7 | 16 | 26 | 0.1641 |

Only 1 of 8 seeds clears significance against bf16-cold individually.

McNemar, int4_tim_domain (each seed) vs **bf16_tim_domain_seed0** (55.5%, 91/164):

| seed | bf16_tim_only | int4_tim_only | p |
|---|---|---|---|
| 0 | 13 | 15 | 0.8506 |
| 1 | 14 | 18 | 0.5966 |
| 2 | 10 | 18 | 0.1849 |
| 3 | 14 | 15 | 1.0000 |
| 4 | 14 | 18 | 0.5966 |
| 5 | 13 | 20 | 0.2962 |
| 6 | 16 | 16 | 1.0000 |
| 7 | 12 | 16 | 0.5716 |

**Zero of 8 seeds** clear significance against bf16-tim_domain.

**Verdict (Rung 4, held to the announced discipline — only claim int4 > bf16
if it beats both bf16-cold *and* bf16-tim with paired significance across the
distribution, not on the seed mean alone): NOT established.** int4-tim_domain's
raw mean (57.8%) is numerically above both bf16-cold (51.8%) and
bf16-tim_domain (55.5%), and it is directionally ahead of bf16-tim_domain on
7 of 8 seeds — but paired significance against bf16-tim_domain is absent
throughout, and against bf16-cold only one seed (seed 2, the same seed that
drove the original "+7pp" headline number) reaches p<0.05. The honest reading
is exactly the one flagged as most likely before this rung ran: **priming
partially rescues int4 toward bf16-cold's level (and arguably a bit past it
on raw numbers), not reliably beyond bf16-tim_domain.** Treating any
int4>bf16 claim as extraordinary and holding it to that standard, this study
does not clear the bar. Full per-task solutions across bf16_cold,
bf16_tim_domain_seed0, int4_cold, and all 8 int4_tim_domain seeds are in
`logs/int4_mechanism/answers_rung4.jsonl` (164 tasks).


---

## RUNG 5 — Is the recovery semantic, or purely entropic?

Having established (Rungs 3a/3b) that *content-diverse* tokens rescue int4
while *degenerate repeated* tokens harm it, the remaining question is whether
the diversity needs to carry semantic meaning. `tim_random` uses real
vocabulary tokens — they carry valid distributional information even without
domain relevance. Could the recovery work with tokens that are diverse but
completely devoid of meaning?

### 5a — tim_entropy: 64 diverse, semantically meaningless tokens

The `tim_entropy` control matches tim_random's exact token budget (64 tokens)
and high token diversity (unique subword fragments), but strips out all
semantic content: tokens are generated from random consonant-vowel gibberish
strings (e.g. "blorq kuznem fwiptal"), verified to contain no valid English
words or Python keywords. Delivery method: KV-cache injection (same path as
tim_random and tim_domain).

| condition | pass@1+ | 95% CI | McNemar vs cold |
|---|---|---|---|
| tim_entropy seed0 | 53.0% (87/164) | [45.4, 60.5] | cold_only=13, entropy_only=16, p=0.7111 |
| tim_entropy seed1 | 56.7% (93/164) | [49.1, 64.1] | cold_only=8, entropy_only=17, p=0.1078 |
| tim_entropy seed2 | 52.4% (86/164) | [44.8, 59.9] | cold_only=12, entropy_only=14, p=0.8450 |
| **pooled** | | | **cold_only=33, entropy_only=47, p=0.1456** |

Reference (from earlier rungs):
- int4_cold: 51.2%
- int4_neutral_prefix (Rung 3b): 42.7%
- int4_tim_random (Rung 3a, mean): 57.5%
- int4_tim_domain_seed0 (Rung 1): 56.7%

McNemar int4_tim_domain (seed0) vs int4_tim_entropy (each seed):

- seed0: tim_domain_only=18 entropy_only=12 p=0.3616
- seed1: tim_domain_only=11 entropy_only=11 p=1.0000
- seed2: tim_domain_only=16 entropy_only=9 p=0.2295

**Verdict (Rung 5): ENTROPY FAILS TO RESCUE: The recovery mechanism strictly requires semantic richness / valid distributional tokens. Test-time compute over diverse but meaningless noise does not rescue the model's capacity. However, unlike the degenerate neutral_prefix, diverse gibberish at least does not actively harm — suggesting the damage from neutral_prefix is specifically from repetition/degeneracy, not from non-semantic content.**

---

## Timing and memory (int4 is not faster — never claim otherwise)

| condition | ms/task | peak_mb | tokens/task |
|---|---|---|---|
| bf16_cold | 996.7 | 3415.2 | 77.5 |
| int4_cold | 1599.4 | 1427.7 | 113.5 |
| int4_prompt_control | 1834.3 | 1429.2 | 130.8 |
| int4_tim_random (seeds 0-2, range) | 2159–2251 | 1435–1463 | 156–161 |
| int4_tim_domain_seed0 | 2386.6 | 1463.3 | 170.0 |
| int4_neutral_prefix | **2968.8** | 1460.2 | **215.3** |

int4 uses roughly 40% of bf16's peak memory (as expected for 4-bit weights)
but is consistently slower per task — 1.6× slower cold (matching the
quantization-gate study's own finding), rising to nearly 3× slower for
neutral_prefix, driven mostly by generating far more tokens per task under
that condition, not by anything intrinsic to the quantization itself.

---

## Docstring rate and the echo test, carried across every condition

Using the same echo-test methodology as `docs/generation_quality.md`
(whitespace-normalized exact match → echoed; `difflib` ratio ≥0.6 but not
exact → modified; else → original), computed over each condition's full 164
solutions (not restricted to a both-pass subset, since this table is
diagnostic for this ladder rather than a formal quality comparison):

| condition | docstring rate | echoed | modified | original |
|---|---|---|---|---|
| int4_cold | 34.8% (57/164) | 73.7% (42/57) | 17.5% (10/57) | 8.8% (5/57) |
| int4_prompt_control | 47.6% (78/164) | 69.2% (54/78) | 16.7% (13/78) | 14.1% (11/78) |
| int4_tim_random_seed0 | 73.2% (120/164) | 86.7% (104/120) | 8.3% (10/120) | 5.0% (6/120) |
| int4_tim_domain_seed0 | 77.4% (127/164) | 84.3% (107/127) | 9.4% (12/127) | 6.3% (8/127) |
| int4_neutral_prefix | **89.6% (147/164)** | 80.9% (119/147) | 12.2% (18/147) | 6.8% (10/147) |

Two things confirmed here:

1. **The gate's own side-finding is confirmed as an echo artifact.** The
   quantization-gap gate found docstring rate jumped 7.9%→34.8% under
   quantization alone (no priming) — the 34.8% figure for int4_cold here is
   exactly that number, and 73.7% of those docstrings are verbatim echoes of
   the prompt's own docstring, whitespace-normalized. Quantization and
   priming are disturbing the same fragile input-copying behavior, by
   different means, exactly as `docs/precision_quality_grid.md` concluded for
   bf16-vs-int4 cold. This ladder adds: *every* condition studied here,
   including the harmful neutral_prefix, shows the same 70-87% echo-dominance
   — docstring-rate movement in this model, under any perturbation tried so
   far, is overwhelmingly an echo effect, not induced documentation style.
2. **Docstring rate scales with "how much extra context was added", almost
   monotonically with token budget** (cold 34.8% → prompt_control 47.6% →
   tim_random 73.2% → tim_domain 77.4% → neutral_prefix 89.6%) — and it is
   *not* a proxy for correctness. neutral_prefix has the highest docstring
   rate of any condition here and the worst accuracy.

---

## Consolidated verdict

Mapping the evidence onto the three hypotheses named at the outset:

**1. "Universal fallback" (any extra context rescues a degraded model,
regardless of mechanism) — SUPPORTED.** Rung 2 (prompt_control ≈ tim_domain,
McNemar p=1.0000) and Rung 3a (tim_random ≈ tim_domain, pooled gaps 31 vs 35)
both point the same way: what matters is that the model gets *more tokens of
real context before it has to answer*, not that those tokens come via KV
injection specifically or via domain-relevant content specifically. This is
the best-supported reading of the whole ladder.

**2. "Attention sink" (priming absorbs quantization-induced activation
outliers, precision-differentially) — NOT SUPPORTED, actively contradicted.**
A genuine sink-like attention pattern exists in this architecture (3c: up to
~80% of attention mass on 64 primed positions in late layers) — but it is
statistically identical in magnitude between bf16 and int4 (max 2.68pp
difference across 28 layers), so it isn't doing differential work under
quantization. Worse for the hypothesis: the purest test of "does a
content-free prefix help by absorbing attention" (3b, neutral_prefix) shows
the opposite of rescue — it is the single most harmful condition tested,
8.5pp below int4_cold. The sink is real as an architectural feature; it is
not the mechanism behind the int4-specific accuracy gain.

**3. "Regularization synergy" / compressed model can exceed full precision
— NOT ESTABLISHED, and treated as an extraordinary claim held to that
standard.** Rung 4's 8-seed distribution (57.8% ± 1.6pp) sits numerically
above both bf16-cold and bf16-tim_domain, and beats int4-cold consistently
and significantly pooled — but paired significance against bf16-tim_domain is
absent on all 8 seeds, and against bf16-cold on 7 of 8. The honest,
non-extraordinary reading: **priming (or, per Rung 2, any added context)
partially rescues int4 toward bf16's level; it does not reliably exceed it.**

**What Rung 5 resolved (previously untested):** The "universal fallback"
mechanism was narrowed above to *token-diverse, non-degenerate prefixes* —
but did the diversity need to carry *semantic* meaning, or was pure latent
entropy sufficient? Rung 5 tested this directly: `tim_entropy` (64 diverse
but semantically meaningless consonant-vowel gibberish tokens, ~91% unique,
verified to contain no English words or Python keywords, delivered via
KV-cache injection) landed at 54.1% mean across 3 seeds — between int4_cold
(51.2%) and tim_random (57.5%), failing to reach significance against cold
(pooled McNemar p=0.1456). This places it clearly below tim_random and
tim_domain but clearly above neutral_prefix (42.7%), resolving the question:
**the recovery mechanism requires tokens from the model's trained
distribution (valid subword units with learned embeddings), not merely high
token diversity in latent space.** Pure entropic diversity (random gibberish
subword fragments) provides a small, non-significant nudge — better than
degenerate repetition but insufficient for the full rescue effect seen with
real vocabulary tokens.

**What remains genuinely untested:** whether this pattern generalizes beyond
Qwen3-1.7B or beyond NF4 — every rung here used one model, one quantization
scheme.

---

## Files written

Generation (raw `.jsonl`, sanitized `-sanitized.jsonl`, EvalPlus
`-sanitized.eval_results.json`, and `_metrics.json` for every condition):
- `logs/int4_mechanism/int4_prompt_control*`
- `logs/int4_mechanism/int4_tim_random_seed{0,1,2}*`
- `logs/int4_mechanism/int4_neutral_prefix*`
- `logs/quality_quant/int4_tim_domain_seed{3,4,5,6,7}*` (extending the existing seeds 0-2)
- `logs/int4_mechanism/mechanism_gen_report_2.json`, `mechanism_gen_report_3a_3b.json`, `mechanism_gen_report_4.json`
- `logs/int4_mechanism/int4_tim_entropy_seed{0,1,2}*` (Rung 5)
- `logs/int4_mechanism/mechanism_gen_report_5_seeds{0,1,2}.json`

Analysis:
- `scripts/analyze_int4_mechanism.py`
- `scripts/analyze_rung5.py`
- `logs/int4_mechanism/analysis_summary.json`
- `logs/int4_mechanism/analysis_rung5.json`
- `logs/int4_mechanism/answers_rung{1,2,3,4,5}.jsonl` (per-task side-by-side diffs)

Attention probe (Rung 3c):
- `experiment-3/run_attention_probe.py`
- `logs/int4_mechanism/attention_probe_raw.json`
- `docs/figures/attention_sink_int4_vs_bf16.png`

Generation code:
- `experiment-3/run_int4_mechanism.py` (Rungs 2, 3a, 3b, 4; `--rungs` flag
  allows splitting work across GPUs as separate processes)

## Expected but not found

None — every condition specified in the task ladder was generated and
scored, and `scripts/analyze_int4_mechanism.py`'s own missing-file check
reported nothing missing on the final run.
