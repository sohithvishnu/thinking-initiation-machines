# Precision x priming grid: does KV-cache priming behave the same way at int4 as at bf16?

Extends [generation_quality.md](generation_quality.md) (bf16 quality panel +
LLM judge) and [quantization_gap.md](quantization_gap.md) (NF4 barely dents
accuracy — gate failed at +0.6 pp) across precision. Model: Qwen3-1.7B
throughout. No existing code was rewritten — `scripts/quality_panel_grid.py`
and `scripts/llm_judge_grid.py` import the metric/judge functions from
`scripts/quality_panel.py` and `scripts/llm_judge.py` unchanged (see each
new file's module docstring for exactly what's imported vs. what's new).
`experiments/int4_tim_domain.py` imports model loading, determinism
checking, and the timed `run_condition` runner from
`experiments/quant_gap.py` rather than reimplementing them.

**Premises, not failures**: bf16 pass@1 is flat under priming (McNemar
p=0.69) and int4 barely dents pass@1 at all (gate: damage=+0.6pp, failed its
own significance threshold). Both are the reason this study exists — quality
differences on the identical-outcome subset are the only differences pass@1
can't already tell you about.

## Locating the four cells

| | bf16 | int4 (NF4) |
|---|---|---|
| cold | `logs/tim/Qwen_Qwen3-1.7B/cold*` (on disk) | `logs/quant_gap/Qwen_Qwen3-1.7B/gate/C_nf4_cold*` (on disk) |
| tim_domain | `logs/tim/Qwen_Qwen3-1.7B/tim_domain_seed{0,1,2}*` (on disk, 3 seeds) | `logs/quality_quant/int4_tim_domain_seed{0,1,2}*` (**generated this session**) |

int4-cold's sanitized solutions and eval_results **were already present**
(`C_nf4_cold-sanitized.jsonl`, `C_nf4_cold-sanitized.eval_results.json`) —
regeneration was not needed, and both int4 cells therefore come from the
same NF4 config and the same shared code path (the gate script and
`int4_tim_domain.py` both load through `tim.models.load_model(quant="nf4")`).

## Step 1 — Generating int4 + tim_domain

**NF4 determinism re-check**: 3 greedy runs of one prompt, byte-identical on
the first attempt (`bnb_4bit_use_double_quant=True`, batch size 1) — same
result as the gate. No retry needed.

Config: `BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16)`,
loaded parameter dtypes `{torch.bfloat16, torch.uint8}` (matches the gate).
Priming: `TIMPrimer(noise_length=64, num_passes=1)` from
`tim/primer.py`, `PYTHON_DOMAIN_WORDS` (the 211-word pool built
from `keyword.kwlist` + `dir(builtins)` in `tim/vocab.py`, the same pool
`quant_gap.py` and the bf16 study both use), seeds 0/1/2, per-task `copy.deepcopy` cache
isolation (no cache shared across tasks). Full 164 tasks, greedy,
`enable_thinking=False`, `max_new_tokens=768`, sanitized and scored with
`evalplus.sanitize` / `evalplus.evaluate`.

| int4 tim_domain | pass@1 (base) | pass@1 (base+extra) | tokens/task | ms/task | peak MB |
|---|---|---|---|---|---|
| seed0 | 62.8% (103/164) | 56.7% (93/164) | 170.0 | 2386.6 | 1463.3 |
| seed1 | 62.8% (103/164) | 57.9% (95/164) | 175.8 | 2452.2 | 1468.8 |
| seed2 | 66.5% (109/164) | 60.4% (99/164) | 178.4 | 2482.0 | 1468.0 |

**Not "faster"**: int4+tim_domain ran 1.49x slower per task than int4-cold
alone (2386.6 vs 1599.4 ms, seed0), and 2.39x slower than bf16-cold
(2386.6 vs 996.7 ms). Quantization's dequant overhead (measured in the gate:
1.6x) and priming's per-task deepcopy+longer-cache cost stack rather than
offset. Peak memory stayed essentially flat vs int4-cold (1463-1469 MB vs
1427.7 MB) — priming a 64-token cache costs little extra memory next to
quantization's ~59% reduction from bf16.

## Step 2 — Objective panel across the grid

`scripts/quality_panel_grid.py` reused `quality_panel.py`'s
`load_solutions`, `load_eval_results`, `compute_condition_metrics`,
`summarize_pair`, `fragile_rate`, and the ruff/radon/ast/Wilcoxon machinery
unchanged; it only adds multi-directory cell loading and the cross-precision
pairing. tim_domain is represented by **seed0 only** at both precisions —
matching the bf16 study's own protocol (its Part 1 panel and Part 2 judge
were both single-seed, seed0, despite 3 seeds existing on disk; the task's
recollection of "3 seeds for the panel" doesn't match what was actually
published in `generation_quality.md`, so seed0-only is used here too for
genuine comparability rather than silently changing methodology).

**EvalPlus fragile rate** (base pass, plus fail — full 164, all four cells):

| condition | fragile / 164 | rate |
|---|---|---|
| bf16 cold | 7 | 4.3% |
| bf16 tim_domain (seed0) | 8 | 4.9% |
| int4 cold | 7 | 4.3% |
| int4 tim_domain (seed0) | 10 | 6.1% |

Fragility never improves under priming, at either precision — same pattern
as the bf16-only study.

### bf16: cold vs tim_domain (n=72 both-pass) — reproduced for side-by-side

| metric | bf16 cold | bf16 tim_domain | p |
|---|---|---|---|
| **lint violations (raw)** | 0.944 ± 2.047 | 1.236 ± 2.458 | **0.0034** * |
| **lint violations/LOC** | 0.153 ± 0.327 | 0.176 ± 0.376 | **0.0167** * |
| type-hint coverage | 0.451 ± 0.423 | 0.451 ± 0.423 | n/a (identical) |
| comment density | 0.003 ± 0.023 | 0.018 ± 0.089 | 0.0679 |
| **docstring presence** | 0.069 ± 0.254 | 0.500 ± 0.500 | **<0.0001** * |
| cyclomatic complexity | 3.139 ± 1.521 | 3.111 ± 1.629 | 0.6664 |
| **maintainability index** | 78.026 ± 12.715 | 83.344 ± 14.147 | **0.0014** * |
| **function length (lines)** | 7.056 ± 5.522 | 10.500 ± 6.614 | **<0.0001** * |

Exact match to `generation_quality.md`'s numbers — a clean reproducibility
check on the shared pipeline.

### int4: cold vs tim_domain (n=72 both-pass) — new

| metric | int4 cold | int4 tim_domain | p |
|---|---|---|---|
| lint violations (raw) | 1.125 ± 2.192 | 1.236 ± 2.452 | 0.0679 |
| lint violations/LOC | 0.178 ± 0.371 | 0.176 ± 0.369 | 0.8017 |
| type-hint coverage | 0.400 ± 0.434 | 0.414 ± 0.445 | 0.1573 |
| comment density | 0.072 ± 0.210 | 0.071 ± 0.207 | 0.7794 |
| **docstring presence** | 0.444 ± 0.497 | 0.792 ± 0.406 | **<0.0001** * |
| cyclomatic complexity | 3.542 ± 2.459 | 3.556 ± 2.254 | 0.6770 |
| **maintainability index** | 80.027 ± 13.564 | 83.543 ± 14.144 | **0.0389** * |
| **function length (lines)** | 12.014 ± 9.845 | 16.542 ± 11.292 | **<0.0001** * |

### cold: bf16 vs int4 (n=66 both-pass) — the cross-precision comparison

| metric | bf16 cold | int4 cold | p |
|---|---|---|---|
| lint violations (raw) | 1.061 ± 2.173 | 1.212 ± 2.280 | 0.0722 |
| **lint violations/LOC** | 0.166 ± 0.338 | 0.194 ± 0.385 | **0.0190** * |
| type-hint coverage | 0.429 ± 0.430 | 0.429 ± 0.430 | n/a (identical) |
| **comment density** | 0.007 ± 0.039 | 0.038 ± 0.115 | **0.0117** * |
| **docstring presence** | 0.091 ± 0.287 | 0.379 ± 0.485 | **<0.0001** * |
| cyclomatic complexity | 3.258 ± 1.570 | 3.144 ± 1.618 | 0.2110 |
| **maintainability index** | 78.111 ± 12.567 | 81.429 ± 13.358 | **0.0330** * |
| **function length (lines)** | 7.409 ± 5.702 | 10.258 ± 7.502 | **<0.0001** * |

**Quantization alone — no priming involved — moves the same metrics in the
same direction priming does at bf16**: docstring presence, comment density,
function length, and maintainability index all rise significantly on
NF4-quantized cold generation vs bf16 cold generation. Fragile rate does
not (4.3% both), and cyclomatic complexity does not (flat).

### Docstring provenance / echo test — all four cells (both-pass-restricted, per pair)

| condition (both-pass set) | docstrings | echoed | modified | original |
|---|---|---|---|---|
| bf16 cold (in bf16 pair, n=72) | 4/72 | 3 (75.0%) | 1 (25.0%) | 0 |
| **bf16 tim_domain (n=72)** | **36/72** | **32 (88.9%)** | 1 (2.8%) | 3 (8.3%) |
| int4 cold (in int4 pair, n=72) | 32/72 | 27 (84.4%) | 4 (12.5%) | 1 (3.1%) |
| **int4 tim_domain (n=72)** | **57/72** | **47 (82.5%)** | 8 (14.0%) | 2 (3.5%) |
| bf16 cold (in cross-precision pair, n=66) | 6/66 | 4 (66.7%) | 1 (16.7%) | 1 (16.7%) |
| **int4 cold (n=66)** | **25/66** | **22 (88.0%)** | 2 (8.0%) | 1 (4.0%) |

**The echo test governs interpretation at both precisions, exactly as it
did at bf16**: 82-89% of docstring-bearing solutions under tim_domain
priming (either precision) are verbatim copies of the prompt's own
docstring, not new text. And critically — **int4-cold's own elevated
docstring rate (32/72 = 44.4% in the int4 pair, 25/66 = 37.9% in the
cross-precision pair, vs bf16-cold's 4-6%) is itself 84-88% echoed.**
Quantization alone reproduces the same input-copying behavior priming
does, by a completely different mechanism (weight perturbation vs
KV-cache context injection). This directly answers the gate's side-finding
(int4-cold docstring rate 34.8% on the full 164, in `quantization_gap.md`):
**yes, that too is input-echoing, not induced documentation style.**

## Step 3 — LLM judge, two pairings

Judge model confirmed via `ollama list`: `gemma4:12b` (same tag the bf16
study used; no `gemma3:12b` tag exists in this environment).
`think: false` was carried over from the bf16 fix (avoids empty `content`
from the model spending its token budget in a thinking trace) — 0
`judge_error` across all 276 calls in both pairings below.
`scripts/llm_judge_grid.py` reused `query_judge`, `get_model_tag`, and
`JUDGE_PROMPT` from `llm_judge.py` unchanged; only the cache-key schema was
extended (pairing name added) since this script runs two pairings against
one shared cache file, and `llm_judge.py`'s own cache/load functions were
made to accept a path parameter (backward compatible — the original bf16
script's behavior is unchanged) rather than duplicated.

### int4: cold vs tim_domain (n=72 both-pass) — direct analog of the bf16 judge run

| | count | % of order-consistent |
|---|---|---|
| order-consistent decisions | 61 / 72 | — |
| int4_tim_domain wins | 21 | 34.4% |
| int4_cold wins | 14 | 23.0% |
| ties | 26 | 42.6% |
| **position_bias** | **11 / 72** | **15.3%** |
| judge_error | 0 / 72 | — |

Binomial test (tim_domain wins 21 vs cold wins 14, ties/bias excluded):
**p = 0.3105, tim_domain win rate = 60.0%, 95% CI [42.1%, 76.1%] — includes
50%, not significant.**

This is close to a carbon copy of the bf16 result (win rate 61.1%, CI
[43.5%, 76.9%], position bias 15.3% — identical to the int4 figure to one
decimal place). Reasons for int4_tim_domain wins: documentation 24/44
(54.5%), robustness 10/44 (22.7%), idiomaticity 2/44, readability 2/44,
simplicity 4/44. Reasons for int4_cold wins: simplicity 13/28 (46.4%),
idiomaticity 6/28 (21.4%), readability 5/28 (17.9%), robustness 2/28,
documentation 2/28. Same shape as bf16: primed wins skew documentation,
cold wins skew simplicity — "robustness" appears somewhat more often for
tim_domain's wins here than it did at bf16 (22.7% vs 6.8%), a difference
too small at this n to read as more than noise, but noted rather than
smoothed over.

### cold: bf16 vs int4 (n=66 both-pass) — does quantization change perceived quality?

| | count | % of order-consistent |
|---|---|---|
| order-consistent decisions | 52 / 66 | — |
| int4_cold wins | 27 | 51.9% |
| bf16_cold wins | 11 | 21.2% |
| ties | 14 | 26.9% |
| **position_bias** | **14 / 66** | **21.2%** |
| judge_error | 0 / 66 | — |

Binomial test (int4_cold wins 27 vs bf16_cold wins 11, ties/bias excluded):
**p = 0.0139, int4_cold win rate = 71.1%, 95% CI [54.1%, 84.6%] — excludes
50%, statistically significant.** This is the one judge result in this
study (bf16 or int4) that clears significance. Reasons for int4_cold wins:
documentation 23/54 (42.6%), idiomaticity 13/54 (24.1%), readability 9/54
(16.7%), robustness 5/54 (9.3%), simplicity 3/54 (5.6%), plus one response
(1/54) that named "efficiency" — outside the fixed six-term rubric; kept
and reported rather than discarded, since the winner field (the load-bearing
one) parsed cleanly. Reasons for bf16_cold wins: idiomaticity 9/22 (40.9%),
simplicity 7/22 (31.8%), robustness 5/22 (22.7%), readability 1/22.

**Position bias was also higher here (21.2%) than in either priming
pairing (15.3% each)** — a real reliability caveat specific to this
comparison, worth weighing before leaning on the significant p-value alone.

**Per the guardrails, this significant judge preference does not override
the objective panel and echo test.** Documentation is the single largest
reason (42.6%) the judge prefers int4-cold, and the echo test above already
showed int4-cold's elevated docstring rate is 84-88% copied text. The
judge's significant preference for int4-cold is corroborating evidence for
"quantization alone shifts style in the priming-like direction," not
independent evidence that quantized code is better-written.

## Step 4 — Precision invariance

| metric | bf16 Δ (tim − cold) | int4 Δ (tim − cold) | same direction & magnitude? |
|---|---|---|---|
| lint violations (raw) | +0.292 (p=0.0034 *) | +0.111 (p=0.0679, n.s.) | same direction, **attenuated to non-significance at int4** |
| lint violations/LOC | +0.023 (p=0.0167 *) | −0.002 (p=0.8017, n.s.) | **diverges** — significant increase at bf16, flat at int4 |
| type-hint coverage | 0.000 (n/a, identical) | +0.014 (p=0.1573, n.s.) | invariant — flat at both |
| comment density | +0.015 (p=0.0679, n.s.) | −0.001 (p=0.7794, n.s.) | invariant — flat (n.s.) at both |
| docstring presence | +0.431 (p<0.0001 *) | +0.348 (p<0.0001 *) | same direction, same order of magnitude (smaller at int4 — ceiling effect, int4-cold baseline already 44.4%) |
| cyclomatic complexity | −0.028 (p=0.6664, n.s.) | +0.014 (p=0.6770, n.s.) | invariant — flat (n.s.) at both |
| maintainability index | +5.318 (p=0.0014 *) | +3.516 (p=0.0389 *) | same direction, same order of magnitude, weaker significance at int4 |
| function length (lines) | +3.444 (p<0.0001 *) | +4.528 (p<0.0001 *) | same direction, similar-to-larger magnitude at int4 |
| fragile rate (full 164) | +0.6 pp | +1.8 pp | same direction (never improves), 3x larger delta at int4 (not significance-tested — small counts) |
| judge win rate (tim vs cold) | 61.1%, CI incl. 50% | 60.0%, CI incl. 50% | **near-identical**, both at chance |
| judge position bias | 15.3% | 15.3% | **identical to one decimal place** |

### Verdict: Precision-invariant, with one attenuated exception

Six of eight objective metrics, the fragile rate, and both judge statistics
replicate cleanly across precision: verbosity (function length, docstring
presence) goes up significantly at both precisions and by similar
magnitudes; complexity, type-hint coverage, and comment density stay flat
(not significant) at both; fragile rate never improves at either precision;
the judge lands at chance with essentially identical win rates (61.1% vs
60.0%) and an identically-measured position-bias rate (15.3% vs 15.3%) at
both precisions. This is the expected result and a genuine external-validity
finding: **priming's behavior on Qwen3-1.7B does not depend on whether the
model is running at bf16 or NF4.**

The one real divergence: **lint violations**. At bf16, priming
significantly worsens both raw violations (p=0.0034) and violations/LOC
(p=0.0167). At int4, neither reaches significance (p=0.068, p=0.80) — the
raw-count trend is still upward but weaker, and the per-LOC rate is flat.
The most likely explanation, visible in the panel above, is a ceiling
effect: int4-cold's own baseline lint rate is already elevated relative to
bf16-cold (1.125 vs 0.944 raw violations, itself part of the "quantization
disturbs the same behavior" pattern below), compressing the headroom left
for priming to add further violations on top. This is worth flagging
explicitly rather than folding into "no effect," but it is an attenuation
of an existing effect, not a new or reversed one, so it does not change the
overall verdict.

### The cross-precision cold question, in plain language

**Yes — 4-bit quantization, entirely on its own, changes generation
behavior in the same direction KV-cache priming does, and through the same
mechanism (input-copying, not induced style).** Cold generation at NF4 vs
cold generation at bf16 (n=66 both-pass, no priming on either side) shows
significantly more docstrings (9.1% -> 37.9%, p<0.0001), significantly more
comments (p=0.0117), significantly longer functions (p<0.0001), and a
significantly higher maintainability index (p=0.0330) — the same four axes
priming moves at bf16, in the same direction, at similar or greater
significance. The LLM judge agrees and is, uniquely in this study,
statistically significant on it (p=0.0139, int4-cold win rate 71.1%,
CI [54.1%, 84.6%]), driven mainly by "documentation." The echo test then
does its job: int4-cold's own docstrings are 84-88% verbatim copies of the
prompt, exactly like priming's docstrings are. **Quantization and priming
are disturbing the same fragile behavior — the model's tendency to echo
its own input back when its output distribution is perturbed, by whatever
means — not two unrelated phenomena that happen to look similar.**

## Guardrails observed

- The objective panel was treated as the backbone throughout. The one
  place the judge reached significance (cold: bf16 vs int4) was
  cross-checked against, and explained by, the objective panel's own
  documentation/comment/function-length findings and the echo test — not
  taken as a freestanding "quantization improves code" claim.
- The echo test was applied to every docstring-rate movement in this
  document, at both precisions, per the guardrail — including the
  cross-precision cold comparison, which was not explicitly a "priming"
  result but needed the same scrutiny.
- pass@1 being flat (bf16) and barely moved (int4 gate) were treated as the
  premises that make this whole study meaningful, not as null results.
  Accuracy numbers are reported in Step 1 for completeness but are not the
  object of this study's verdict — the identical-outcome (both-pass) subset
  is.
- int4 is never called "faster." It is 1.6x slower than bf16 for cold
  generation (established in the gate) and int4+tim_domain is 2.39x slower
  than bf16-cold and 1.49x slower than int4-cold alone.

## Files written

- `experiments/int4_tim_domain.py` — generates the missing cell
  (imports model loading / determinism / `run_condition` from
  `experiments/quant_gap.py`).
- `scripts/quality_panel_grid.py` — Part 1 across all four cells (imports
  metric functions from `scripts/quality_panel.py`).
- `scripts/llm_judge_grid.py` — Part 2, two pairings (imports judge/cache
  functions from `scripts/llm_judge.py`; `llm_judge.py`'s `load_cache`/
  `append_cache` were extended with an optional path parameter, default
  unchanged, to support a second cache file).
- `logs/quality_quant/int4_tim_domain_seed{0,1,2}.jsonl` and
  `-sanitized.jsonl` / `-sanitized.eval_results.json` / `_metrics.json` —
  raw generations, sanitized solutions, EvalPlus results, and timing per
  seed.
- `logs/quality_quant/int4_tim_domain_report.json` — determinism check +
  quant config + per-seed summary.
- `logs/quality_quant/panel_grid_raw.json` — full per-task metrics for all
  four cells plus the three pairwise summaries.
- `logs/quality_quant/judge_raw.jsonl` — all 276 raw judge responses
  (cached; re-running `llm_judge_grid.py` will not re-query them).
- `logs/quality_quant/judge_grid_summary.json` — aggregated judge
  statistics for both pairings.
- `docs/precision_quality_grid.md` — this file.

## Expected but not found / noted gaps

- **bf16 tim_domain has no timing/memory instrumentation.** It was
  generated by the original `experiments/conditions.py` (root), which predates the
  timing/peak-memory instrumentation added in `experiments/pass_sweep.py` and
  `experiments/quant_gap.py`. Only `pass@1_base` / `pass@1_base_plus_extra`
  are available for that condition (`logs/tim/Qwen_Qwen3-1.7B/tim_domain_seed0_results.json`).
  ms/task, tokens/task, and peak_memory_mb for bf16 tim_domain are not
  reported anywhere in this document because they do not exist on disk —
  not fabricated to fill the gap.
- No condition files were otherwise missing; `quality_panel_grid.py` and
  `llm_judge_grid.py` both reported zero missing files/conditions on their
  respective runs.
- `flake8` was not used (matching the bf16 study — `ruff check` covers it).
