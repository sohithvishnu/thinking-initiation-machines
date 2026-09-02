# Generation quality: does priming change code quality along axes pass@1 can't see?

Companion study to [README.md](../README.md) and [quantization_gap.md](quantization_gap.md).
The main study established that KV-cache priming does not move HumanEval+
pass@1 (McNemar p = 0.69) but does change output behavior (docstring rate
18.3% -> 51.8%, p ~ 2e-10; +47% tokens). **pass@1 being flat is the premise
of this study, not a failure to find an effect** — it is what makes "does
priming change quality along dimensions pass@1 cannot see" a real question
rather than a moot one.

Everything below operates on `.jsonl`/`.json` files already on disk in
`logs/`. No model generation happened for this study. Runners:
`scripts/quality_panel.py` (Part 1, objective) and `scripts/llm_judge.py`
(Part 2, LLM judge).

## Condition files found

For Qwen3-1.7B, under `logs/tim/Qwen_Qwen3-1.7B/`:

| condition | sanitized solutions | eval_results.json |
|---|---|---|
| cold | `cold-sanitized.jsonl` | `cold-sanitized.eval_results.json` |
| prompt_control | `prompt_control-sanitized.jsonl` | `prompt_control-sanitized.eval_results.json` |
| tim_domain | `tim_domain_seed{0,1,2}-sanitized.jsonl` (3 seeds) | `tim_domain_seed{0,1,2}-sanitized.eval_results.json` |
| tim_random | `tim_random_seed{0,1,2}-sanitized.jsonl` (3 seeds) | `tim_random_seed{0,1,2}-sanitized.eval_results.json` |

All 164 tasks present in every file (confirmed by `quality_panel.py`'s
startup report). tim_domain and tim_random are each represented by **seed0
only** in the analysis below — using a single concrete seed keeps every
paired comparison unambiguous rather than conflating three differently
seeded primes into one number. seed0 was chosen arbitrarily (first seed),
not cherry-picked; it is not the seed used for the main study's headline
McNemar table (see the both-pass-set note below).

## Both-pass set sizes

Every quality comparison is restricted to the both-pass set for that
specific pair (`base_status == 'pass' AND plus_status == 'pass'` in BOTH
conditions), rebuilt independently per pair from the eval_results JSON —
never reused across pairs, since each condition passes a different subset
of the 164 tasks.

| pair | both-pass n |
|---|---|
| cold ∩ prompt_control | 77 |
| cold ∩ tim_domain_seed0 | **72** |
| cold ∩ tim_random_seed0 | 70 |

**Note on the ~80 recalled in the task prompt**: the real cold∩tim_domain
both-pass count is 72, not ~80. Across the three tim_domain seeds (same
base_status AND plus_status criterion throughout), cold∩tim_domain both-pass
is 72 (seed0), 65 (seed1), 73 (seed2) — all noticeably below 80. The figure
closest to 80 in this repo's logs is **prompt_control∩tim_domain, which is
78 for seed0 and seed2** (74 for seed1) — a different pair entirely, not
involving cold. The ~80 recollection most likely refers to that pairing,
not cold∩tim_domain. This is stated plainly rather than silently adjusting
the real number to match.

## Part 1 — Objective quality panel

Ruff and radon were not installed; both installed cleanly via pip
(`ruff==0.16.5`, `radon==6.0.1`) — no metric was skipped.

**EvalPlus fragile rate** (base pass, plus fail — full 164 per condition,
not both-pass-restricted; this is itself a correctness-robustness signal):

| condition | fragile / 164 | rate |
|---|---|---|
| cold | 7 | 4.3% |
| prompt_control | 8 | 4.9% |
| tim_domain_seed0 | 8 | 4.9% |
| tim_random_seed0 | 11 | 6.7% |

Fragility is flat-to-slightly-worse under priming, never better.

**Pairwise metrics** (mean ± std on the both-pass set; `*` = paired
Wilcoxon signed-rank p < 0.05 vs cold on that same both-pass set; `n/a` =
no variance to test, e.g. identical values for every paired task):

### prompt_control vs cold (n=77)

| metric | cold | prompt_control | p |
|---|---|---|---|
| lint violations (raw) | 0.935 ± 2.060 | 1.026 ± 2.113 | 0.1245 |
| lint violations/LOC | 0.146 ± 0.319 | 0.153 ± 0.330 | 0.3270 |
| type-hint coverage | 0.429 ± 0.429 | 0.433 ± 0.433 | 0.3173 |
| comment density | 0.006 ± 0.036 | 0.019 ± 0.091 | 0.0679 |
| **docstring presence** | 0.078 ± 0.268 | 0.169 ± 0.375 | **0.0082** * |
| cyclomatic complexity | 3.221 ± 1.617 | 3.221 ± 1.648 | 1.0000 |
| maintainability index | 78.120 ± 12.519 | 79.768 ± 13.023 | 0.2735 |
| **function length (lines)** | 7.169 ± 5.563 | 8.143 ± 5.817 | **0.0016** * |

### tim_domain_seed0 vs cold (n=72)

| metric | cold | tim_domain_seed0 | p |
|---|---|---|---|
| **lint violations (raw)** | 0.944 ± 2.047 | 1.236 ± 2.458 | **0.0034** * |
| **lint violations/LOC** | 0.153 ± 0.327 | 0.176 ± 0.376 | **0.0167** * |
| type-hint coverage | 0.451 ± 0.423 | 0.451 ± 0.423 | n/a (identical) |
| comment density | 0.003 ± 0.023 | 0.018 ± 0.089 | 0.0679 |
| **docstring presence** | 0.069 ± 0.254 | 0.500 ± 0.500 | **<0.0001** * |
| cyclomatic complexity | 3.139 ± 1.521 | 3.111 ± 1.629 | 0.6664 |
| **maintainability index** | 78.026 ± 12.715 | 83.344 ± 14.147 | **0.0014** * |
| **function length (lines)** | 7.056 ± 5.522 | 10.500 ± 6.614 | **<0.0001** * |

### tim_random_seed0 vs cold (n=70)

| metric | cold | tim_random_seed0 | p |
|---|---|---|---|
| **lint violations (raw)** | 0.971 ± 2.070 | 1.143 ± 2.244 | **0.0480** * |
| lint violations/LOC | 0.158 ± 0.331 | 0.167 ± 0.341 | 0.1394 |
| type-hint coverage | 0.443 ± 0.433 | 0.443 ± 0.433 | n/a (identical) |
| comment density | 0.003 ± 0.024 | 0.013 ± 0.082 | 0.2850 |
| **docstring presence** | 0.071 ± 0.258 | 0.486 ± 0.500 | **<0.0001** * |
| cyclomatic complexity | 3.200 ± 1.573 | 3.243 ± 1.651 | 0.3657 |
| maintainability index | 78.386 ± 12.629 | 81.532 ± 14.425 | 0.0780 |
| **function length (lines)** | 6.986 ± 5.497 | 10.343 ± 5.901 | **<0.0001** * |

**Reading these together**: type-hint coverage never moves (identical or
non-significant in all three pairs — priming does not make the model
annotate more). Cyclomatic complexity never moves. Comment density trends
up but is never significant at this n. Docstring presence and function
length move significantly and substantially in every primed condition
(prompt_control included, more weakly). Lint violations move the *wrong*
way for tim_domain and tim_random — more violations, not fewer, under
priming — which argues against any "cleaner code" reading. The
maintainability-index increase for tim_domain (p=0.0014) should not be read
as independent evidence of better code: radon's MI formula includes a
positive term for comment/documentation density, so a docstring increase
mechanically raises MI on its own; it is very likely downstream of the same
documentation effect, not a separate signal.

### Docstring provenance — the echo test

This is the single most important number in Part 1. For solutions with a
docstring, whitespace-normalized character identity to the prompt's own
docstring classifies it as **echoed** (copied), **modified** (similar but
edited, similarity ratio >= 0.6), or **original** (new text, or the prompt
had no docstring to copy).

| condition (both-pass set) | docstrings | echoed | modified | original |
|---|---|---|---|---|
| cold (in prompt_control's n=77) | 5/77 | 4 (80.0%) | 1 (20.0%) | 0 |
| prompt_control (n=77) | 13/77 | 12 (92.3%) | 0 (0.0%) | 1 (7.7%) |
| cold (in tim_domain's n=72) | 4/72 | 3 (75.0%) | 1 (25.0%) | 0 |
| **tim_domain_seed0 (n=72)** | **36/72** | **32 (88.9%)** | 1 (2.8%) | 3 (8.3%) |
| cold (in tim_random's n=70) | 4/70 | 3 (75.0%) | 1 (25.0%) | 0 |
| tim_random_seed0 (n=70) | 34/70 | 31 (91.2%) | 1 (2.9%) | 2 (5.9%) |

**Across every condition, roughly 89-92% of docstring-bearing solutions
have simply copied the prompt's docstring back verbatim.** Fewer than 1 in
10 docstrings are genuinely new or edited text. This means the headline
docstring-rate finding (18.3% -> 51.8% in the main study) is overwhelmingly
an **input-copying effect, not induced Pythonic documentation style** — the
model becomes more likely to echo the problem statement's docstring into
its answer, not more likely to write its own. This is discounted
accordingly in the verdict below; it does not by itself support any claim
about the model having learned better documentation habits.

## Part 2 — Pairwise LLM judge

**Judge model**: `ollama list` showed `muse-glimmer:latest`,
`deepseek-coder-v2:16b`, `gemma4:12b`, `llama3.1:8b` — no tag literally
named `gemma3:12b` was present. `gemma4:12b` (11.9B params, Q4_K_M) is the
Gemma-class 12B-scale model actually available and is what was used; the
task's mention of "gemma3:12b" was a guessed tag, not a directive, and is
noted here rather than silently substituted without comment.

Each of the 72 cold∩tim_domain_seed0 both-pass tasks was judged twice, with
solution order swapped (cold-first, then tim_domain-first), position labels
hidden from the model. `think: false` was set after an initial smoke test
showed the model could return empty `content` when its whole token budget
went into an internal reasoning trace rather than the final answer (0/6
became 0/144 parse errors after the fix; no judge_error cases occurred in
the full run — context-length truncation was never observed either, so the
context window did not need to be enlarged beyond Ollama's default). All
144 raw responses are cached in `logs/quality/judge_raw.jsonl`; re-running
`scripts/llm_judge.py` will not re-query them.

| | count | % of order-consistent |
|---|---|---|
| order-consistent decisions | 61 / 72 | — |
| tim_domain wins | 22 | 36.1% |
| cold wins | 14 | 23.0% |
| ties | 25 | 41.0% |
| **position_bias** (order flips the verdict) | **11 / 72** | **15.3% of all valid pairs** |
| judge_error | 0 / 72 | — |

**Binomial test, tim_domain wins (22) vs cold wins (14), ties and
position-bias pairs excluded**: p = 0.243, tim_domain win rate = 61.1%,
95% CI [43.5%, 76.9%]. **The CI includes 50% — the raw win-rate lean toward
tim_domain is not statistically distinguishable from chance** at this
sample size (n=36 decisive pairs).

Position bias at 15.3% is a real reliability caveat on this judge: better
than a coin flip's worth of pairs would have been called differently had
the order not been swapped-and-checked. This is exactly why the swap
protocol was not optional.

**Stated reasons** (each decisive pair contributes 2 reason mentions, one
per order):

| reason | tim_domain wins (44 mentions) | cold wins (28 mentions) |
|---|---|---|
| documentation | 28 (63.6%) | 0 |
| idiomaticity | 8 (18.2%) | 5 (17.9%) |
| readability | 3 (6.8%) | 2 (7.1%) |
| robustness | 3 (6.8%) | 9 (32.1%) |
| simplicity | 2 (4.5%) | 12 (42.9%) |

When the judge prefers tim_domain, it says so because of documentation
nearly two-thirds of the time. When the judge prefers cold, it is almost
entirely because cold is simpler or more robust-looking — the exact
opposite axes from where tim_domain "wins."

## Verdict

**Style-only, and even that is substantially an echo artifact — not
substance, and not clearly distinguishable from chance once position bias
is controlled for.**

Matching against the four possible verdicts:

- *Substance* is not supported: complexity is flat (p=0.67), fragile rate
  is flat-to-worse (never better), and lint violations are significantly
  **worse**, not better, under tim_domain and tim_random. The judge's wins
  for tim_domain are documentation-driven, not readability- or
  robustness-driven — the opposite of what the substance verdict requires.
- *Style-only* is the closest fit on the objective panel alone: documentation
  metrics (docstring presence, function length) move significantly;
  complexity/fragile-rate do not. The judge's reason distribution — 63.6%
  "documentation" for tim_domain's wins, versus "simplicity"/"robustness"
  for cold's wins — matches this pattern, and ties are the single largest
  outcome (41.0%).
- *Echo artifact* applies on top of that and is the dominant caveat: ~89-92%
  of the "induced" docstrings across every primed condition are verbatim
  copies of the prompt's own docstring. The style effect is real but it is
  mostly "the model echoes the question back," not "the model was primed
  into writing better-documented code."
- *No effect* is not quite right either — the panel is not flat (docstring
  presence and function length move with strong significance, p < 0.0001
  in two of three pairs) — but the judge's own binomial test lands at
  chance (CI includes 50%), so "no effect" is the right read specifically
  for the LLM-judge evidence taken alone.

**Two cautions, as instructed:**

1. The LLM judge is contestable evidence — a single 12B model, one prompt
   template, with a measured 15.3% position-bias rate. The objective panel
   is the backbone of this study. Here they happen to agree (both point to
   a documentation-driven, not-quite-significant style effect), but if they
   had disagreed, the judge's win rate would not have been allowed to
   override a flat objective panel.
2. HumanEval+ pass@1 being unchanged is the premise this whole study
   depends on, not a null result to apologize for — the entire point was to
   look for quality differences in the identical-outcome, both-pass subset
   where pass@1 has nothing left to say.

## Files written

- `scripts/quality_panel.py` — Part 1 runner (ruff + radon + ast metrics,
  Wilcoxon tests, docstring provenance).
- `scripts/llm_judge.py` — Part 2 runner (Ollama pairwise judge, position-
  bias-controlled).
- `logs/quality/panel_raw.json` — full per-task metrics for all 164 tasks x
  4 conditions, plus the three pairwise summaries.
- `logs/quality/judge_raw.jsonl` — all 144 raw judge responses (cached;
  re-running the judge script will not re-query these).
- `logs/quality/judge_summary.json` — aggregated judge statistics.
- `docs/generation_quality.md` — this file.

## Tools/conditions expected but not found

- `flake8` was not installed or used — `ruff check` was preferred per the
  task instructions and worked without issue, so flake8 was never needed.
- No condition files were missing for Qwen3-1.7B; `tim_random_seed{0,1,2}`
  was present and used in Part 1 (not requested for Part 2, which is
  scoped to cold vs tim_domain only).
- This study used Qwen3-1.7B only, matching the model the both-pass-set
  sizes in the task prompt were benchmarked against. Qwen3-4B/8B have their
  own `logs/tim/` directories with the same file layout if this analysis is
  extended to them later.
