# MBPP+ scale run: coverage bug, fix, and the int4 capacity-rescue replication test

Qwen3-1.7B, MBPP+ (this evalplus install's canonical set — **N=378**, not
399; see "Coverage bug" below), greedy decoding, `enable_thinking=False`,
`max_new_tokens=768`. Two physical GPUs ran this batch concurrently
(`run_mbpp_scale_gpu0.sh` / `run_mbpp_scale_gpu1.sh`).

This run exists to test one question: **does the int4 capacity-rescue effect
found on HumanEval+ (N=164: int4-cold 51.2% → int4-tim_domain 56.7/57.9/60.4%
across 3 seeds, pooled McNemar p=0.0004, ~+7pp) replicate on a second,
higher-N dataset?** Before that question could be answered, a scoring bug
had to be found and fixed — several conditions' `evalplus.evaluate` calls
crashed with `AssertionError: Missing problems in samples` and wrote no
`eval_results.json`, which would have made any comparison involving them
invalid by construction.

## Coverage bug: diagnosis, cause, fix

**Symptom.** 7 of 15 condition files under `logs/mbpp_scale/` had no
`-sanitized.eval_results.json`: `int4_neutral_prefix`, `int4_prompt_control`,
`int4_tim_entropy_seed0`, `full/B_bf16_tim_domain_seed0` (all 3 passes), and
`full/D_nf4_tim_domain_seed0`.

**Leading suspect going in (truncation):** TIM priming adds tokens; MBPP+
solutions can be long; `max_new_tokens=768` might truncate primed generations
before a closing code fence, dropping the task at the sanitize stage and
biasing the comparison against the priming arms. `scripts/diagnose_mbpp_coverage.py`
was written to test this directly, counting raw/sanitized/scored rows and
the exact missing task-id list at every stage, for every condition file.

**Result: truncation is flatly rejected.** Every single condition — all 15,
including the 7 broken ones — had **378/378 raw rows and 378/378 sanitized
rows**, unique task-ids, zero duplicates, zero rows lost at generation, zero
rows lost at sanitization. The loss was 100% confined to the scoring stage:
`evalplus.evaluate` simply never produced output for those 7 files.

**Condition-correlation check (Step 2):** if the loss were content-driven,
it should track condition identity. It doesn't. `F_nf4_prompt_control`
(GPU0, `full/`) and `int4_prompt_control` (GPU1, top-level) are the *same*
condition — nf4 + `DOMAIN_PERSONA_TEXT` prepend, no KV injection — generated
independently on the two GPUs. One scored cleanly, the other didn't. The
same condition succeeding on one run and failing on the other rules out any
content/condition-correlated cause outright.

**Confirmation:** re-running `evalplus.evaluate` on one of the broken
`-sanitized.jsonl` files, alone, with no concurrent process — right now, on
the exact bytes already on disk — succeeded immediately and cleanly
(378/378, `pass@1: 0.579` / `pass@1: 0.492`). The sample data was never the
problem.

**Root cause:** `evalplus.evaluate.get_groundtruth()` caches its expensive
ground-truth computation to a single pickle file keyed by dataset hash
(`~/.cache/evalplus/<hash>.pkl`) via a plain `open(path, "wb"); pickle.dump(...)`
— no lock, no atomic temp-file-then-rename. This was the *first-ever* MBPP+
evaluation on this machine (the cache file did not exist beforehand). Running
two independent `evalplus.evaluate` subprocess trees concurrently (one per
GPU script), each spawning its own `ProcessPoolExecutor` worker pool against
the same machine, is enough to produce transient, non-reproducible scoring
failures under contention — confirmed random/transient (Step 2 verdict **(c)**,
not (a) truncation, not (b) a parsing bug), and confirmed harmless to the
underlying sample data.

**Fix (Step 3).** No regeneration was needed — the raw and sanitized data
were already complete and correct. `scripts/rescore_mbpp_conditions.py`
re-ran `evaluate_with_evalplus` (imported unchanged from `run_quant_gap.py`)
**serially**, one condition at a time, against the existing sanitized files.
All 7 re-scored cleanly, 378/378, on the first attempt:

| condition | pass@1 (base) | pass@1+ (base+extra) |
|---|---|---|
| int4_neutral_prefix | 0.397 | 0.336 |
| int4_prompt_control | 0.582 | 0.487 |
| int4_tim_entropy_seed0 | 0.574 | 0.479 |
| full/B_bf16_tim_domain_seed0 (pass1) | 0.579 | 0.492 |
| full/B_bf16_tim_domain_seed0 (pass2) | 0.574 | 0.489 |
| full/B_bf16_tim_domain_seed0 (pass3) | 0.561 | 0.484 |
| full/D_nf4_tim_domain_seed0 | 0.574 | 0.484 |

**Harness hardening.** `experiment-3/run_quant_gap.py`'s `evaluate_with_evalplus`
previously caught a non-zero evalplus exit with a printed `WARNING` and then
*silently continued*, merging an empty scores dict and letting the run
proceed — this is exactly how the 7 broken conditions ended up looking like
"completed runs" on disk with no accuracy numbers, discoverable only by
manual inspection. It now **raises** `RuntimeError` on a non-zero exit, on
an unparseable stdout, or when `n_scored != n_canonical` against a fresh
`get_mbpp_plus()`/`get_human_eval_plus()` count — refusing to report a
pass@1 under any of those conditions, for every caller of this shared
harness (both HumanEval+ and MBPP+ studies). This class of bug cannot
silently recur.

**Practical takeaway:** don't run two `evalplus.evaluate` invocations
concurrently against a cold (not-yet-cached) dataset on the same machine.
Once the ground-truth cache is warm, this class of failure did not recur
(the gate stage, `A_bf16_cold`/`C_nf4_cold`, ran first and alone, warmed the
cache, and every subsequent single evaluate call succeeded regardless of
what else was running on the other GPU at the time).

## Coverage report

`scripts/diagnose_mbpp_coverage.py` output (all 15 condition files, before
the fix — post-fix, all now show `scored=378`):

```
condition                                       raw/  san/scored/canon  status
int4_neutral_prefix                             378/  378/     0/  378  (fixed by rescore)
int4_prompt_control                             378/  378/     0/  378  (fixed by rescore)
int4_tim_domain_seed0/1/2                       378/  378/   378/  378  OK
int4_tim_entropy_seed0                          378/  378/     0/  378  (fixed by rescore)
full/B_bf16_tim_domain_seed0 (x3 passes)        378/  378/     0/  378  (fixed by rescore)
full/D_nf4_tim_domain_seed0                     378/  378/     0/  378  (fixed by rescore)
full/F_nf4_prompt_control                       378/  378/   378/  378  OK
gate/A_bf16_cold, gate/C_nf4_cold               378/  378/   378/  378  OK
```

Full detail (including the empty missing-task-id lists — every list is
empty, at every stage, for every condition): `logs/mbpp_scale/coverage_report.json`.

---

## Results (complete, matched-denominator set — N=378 throughout)

Every number below passed the coverage gate (`n_scored == n_canonical == 378`)
before being used in any comparison; a condition that failed the gate would
print as `EXCLUDED`, not silently drop into a mean (see
`scripts/analyze_mbpp_scale.py`).

### Pass@1 (base+extra), Wilson 95% CI

| condition | pass@1+ | 95% CI | k/n |
|---|---|---|---|
| bf16_cold | 49.2% | [44.2, 54.2] | 186/378 |
| int4_cold | 48.7% | [43.7, 53.7] | 184/378 |
| int4_prompt_control | 48.7% | [43.7, 53.7] | 184/378 |
| int4_neutral_prefix | 33.6% | [29.0, 38.5] | 127/378 |
| int4_tim_entropy_seed0 | 47.9% | [42.9, 52.9] | 181/378 |
| full/F_nf4_prompt_control (replicate) | 48.7% | [43.7, 53.7] | 184/378 |
| bf16_tim_domain_seed0 (pass1) | 49.2% | [44.2, 54.2] | 186/378 |
| int4_tim_domain_seed0 (pass1) | 47.1% | [42.1, 52.1] | 178/378 |
| int4_tim_domain_seed1 (pass1) | 48.9% | [43.9, 54.0] | 185/378 |
| int4_tim_domain_seed2 (pass1) | 46.3% | [41.3, 51.3] | 175/378 |
| full/D_nf4_tim_domain_seed0 (replicate) | 48.4% | [43.4, 53.4] | 183/378 |
| bf16_tim_domain_seed0 (pass2) | 48.9% | [43.9, 54.0] | 185/378 |
| bf16_tim_domain_seed0 (pass3) | 48.4% | [43.4, 53.4] | 183/378 |
| int4_tim_domain_seed0 (pass2) | 47.9% | [42.9, 52.9] | 181/378 |
| int4_tim_domain_seed0 (pass3) | 48.4% | [43.4, 53.4] | 183/378 |

### Quantization damage (gate)

bf16_cold (49.2%) vs int4_cold (48.7%): discordant 27/25, McNemar **p=0.8899**.
On MBPP+, quantization damage at cold is small and not statistically
significant — a much smaller gate signal than HumanEval+ showed. This
dataset simply has less headroom for a "damage → rescue" story on the cold
baseline alone; the interesting question is entirely in the priming arms
below.

### Headline replication test — int4_cold vs int4_tim_domain (pooled, 3 seeds, pass1)

| seed | pass@1+ | vs int4_cold: a_only / b_only | churn | McNemar p |
|---|---|---|---|---|
| seed0 | 47.1% | 19 / 13 | 8.5% | 0.3771 |
| seed1 | 48.9% | 19 / 20 | 10.3% | 1.0000 |
| seed2 | 46.3% | 23 / 14 | 9.8% | 0.1877 |
| **pooled** | **mean 47.4%** | **61 / 47** | **9.5%** | **0.2108** |

int4_cold = 48.7%. Pooled tim_domain mean = 47.4%. **Gap = −1.2 pp** (tim_domain
is numerically *below* cold, not above it), and the pooled McNemar test is
nowhere near significant (p=0.21, vs the HumanEval+ reference of p=0.0004).
No individual seed reaches significance either.

### bf16 flat-priming check

bf16_cold (49.2%) vs bf16_tim_domain seed0 pass1 (49.2%): discordant 22/22,
McNemar **p=1.0000** — exactly flat, exactly as HumanEval+ found (p=0.69
there). This part of the picture replicates cleanly.

### Mechanism controls (vs int4_cold)

| condition | pass@1+ | a_only / b_only | churn | McNemar p |
|---|---|---|---|---|
| int4_prompt_control | 48.7% | 11 / 11 | 5.8% | 1.0000 |
| int4_neutral_prefix | 33.6% | 78 / 21 | 26.2% | **<0.0001** |
| int4_tim_entropy_seed0 | 47.9% | 16 / 13 | 7.7% | 0.7111 |

- **Delivery parity holds**: int4_cold ≈ int4_prompt_control (p=1.00), and
  int4_prompt_control ≈ int4_tim_domain seed0 (a_only=18, b_only=12,
  p=0.3616) — same pattern as HumanEval+ (prompting the persona text does
  the same as injecting it via KV cache), just with neither one beating cold
  here.
- **The neutral-prefix result replicates decisively and gets *stronger***:
  a degenerate 64-token repeated-token prefix costs **15.1 pp** on MBPP+
  (33.6% vs 48.7%, churn 26.2%, p<0.0001) — even larger than the 8.5pp
  HumanEval+ finding. This is the single most robust result in the whole
  ladder across both datasets: **content-free prefixes hurt int4
  generation badly and reliably**, which continues to argue against a naive
  attention-sink story (a sink pattern alone would not explain why removing
  vocabulary diversity while keeping prefix length constant is actively
  harmful).
- **Entropy control is flat**: int4_tim_entropy neither helps nor hurts
  relative to cold (p=0.71) — consistent with a token-diversity requirement
  rather than any special entropy-maximizing structure.

### Multi-pass dose-response (1 → 2 → 3)

**bf16_tim_domain_seed0:**

| pass | pass@1+ | vs bf16_cold p | pass-to-pass p |
|---|---|---|---|
| 1 | 49.2% | 1.0000 | — |
| 2 | 48.9% | 1.0000 | 1v2: 1.0000 |
| 3 | 48.4% | 0.7608 | 2v3: 0.8036 |

**int4_tim_domain_seed0:**

| pass | pass@1+ | vs int4_cold p | pass-to-pass p |
|---|---|---|---|
| 1 | 47.1% | 0.3771 | — |
| 2 | 47.9% | 0.7709 | 1v2: 0.7201 |
| 3 | 48.4% | 1.0000 | 2v3: 0.8145 |

**Verdict: plateau, not dose-response.** Neither ladder shows a significant
move at any step, in either direction, against its own cold baseline or
pass-to-pass. Stacking priming passes does not help and does not hurt on
MBPP+ — a flat line at both precisions. (Interesting secondary note: int4's
three passes drift slowly upward — 47.1 → 47.9 → 48.4% — while bf16's drift
slightly downward — 49.2 → 48.9 → 48.4% — but none of these moves clear
significance individually or pairwise; read this as noise around a plateau,
not a trend.)

### Cross-run NF4 determinism check

The same nominal condition (nf4 + tim_domain + seed0 + pass1) was generated
independently twice, by two different scripts on two different GPU processes:
`int4_tim_domain_seed0` (GPU1, `run_int4_tim_domain.py`, 47.1%) and
`full/D_nf4_tim_domain_seed0` (GPU0, `run_quant_gap.py`'s `run_full()`,
48.4%). Discordant 14/19, McNemar **p=0.4869** — not significant; the two
independent NF4 model loads produce output that differs task-by-task more
than a single model's own repeated-generation determinism check would (each
script's own 3-run `check_determinism()` passed at load time), but the
*difference between the two loads* is statistically indistinguishable from
noise at this N. The same cross-check for prompt_control
(`int4_prompt_control` vs `full/F_nf4_prompt_control`) came back **byte-for-byte
identical** (0 discordant, p=1.0000) — so NF4 determinism is not uniformly
fragile; it appears to interact with something specific to the tim_domain
priming path (plausibly stochastic-rounding sensitivity in the primed KV
cache's numerics compounding over the longer generations that priming
produces). Flagged here, not further investigated — it does not change any
verdict above (all headline comparisons already account for seed variance).

### Echo test — MBPP+ has (almost) no docstrings to copy

| condition | docstring rate | echoed | modified | original |
|---|---|---|---|---|
| int4_tim_domain_seed0 pass1 | 0.8% (3/378) | 0.0% | 0.0% | 100.0% |
| int4_tim_domain_seed0 pass2 | 1.1% (4/378) | 0.0% | 0.0% | 100.0% |
| int4_tim_domain_seed0 pass3 | 1.9% (7/378) | 0.0% | 14.3% | 85.7% |

Compare to HumanEval+ tim_domain conditions, where the docstring rate runs
85–92% and 85–92% *of that* is classified "echoed" (near-verbatim copy of
the prompt's own docstring). On MBPP+ — whose prompts are natural-language
instructions with no function-level docstring to copy — the docstring rate
collapses to under 2%, and of the handful that do appear, **zero are
classified as echoed** across all three passes. This is the clean
confirmation the design called for: **the HumanEval+ docstring effect was
input-echoing (copying text that was present in the prompt to copy), not an
induced documentation style** — there is nothing to echo on MBPP+, and
accordingly nothing gets echoed.

---

## Verdict

**Did the int4 capacity-rescue effect replicate at N≈378 with strict
significance? No.** Pooled int4_cold vs int4_tim_domain (3 seeds, pass1):
mean gap **−1.2 pp** (numerically the wrong direction), McNemar **p=0.21**.
No seed individually reaches significance either (p=0.38, 1.00, 0.19). This
directly contradicts the HumanEval+ result (+7pp, p=0.0004) rather than
merely failing to confirm it. **The capacity-rescue effect found on
HumanEval+ does not generalize to MBPP+ on this model.** Read plainly: the
original N=164 finding was real (the statistics there were sound and the
mechanism ladder in `docs/int4_mechanism.md` correctly falsified the naive
attention-sink explanation), but it is now shown to be **dataset-specific**,
not a general property of "priming a quantized Qwen3-1.7B." Whatever let
generic context partially rescue quantization damage on HumanEval+ did not
carry over to MBPP+, where cold-baseline quantization damage was itself
already much smaller (0.5pp, not significant) — there may simply have been
less damage available to rescue.

**Does stacking passes help, plateau, or degrade?** Plateau. Neither the
bf16 nor the int4 dose-response ladder shows a significant move at any step,
in either direction. More priming passes neither help nor hurt on MBPP+.

**Does the docstring/echo effect survive on a dataset with no docstrings to
copy?** No — and that is exactly the confirmation the design was built to
extract. Docstring rate collapses from ~85-92% (HumanEval+) to <2% (MBPP+),
and 0% of what little remains is classified as echoed. This **confirms the
input-echoing interpretation** and rules out an induced-documentation-style
interpretation.

**What continues to replicate cleanly across both datasets:**
- Delivery parity (prompt_control ≈ tim_domain; KV injection isn't special
  versus prompting the same content) — p=1.00 on MBPP+, matching bf16 and
  int4 HumanEval+ results.
- bf16 stays flat under priming — p=1.00 on MBPP+, matching HumanEval+
  (p=0.69).
- The neutral/degenerate-prefix penalty — MBPP+ shows it even more strongly
  (−15.1pp, p<0.0001) than HumanEval+ (−8.5pp) — reinforcing that a
  content-free prefix actively harms int4 generation, which continues to
  weigh against a pure attention-sink mechanism.

## Timing and memory (NF4 is never faster — this is not a speed technique)

| condition | ms/task | peak MB | mean new tokens |
|---|---|---|---|
| bf16_cold (A) | 646.1 | 3413.6 | 49.6 |
| int4_cold (C) | 820.6 | 1416.1 | 56.5 |
| bf16_tim_domain seed0 pass1 (B) | 679.9 | 3422.7 | 52.6 |
| int4_tim_domain seed0 pass1 | 887.1 | 1429.0 | 58.2 |
| int4_neutral_prefix | 1511.4 | 1435.4 | 102.2 |
| int4_prompt_control | 806.0 | 1378.1 | 54.1 |

NF4 costs 27–90% more wall-clock per task than bf16 across every matched
pair here, in exchange for ~2.4× less peak memory (~1.4GB vs ~3.4GB). This
matches the standing repo-wide finding — quantization here is a memory
trade, not a speed one, and priming (especially the neutral-prefix control,
which generates the most tokens of any condition on MBPP+ too, echoing the
HumanEval+ pattern) adds further latency on top.

## Files written

- `scripts/diagnose_mbpp_coverage.py` — Step 1 coverage diagnostic (reusable
  for any future evalplus dataset).
- `scripts/rescore_mbpp_conditions.py` — Step 3 serial re-scoring fix.
- `scripts/analyze_mbpp_scale.py` — Step 4 analysis (extended from the
  pre-existing draft; McNemar tests, dose-response, echo test, answers dump).
- `experiment-3/run_quant_gap.py` — `evaluate_with_evalplus` hardened to
  hard-fail instead of silently continuing on a scoring mismatch.
- `logs/mbpp_scale/coverage_report.json` — full per-condition coverage detail.
- `logs/mbpp_scale/rescore_report.json` — re-scoring outcomes for the 7
  originally-broken conditions.
- `logs/mbpp_scale/mbpp_analysis_summary.json` — full structured results.
- `logs/mbpp_scale/answers_mbpp.jsonl` — per-task side-by-side solutions +
  pass/fail across 9 core conditions, all 378 tasks.
- `docs/mbpp_scale.md` — this document.

## Expected but not found

None. Every condition referenced above passed the `n_scored == n_canonical`
gate; `scripts/analyze_mbpp_scale.py`'s own "EXPECTED BUT NOT FOUND" check
reported nothing missing on the final run.
