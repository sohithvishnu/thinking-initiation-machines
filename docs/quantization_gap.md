# Quantization gap: can KV-cache priming recover accuracy that quantization removes?

Companion study to the main TIM results in [README.md](../README.md). Runner:
`experiments/quant_gap.py`. Analysis: `scripts/analyze_quant_gap.py`.
Raw outputs: `logs/quant_gap/`.

## Design

2x2, gated on Step 1 before running the rest:

|            | bf16 | 4-bit (bnb NF4) |
|---|---|---|
| cold | A | C |
| tim_domain | B | D |
| prompt_control | E | F |

`damage = A - C` (what 4-bit costs — the gate). `recovery = D - C` (what TIM
buys on the quantized model). Success condition: `D >= A`. `F - C` checks
whether plain prompting recovers as much as TIM does.

## Step 1 — Gate result

Model: Qwen/Qwen3-1.7B. HumanEval+, n=164, greedy decoding, both conditions
scored end to end (generate -> sanitize -> evaluate).

- **A (bf16 cold)**: loaded with `torch_dtype=torch.bfloat16` explicitly;
  confirmed parameter dtype `torch.bfloat16`.
- **C (nf4 cold)**: `BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
  bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)`;
  loaded parameter dtypes were `{torch.bfloat16, torch.uint8}` (compute path
  bf16, quantized weights uint8), as expected for NF4.
- **Determinism check (nf4)**: 3 greedy generations of the same prompt (64
  new tokens) were byte-identical on the first attempt
  (`bnb_4bit_use_double_quant=True`, batch size 1). No retry with
  `use_double_quant=False` was needed.

| condition | HumanEval (base) pass@1 | HumanEval+ (base+extra) pass@1 | peak memory | ms/task | tokens/task |
|---|---|---|---|---|---|
| A: bf16 cold | 56.1% [48.4, 63.5] (92/164) | 51.8% [44.2, 59.3] (85/164) | 3415.2 MB | 996.7 | 77.5 |
| C: nf4 cold  | 55.5% [47.8, 62.9] (91/164) | 51.2% [43.6, 58.8] (84/164) | 1427.7 MB | 1599.4 | 113.5 |

(95% Wilson CIs in brackets.) A's HumanEval+ figure (51.8%, 85/164) matches
the `cold` condition in the main study exactly — same model, same benchmark,
same decoding — which is a useful reproducibility check on this pipeline.

**damage (A - C) = +0.6 pp.**

### Gate decision: FAILED

The pre-registered threshold was damage >= 5.0 pp. At n=164 the 95% CI
half-width on a proportion near 50% is about 7.5 pp, and the two arms'
Wilson intervals overlap almost completely (`[44.2, 59.3]` vs
`[43.6, 58.8]`). NF4 quantization did not produce a measurable accuracy cost
on this model/benchmark — 4-bit and bf16 differ by exactly one task out of
164. The full 2x2 was not run: with damage this close to zero, "does TIM
recover the accuracy quantization removed" has nothing to recover, and any
observed `D >= A` would be indistinguishable from the same seed-to-seed
churn the main study already documents at full precision.

**Memory**: NF4 cut peak allocation by 1987.5 MB (58.2%, 3415.2 -> 1427.7 MB),
in the range expected for a ~1.7B-parameter model's weights moving from 2
bytes/param to ~0.5 bytes/param plus overhead.

**Timing**: NF4 was 1.60x *slower* per task (996.7 -> 1599.4 ms), not
faster — consistent with dequantization overhead on this consumer GPU
outweighing the memory-bandwidth savings from smaller weights. Tokens/task
also rose (77.5 -> 113.5), which is part of why per-task wall time rose too
(see the docstring-rate observation below); mean `generate_ms` alone rose by
a similar 1.6x, so the token-count increase is not the whole explanation.

### An unanticipated side-finding

Docstring reproduction rate (same regex as the main study) rose sharply
under quantization alone, with no priming involved: **7.9% (13/164) at bf16
cold vs 34.8% (57/164) at nf4 cold.** This was not something the gate was
designed to test, but it echoes the main study's finding that a distribution
lever (there, KV priming; here, weight quantization) can shift *what kind of
text* the model produces without moving pass@1 — accuracy-neutral, but not
behavior-neutral. This is noted for context, not analyzed with a formal test
here, since it falls outside the gate's own hypothesis.

## Step 2 — Full 2x2

**Not run.** The gate failed its own pre-registered stopping rule. Running
B, D, E, F anyway would not answer the "does TIM recover what quantization
removed" question, since quantization removed essentially nothing to
recover on this model/benchmark; the study would be gated on `damage < 5 pp`.

## Verdict: Underpowered

> Quantization perturbs weights; priming perturbs context. This gate does
> not show whether the second can correct the first, because on Qwen3-1.7B
> at NF4, the first barely perturbs anything measurable at n=164 (damage =
> +0.6 pp, inside a ~7.5 pp CI half-width). The honest reading is not
> "priming would recover the loss" or "priming would reshuffle outcomes" —
> it is that this experiment, on this model, cannot distinguish either
> outcome from noise, because there is no damage to recover from in the
> first place.

This does not establish that NF4 is harmless in general — only that on this
model, this benchmark, and this n, no cost was measurable. A larger model
(known to compress less gracefully at 4-bit in some reports), a harder
benchmark, or a larger n could all produce a different gate result. If this
study is revisited, the most direct next step is re-running the gate on
Qwen3-4B or Qwen3-8B rather than proceeding to the 2x2 on 1.7B.

## Files written

- `experiments/quant_gap.py` — gate + full-2x2 runner.
- `scripts/analyze_quant_gap.py` — analysis (pass@1 table, damage/recovery,
  McNemar C vs D, overlap 2x2, docstring rate, timing) — reads whatever of
  gate/full exists and reports what's missing.
- `logs/quant_gap/Qwen_Qwen3-1.7B/gate/` — gate run artifacts: `A_bf16_cold*`,
  `C_nf4_cold*`, `gate_report.json`.
- `logs/quant_gap/Qwen_Qwen3-1.7B/full/` — not created (Step 2 not run).
