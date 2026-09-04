"""
MoE router probe, Gemma-4-26B-A4B replication: does TIM priming move expert
routing, and does the expert set activated during priming predict the expert
set activated during generation?

Same methodology, same falsifiable questions, as tim-moe/moe_router_probe.py
(OLMoE-1B-7B-0924-Instruct, bf16). This is the same probe run against a
second, structurally different MoE model, to see whether the OLMoE verdict
(Q1 yes/non-specific, Q2 no/below-chance) generalizes. See
tim-moe/docs/moe_gemma_comparison.md for the side-by-side writeup.

  Q1 (router shift): does priming change which experts the router selects,
      vs. a cold prompt?
  Q2 (prediction, the linchpin): do the experts activated during the priming
      pass overlap with the experts activated during generation of the real
      task, above a cross-task chance baseline?

Model: cyankiwi/gemma-4-26B-A4B-it-AWQ-4bit (Gemma4ForConditionalGeneration,
text_config: 30 layers, all MoE, 128 experts, top-8, hidden 2816). Despite the
"AWQ" name, config.json's quantization_config is a genuine compressed-tensors
pack-quantized int4 scheme (group_size 32, symmetric) applied per-expert
(model.language_model.layers.{i}.experts.{j}.{gate,up,down}_proj.weight_packed
+ weight_scale, ~34.5k expert tensors) — NOT bitsandbytes, which (per
moe_router_probe.py's docstring, confirmed again for this architecture) is a
structural no-op for batched/per-expert MoE weights.

IMPORTANT — checkpoint/library format mismatch, and how it's fixed here (see
tim-moe/docs/moe_gemma_comparison.md for the full account with load-report
evidence and verification numbers): installed transformers (5.13.1)'s
Gemma4TextExperts expects one batched [num_experts, ...] Parameter per
projection (gate_up_proj, down_proj), but this checkpoint stores one small
compressed-tensors tensor triple per expert per projection, and Gemma4's
model class registers no checkpoint-conversion mapping to bridge the two
(unlike e.g. Qwen3-MoE, which has exactly this kind of per-expert-merge
mapping registered in transformers/conversion_mapping.py). A plain
`AutoModelForCausalLM.from_pretrained(...)` on this checkpoint therefore
reports all ~11520 per-expert tensors as UNEXPECTED and the model's real
`experts.gate_up_proj`/`experts.down_proj` Parameters as MISSING — silently
randomly initialized; every expert's actual computation would run on random
noise, not real weights, with no error raised. Attention, embeddings,
lm_head, the per-layer dense `mlp.*`, and the router all load correctly
(ordinary `nn.Linear`-parented tensors, unaffected).

load_moe_model() below fixes this for real (not a workaround/approximation):
it delegates to gemma_moe_loader.load_moe_model(), which (1) monkeypatches
Gemma4TextExperts to a zero-parameter placeholder before `from_pretrained` so
nothing gets silently randomly initialized, then (2) replaces each layer's
placeholder with a GemmaExpertsInt4 module populated by
gemma_expert_convert.py — which reads the per-expert
(weight_packed/weight_scale/weight_shape) triples directly off the
checkpoint's safetensors shards and dequantizes them with compressed_tensors'
own primitives, verified byte-for-byte (`torch.equal`, max abs diff 0.0)
against compressed_tensors' own top-level decompress API for 5 experts across
4 layers. GemmaExpertsInt4 keeps each expert's *original* packed int4 weights
resident on GPU (~12.9GB across all 30 layers, split across both GPUs) and
dequantizes only the router-selected experts on-the-fly per forward call —
see gemma_moe_loader.py's module docstring for the full memory/quantization
design rationale (and why this was chosen over building 11520 individual
bitsandbytes Linear4bit submodules).

Router capture: forward hooks on each layer's router submodule, found by
walking model.named_modules() for Gemma4TextRouter instances (not assumed
from the checkpoint's tensor-name path — see find_router_modules() below).
Gemma4TextRouter.forward returns (router_probabilities, top_k_weights,
top_k_index) — NOT the (router_logits, router_scores, router_indices) shape
OLMoE's OlmoeTopKRouter returns. router_probabilities is already a full
softmax over all num_experts (computed in fp32 inside the router), so unlike
the OLMoE hook this one does NOT re-apply softmax — see RouterCapture below.
Same reasoning as the OLMoE probe for using hooks over
output_router_logits=True: TIMPrimer.prime()'s private _forward_with_kv only
returns (logits, past_key_values), so a hook is the only way to see routing
during both the priming and generation phases uniformly.

Reused unmodified from the existing repo: TIMPrimer (src/tim/primer.py),
PYTHON_DOMAIN_WORDS (copied verbatim, same convention as moe_router_probe.py
and experiment-3/run_quant_gap.py), download_model/model_path_for
(src/tim/models.py), get_human_eval_plus (evalplus.data).

Does NOT touch or import from moe_router_probe.py — kept fully independent so
the OLMoE run's already-verified script and its logs/moe_probe/ output are
never at risk of being altered by this file.
"""

import argparse
import builtins
import json
import keyword
import re
import sys
import time
from pathlib import Path

import torch
from dotenv import load_dotenv
from evalplus.data import get_human_eval_plus

TIM_MOE_DIR = Path(__file__).resolve().parent
ROOT_DIR = TIM_MOE_DIR.parent
SRC_DIR = ROOT_DIR / "src"

load_dotenv(dotenv_path=str(ROOT_DIR / ".env"))

sys.path.insert(0, str(TIM_MOE_DIR))
import gemma_moe_loader as gml  # noqa: E402 — the real per-expert-format fix, see module docstring

sys.path.insert(0, str(SRC_DIR))
from tim.models import download_model, model_path_for  # noqa: E402
from tim.primer import TIMPrimer  # noqa: E402

LOGS_DIR = TIM_MOE_DIR / "logs" / "moe_probe_gemma"
LOGS_DIR.mkdir(parents=True, exist_ok=True)

MODEL_ID = "cyankiwi/gemma-4-26B-A4B-it-AWQ-4bit"
NOISE_LENGTH = 64
MAX_NEW_TOKENS = 768
N_TASKS = 20

# Copied verbatim from experiment-2/tim.py / moe_router_probe.py — see those
# files' comments for why this is duplicated rather than imported across
# directories.
PYTHON_DOMAIN_WORDS = (
    list(keyword.kwlist)
    + [name for name in dir(builtins) if not name.startswith("_")]
    + [
        "__init__", "__str__", "__repr__", "__len__", "__eq__", "__call__",
        "self", "cls", "->", "==", "!=", ">=", "<=", "+=", "-=",
        "return", "yield", "raise", "except", "finally", "assert",
        "def", "class", "lambda", "async", "await",
    ]
)


# --------------------------------------------------------------------------- #
# Model loading
# --------------------------------------------------------------------------- #

def load_moe_model():
    """Loads the compressed-tensors int4 checkpoint through the real fix for
    the per-expert/batched-Parameter format mismatch — see module docstring
    and gemma_moe_loader.py / gemma_expert_convert.py for the full account."""
    model_dir = model_path_for(MODEL_ID)
    if not model_dir.exists() or not (model_dir / "config.json").exists():
        res = download_model(MODEL_ID)
        if res.get("status") == "failed":
            raise RuntimeError(f"Failed to download {MODEL_ID}: {res.get('error')}")

    return gml.load_moe_model(str(model_dir))


def stop_token_ids(tokenizer, model) -> set:
    """Gemma-4's chat template ends an assistant turn with the '<turn|>'
    special token, not (only) '<eos>' — config.json's eos_token_id is a list
    ([1, 106]) and generation_config.json's is [1, 106, 50]. Unlike OLMoE
    (single eos id), the manual greedy-decode loop here must stop on ANY of
    these, or it will run every generation to the max_new_tokens cap."""
    ids = set()
    cfg_eos = getattr(model.config, "eos_token_id", None)
    gen_eos = getattr(getattr(model, "generation_config", None), "eos_token_id", None)
    for src in (cfg_eos, gen_eos, tokenizer.eos_token_id):
        if src is None:
            continue
        if isinstance(src, (list, tuple, set)):
            ids.update(int(x) for x in src)
        else:
            ids.add(int(src))
    return ids


def model_report(model, tokenizer) -> dict:
    cfg = model.config.get_text_config()
    routers = find_router_modules(model)
    param_dtypes = sorted({str(p.dtype) for p in model.parameters()})
    buffer_dtypes = sorted({str(b.dtype) for name, b in model.named_buffers() if "packed" in name or "scale" in name})
    expert_devices = sorted({
        str(m.gate_packed.device) for m in model.modules() if isinstance(m, gml.GemmaExpertsInt4)
    })
    expert_bytes = sum(
        t.numel() * t.element_size()
        for m in model.modules() if isinstance(m, gml.GemmaExpertsInt4)
        for t in (m.gate_packed, m.gate_scale, m.up_packed, m.up_scale, m.down_packed, m.down_scale)
    )
    return {
        "model_id": MODEL_ID,
        "architecture": model.config.architectures[0] if getattr(model.config, "architectures", None) else None,
        "num_experts": getattr(cfg, "num_experts", None),
        "num_experts_per_tok": getattr(cfg, "top_k_experts", None),
        "num_hidden_layers": getattr(cfg, "num_hidden_layers", None),
        "hidden_size": getattr(cfg, "hidden_size", None),
        "moe_layer_indices": sorted(routers.keys()),
        "n_moe_layers": len(routers),
        "precision": (
            "non-expert weights (attention/embed/lm_head/router/dense-mlp): bf16, resident on "
            f"{next(model.parameters()).device}. Expert weights (gate/up/down_proj, all 30 layers x "
            "128 experts): genuine int4 (compressed-tensors pack-quantized, group_size=32, symmetric), "
            "kept in their ORIGINAL packed-int4 form resident on GPU "
            f"({expert_bytes/1e9:.2f} GB across {sorted(expert_devices)}) and dequantized on-the-fly, "
            "per router-selected expert, at each forward call — see gemma_moe_loader.py / "
            "gemma_expert_convert.py and tim-moe/docs/moe_gemma_comparison.md for the fix this "
            "replaces (plain from_pretrained silently randomly-initializes all expert weights for "
            "this checkpoint) and its verification evidence."
        ),
        "param_dtypes_loaded": param_dtypes,
        "expert_buffer_dtypes": buffer_dtypes,
        "expert_bytes_resident": expert_bytes,
        "expert_devices": expert_devices,
        "device_map": {n: str(d) for n, d in getattr(model, "hf_device_map", {}).items()} or None,
        "router_logits_extractable": True,
        "router_logits_method": (
            "forward_hook on the Gemma4TextRouter module of each decoder layer, found via "
            "model.named_modules() (not an assumed path — see find_router_modules()); "
            "forward returns (router_probabilities, top_k_weights, top_k_index), "
            "router_probabilities already a full softmax over all experts computed in fp32"
        ),
    }


# --------------------------------------------------------------------------- #
# Router capture — forward hooks on each layer's Gemma4TextRouter module,
# discovered by walking named_modules() rather than assuming a path.
# --------------------------------------------------------------------------- #

def find_router_modules(model) -> dict:
    """layer_idx -> Gemma4TextRouter submodule, for every layer that has one.
    Walks model.named_modules() and matches on class name + a trailing
    '...layers.<N>.router' name pattern, rather than assuming the runtime
    module path mirrors the checkpoint's tensor-name path exactly (it may
    not, once wrapped by device_map="auto")."""
    routers = {}
    pat = re.compile(r"layers\.(\d+)\.router$")
    for name, module in model.named_modules():
        if module.__class__.__name__ == "Gemma4TextRouter":
            m = pat.search(name)
            if m:
                routers[int(m.group(1))] = module
    return routers


class RouterCapture:
    """Same accumulation contract as moe_router_probe.py's RouterCapture:
    start()/stop() toggle recording, snapshot() reads accumulated per-layer
    expert_set + mean router distribution out. Adapted only for Gemma4TextRouter's
    output shape (router_probabilities is already softmax'd — no re-softmax)."""

    def __init__(self, model):
        self.routers = find_router_modules(model)
        self.active = False
        self.layer_stats = {}
        self.handles = [r.register_forward_hook(self._make_hook(i)) for i, r in self.routers.items()]

    def _make_hook(self, layer_idx):
        def hook(module, inputs, output):
            if not self.active:
                return
            router_probabilities, top_k_weights, top_k_index = output
            probs = router_probabilities.float()  # already softmax over all experts, fp32
            stat = self.layer_stats.setdefault(
                layer_idx, {"expert_set": set(), "sum_probs": None, "n_tokens": 0}
            )
            idx = top_k_index.detach().cpu()
            for row in idx.tolist():
                stat["expert_set"].update(row)
            sp = probs.sum(dim=0).detach().cpu()
            stat["sum_probs"] = sp if stat["sum_probs"] is None else stat["sum_probs"] + sp
            stat["n_tokens"] += router_probabilities.shape[0]
        return hook

    def start(self):
        self.layer_stats = {}
        self.active = True

    def stop(self):
        self.active = False

    def snapshot(self) -> dict:
        out = {}
        for layer_idx, stat in self.layer_stats.items():
            n = stat["n_tokens"]
            mean_dist = (stat["sum_probs"] / n).tolist() if n > 0 else []
            out[str(layer_idx)] = {
                "expert_set": sorted(int(x) for x in stat["expert_set"]),
                "mean_dist": mean_dist,
                "n_tokens": n,
            }
        return out

    def remove(self):
        for h in self.handles:
            h.remove()


# --------------------------------------------------------------------------- #
# Prompting / generation (chat template + code-fence extraction convention
# copied from moe_router_probe.py / experiment-2/tim.py).
# --------------------------------------------------------------------------- #

def build_chat_prompt(tokenizer, problem_prompt: str) -> str:
    content = (
        "Complete the following Python function. "
        "Return ONLY the code in a single ```python fenced block, "
        "with no explanation before or after.\n\n"
        f"{problem_prompt}"
    )
    messages = [{"role": "user", "content": content}]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
    )


def manual_greedy_decode(model, tokenizer, device, input_ids, past_key_values, capture,
                          max_new_tokens, eos_ids):
    """Same prefill/generate split as moe_router_probe.py's
    manual_greedy_decode: the prefill forward call runs with `capture`
    inactive, so "generation-phase routing" means routing for tokens the
    model actually emits. Stops on ANY id in eos_ids (see stop_token_ids()),
    not a single eos id."""
    attention_mask = None
    if past_key_values is not None:
        cache_len = (
            past_key_values.get_seq_length() if hasattr(past_key_values, "get_seq_length")
            else past_key_values[0][0].shape[-2]
        )
        full_len = cache_len + input_ids.shape[-1]
        attention_mask = torch.ones((1, full_len), dtype=torch.long, device=device)

    with torch.no_grad():
        if past_key_values is not None:
            out = model(input_ids=input_ids, past_key_values=past_key_values,
                        attention_mask=attention_mask, use_cache=True)
        else:
            out = model(input_ids=input_ids, use_cache=True)
    kv = out.past_key_values
    next_token = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
    generated = [int(next_token.item())]

    capture.start()
    for _ in range(max_new_tokens - 1):
        if generated[-1] in eos_ids:
            break
        with torch.no_grad():
            out = model(input_ids=next_token, past_key_values=kv, use_cache=True)
        kv = out.past_key_values
        next_token = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        generated.append(int(next_token.item()))
    capture.stop()

    routing = capture.snapshot()
    text = tokenizer.decode(generated, skip_special_tokens=True)
    return text, routing, len(generated)


# --------------------------------------------------------------------------- #
# Per-task, per-condition run
# --------------------------------------------------------------------------- #

def run_task(model, tokenizer, device, primer, capture, eos_ids, task_id, prompt, seed=0):
    """Returns dict: {condition: {phase: routing_snapshot}}"""
    result = {}

    # ---- cold: no priming, generation-phase routing only ------------------
    chat_text = build_chat_prompt(tokenizer, prompt)
    input_ids = tokenizer(chat_text, return_tensors="pt", add_special_tokens=False).input_ids.to(device)
    _, routing, n_gen = manual_greedy_decode(model, tokenizer, device, input_ids, None, capture,
                                              MAX_NEW_TOKENS, eos_ids)
    result["cold"] = {"generation": routing, "n_generated_tokens": n_gen}

    # ---- tim_domain: prime with 70/30 domain/random noise ------------------
    domain_ids = primer.get_domain_token_ids_from_words(PYTHON_DOMAIN_WORDS)
    capture.start()
    kv = primer.prime(domain_tokens=domain_ids, seed=seed)
    priming_routing = capture.snapshot()
    capture.stop()

    input_ids = tokenizer(chat_text, return_tensors="pt", add_special_tokens=False).input_ids.to(device)
    _, gen_routing, n_gen = manual_greedy_decode(model, tokenizer, device, input_ids, kv, capture,
                                                  MAX_NEW_TOKENS, eos_ids)
    result["tim_domain"] = {
        "priming": priming_routing,
        "generation": gen_routing,
        "n_generated_tokens": n_gen,
    }
    del kv

    # ---- tim_random: pure random-vocab noise, same seed --------------------
    capture.start()
    kv = primer.prime(domain_tokens=None, seed=seed)
    priming_routing = capture.snapshot()
    capture.stop()

    input_ids = tokenizer(chat_text, return_tensors="pt", add_special_tokens=False).input_ids.to(device)
    _, gen_routing, n_gen = manual_greedy_decode(model, tokenizer, device, input_ids, kv, capture,
                                                  MAX_NEW_TOKENS, eos_ids)
    result["tim_random"] = {
        "priming": priming_routing,
        "generation": gen_routing,
        "n_generated_tokens": n_gen,
    }
    del kv

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result


# --------------------------------------------------------------------------- #
# Self-test — mandatory gate, must pass before any real experiment runs.
# --------------------------------------------------------------------------- #

def run_self_test(model, tokenizer, device, eos_ids, prompt):
    print("\n=== SELF-TEST: coherent generation on one HumanEval prompt? ===")
    chat_text = build_chat_prompt(tokenizer, prompt)
    input_ids = tokenizer(chat_text, return_tensors="pt", add_special_tokens=False).input_ids.to(device)
    with torch.no_grad():
        out = model(input_ids=input_ids, use_cache=True)
    kv = out.past_key_values
    next_token = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
    generated = [int(next_token.item())]
    for _ in range(149):
        if generated[-1] in eos_ids:
            break
        with torch.no_grad():
            out = model(input_ids=next_token, past_key_values=kv, use_cache=True)
        kv = out.past_key_values
        next_token = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        generated.append(int(next_token.item()))
    text = tokenizer.decode(generated, skip_special_tokens=True)
    print(f"--- raw output ({len(generated)} tokens) ---\n{text}\n--- end raw output ---")
    return text


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_tasks", type=int, default=N_TASKS)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--self_test_only", action="store_true")
    args = ap.parse_args()

    print(f"[load] {MODEL_ID} ...")
    t0 = time.perf_counter()
    model, tokenizer, device = load_moe_model()
    print(f"[load] done in {time.perf_counter()-t0:.1f}s, device={device}")

    eos_ids = stop_token_ids(tokenizer, model)
    print(f"[eos] stop token ids: {sorted(eos_ids)}")

    report = model_report(model, tokenizer)
    print("=== MODEL REPORT ===")
    print(json.dumps(report, indent=2))

    problems = get_human_eval_plus()
    task_items = list(problems.items())[: args.n_tasks]
    task_ids = [tid for tid, _ in task_items]

    run_self_test(model, tokenizer, device, eos_ids, task_items[0][1]["prompt"])
    if args.self_test_only:
        print("[done] --self_test_only set, exiting after self-test.")
        return

    capture = RouterCapture(model)
    primer = TIMPrimer(model, tokenizer, device, noise_length=NOISE_LENGTH, num_passes=1)

    print(f"[tasks] {len(task_items)} tasks: {task_ids}")

    all_results = {}
    t_start = time.perf_counter()
    for i, (task_id, problem) in enumerate(task_items):
        t0 = time.perf_counter()
        all_results[task_id] = run_task(
            model, tokenizer, device, primer, capture, eos_ids, task_id, problem["prompt"], seed=args.seed
        )
        dt = time.perf_counter() - t0
        print(f"[{i+1}/{len(task_items)}] {task_id} done in {dt:.1f}s")

    total_dt = time.perf_counter() - t_start
    print(f"[done] all tasks in {total_dt:.1f}s")

    out = {
        "model_report": report,
        "noise_length": NOISE_LENGTH,
        "max_new_tokens": MAX_NEW_TOKENS,
        "seed": args.seed,
        "task_ids": task_ids,
        "results": all_results,
    }
    out_path = LOGS_DIR / "routing_raw.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"[write] {out_path} ({out_path.stat().st_size / 1e6:.1f} MB)")

    capture.remove()


if __name__ == "__main__":
    main()
