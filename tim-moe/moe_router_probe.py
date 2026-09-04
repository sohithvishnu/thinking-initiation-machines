"""
MoE router probe: does TIM priming move expert routing, and does the expert
set activated during priming predict the expert set activated during
generation?

This is a routing-mechanics probe, not a performance/accuracy study. It opens
the MoE line of work (see tim-moe/docs/moe_router_probe.md for the full
write-up) with the cheapest, most decisive first question:

  Q1 (router shift): does priming change which experts the router selects,
      vs. a cold prompt?
  Q2 (prediction, the linchpin): do the experts activated during the priming
      pass overlap with the experts activated during generation of the real
      task, above a cross-task chance baseline? Expert prefetching only makes
      sense if the answer is yes.

Model: allenai/OLMoE-1B-7B-0924-Instruct (OlmoeForCausalLM), loaded in plain bf16 —
NOT 4-bit. Two candidates were ruled out before writing any experiment code
(Step 0):

  - Qwen/Qwen3-30B-A3B: safetensors total ~61.1GB, doesn't fit this
    machine's ~52-55GB free disk (checked via
    HfApi().model_info(..., files_metadata=True) before downloading
    anything).
  - Qwen/Qwen1.5-MoE-A2.7B (the spec's documented fallback): downloaded
    (28.6GB) and its config verified cleanly (60 experts, top-4, 24 layers,
    all-MoE), but loading it with BitsAndBytesConfig(4-bit) still tried to
    allocate ~14.3GB of VRAM and OOM'd. Root cause, confirmed by reading the
    installed transformers 5.13.1 source: this version's Qwen2MoE (and
    Mixtral, and every other current MoE class checked) stores all experts'
    weights as two batched `nn.Parameter` tensors (`gate_up_proj`,
    `down_proj`) inside a custom `*Experts` module, not as individual
    `nn.Linear` submodules. bitsandbytes' automatic 4-bit replacement only
    targets `nn.Linear` instances
    (`Bnb4BitHfQuantizer.param_needs_quantization` checks
    `isinstance(module, bnb.nn.Linear4bit)`), so those batched expert
    tensors are silently never quantized — the "4-bit" load was actually
    loading ~24GB of expert weights in bf16. This is a transformers-version
    / bitsandbytes incompatibility affecting every current-generation MoE
    architecture in this environment, not a Qwen1.5-MoE-specific issue, so
    swapping to a different large MoE model wouldn't have fixed it. The
    28.6GB download was deleted after confirming this (see git history /
    session log — not committed).

Given quantization is a structural no-op for MoE experts here, the real
constraint became: which well-supported MoE model's *unquantized* bf16
footprint fits this machine's ~22GB combined free VRAM (a 16GB + an 8GB
card, both partially occupied by other processes)? allenai/OLMoE-1B-7B-0924-Instruct
(6.9B total params, 1B active, 64 experts, top-8 routing, 16 layers, safetensors
~13.8GB) fits comfortably with headroom for KV-cache and activations, and
its `OlmoeSparseMoeBlock`/`OlmoeTopKRouter` are structurally identical to
Qwen2MoE's (same batched-experts pattern, same router forward signature
`(router_logits, router_scores, router_indices)`), so the router-capture
approach below needed no changes across the swap. The -Instruct variant
(not the plain -0924 base checkpoint) was used specifically because the
base checkpoint's tokenizer ships with no `chat_template` at all — this
repo's prompting convention (build_chat_prompt, copied from
experiment-2/tim.py) is chat-template-based, and the Instruct variant is
the same size and architecture with a template included.

Router capture: implemented via forward hooks on each layer's
`model.model.layers[i].mlp.gate` module (a Qwen2MoeTopKRouter instance),
NOT via the native `output_router_logits=True` kwarg. Reason: that kwarg
only reaches you through the top-level model output, but TIMPrimer.prime()
(experiment-2/tim_primer.py, reused unmodified here) makes its own internal
forward calls via a private `_forward_with_kv` that only returns
(logits, past_key_values) — there's no way to get router_logits back out of
it without editing TIMPrimer, which the task explicitly rules out ("Do not
modify dense-model code"). A forward hook on the router submodule instead
captures routing on *every* forward call transparently, regardless of who
calls the model (TIMPrimer internally, or this script's own manual decode
loop) — same mechanism, no dense-model file touched, and it uniformly covers
both the priming phase and the generation phase.

Reused unmodified from the existing repo: TIMPrimer (experiment-2/tim_primer.py),
PYTHON_DOMAIN_WORDS (copied verbatim from experiment-2/tim.py, matching the
convention experiment-3/run_quant_gap.py already established rather than
importing across directories), download_model/model_path_for (utils.py),
get_human_eval_plus (evalplus.data).
"""

import argparse
import builtins
import json
import keyword
import sys
import time
from pathlib import Path

import torch
from dotenv import load_dotenv
from evalplus.data import get_human_eval_plus
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

TIM_MOE_DIR = Path(__file__).resolve().parent
ROOT_DIR = TIM_MOE_DIR.parent
SRC_DIR = ROOT_DIR / "src"

load_dotenv(dotenv_path=str(ROOT_DIR / ".env"))

sys.path.insert(0, str(SRC_DIR))
from tim.models import download_model, model_path_for  # noqa: E402
from tim.primer import TIMPrimer  # noqa: E402

LOGS_DIR = TIM_MOE_DIR / "logs" / "moe_probe"
LOGS_DIR.mkdir(parents=True, exist_ok=True)

MODEL_ID = "allenai/OLMoE-1B-7B-0924-Instruct"
NOISE_LENGTH = 64
MAX_NEW_TOKENS = 768
N_TASKS = 20

# Copied verbatim from experiment-2/tim.py — see that file's comment for why
# this is duplicated rather than imported across directories.
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

def load_moe_model(quant: str = "bf16"):
    """quant='nf4' is kept for completeness but is a documented no-op for this
    model family in the installed transformers version — see the module
    docstring's Step 0 write-up. Default is plain bf16, which is what
    actually fits this machine for a MoE model at this scale."""
    model_dir = model_path_for(MODEL_ID)
    if not model_dir.exists() or not (model_dir / "config.json").exists():
        res = download_model(MODEL_ID)
        if res.get("status") == "failed":
            raise RuntimeError(f"Failed to download {MODEL_ID}: {res.get('error')}")

    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    if quant == "nf4":
        # Single GPU: the earlier (abandoned) Qwen1.5-MoE-A2.7B attempt at
        # this quant path hit a caching_allocator_warmup OOM under
        # device_map="auto" on this machine's uneven two-GPU setup; kept as
        # {"": 0} here for anyone re-testing whether a future
        # transformers/bitsandbytes release actually quantizes batched
        # experts.
        device_map = {"": 0} if torch.cuda.is_available() else None
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_dir, local_files_only=True, trust_remote_code=True,
            device_map=device_map, quantization_config=bnb_config,
        )
    elif quant == "bf16":
        # Load to CPU first, then move the fully-materialized model to GPU0
        # in one shot. Two GPU-resident device_map paths were tried first
        # and both failed on this machine: "auto" mis-split the model across
        # the uneven two-GPU setup (16GB + 8GB), overloading the smaller
        # card; {"": 0} hit transformers' caching_allocator_warmup
        # pre-reserving ~14GB of the 16GB card up front (sized almost
        # exactly to the model's own footprint, leaving no margin), then
        # OOMing on the *extra* scratch space needed mid-load to concatenate
        # split checkpoint shards (e.g. mlp.experts.gate_up_proj, stored as
        # 2 tensors in the checkpoint) into their final merged tensors.
        # Loading to CPU sidesteps both: the concatenation scratch is CPU
        # RAM (24GB available, comfortable for a 13.8GB model), and the
        # subsequent .to(device) is a straight copy of already-final tensors
        # with no extra merge overhead on the GPU side.
        model = AutoModelForCausalLM.from_pretrained(
            model_dir, local_files_only=True, trust_remote_code=True,
            device_map=None, torch_dtype=torch.bfloat16,
        )
        if torch.cuda.is_available():
            model = model.to("cuda:0")
    else:
        raise ValueError(f"unknown quant {quant!r}")

    device = next(model.parameters()).device
    model.eval()
    return model, tokenizer, device


def model_report(model, tokenizer, quant: str) -> dict:
    cfg = model.config
    moe_gates = find_router_modules(model)
    param_dtypes = sorted({str(p.dtype) for p in model.parameters()})
    return {
        "model_id": MODEL_ID,
        "architecture": cfg.architectures[0] if getattr(cfg, "architectures", None) else None,
        "num_experts": getattr(cfg, "num_experts", None),
        "num_experts_per_tok": getattr(cfg, "num_experts_per_tok", None),
        "num_hidden_layers": getattr(cfg, "num_hidden_layers", None),
        "hidden_size": getattr(cfg, "hidden_size", None),
        "moe_layer_indices": sorted(moe_gates.keys()),
        "n_moe_layers": len(moe_gates),
        "active_params_approx_B": 1.3,
        "total_params_approx_B": 6.9,
        "precision": quant,
        "param_dtypes_loaded": param_dtypes,
        "router_logits_extractable": True,
        "router_logits_method": (
            "forward_hook on model.model.layers[i].mlp.gate "
            "(OlmoeTopKRouter; forward returns (router_logits, router_scores, router_indices)) "
            "— not native output_router_logits=True, see module docstring for why"
        ),
    }


# --------------------------------------------------------------------------- #
# Router capture — forward hooks on each layer's router module
# --------------------------------------------------------------------------- #

def find_router_modules(model) -> dict:
    """layer_idx -> gate submodule, for every layer that has one (MoE layer)."""
    gates = {}
    for i, layer in enumerate(model.model.layers):
        gate = getattr(layer.mlp, "gate", None)
        if gate is not None and hasattr(gate, "top_k"):
            gates[i] = gate
    return gates


class RouterCapture:
    """Accumulates per-layer expert-selection stats over whatever forward
    calls happen while `active`. Call start() to reset+begin recording,
    stop() to end, snapshot() to read the accumulated per-layer stats out."""

    def __init__(self, model):
        self.gates = find_router_modules(model)
        self.active = False
        self.layer_stats = {}
        self.handles = [gate.register_forward_hook(self._make_hook(i)) for i, gate in self.gates.items()]

    def _make_hook(self, layer_idx):
        def hook(module, inputs, output):
            if not self.active:
                return
            router_logits, router_scores, router_indices = output
            probs = torch.softmax(router_logits.float(), dim=-1)  # [n_tok_this_call, num_experts]
            stat = self.layer_stats.setdefault(
                layer_idx, {"expert_set": set(), "sum_probs": None, "n_tokens": 0}
            )
            idx = router_indices.detach().cpu()
            for row in idx.tolist():
                stat["expert_set"].update(row)
            sp = probs.sum(dim=0).detach().cpu()
            stat["sum_probs"] = sp if stat["sum_probs"] is None else stat["sum_probs"] + sp
            stat["n_tokens"] += router_logits.shape[0]
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
# Prompting / generation (chat template + code-fence extraction copied from
# experiment-2/tim.py's build_chat_prompt / extract_code — same convention).
# --------------------------------------------------------------------------- #

def build_chat_prompt(tokenizer, problem_prompt: str) -> str:
    content = (
        "Complete the following Python function. "
        "Return ONLY the code in a single ```python fenced block, "
        "with no explanation before or after.\n\n"
        f"{problem_prompt}"
    )
    messages = [{"role": "user", "content": content}]
    # No enable_thinking kwarg here (unlike the Qwen dense-model scripts this
    # is copied from) — OLMoE-Instruct's chat template doesn't define a
    # thinking mode, and apply_chat_template only accepts kwargs its
    # specific Jinja template actually references.
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
    )


def manual_greedy_decode(model, tokenizer, device, input_ids, past_key_values, capture, max_new_tokens):
    """
    Greedy decode with explicit control over exactly which forward calls get
    captured by `capture`: the prompt-prefill call (processing the full
    input_ids in one shot, possibly on top of a warm primed cache) is
    deliberately done with capture inactive, so "generation-phase routing"
    means routing for the tokens the model actually emits, not the prompt it
    read. capture.start() is called only once the prefill is done.
    """
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

    eos_id = tokenizer.eos_token_id
    capture.start()
    for _ in range(max_new_tokens - 1):
        if generated[-1] == eos_id:
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

def run_task(model, tokenizer, device, primer, capture, task_id, prompt, seed=0):
    """Returns dict: {condition: {phase: routing_snapshot}}"""
    result = {}

    # ---- cold: no priming, generation-phase routing only ------------------
    chat_text = build_chat_prompt(tokenizer, prompt)
    input_ids = tokenizer(chat_text, return_tensors="pt", add_special_tokens=False).input_ids.to(device)
    _, routing, n_gen = manual_greedy_decode(model, tokenizer, device, input_ids, None, capture, MAX_NEW_TOKENS)
    result["cold"] = {"generation": routing, "n_generated_tokens": n_gen}

    # ---- tim_domain: prime with 70/30 domain/random noise ------------------
    domain_ids = primer.get_domain_token_ids_from_words(PYTHON_DOMAIN_WORDS)
    capture.start()
    kv = primer.prime(domain_tokens=domain_ids, seed=seed)
    priming_routing = capture.snapshot()
    capture.stop()

    input_ids = tokenizer(chat_text, return_tensors="pt", add_special_tokens=False).input_ids.to(device)
    _, gen_routing, n_gen = manual_greedy_decode(model, tokenizer, device, input_ids, kv, capture, MAX_NEW_TOKENS)
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
    _, gen_routing, n_gen = manual_greedy_decode(model, tokenizer, device, input_ids, kv, capture, MAX_NEW_TOKENS)
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
# Main
# --------------------------------------------------------------------------- #

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quant", type=str, choices=["nf4", "bf16"], default="bf16")
    ap.add_argument("--n_tasks", type=int, default=N_TASKS)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit_layers_print", type=int, default=5)
    args = ap.parse_args()

    print(f"[load] {MODEL_ID} quant={args.quant} ...")
    t0 = time.perf_counter()
    model, tokenizer, device = load_moe_model(quant=args.quant)
    print(f"[load] done in {time.perf_counter()-t0:.1f}s, device={device}")

    report = model_report(model, tokenizer, args.quant)
    print("=== MODEL REPORT ===")
    print(json.dumps(report, indent=2))

    capture = RouterCapture(model)

    primer = TIMPrimer(model, tokenizer, device, noise_length=NOISE_LENGTH, num_passes=1)

    problems = get_human_eval_plus()
    task_items = list(problems.items())[: args.n_tasks]
    task_ids = [tid for tid, _ in task_items]
    print(f"[tasks] {len(task_items)} tasks: {task_ids}")

    all_results = {}
    t_start = time.perf_counter()
    for i, (task_id, problem) in enumerate(task_items):
        t0 = time.perf_counter()
        all_results[task_id] = run_task(
            model, tokenizer, device, primer, capture, task_id, problem["prompt"], seed=args.seed
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
