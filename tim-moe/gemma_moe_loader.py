"""
Shared model-loading path for the Gemma-4-26B-A4B MoE replication scripts
(moe_router_probe_gemma.py, moe_expert_lock_gemma.py).

Fixes the checkpoint/library format mismatch documented in
gemma_expert_convert.py's module docstring and tim-moe/docs/moe_gemma_comparison.md:
cyankiwi/gemma-4-26B-A4B-it-AWQ-4bit stores MoE expert weights as one
compressed-tensors pack-quantized int4 tensor triple
(weight_packed/weight_scale/weight_shape) per expert per projection, but
installed transformers (5.13.1)'s Gemma4TextExperts expects one batched
[num_experts, ...] Parameter per projection and has no checkpoint-conversion
mapping registered to bridge the two — so a plain `AutoModelForCausalLM.
from_pretrained(...)` call silently leaves every expert's real weights
randomly initialized (see gemma_expert_convert.py docstring for the LOAD
REPORT evidence: 11520 UNEXPECTED expert tensors, 60 MISSING
experts.{gate_up_proj,down_proj} Parameters, zero mismatches elsewhere).

Fix, in two steps:

1. Monkeypatch `Gemma4TextExperts` -> a zero-parameter placeholder BEFORE
   calling `from_pretrained`, so the model never allocates/randomly-inits the
   native batched Parameters at all (avoids ~46GB of wasted bf16 allocation
   for parameters we're about to throw away) and the loader's mismatch
   report is limited to "expert tensors are unexpected" (true and harmless —
   we load them ourselves) rather than also claiming missing data. Every
   other tensor (attention, embeddings, lm_head, router, the per-layer dense
   `mlp.*`) loads through the normal compressed-tensors path exactly as
   before, unaffected by this patch.

2. After `from_pretrained` returns, walk the model for decoder layers and
   replace each placeholder `.experts` with a `GemmaExpertsInt4` module,
   populated via gemma_expert_convert.py's verified dequantization
   primitives.

Memory/quantization design (see moe_gemma_comparison.md for the full
rationale and the alternative considered — 128 individual bitsandbytes
Linear4bit submodules per layer): `GemmaExpertsInt4` keeps each expert's
*original* compressed-tensors packed int4 weights (weight_packed +
weight_scale, unmodified from the checkpoint) resident on GPU as plain
buffers — ~12.9GB total for all 30 layers x 128 experts x 3 projections,
measured directly (see attach_int4_experts()'s printed report) — and
dequantizes only the handful of experts actually selected by the router for
each forward call, on-GPU, using the exact same unpack_from_int32 +
dequantize primitives verified byte-for-byte against compressed_tensors' own
decompress API in gemma_expert_convert.verify(). This was chosen over
building 11520 bitsandbytes Linear4bit submodules (config.num_experts=128 x
config.num_hidden_layers=30 x 3 projections) because: (a) it reuses
already-verified, exact (non-lossy) dequantization instead of introducing a
second, different int4 quantization scheme on top of the first; (b) it
avoids constructing/quantizing 11520 separate nn.Module objects; (c)
benchmarking showed the on-GPU dequant overhead is ~4.7ms per layer for 8
active experts (config.top_k_experts=8) x 3 projections, i.e. ~140ms/token
across all 30 layers — an acceptable fraction of total per-token generation
time for a 20-task research run, not a bottleneck. Gemma4TextExperts.forward
already only computes experts that were actually hit by the router (an
`expert_hit` loop, not a dense sweep over all num_experts), so this mirrors
the reference eager implementation's compute pattern exactly; we've just
added one more step (dequantize this expert's weight right before its
matmul) inside that same per-hit loop.
"""

from __future__ import annotations

import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

import transformers.models.gemma4.modeling_gemma4 as gemma4_mod

from gemma_expert_convert import CheckpointReader, dequantize_packed, read_packed_stack


# --------------------------------------------------------------------------- #
# Step 1: placeholder that replaces Gemma4TextExperts before from_pretrained.
# --------------------------------------------------------------------------- #

class _PlaceholderExperts(nn.Module):
    """Zero-parameter stand-in for Gemma4TextExperts. Holds no
    gate_up_proj/down_proj, so from_pretrained never allocates or randomly
    initializes them (the checkpoint's per-expert tensors become harmless
    UNEXPECTED keys instead of silently-wrong MISSING ones). Must be
    replaced with a populated GemmaExpertsInt4 via attach_int4_experts()
    before the model is used for anything."""

    def __init__(self, config):
        super().__init__()
        self.num_experts = config.num_experts
        # Gemma4PreTrainedModel._init_weights() does `isinstance(module,
        # Gemma4TextExperts)` — a global name lookup that now resolves to
        # THIS class (see patch_gemma4_experts_for_loading()), so it matches
        # placeholders too and tries `init.normal_(module.gate_up_proj, ...)`
        # / `.down_proj`. Give it real (but zero-size, ~free) Parameters so
        # that succeeds as a no-op instead of raising AttributeError.
        self.gate_up_proj = nn.Parameter(torch.empty(0))
        self.down_proj = nn.Parameter(torch.empty(0))

    def forward(self, *args, **kwargs):
        raise RuntimeError(
            "_PlaceholderExperts.forward() called — attach_int4_experts() was "
            "never run to replace this layer's placeholder with real expert weights."
        )


_original_gemma4_text_experts = gemma4_mod.Gemma4TextExperts


def patch_gemma4_experts_for_loading():
    """Call before AutoModelForCausalLM.from_pretrained(). Idempotent."""
    gemma4_mod.Gemma4TextExperts = _PlaceholderExperts


def unpatch_gemma4_experts():
    """Restore the original class (only matters if something else in-process
    constructs a fresh Gemma4 model after we're done needing the patch)."""
    gemma4_mod.Gemma4TextExperts = _original_gemma4_text_experts


# --------------------------------------------------------------------------- #
# Step 2: the real int4 experts module, populated after from_pretrained.
# --------------------------------------------------------------------------- #

class GemmaExpertsInt4(nn.Module):
    """Drop-in replacement for Gemma4TextExperts with identical forward
    semantics (same expert_hit loop, same index_add_ accumulation, same
    top_k_weights application) but holding each expert's original
    compressed-tensors packed int4 (weight_packed + weight_scale) instead of
    a dequantized batched Parameter. See module docstring for the full
    rationale."""

    def __init__(self, config):
        super().__init__()
        self.num_experts = config.num_experts
        self.hidden_dim = config.hidden_size
        self.intermediate_dim = config.moe_intermediate_size
        self.act_fn = gemma4_mod.ACT2FN[config.hidden_activation]
        # buffers gate_packed/gate_scale/up_packed/up_scale/down_packed/
        # down_scale are registered directly by load_expert_weights() below
        # (register_buffer rejects pre-existing plain attributes of the same
        # name, so we don't pre-declare them here).
        self.gate_shape = (self.intermediate_dim, self.hidden_dim)
        self.up_shape = (self.intermediate_dim, self.hidden_dim)
        self.down_shape = (self.hidden_dim, self.intermediate_dim)

    def load_expert_weights(self, reader: CheckpointReader, layer_idx: int, device: torch.device):
        gp, gs, gshape = read_packed_stack(reader, layer_idx, self.num_experts, "gate_proj")
        up, us, ushape = read_packed_stack(reader, layer_idx, self.num_experts, "up_proj")
        dp, ds, dshape = read_packed_stack(reader, layer_idx, self.num_experts, "down_proj")
        assert gshape == self.gate_shape, (gshape, self.gate_shape)
        assert ushape == self.up_shape, (ushape, self.up_shape)
        assert dshape == self.down_shape, (dshape, self.down_shape)
        self.register_buffer("gate_packed", gp.to(device), persistent=False)
        self.register_buffer("gate_scale", gs.to(device), persistent=False)
        self.register_buffer("up_packed", up.to(device), persistent=False)
        self.register_buffer("up_scale", us.to(device), persistent=False)
        self.register_buffer("down_packed", dp.to(device), persistent=False)
        self.register_buffer("down_scale", ds.to(device), persistent=False)

    def _dequant(self, packed_e, scale_e, shape, out_dtype):
        return dequantize_packed(packed_e, scale_e, shape).to(out_dtype)

    def forward(self, hidden_states, top_k_index, top_k_weights):
        final_hidden_states = torch.zeros_like(hidden_states)
        with torch.no_grad():
            expert_mask = F.one_hot(top_k_index, num_classes=self.num_experts)
            expert_mask = expert_mask.permute(2, 1, 0)
            expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()

        for expert_idx in expert_hit:
            e = int(expert_idx[0])
            if e == self.num_experts:
                continue
            top_k_pos, token_idx = torch.where(expert_mask[e])
            current_state = hidden_states[token_idx]
            expert_device = self.gate_packed.device
            orig_device = current_state.device
            if expert_device != orig_device:
                current_state = current_state.to(expert_device)

            gate_w = self._dequant(self.gate_packed[e], self.gate_scale[e], self.gate_shape, current_state.dtype)
            up_w = self._dequant(self.up_packed[e], self.up_scale[e], self.up_shape, current_state.dtype)
            gate = F.linear(current_state, gate_w)
            up = F.linear(current_state, up_w)
            current_hidden_states = self.act_fn(gate) * up

            down_w = self._dequant(self.down_packed[e], self.down_scale[e], self.down_shape, current_state.dtype)
            current_hidden_states = F.linear(current_hidden_states, down_w)

            if expert_device != orig_device:
                current_hidden_states = current_hidden_states.to(orig_device)
            current_hidden_states = current_hidden_states * top_k_weights[token_idx, top_k_pos, None]
            final_hidden_states.index_add_(0, token_idx, current_hidden_states.to(final_hidden_states.dtype))

        return final_hidden_states


def find_placeholder_experts(model) -> dict:
    """layer_idx -> (parent_decoder_layer_module, '_PlaceholderExperts' instance)."""
    import re
    pat = re.compile(r"layers\.(\d+)\.experts$")
    out = {}
    for name, module in model.named_modules():
        if isinstance(module, _PlaceholderExperts):
            m = pat.search(name)
            if m:
                out[int(m.group(1))] = (name, module)
    return out


# Measured exactly from tensor shapes: 128 experts x (gate+up+down) packed
# int32 + fp16 group scales = ~408.4 MiB/layer. Add ~12% margin for the CUDA
# caching allocator's per-allocation overhead (7 separate tensors/layer) —
# an earlier attempt at this without margin OOM'd 1 tensor short of the end
# of a 30-layer plan (see moe_gemma_comparison.md).
BYTES_PER_LAYER_EXPERTS = int(458 * 1024 * 1024)
DEFAULT_DEVICE_RESERVE_MIB = 500  # flat per-device reserve (CUDA context + allocator slack)


def plan_layer_devices(num_layers: int, reserve_mib: dict[int, int] | None = None) -> dict[int, torch.device]:
    """Balance the ~430MB/layer of packed int4 expert buffers across all
    visible CUDA devices by currently-free memory, reserving `reserve_mib[i]`
    (plus a flat DEFAULT_DEVICE_RESERVE_MIB on every device, for CUDA context
    + allocator overhead) MiB of headroom on device i for KV cache /
    activations / logits (the non-expert weights + all runtime activations
    live on device 0, which is where device_map={"": "cuda:0"} put
    everything else — see load_moe_model below — so device 0 needs a much
    bigger reserve than any other device that's only holding static expert
    buffers). Fills each device to its own budget (largest-free-first) rather
    than round-robin, so a device never gets asked to hold "half a layer"
    worth more than its true remaining budget."""
    reserve_mib = reserve_mib or {}
    n_gpus = torch.cuda.device_count()
    if n_gpus == 0:
        return {i: torch.device("cpu") for i in range(num_layers)}

    free_bytes = []
    for i in range(n_gpus):
        free, _total = torch.cuda.mem_get_info(i)
        reserve = (reserve_mib.get(i, 0) + DEFAULT_DEVICE_RESERVE_MIB) * 1024 * 1024
        free_bytes.append(max(0, free - reserve))

    plan = {}
    for layer_idx in range(num_layers):
        eligible = [i for i in range(n_gpus) if free_bytes[i] >= BYTES_PER_LAYER_EXPERTS]
        dev_idx = max(eligible, key=lambda i: free_bytes[i]) if eligible else max(range(n_gpus), key=lambda i: free_bytes[i])
        plan[layer_idx] = torch.device(f"cuda:{dev_idx}")
        free_bytes[dev_idx] -= BYTES_PER_LAYER_EXPERTS

    if any(free_bytes[i] < 0 for i in range(n_gpus)):
        # All devices exhausted their true budget and we still had to place
        # more layers — report clearly rather than silently risking an OOM.
        deficit = -min(free_bytes)
        print(f"[plan_layer_devices] WARNING: ran out of reserved headroom by "
              f"~{deficit/1e6:.0f}MB while placing {num_layers} layers; "
              f"proceeding, but attach_int4_experts may hit a real CUDA OOM.")
    return plan


def attach_int4_experts(model, model_dir, layer_device_plan: dict[int, torch.device] | None = None,
                         verbose: bool = True):
    """Replace every layer's placeholder .experts with a populated
    GemmaExpertsInt4. `layer_device_plan` (layer_idx -> torch.device)
    controls where each layer's ~430MB of packed expert buffers live; if not
    given, falls back to the layer's router.proj.weight device (fine for
    single-GPU or when device_map already spread layers across devices with
    room to spare for their experts too)."""
    reader = CheckpointReader(model_dir)
    placeholders = find_placeholder_experts(model)
    cfg = model.config.get_text_config()

    total_bytes = 0
    per_device_bytes: dict[str, int] = {}
    t_start = time.perf_counter()
    for layer_idx in sorted(placeholders.keys()):
        full_name, placeholder = placeholders[layer_idx]
        parent_path = full_name.rsplit(".", 1)[0]
        parent = model.get_submodule(parent_path)

        if layer_device_plan is not None:
            device = layer_device_plan[layer_idx]
        else:
            router = model.get_submodule(parent_path + ".router")
            device = router.proj.weight.device

        new_experts = GemmaExpertsInt4(cfg)
        new_experts.load_expert_weights(reader, layer_idx, device)
        parent.experts = new_experts

        layer_bytes = 0
        for t in (new_experts.gate_packed, new_experts.gate_scale, new_experts.up_packed,
                  new_experts.up_scale, new_experts.down_packed, new_experts.down_scale):
            layer_bytes += t.numel() * t.element_size()
        total_bytes += layer_bytes
        per_device_bytes[str(device)] = per_device_bytes.get(str(device), 0) + layer_bytes

        if verbose:
            print(f"[attach_int4_experts] layer {layer_idx:>2} -> device={device} "
                  f"({time.perf_counter()-t_start:.1f}s elapsed)")

    reader.close()
    if verbose:
        per_dev_str = ", ".join(f"{d}: {b/1e9:.2f}GB" for d, b in per_device_bytes.items())
        print(f"[attach_int4_experts] done: {len(placeholders)} layers, "
              f"{total_bytes/1e9:.2f} GB of packed int4 expert weights resident "
              f"({per_dev_str}), {time.perf_counter()-t_start:.1f}s total")
    return total_bytes


# --------------------------------------------------------------------------- #
# Top-level entry point used by moe_router_probe_gemma.py / moe_expert_lock_gemma.py.
# --------------------------------------------------------------------------- #

def load_moe_model(model_dir, primary_device: str = "cuda:0", primary_reserve_mib: int = 2500, verbose: bool = True):
    """Full fixed loading path:
      1. patch Gemma4TextExperts -> placeholder (no wasted alloc, no silent
         random-init of expert weights)
      2. from_pretrained onto a single device (device_map="auto"'s balanced
         multi-GPU dispatch hits an unrelated accelerate bug for this
         checkpoint — `check_device_map` rejects the inferred map because
         two of Gemma4VisionModel's top-level buffers, std_bias/std_scale,
         end up with no assigned device once the map spans >1 GPU; loading
         onto a single device sidesteps it entirely, and the non-expert
         model is small enough (~4.6GB measured) that this is not a memory
         problem)
      3. attach_int4_experts with a balanced layer->device plan across all
         visible GPUs, reserving `primary_reserve_mib` on `primary_device`
         for KV cache / activations / lm_head logits (everything non-expert,
         plus all runtime activations, lives on primary_device)

    Returns (model, tokenizer, device) — device is primary_device, matching
    the (model, tokenizer, device) contract the rest of the probe/lock
    scripts (input_ids.to(device), etc.) already assume.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    patch_gemma4_experts_for_loading()

    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    t0 = time.perf_counter()
    model = AutoModelForCausalLM.from_pretrained(
        model_dir, local_files_only=True, trust_remote_code=True,
        dtype="auto", device_map={"": primary_device},
    )
    if verbose:
        print(f"[load_moe_model] base (non-expert) weights loaded in {time.perf_counter()-t0:.1f}s "
              f"onto {primary_device}")

    cfg = model.config.get_text_config()
    num_layers = cfg.num_hidden_layers
    primary_idx = int(primary_device.split(":")[-1]) if ":" in primary_device else 0
    reserve = {primary_idx: primary_reserve_mib}
    layer_plan = plan_layer_devices(num_layers, reserve_mib=reserve)

    attach_int4_experts(model, model_dir, layer_device_plan=layer_plan, verbose=verbose)

    model.eval()
    device = torch.device(primary_device)
    return model, tokenizer, device
