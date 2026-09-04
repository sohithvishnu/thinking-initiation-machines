"""
Conversion utilities: cyankiwi/gemma-4-26B-A4B-it-AWQ-4bit's per-expert
compressed-tensors pack-quantized int4 checkpoint format -> the tensors
needed to run Gemma4's *actual* MoE compute (Gemma4TextExperts expects one
batched [num_experts, ...] Parameter per projection, not one small tensor
per expert per projection).

Background (see tim-moe/docs/moe_gemma_comparison.md for the full writeup):
transformers 5.13.1's Gemma4 model class does not register a checkpoint
conversion mapping for its MoE experts (unlike e.g. Qwen3-MoE, which has a
`"qwen2_moe"` entry in transformers/conversion_mapping.py merging
`mlp.experts.*.{gate,up}_proj.weight` -> `mlp.experts.gate_up_proj` via
MergeModulelist+Concatenate, and `CompressedTensorsHfQuantizer` extends that
merge with an int4-decompress op automatically). Gemma4 ships no equivalent
entry, so `AutoModelForCausalLM.from_pretrained(...)` on this checkpoint
reports every `experts.{j}.{gate,up,down}_proj.weight_packed/scale/shape`
tensor as UNEXPECTED and the model's real `experts.gate_up_proj` /
`experts.down_proj` Parameters as MISSING (silently randomly initialized).
Confirmed directly by loading onto `device_map={"": "meta"}` with
`output_loading_info=True` — see the LOAD REPORT this prints; 11520
UNEXPECTED expert tensors, 60 MISSING `experts.{gate_up_proj,down_proj}`
Parameters (30 layers x 2), zero mismatches anywhere else (attention,
embeddings, lm_head, router, and the per-layer dense `mlp.*` all load
correctly — they are ordinary `nn.Linear`-parented tensors the generic
compressed-tensors module-wrapping path handles directly, unrelated to the
checkpoint-conversion-mapping machinery that experts need and don't have).

This module provides the from-scratch fix: read the per-expert
(weight_packed, weight_scale, weight_shape) triples directly off the
checkpoint's safetensors shards, dequantize them with compressed_tensors'
own primitives, and assemble them in the exact shape/order
transformers.models.gemma4.modeling_gemma4.Gemma4TextExperts expects.

Shape/order derivation (read directly from installed transformers source,
transformers/models/gemma4/modeling_gemma4.py, Gemma4TextExperts.forward,
and confirmed identical in the ALL_EXPERTS_FUNCTIONS
`batched_mm_experts_forward`/`grouped_mm_experts_forward` reference
implementations in transformers/integrations/moe.py):

    gate, up = nn.functional.linear(current_state, self.gate_up_proj[e]).chunk(2, dim=-1)
    ...
    current_hidden_states = nn.functional.linear(current_hidden_states, self.down_proj[e])

`nn.functional.linear(x, W)` computes `x @ W.T`; chunking the *output* into
two halves along the last dim means the *rows* of `gate_up_proj[e]` are
`[gate_out_features; up_out_features]` stacked along dim 0. So:

    gate_up_proj[e] = torch.cat([gate_proj_weight_e, up_proj_weight_e], dim=0)   # [2*I, H]
    down_proj[e]    = down_proj_weight_e                                        # [H, I]

where `gate_proj_weight_e`/`up_proj_weight_e` are the checkpoint's per-expert
`gate_proj`/`up_proj` weights in standard nn.Linear (out_features,
in_features) = (intermediate_dim, hidden_dim) layout, and `down_proj_weight_e`
is (hidden_dim, intermediate_dim). The router's `top_k_index` indexes
directly into 0..num_experts-1 with no offset (Gemma4TextRouter.proj is a
plain `nn.Linear(hidden_size, num_experts)` over the full expert set, and
Gemma4TextExperts.forward uses `top_k_index` directly as the first dim of
`gate_up_proj`/`down_proj` — no separate index-space remapping).

Dequantization: compressed_tensors 0.17.0's
`compressors/pack_quantized/helpers.py:unpack_from_int32` unpacks
`weight_packed` (int32, `packed_dim=1`, `num_bits=4`, matching this
checkpoint's `quantization_config.config_groups.group_0` — pack-quantized,
group_size 32, symmetric int4) back to int8, then
`quantization/lifecycle/forward.py:dequantize` applies `weight_scale`
(letting it auto-infer the GROUP strategy from `scale.ndim==2`, exactly as
compressed_tensors' own `dequantize(args=None, ...)` path does when no
explicit QuantizationArgs are passed). Verified byte-for-byte against
compressed_tensors' own `PackedQuantizationCompressor.decompress` (the
documented top-level decompress API) for 5 experts across 4 different
layers (0, 5, 15, 29) and all three projections (gate/up/down) — see
`verify()` below and its printed output. `torch.equal` is True in all cases
(max abs diff 0.0), which is expected: dequantization here is a deterministic
bit-unpack + affine rescale, not a lossy/approximate operation.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import torch
from safetensors import safe_open

from compressed_tensors.compressors.pack_quantized.helpers import unpack_from_int32
from compressed_tensors.quantization.lifecycle.forward import dequantize

NUM_BITS = 4  # group_size=32, symmetric int4 — from config.json's quantization_config


class CheckpointReader:
    """Lazy, shard-aware reader for one local model directory's safetensors
    files, keyed by the tensor names in model.safetensors.index.json. Keeps
    shard file handles open (safe_open is a context manager under the hood
    but also usable as a persistent handle) so repeated per-expert reads
    don't reopen the same multi-GB shard file each time."""

    def __init__(self, model_dir: str | Path):
        self.model_dir = Path(model_dir)
        index_path = self.model_dir / "model.safetensors.index.json"
        with open(index_path) as f:
            self.weight_map: dict[str, str] = json.load(f)["weight_map"]
        self._handles: dict[str, "safe_open"] = {}

    def _handle_for(self, shard_name: str):
        h = self._handles.get(shard_name)
        if h is None:
            h = safe_open(str(self.model_dir / shard_name), framework="pt")
            self._handles[shard_name] = h
        return h

    def get_tensor(self, name: str) -> torch.Tensor:
        shard = self.weight_map[name]
        return self._handle_for(shard).get_tensor(name)

    def has(self, name: str) -> bool:
        return name in self.weight_map

    def close(self):
        # safe_open handles are released on GC / process exit; nothing to
        # explicitly close via its Python API.
        self._handles.clear()


def dequantize_packed(packed: torch.Tensor, scale: torch.Tensor, shape: torch.Tensor | tuple) -> torch.Tensor:
    """packed: int32 [rows, cols/pack_factor]; scale: fp16 [rows, n_groups];
    shape: the original (rows, cols) shape (tensor or tuple). Returns the
    dequantized weight in scale's dtype (fp16 in this checkpoint)."""
    if isinstance(shape, torch.Tensor):
        shape = tuple(int(x) for x in shape.tolist())
    unpacked = unpack_from_int32(packed, NUM_BITS, shape, packed_dim=1)
    return dequantize(x_q=unpacked, scale=scale, zero_point=None, args=None)


def read_expert_projection(reader: CheckpointReader, layer_idx: int, expert_idx: int, proj: str):
    """proj in {'gate_proj', 'up_proj', 'down_proj'}. Returns (packed, scale, shape) raw tensors, undequantized."""
    base = f"model.language_model.layers.{layer_idx}.experts.{expert_idx}.{proj}"
    packed = reader.get_tensor(f"{base}.weight_packed")
    scale = reader.get_tensor(f"{base}.weight_scale")
    shape = reader.get_tensor(f"{base}.weight_shape")
    return packed, scale, shape


def dequantize_expert_projection(reader: CheckpointReader, layer_idx: int, expert_idx: int, proj: str) -> torch.Tensor:
    packed, scale, shape = read_expert_projection(reader, layer_idx, expert_idx, proj)
    return dequantize_packed(packed, scale, shape)


def build_batched_expert_tensors(reader: CheckpointReader, layer_idx: int, num_experts: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Fully-dequantized fallback path (approach (a) from the task brief):
    returns (gate_up_proj, down_proj) as plain fp16 batched tensors in the
    exact shape Gemma4TextExperts expects: gate_up_proj [E, 2I, H],
    down_proj [E, H, I]. NOT used by the main run path (see
    GemmaExpertsInt4 in moe_router_probe_gemma.py / moe_expert_lock_gemma.py
    for the memory-efficient GPU-resident-packed path actually used) but
    kept here as the ground-truth assembly function the verify() self-check
    below and any ad-hoc debugging can lean on.
    """
    gate_up_rows = []
    down_rows = []
    for e in range(num_experts):
        gate_w = dequantize_expert_projection(reader, layer_idx, e, "gate_proj")
        up_w = dequantize_expert_projection(reader, layer_idx, e, "up_proj")
        down_w = dequantize_expert_projection(reader, layer_idx, e, "down_proj")
        gate_up_rows.append(torch.cat([gate_w, up_w], dim=0))
        down_rows.append(down_w)
    gate_up_proj = torch.stack(gate_up_rows, dim=0)
    down_proj = torch.stack(down_rows, dim=0)
    return gate_up_proj, down_proj


def read_packed_stack(reader: CheckpointReader, layer_idx: int, num_experts: int, proj: str):
    """Returns (packed_stack, scale_stack, shape) for one projection across
    all experts of a layer, stacked along a new leading expert dim, WITHOUT
    dequantizing — this is the raw form the GPU-resident int4 experts module
    keeps permanently resident (packed int4 + fp16 group scales), matching
    the checkpoint's on-disk memory footprint almost exactly."""
    packed_list, scale_list, shape_ref = [], [], None
    for e in range(num_experts):
        packed, scale, shape = read_expert_projection(reader, layer_idx, e, proj)
        packed_list.append(packed)
        scale_list.append(scale)
        if shape_ref is None:
            shape_ref = tuple(int(x) for x in shape.tolist())
    return torch.stack(packed_list, dim=0), torch.stack(scale_list, dim=0), shape_ref


# --------------------------------------------------------------------------- #
# Verification: dequantize_packed() vs compressed_tensors' own top-level
# PackedQuantizationCompressor.decompress(), for several experts across
# several layers and all three projection kinds. Prints max abs diff for
# each — must show 0.0 (exact) before this conversion is trusted at scale.
# --------------------------------------------------------------------------- #

def verify(model_dir: str | Path, cases: list[tuple[int, int, str]] | None = None) -> bool:
    from compressed_tensors.compressors.pack_quantized.base import PackedQuantizationCompressor
    from compressed_tensors.quantization import (
        QuantizationArgs, QuantizationScheme, QuantizationStrategy, QuantizationType,
    )

    reader = CheckpointReader(model_dir)
    scheme = QuantizationScheme(
        targets=["Linear"],
        weights=QuantizationArgs(
            num_bits=NUM_BITS, group_size=32, strategy=QuantizationStrategy.GROUP,
            symmetric=True, type=QuantizationType.INT,
        ),
    )

    if cases is None:
        cases = [
            (0, 0, "gate_proj"), (0, 5, "up_proj"), (5, 50, "down_proj"),
            (15, 100, "gate_proj"), (29, 127, "down_proj"),
        ]

    print(f"=== gemma_expert_convert.verify() — {len(cases)} cases ===")
    all_ok = True
    for layer_idx, expert_idx, proj in cases:
        packed, scale, shape = read_expert_projection(reader, layer_idx, expert_idx, proj)
        mine = dequantize_packed(packed, scale, shape)

        sd = {"weight_packed": packed.clone(), "weight_scale": scale.clone(), "weight_shape": shape.clone()}
        official = PackedQuantizationCompressor.decompress(sd, scheme)["weight"]

        max_diff = (mine.float() - official.float()).abs().max().item()
        exact = torch.equal(mine, official)
        all_ok = all_ok and exact
        print(
            f"layer={layer_idx:>2} expert={expert_idx:>3} {proj:<10} "
            f"shape={tuple(int(x) for x in shape.tolist())!s:<14} "
            f"max_abs_diff={max_diff:.6g} exact_match={exact}"
        )

    print(f"=== verify() {'PASSED' if all_ok else 'FAILED'} ===")
    return all_ok


if __name__ == "__main__":
    import sys
    model_dir = sys.argv[1] if len(sys.argv) > 1 else (
        Path(__file__).resolve().parent.parent / "models" / "cyankiwi_gemma-4-26B-A4B-it-AWQ-4bit"
    )
    ok = verify(model_dir)
    sys.exit(0 if ok else 1)
