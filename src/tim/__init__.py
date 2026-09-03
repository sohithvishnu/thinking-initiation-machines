"""
TIM — Thinking Initiation Machines.

Inference-time latent conditioning: prefill a language model's KV cache with
sampled vocabulary noise *before* the query arrives, instead of putting the
same content in the prompt.

Layout:
    tim.config      model list and repository paths
    tim.vocab       the injected content (domain pools, persona text)
    tim.primer      TIMPrimer — noise sampling and KV-cache priming
    tim.models      local model mirroring and (optionally quantized) loading
    tim.generation  prompt construction, instrumented generation, CUDA timing
    tim.evaluation  generate -> sanitize -> score -> record, per condition
    tim.stats       Wilson CIs, exact McNemar, eval-artifact readers
    tim.audit       environment / hardware / model-config snapshot

Re-exports are resolved lazily so that `tim.stats` and `tim.config` stay
importable without torch — the analysis scripts under `scripts/` read
committed JSON and never touch a GPU.
"""

__version__ = "0.1.0"

__all__ = ["TIMPrimer", "DOMAIN_NAME", "DOMAIN_PERSONA_TEXT", "PYTHON_DOMAIN_WORDS"]

_LAZY = {
    "TIMPrimer": ("tim.primer", "TIMPrimer"),
    "DOMAIN_NAME": ("tim.vocab", "DOMAIN_NAME"),
    "DOMAIN_PERSONA_TEXT": ("tim.vocab", "DOMAIN_PERSONA_TEXT"),
    "PYTHON_DOMAIN_WORDS": ("tim.vocab", "PYTHON_DOMAIN_WORDS"),
}


def __getattr__(name):
    if name in _LAZY:
        import importlib
        module_name, attr = _LAZY[name]
        return getattr(importlib.import_module(module_name), attr)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(__all__)
