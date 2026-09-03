"""
The content that gets injected — as KV cache (TIM arms) or as prompt text
(the `prompt_control` arm).

Two domain pools exist because the study used two, and the difference is a
documented methodological change rather than an oversight:

  DOMAIN_NAME          one short descriptive *sentence about* Python. Used by
                       the main 3-seed HumanEval+ experiment
                       (`experiments/conditions.py`, `logs/tim/`).
  PYTHON_DOMAIN_WORDS  211 entries of real Python *syntax* — keywords,
                       builtins, dunders, operators — each tokenized
                       independently so short syntax tokens are not diluted
                       by surrounding sentence context. Used by everything
                       from the pass sweep onward.

DOMAIN_PERSONA_TEXT is plain English delivered through the prompt. It is the
`prompt_control` arm: prompt tokens become KV entries during prefill, so
this is the ablation that decides whether KV injection buys anything that
prompting does not.
"""

import builtins
import keyword

DOMAIN_NAME = "python programming code function algorithm"

DOMAIN_PERSONA_TEXT = (
    "You are an expert Python programmer who writes clean, correct, "
    "well-tested functions."
)

# Built from the standard library so the pool is reproducible on any
# interpreter rather than hand-curated and subtly wrong.
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
