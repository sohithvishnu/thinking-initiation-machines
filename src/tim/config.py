"""
Repository-level configuration: which models are studied, and where model
mirrors and run artifacts live.

Paths resolve relative to the repository root so that every entrypoint —
whether run from the root, from `experiments/`, or from `scripts/` — writes
to the same `logs/` tree the committed artifacts already occupy. Set
`TIM_ROOT` to override (useful if the package is installed non-editable).
"""

import os
from pathlib import Path

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:  # analysis scripts only need the paths below, not a token
    pass

# The three Qwen3 instruct sizes the study covers, largest first (largest is
# the most likely to OOM, so failing fast on it saves a long run).
qwen_models = ["Qwen/Qwen3-8B", "Qwen/Qwen3-4B", "Qwen/Qwen3-1.7B"]

ROOT_DIR = Path(os.environ.get("TIM_ROOT", Path(__file__).resolve().parents[2]))
MODELS_DIR = ROOT_DIR / "models"
LOGS_DIR = ROOT_DIR / "logs"
DOCS_DIR = ROOT_DIR / "docs"
FIGURES_DIR = DOCS_DIR / "figures"


def all_models() -> list[str]:
    return list(qwen_models)


def hf_token() -> str | None:
    """HuggingFace token from the environment (`.env` is loaded above).

    Returned lazily rather than read at import time: the analysis scripts
    import this module and must not require a token to run offline.
    """
    return os.environ.get("hf_token") or os.environ.get("HF_TOKEN")
