"""
Model mirroring and loading.

Every experiment loads from a local snapshot under `models/` rather than the
HF cache, so a run is reproducible against a pinned set of files and works
offline. `load_model()` is the single loader for the whole repo; the `quant`
argument is what previously forced the quantization studies to keep their
own copy of it.
"""

import gc
import json
from datetime import datetime, timezone
from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoProcessor,
    AutoTokenizer,
    BitsAndBytesConfig,
)

from tim.config import LOGS_DIR, MODELS_DIR, all_models, hf_token


def model_path_for(model_id: str) -> Path:
    safe_name = model_id.replace("/", "_").replace("\\", "_").replace(" ", "_")
    return MODELS_DIR / safe_name


def check_model_exists(model_id: str) -> bool:
    """True if the model has already been mirrored into `models/`."""
    model_path = model_path_for(model_id)
    return model_path.exists() and any(model_path.rglob("*.safetensors"))


def download_model(model_id: str) -> dict[str, str]:
    model_path = model_path_for(model_id)

    if check_model_exists(model_id):
        return {"model_id": model_id, "status": "already_exists",
                "cache_path": str(model_path)}

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    try:
        cache_path = snapshot_download(
            repo_id=model_id,
            token=hf_token(),
            local_dir=str(model_path),
            local_dir_use_symlinks=False,
        )
        return {"model_id": model_id, "status": "ok", "cache_path": str(cache_path)}
    except Exception as exc:
        return {"model_id": model_id, "status": "failed", "error": str(exc)}


def verify_model_load(model_id: str) -> dict[str, str]:
    model_path = model_path_for(model_id)
    try:
        AutoConfig.from_pretrained(model_path, local_files_only=True, trust_remote_code=True)
        AutoTokenizer.from_pretrained(model_path, local_files_only=True, trust_remote_code=True)
        try:
            AutoProcessor.from_pretrained(model_path, local_files_only=True, trust_remote_code=True)
        except Exception:
            pass  # text-only checkpoints have no processor; not an error
        return {"model_id": model_id, "status": "loadable", "cache_path": str(model_path)}
    except Exception as exc:
        return {"model_id": model_id, "status": "unloadable",
                "cache_path": str(model_path), "error": str(exc)}


def get_models() -> None:
    """Mirror every configured model locally, then verify each one loads."""
    report = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "mode": "download",
        "results": [],
    }

    models = all_models()
    for model_id in models:
        result = download_model(model_id)
        report["results"].append(result)
        if result["status"] == "already_exists":
            print(f"{model_id}: SKIPPED (already mirrored locally)")
        else:
            print(f"{model_id}: {result['status']}")

    report["verification"] = []
    for model_id in models:
        verify_result = verify_model_load(model_id)
        report["verification"].append(verify_result)
        print(f"{model_id}: {verify_result['status']}")

    if all(r["status"] == "already_exists" for r in report["results"]):
        print("All models are mirrored locally")
        report["summary"] = {"message": "All models are mirrored locally",
                             "skipped": len(models)}

    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = LOGS_DIR / f"model_download_{stamp}.json"
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Wrote report to {output_path}")


def load_model(model_id: str, quant: str = "bf16", use_double_quant: bool = True,
               attn_implementation: str | None = None):
    """
    Load a mirrored model, downloading it first if `models/` has no copy.

    quant:
      "auto"  torch_dtype="auto" — the checkpoint's own dtype. What the
              HumanEval+ baseline and the main 3-seed experiment used.
      "bf16"  forced bfloat16 — the full-precision arm of the quantization
              study, so that arm is not at the mercy of checkpoint dtype.
      "nf4"   bitsandbytes 4-bit NF4 with bfloat16 compute.

    `attn_implementation="eager"` is required for `output_attentions=True`
    (sdpa and flash do not return per-head weights).

    Returns (model, tokenizer, device, quant_report).
    """
    model_dir = model_path_for(model_id)
    if not model_dir.exists() or not (model_dir / "config.json").exists():
        result = download_model(model_id)
        if result.get("status") == "failed":
            raise RuntimeError(f"Failed to download {model_id}: {result.get('error')}")
    if not (model_dir / "config.json").exists():
        raise FileNotFoundError(
            f"Missing model files in {model_dir}. Run `python -m tim.models` first."
        )

    tokenizer = AutoTokenizer.from_pretrained(
        model_dir, local_files_only=True, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    device_map = "sequential" if torch.cuda.is_available() else None
    kwargs = dict(local_files_only=True, trust_remote_code=True, device_map=device_map)
    if attn_implementation is not None:
        kwargs["attn_implementation"] = attn_implementation

    if quant == "auto":
        model = AutoModelForCausalLM.from_pretrained(model_dir, torch_dtype="auto", **kwargs)
        quant_report = {"quant": "auto", "bnb_config": None}
    elif quant == "bf16":
        model = AutoModelForCausalLM.from_pretrained(model_dir, torch_dtype=torch.bfloat16, **kwargs)
        quant_report = {"quant": "bf16", "bnb_config": None}
    elif quant == "nf4":
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=use_double_quant,
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_dir, quantization_config=bnb_config, **kwargs)
        quant_report = {
            "quant": "nf4",
            "bnb_config": {
                "load_in_4bit": True,
                "bnb_4bit_quant_type": "nf4",
                "bnb_4bit_compute_dtype": "bfloat16",
                "bnb_4bit_use_double_quant": use_double_quant,
            },
        }
    else:
        raise ValueError(f"unknown quant {quant!r} (expected auto, bf16 or nf4)")

    device = next(model.parameters()).device
    model.eval()

    quant_report["param_dtypes_loaded"] = sorted({str(p.dtype) for p in model.parameters()})
    quant_report["device_map"] = device_map
    quant_report["attn_implementation"] = attn_implementation
    print(f"[load] {model_id} quant={quant} "
          f"param_dtypes={quant_report['param_dtypes_loaded']} device={device}")
    return model, tokenizer, device, quant_report


def unload_model(model) -> None:
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    get_models()
