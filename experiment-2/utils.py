import json
from datetime import datetime, timezone
from pathlib import Path

from huggingface_hub import snapshot_download
from transformers import AutoConfig, AutoProcessor, AutoTokenizer

from config import qwen_models
from dotenv import load_dotenv
import os

load_dotenv()
hf_token = os.environ["hf_token"]
root_dir = Path(__file__).resolve().parent
models_dir = root_dir / "models"
logs_dir = root_dir / "logs"
logs_dir.mkdir(exist_ok=True)
models_dir.mkdir(exist_ok=True)

def all_models() -> list[str]:
    return qwen_models 


def model_path_for(model_id: str) -> Path:
    safe_name = model_id.replace("/", "_").replace("\\", "_").replace(" ", "_")
    return models_dir / safe_name

def check_model_exists(model_id: str) -> bool:
    """Check if the model has already been mirrored into the local models folder."""
    model_path = model_path_for(model_id)
    return model_path.exists() and any(model_path.rglob("*.safetensors"))

def download_model(model_id: str) -> dict[str, str]:
    model_path = model_path_for(model_id)

    if check_model_exists(model_id):
        return {
            "model_id": model_id,
            "status": "already_exists",
            "cache_path": str(model_path),
        }

    try:
        cache_path = snapshot_download(
            repo_id=model_id,
            token=hf_token,
            local_dir=str(model_path),
            local_dir_use_symlinks=False,
        )
        return {
            "model_id": model_id,
            "status": "ok",
            "cache_path": str(cache_path),
        }
    except Exception as exc:
        return {
            "model_id": model_id,
            "status": "failed",
            "error": str(exc),
        }


def verify_model_load(model_id: str) -> dict[str, str]:
    model_path = model_path_for(model_id)

    try:
        AutoConfig.from_pretrained(model_path, local_files_only=True, trust_remote_code=True)
        AutoTokenizer.from_pretrained(model_path, local_files_only=True, trust_remote_code=True)

        try:
            AutoProcessor.from_pretrained(model_path, local_files_only=True, trust_remote_code=True)
        except Exception:
            pass

        return {
            "model_id": model_id,
            "status": "loadable",
            "cache_path": str(model_path),
        }
    except Exception as exc:
        return {
            "model_id": model_id,
            "status": "unloadable",
            "cache_path": str(model_path),
            "error": str(exc),
        }

def get_models() -> None:
    report = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "mode": "download",
        "results": [],
    }

    skipped_count = 0
    total_count = len(all_models())

    for model_id in all_models():
        result = download_model(model_id)
        report["results"].append(result)

        if result["status"] == "already_exists":
            print(f"{model_id}: ⏭ SKIPPED (already mirrored locally)")
            skipped_count += 1
        else:
            print(f"{model_id}: {result['status']}")

    report["verification"] = []
    for model_id in all_models():
        verify_result = verify_model_load(model_id)
        report["verification"].append(verify_result)
        print(f"{model_id}: {verify_result['status']}")

    total_skipped = sum(1 for r in report["results"] if r["status"] == "already_exists")
    if total_skipped == total_count:
        print("All models are mirrored locally")
        report["summary"] = {"message": "All models are mirrored locally", "skipped": total_count}

    output_path = logs_dir / f"model_download_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Wrote report to {output_path}")