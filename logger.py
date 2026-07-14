import os
import sys
import json
import platform
import subprocess
from datetime import datetime, timezone

from config import qwen_models, hf_token

logs_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
os.makedirs(logs_dir, exist_ok=True)

def get_hardware_info():
    info = {
        "platform": platform.system(),
        "platform_version": platform.version(),
        "architecture": platform.architecture(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
        "gpus": []
    }

    try: 
        import torch
        info["torch_version"] = torch.__version__
        info["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            info["cuda_version"] = torch.version.cuda
            info["gpu_count"] = torch.cuda.device_count()
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                info["gpus"].append({
                    "name": props.name,
                    "total_memory": props.total_memory,
                    "multi_processor_count": props.multi_processor_count,
                })
    except ImportError:
        info["torch_version"] = None
        info["cuda_available"] = False

    try: 
        smi = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"], capture_output=True, text=True)
        if smi.returncode == 0:
            info["nvidia_smi"] = smi.stdout.strip()
    except Exception as e:
        info["nvidia_smi_error"] = "nvidia-smi not found or failed to execute: " + str(e)
    
    return info

def get_package_versions():
    packages = ["torch", "transformers", "accelerate", "huggingface_hub",
                "tokenizers", "safetensors", "bitsandbytes", "vllm"]
    versions = {}
    for package in packages:
        try:
            module = __import__(package)
            versions[package] = module.__version__
        except ImportError:
            versions[package] = None
    return versions

def audit_models(model_id,  hf_token):

    result = {
        "model_id": model_id,
        "status": "pending"
    }

    try:
        from huggingface_hub import HfApi, model_info
        from transformers import AutoConfig


        api = HfApi(token=hf_token)

        info = model_info(model_id, token=hf_token)
        result["sha"] = info.sha
        result["gated"] = getattr(info, "gated", None)
        result["last_modified"] = str(getattr(info, "lastModified", None))
        result["tags"] = getattr(info, "tags", [])
 
        # Config-level details: architecture, attention, context length
        config = AutoConfig.from_pretrained(model_id, token=hf_token)
        cfg_dict = config.to_dict()

        # Composite multimodal models (e.g. Gemma-4) nest decoder params under text_config
        text_cfg = cfg_dict.get("text_config", cfg_dict)

        result["architectures"] = cfg_dict.get("architectures")
        result["model_type"] = cfg_dict.get("model_type")
        result["vocab_size"] = text_cfg.get("vocab_size")
        result["max_position_embeddings"] = text_cfg.get("max_position_embeddings")
        result["hidden_size"] = text_cfg.get("hidden_size")
        result["num_hidden_layers"] = text_cfg.get("num_hidden_layers")
        result["num_attention_heads"] = text_cfg.get("num_attention_heads")
        result["num_key_value_heads"] = text_cfg.get("num_key_value_heads")
        result["sliding_window"] = text_cfg.get("sliding_window", None)
        matched_keys = [k for k in ["num_experts", "num_local_experts", "linear_attn", "delta_net"] if k in text_cfg]
        result["moe_or_hybrid_matched_keys"] = matched_keys
        result["has_vision_config"] = "vision_config" in cfg_dict
        result["has_audio_config"] = "audio_config" in cfg_dict
        result["num_experts_value"] = text_cfg.get("num_experts")
        result["num_experts_per_tok_value"] = text_cfg.get("num_experts_per_tok")  # if present, also log this
        result["is_moe_active"] = bool(text_cfg.get("num_experts")) and text_cfg.get("num_experts", 0) > 1
        
        result["status"] = "ok"
 
    except Exception as e:
        result["status"] = "FAILED"
        result["error"] = str(e)
 
    return result

def check_torch_determinism_settings():
    try:
        import torch
        return {
            "cudnn_deterministic": torch.backends.cudnn.deterministic,
            "cudnn_benchmark": torch.backends.cudnn.benchmark,
            "note": (
                "For Phase 3 determinism tests, set "
                "torch.backends.cudnn.deterministic=True and "
                "torch.backends.cudnn.benchmark=False, and use do_sample=False."
            ),
        }
    except ImportError:
        return {"error": "torch not installed"}
 
 
def main():
    log = {
        "run_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "hardware": get_hardware_info(),
        "package_versions": get_package_versions(),
        "determinism_settings": check_torch_determinism_settings(),
        "models": {},
    }

    print("Environment and Model Config Audit")
    for k, v in log["package_versions"].items():
        print(f"    {k}: {v}")
    
    print("Model Config Audit")
    for model_id in qwen_models:
        log["models"][model_id] = audit_models(model_id, hf_token)


    log_filename = os.path.join(logs_dir, f"model_audit_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    with open(log_filename, "w") as f:
        json.dump(log, f, indent=4)


if __name__ == "__main__":
    main()