"""Model factory for instantiating the appropriate model adapter."""

from .llava_adapter import LLaVAAdapter
from .instructblip_adapter import InstructBLIPAdapter
from .qwen25vl_adapter import Qwen25VLAdapter


MODEL_REGISTRY = {
    "llava-1.5-7b": {
        "class": LLaVAAdapter,
        "default_path": "liuhaotian/llava-v1.5-7b",        # 官方仓库格式
    },
    "instructblip": {
        "class": InstructBLIPAdapter,
        "default_path": "Salesforce/instructblip-vicuna-7b", # HF 格式
    },
    "qwen2.5-vl": {
        "class": Qwen25VLAdapter,
        "default_path": "Qwen/Qwen2.5-VL-7B-Instruct",     # HF 格式
    },
}


def create_model(model_name: str, model_path: str = None, device: str = "cuda",
                 **adapter_kwargs):
    """
    Create a model adapter.

    Args:
        model_name: One of 'llava-1.5-7b', 'instructblip', 'qwen2.5-vl'
        model_path: Override default model path (local path or HF hub id)
        device: Device to load model on
        **adapter_kwargs: Model-specific adapter options

    Returns:
        Model adapter instance (call .load_model() to actually load weights)
    """
    if model_name not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model: {model_name}. "
            f"Available: {list(MODEL_REGISTRY.keys())}"
        )

    info = MODEL_REGISTRY[model_name]
    path = model_path or info["default_path"]
    adapter = info["class"](
        model_path=path, device=device, **adapter_kwargs
    )
    return adapter
