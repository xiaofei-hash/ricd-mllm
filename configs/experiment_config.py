"""Portable path configuration for RiCD experiment runners."""

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODEL_PATHS = {
    "llava-1.5-7b": os.environ.get(
        "LLAVA_MODEL_PATH", "liuhaotian/llava-v1.5-7b"
    ),
    "instructblip": os.environ.get(
        "INSTRUCTBLIP_MODEL_PATH", "Salesforce/instructblip-vicuna-7b"
    ),
    "qwen2.5-vl": os.environ.get(
        "QWEN_VL_MODEL_PATH", "Qwen/Qwen2.5-VL-7B-Instruct"
    ),
}

OUTPUT_ROOT = os.environ.get("RICD_OUTPUT_ROOT", str(PROJECT_ROOT / "outputs"))
