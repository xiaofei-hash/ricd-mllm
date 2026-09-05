#!/usr/bin/env python3
"""Run one resumable AMBER generative job without changing the official evaluator."""

import argparse
import fcntl
import hashlib
import json
import os
import random
import re
import sys
from pathlib import Path

for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    try:
        if int(os.environ.get(_name, "1")) <= 0:
            os.environ[_name] = "1"
    except ValueError:
        os.environ[_name] = "1"

import numpy as np
import torch
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from configs.experiment_config import MODEL_PATHS, OUTPUT_ROOT  # noqa: E402
from src.models.factory import MODEL_REGISTRY, create_model  # noqa: E402
from src.pipeline import BaselinePipeline, RCCRPipeline  # noqa: E402
from src.utils.data_loaders import load_image_safe  # noqa: E402

DEFAULT_CONFIG = ROOT / "configs" / "amber_recovery_quality_v1.json"


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_jsonl(path):
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(str(temporary), str(path))


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def save_rng(path, completed):
    state = {
        "completed": completed,
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(state, str(temporary))
    os.replace(str(temporary), str(path))


def restore_rng(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if torch.cuda.is_available() and state["cuda"] is not None:
        torch.cuda.set_rng_state_all(state["cuda"])


def resolve(root, path):
    value = Path(path)
    return value if value.is_absolute() else root / value


def resolve_model_path(model):
    configured = MODEL_PATHS[model]
    return configured if os.path.exists(configured) else MODEL_REGISTRY[model]["default_path"]


def validate_qwen_caption(text, item_id):
    """Fail closed before an invalid Qwen caption contaminates a long run."""
    value = (text or "").strip()
    if not value:
        raise RuntimeError("Qwen generated an empty caption at AMBER item %s" % item_id)
    invalid_markers = (
        "addCriterion", "<tool_call>", "</tool_call>",
        "<|im_start|>", "<|im_end|>", "<|vision_start|>",
    )
    marker = next((token for token in invalid_markers if token in value), None)
    if marker is not None:
        raise RuntimeError(
            "Qwen generated control/template token %r at AMBER item %s" %
            (marker, item_id)
        )
    words = re.findall(r"[A-Za-z0-9_]+", value.lower())
    if any(words[index:index + 4] == [words[index]] * 4
           for index in range(max(len(words) - 3, 0))):
        raise RuntimeError(
            "Qwen generated a repeated-token loop at AMBER item %s" % item_id
        )
    if len(re.findall(r"[\u4e00-\u9fff]", value)) >= 10:
        raise RuntimeError(
            "Qwen switched to a non-English template at AMBER item %s" % item_id
        )


def build_pipeline(adapter, method, config, model):
    generation = config["generation"]
    common = {key: generation[key] for key in ("do_sample", "temperature", "top_p", "top_k")}
    if method == "regular":
        return BaselinePipeline(adapter, **common)
    return RCCRPipeline(adapter, **config["models"][model]["ricd"], **common)


def build_adapter_kwargs(model, config):
    model_config = config["models"][model]
    kwargs = {}
    # LLaVA's RiCD branch is fixed to visual layer 4 internally.  The other
    # adapters expose the same choice explicitly.
    if model != "llava-1.5-7b" and model_config.get("shallow_layer_index") is not None:
        kwargs["shallow_layer_index"] = int(model_config["shallow_layer_index"])
    if model == "qwen2.5-vl":
        for key in ("max_pixels", "min_new_tokens"):
            if model_config.get(key) is not None:
                kwargs[key] = int(model_config[key])
        if model_config.get("response_prefix") is not None:
            kwargs["response_prefix"] = model_config["response_prefix"]
    return kwargs


def prepare_resume(job_dir, expected_ids, expected_version, overwrite):
    predictions_path = job_dir / "predictions.jsonl"
    state_path = job_dir / "rng_state.pt"
    responses_path = job_dir / "responses.json"
    if overwrite:
        for path in (predictions_path, state_path, responses_path, job_dir / "amber_metrics.json"):
            if path.exists():
                path.unlink()
    rows = read_jsonl(predictions_path)
    actual_ids = [int(row["item_id"]) for row in rows]
    if actual_ids != expected_ids[:len(actual_ids)]:
        raise RuntimeError("Existing predictions are not an exact prefix of the official AMBER query order")
    versions = {row.get("protocol_version") for row in rows}
    if rows and versions != {expected_version}:
        raise RuntimeError(
            "Existing predictions use protocol version(s) %s, expected %s; "
            "restart this job with --overwrite" %
            (sorted(str(version) for version in versions), expected_version)
        )
    state = None
    if 0 < len(rows) < len(expected_ids):
        if not state_path.exists():
            raise RuntimeError("Partial output has no RNG checkpoint; use --overwrite")
        state = torch.load(str(state_path), map_location="cpu")
        if int(state["completed"]) != len(rows):
            raise RuntimeError("RNG checkpoint does not match the partial output")
    return predictions_path, state_path, responses_path, rows, state


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=["llava-1.5-7b", "instructblip", "qwen2.5-vl"])
    parser.add_argument("--method", required=True, choices=["regular", "ricd"])
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--namespace", default="recovery_quality_v1")
    parser.add_argument("--max_samples", type=int, help="Smoke-test only; uses a separate output namespace")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--validate_only", action="store_true")
    parser.add_argument("--require_images", action="store_true")
    args = parser.parse_args()

    config_path = resolve(ROOT, args.config)
    config = read_json(config_path)
    if args.seed != int(config["seed"]):
        parser.error("This registered comparison protocol is single-seed (seed=1)")
    query_path = resolve(ROOT, config["amber"]["query_file"])
    image_dir = resolve(ROOT, os.environ.get("AMBER_IMAGE_DIR", config["amber"]["image_dir"]))
    queries = read_json(query_path)
    prompt_policy = config["amber"].get("prompt_policy", "configured_or_query")
    if prompt_policy == "official_query_exact":
        if "prompt" in config["amber"]:
            raise ValueError("official_query_exact forbids an AMBER prompt override")
        official_prompt = config["amber"].get("official_prompt", "Describe this image.")
        if any(row.get("query") != official_prompt for row in queries):
            raise ValueError("AMBER query file does not contain the registered official prompt")
    expected = int(config["amber"]["samples"])
    if len(queries) != expected or [int(row["id"]) for row in queries] != list(range(1, expected + 1)):
        raise ValueError("AMBER generative queries must contain the official ordered IDs 1..1004")
    missing = [row["image"] for row in queries if not (image_dir / row["image"]).is_file()]
    status = {
        "config": str(config_path), "version": config["version"], "model": args.model,
        "method": args.method, "seed": args.seed, "queries": len(queries),
        "max_new_tokens": int(config["generation"]["max_new_tokens"]),
        "max_pixels": config["models"][args.model].get("max_pixels"),
        "image_dir": str(image_dir), "images_present": len(queries) - len(missing),
        "images_missing": len(missing), "sample_missing": missing[:3],
        "inference_started": False,
    }
    print(json.dumps(status, indent=2, ensure_ascii=False))
    if args.validate_only:
        if args.require_images and missing:
            raise FileNotFoundError("AMBER images are incomplete")
        return 0
    if missing:
        raise FileNotFoundError("AMBER images are incomplete; run scripts/prepare_amber_environment.sh --download-images")

    if args.max_samples:
        queries = queries[:args.max_samples]
        namespace = "recovery_quality_smoke_v1"
    else:
        namespace = args.namespace
    job_dir = Path(OUTPUT_ROOT) / namespace / args.model / "amber" / args.method / ("seed%d" % args.seed)
    if args.max_samples:
        job_dir = job_dir / ("n%d" % args.max_samples)
    job_dir.mkdir(parents=True, exist_ok=True)
    # A second GPU queue may reach the same registered job.  Serialize at the
    # job directory so only one process can create or resume its JSONL output.
    job_lock = (job_dir / ".run.lock").open("a+")
    fcntl.flock(job_lock.fileno(), fcntl.LOCK_EX)
    expected_ids = [int(row["id"]) for row in queries]
    paths = prepare_resume(
        job_dir, expected_ids, config["version"], args.overwrite
    )
    predictions_path, state_path, responses_path, predictions, resume_state = paths

    if len(predictions) == len(queries):
        atomic_json(responses_path, [{"id": row["item_id"], "response": row["prediction"]} for row in predictions])
        print("Complete output already exists:", responses_path)
        return 0

    set_seed(args.seed)
    adapter_kwargs = build_adapter_kwargs(args.model, config)
    adapter = create_model(
        args.model, model_path=resolve_model_path(args.model),
        **adapter_kwargs
    )
    adapter.load_model()
    pipeline = build_pipeline(adapter, args.method, config, args.model)
    adapter_sources = {
        "llava-1.5-7b": "llava_adapter.py",
        "instructblip": "instructblip_adapter.py",
        "qwen2.5-vl": "qwen25vl_adapter.py",
    }
    source_files = [
        Path(__file__).resolve(), ROOT / "src" / "pipeline.py",
        ROOT / "src" / "decoder" / "rccr_decoder.py",
        ROOT / "src" / "proxy" / "conflict_proxies.py",
        ROOT / "src" / "risk" / "risk_surrogates.py",
        ROOT / "src" / "models" / adapter_sources[args.model],
    ]
    manifest = {
        "protocol_version": config["version"],
        "config": str(config_path),
        "query_sha256": sha256_file(query_path),
        "model": args.model, "method": args.method, "seed": args.seed,
        "prompt_policy": prompt_policy, "adapter_kwargs": adapter_kwargs,
        "source_sha256": {str(path.relative_to(ROOT)): sha256_file(path) for path in source_files},
    }
    atomic_json(job_dir / "manifest.json", manifest)
    if resume_state is not None:
        restore_rng(resume_state)

    mode = "a" if predictions else "w"
    with predictions_path.open(mode, encoding="utf-8") as handle:
        for index in tqdm(range(len(predictions), len(queries)), desc="amber/%s/%s" % (args.model, args.method)):
            sample = queries[index]
            image = load_image_safe(str(image_dir / sample["image"]))
            prompt = sample["query"] if prompt_policy == "official_query_exact" else config["amber"].get("prompt", sample["query"])
            result = pipeline.generate(
                image=image, prompt=prompt,
                max_new_tokens=config["generation"]["max_new_tokens"],
                sample_id="amber_%s_%s_seed%d" % (args.model, sample["id"], args.seed),
            )
            if args.model == "qwen2.5-vl":
                validate_qwen_caption(result["text"], sample["id"])
            row = {
                "item_id": int(sample["id"]), "image": sample["image"], "prompt": prompt,
                "prediction": result["text"], "caption": result["text"], "response": result["text"],
                "num_tokens": result.get("num_tokens"), "num_changed": result.get("num_changed"),
                "change_ratio": result.get("change_ratio"),
                "num_safety_fallbacks": result.get("num_safety_fallbacks", 0),
                "time_s": result.get("time_s"),
                "seed": args.seed, "method": args.method, "model": args.model,
                "protocol_version": config["version"],
            }
            predictions.append(row)
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            handle.flush()
            save_rng(state_path, len(predictions))
    atomic_json(responses_path, [{"id": row["item_id"], "response": row["prediction"]} for row in predictions])
    print("Saved official AMBER responses:", responses_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
