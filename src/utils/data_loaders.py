"""
Data loaders for POPE, MME, CHAIR benchmarks.

Expected data structure:
  data/pope/coco_pope_{split}.json  (split: random, popular, adversarial)
  data/mme/  (MME benchmark images + questions)
  data/chair/  (COCO images for captioning)
"""

import os
import json
from typing import List, Dict, Tuple
from PIL import Image


# ============================================================
# POPE Loader
# ============================================================

def load_pope_data(data_dir: str, split: str, image_dir: str) -> List[Dict]:
    """
    Load POPE benchmark data.

    Args:
        data_dir: Path to POPE data directory
        split: One of 'random', 'popular', 'adversarial'
        image_dir: Path to COCO images (val2014)

    Returns:
        List of dicts with keys: question_id, image, question, label
    """
    # Try multiple common filename patterns
    candidates = [
        os.path.join(data_dir, f"coco_pope_{split}.json"),
        os.path.join(data_dir, f"pope_{split}.json"),
        os.path.join(data_dir, f"{split}.json"),
        os.path.join(data_dir, f"coco_pope_{split}.jsonl"),
    ]

    filepath = None
    for c in candidates:
        if os.path.exists(c):
            filepath = c
            break

    if filepath is None:
        raise FileNotFoundError(
            f"POPE {split} data not found. Tried: {candidates}\n"
            f"Please download POPE data and place in {data_dir}/"
        )

    samples = []
    with open(filepath, "r") as f:
        if filepath.endswith(".jsonl"):
            lines = f.readlines()
            raw_data = [json.loads(line) for line in lines]
        else:
            raw_data = json.load(f)

    for item in raw_data:
        image_file = item.get("image", "")
        question = item.get("text", item.get("question", ""))
        label = item.get("label", item.get("answer", ""))

        # Normalize label
        if isinstance(label, str):
            label = label.strip().lower()

        image_path = os.path.join(image_dir, image_file)

        samples.append({
            "question_id": item.get("question_id", len(samples)),
            "image_path": image_path,
            "image_file": image_file,
            "question": question,
            "label": label,
        })

    return samples


# ============================================================
# MME Loader
# ============================================================

def load_mme_data(data_dir: str) -> Dict[str, List[Dict]]:
    """
    Load MME benchmark data.

    Args:
        data_dir: Path to MME data directory

    Returns:
        Dict mapping subtask name -> list of samples
    """
    subtasks = {}

    # MME has subdirectories per subtask
    for subtask_name in sorted(os.listdir(data_dir)):
        subtask_dir = os.path.join(data_dir, subtask_name)
        if not os.path.isdir(subtask_dir):
            continue

        images_dir = os.path.join(subtask_dir, "images") if os.path.isdir(os.path.join(subtask_dir, "images")) else subtask_dir
        questions_file = os.path.join(subtask_dir, "questions.txt")

        # Also try reading from pairs directly
        if not os.path.exists(questions_file):
            # Try reading from individual text files paired with images
            samples = []
            for fname in sorted(os.listdir(subtask_dir)):
                if fname.endswith(".txt"):
                    txt_path = os.path.join(subtask_dir, fname)
                    img_name = fname.replace(".txt", ".jpg")
                    img_candidates = [
                        os.path.join(subtask_dir, img_name),
                        os.path.join(subtask_dir, fname.replace(".txt", ".png")),
                    ]
                    img_path = None
                    for ic in img_candidates:
                        if os.path.exists(ic):
                            img_path = ic
                            break
                    if img_path is None:
                        continue

                    with open(txt_path, "r") as f:
                        for line in f:
                            line = line.strip()
                            if not line:
                                continue
                            parts = line.split("\t")
                            if len(parts) >= 2:
                                question, answer = parts[0], parts[1]
                            else:
                                question = parts[0]
                                answer = ""
                            samples.append({
                                "image_path": img_path,
                                "image_id": os.path.splitext(os.path.basename(img_path))[0],
                                "question": question,
                                "label": answer.strip().lower(),
                                "subtask": subtask_name,
                            })
            if samples:
                subtasks[subtask_name] = samples
        else:
            samples = []
            with open(questions_file, "r") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    parts = line.split("\t")
                    img_file = parts[0]
                    question = parts[1] if len(parts) > 1 else ""
                    answer = parts[2] if len(parts) > 2 else ""

                    img_path = os.path.join(images_dir, img_file)
                    samples.append({
                        "image_path": img_path,
                        "image_id": os.path.splitext(os.path.basename(img_path))[0],
                        "question": question,
                        "label": answer.strip().lower(),
                        "subtask": subtask_name,
                    })
            subtasks[subtask_name] = samples

    return subtasks


# ============================================================
# CHAIR Loader
# ============================================================

def load_chair_data(image_dir: str, annotation_file: str = None,
                    max_samples: int = 500,
                    image_ids: List[int] = None) -> List[Dict]:
    """
    Load COCO images for caption generation (CHAIR evaluation).

    Args:
        image_dir: Path to COCO val2014 images
        annotation_file: Optional path to COCO captions annotation JSON
        max_samples: Maximum number of samples
        image_ids: Optional ordered image-ID allowlist. When supplied, samples
            are returned in exactly this order and every ID must exist.

    Returns:
        List of dicts with image_path and image_id
    """
    samples = []

    if annotation_file and os.path.exists(annotation_file):
        with open(annotation_file, "r") as f:
            coco_data = json.load(f)

        image_by_id = {int(item["id"]): item for item in coco_data.get("images", [])}

        if image_ids is not None:
            missing_ids = [int(image_id) for image_id in image_ids
                           if int(image_id) not in image_by_id]
            if missing_ids:
                raise ValueError(f"CHAIR IDs absent from COCO annotations: {missing_ids[:10]}")
            for image_id in image_ids[:max_samples if max_samples else None]:
                img_info = image_by_id[int(image_id)]
                img_path = os.path.join(image_dir, img_info["file_name"])
                if not os.path.exists(img_path):
                    raise FileNotFoundError(f"CHAIR image missing: {img_path}")
                samples.append({
                    "image_id": int(image_id),
                    "image_path": img_path,
                    "file_name": img_info["file_name"],
                })
            return samples

        # Get unique images in annotation-file order (legacy behavior).
        seen_ids = set()
        for img_info in coco_data.get("images", []):
            img_id = img_info["id"]
            if img_id in seen_ids:
                continue
            seen_ids.add(img_id)

            img_path = os.path.join(image_dir, img_info["file_name"])
            if os.path.exists(img_path):
                samples.append({
                    "image_id": img_id,
                    "image_path": img_path,
                    "file_name": img_info["file_name"],
                })
            if len(samples) >= max_samples:
                break
    else:
        # Fallback: just list images in directory
        for fname in sorted(os.listdir(image_dir)):
            if fname.endswith((".jpg", ".png", ".jpeg")):
                # Extract image_id from COCO filename: COCO_val2014_000000XXXXXX.jpg
                try:
                    img_id = int(fname.split("_")[-1].split(".")[0])
                except ValueError:
                    img_id = len(samples)

                samples.append({
                    "image_id": img_id,
                    "image_path": os.path.join(image_dir, fname),
                    "file_name": fname,
                })
                if len(samples) >= max_samples:
                    break

    return samples


def load_image_safe(path: str, allow_placeholder: bool = False) -> Image.Image:
    """Load an RGB image.

    Benchmark runs fail closed by default.  A black placeholder changes the
    model input and silently corrupts reported metrics, so it is available
    only for explicitly requested UI/debug use.
    """
    try:
        img = Image.open(path).convert("RGB")
        return img
    except Exception as e:
        if allow_placeholder:
            print(f"[WARN] Failed to load image {path}; using placeholder: {e}")
            return Image.new("RGB", (224, 224), (0, 0, 0))
        raise RuntimeError(f"Failed to load benchmark image {path}: {e}") from e
