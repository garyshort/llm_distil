#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Gary Short
"""
Demo 5: Create training data for vision distillation — Cats vs Dogs.

Builds a self-contained dataset by:
1. Downloading the Oxford-IIIT Pet Dataset (CC BY-SA 4.0)
2. Parsing annotations to build balanced cat/dog image pools
3. Copying sampled images into a local dataset folder
4. Sending each image to an Azure OpenAI vision teacher for labeling
5. Writing training.jsonl with dataset-relative image paths

Oxford-IIIT labels are used ONLY for image organisation, balanced sampling,
and optional teacher-vs-dataset diagnostics. They are NEVER used as training
labels. The canonical training labels come exclusively from the teacher model.

Pipeline:
    image + prompt → teacher vision model → validated JSON → training.jsonl

Output layout (--dataset-dir, default ./dataset):
  dataset/
  ├── training.jsonl          # dataset-relative image paths for Qwen fine-tuning
  ├── training_metadata.jsonl # per-image audit trail; used for resume
  ├── images/
  │   ├── cat/cat_000001.jpg ...
  │   └── dog/dog_000001.jpg ...
  └── source/
      └── oxford-iiit-pet/    # downloaded once; kept for re-runs

Attribution: O. M. Parkhi, A. Vedaldi, A. Zisserman, C. V. Jawahar,
"Cats and Dogs", CVPR 2012. CC BY-SA 4.0.

Uses .env for GPT54_ENDPOINT_* (or OPENAI_* with --use-openai-endpoint).
"""

import argparse
import base64
import hashlib
import json
import logging
import random
import shutil
import tarfile
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv
from openai import APIStatusError, AzureOpenAI

try:
    import tqdm as _tqdm
except ImportError:
    _tqdm = None

# Oxford-IIIT Pet Dataset (CC BY-SA 4.0)
OXFORD_IMAGES_URL = "https://thor.robots.ox.ac.uk/~vgg/data/pets/images.tar.gz"
OXFORD_ANNOTATIONS_URL = (
    "https://thor.robots.ox.ac.uk/~vgg/data/pets/annotations.tar.gz"
)
OXFORD_SOURCE_NAME = "oxford-iiit-pet"
OXFORD_ANNOTATION_FILE = "annotations/list.txt"

# Supported image extensions
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

EXT_TO_MIME = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".webp": "image/webp",
    ".bmp": "image/bmp",
}

# Deterministic class order
CLASSES = ["cat", "dog"]
VALID_CLASSES = set(CLASSES)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


def load_config(use_openai_endpoint: bool = False) -> dict:
    """Load .env, validate required vision API vars, return config dict."""
    demos_dir = Path(__file__).resolve().parent.parent
    load_dotenv(demos_dir / ".env")

    import os

    if use_openai_endpoint:
        required = ["OPENAI_API_KEY", "OPENAI_BASE_URL"]
    else:
        required = [
            "GPT54_ENDPOINT_API_KEY",
            "GPT54_ENDPOINT_BASE_URL",
            "GPT54_ENDPOINT_API_VERSION",
        ]

    config: dict = {}
    missing = []
    for key in required:
        val = os.getenv(key)
        if not val:
            missing.append(key)
        config[key] = val

    if missing:
        raise SystemExit(
            f"Missing required env vars: {', '.join(missing)}. Add them to demos/.env"
        )

    if use_openai_endpoint:
        config["VISION_BASE_URL"] = config["OPENAI_BASE_URL"]
        config["VISION_API_KEY"] = config["OPENAI_API_KEY"]
        config["VISION_API_VERSION"] = os.getenv(
            "OPENAI_API_VERSION", "2024-12-01-preview"
        )
        config["VISION_MODEL"] = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    else:
        config["VISION_BASE_URL"] = config["GPT54_ENDPOINT_BASE_URL"]
        config["VISION_API_KEY"] = config["GPT54_ENDPOINT_API_KEY"]
        config["VISION_API_VERSION"] = config["GPT54_ENDPOINT_API_VERSION"]
        config["VISION_MODEL"] = os.getenv("GPT54_ENDPOINT_MODEL") or os.getenv(
            "OPENAI_MODEL", "gpt-4o-mini"
        )

    return config


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Build cats-vs-dogs distillation dataset using Oxford-IIIT Pet images "
            "and an Azure OpenAI vision teacher model."
        )
    )
    parser.add_argument(
        "--max-per-class",
        type=str,
        default="1000",
        help=(
            "Max images per class: integer (e.g. 1000) or class:N pairs "
            "(e.g. cat:500,dog:100)"
        ),
    )
    parser.add_argument(
        "--dataset-dir",
        default="dataset",
        help="Root directory for output (default: dataset/)",
    )
    parser.add_argument(
        "--prompt",
        default="cat_dog_prompt.txt",
        help="Prompt filename relative to script directory (default: cat_dog_prompt.txt)",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=3,
        help="Max retries per image on API or validation failure (default: 3)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Deterministic sampling seed (default: 42)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip already-completed images using metadata sidecar",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Verify dataset download/extraction, parse annotations, build image pools, "
            "load prompt, preview rows — no API calls"
        ),
    )
    parser.add_argument(
        "--use-openai-endpoint",
        action="store_true",
        help="Use OPENAI_BASE_URL/OPENAI_API_KEY/OPENAI_MODEL instead of GPT54_ENDPOINT_*",
    )
    return parser.parse_args()


def parse_max_per_class(value: str, classes: list[str]) -> dict[str, int]:
    """
    Parse --max-per-class into class -> max_count.
      "1000"           -> {cls: 1000 for each class}
      "cat:500,dog:100" -> {"cat": 500, "dog": 100}
    Unknown class names raise SystemExit.
    """
    value = value.strip()
    if not value:
        return {c: 1000 for c in classes}

    if ":" not in value:
        try:
            n = int(value)
            return {c: n for c in classes}
        except ValueError:
            raise SystemExit(
                f"Invalid --max-per-class '{value}': use an integer or class:N pairs"
            )

    cls_to_max: dict[str, int] = {}
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        if ":" not in part:
            raise SystemExit(
                f"Invalid --max-per-class segment '{part}': expected class:N"
            )
        cls, _, max_str = part.partition(":")
        cls = cls.strip()
        if cls not in VALID_CLASSES:
            raise SystemExit(
                f"Unknown class '{cls}' in --max-per-class. Valid classes: {sorted(VALID_CLASSES)}"
            )
        try:
            cls_to_max[cls] = int(max_str.strip())
        except ValueError:
            raise SystemExit(
                f"Invalid count in --max-per-class '{part}': expected integer after ':'"
            )

    return {c: cls_to_max.get(c, 1000) for c in classes}


# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------


def load_prompt(path: Path) -> str:
    """Read prompt file and return full text. Fails fast if missing or empty."""
    if not path.exists():
        raise SystemExit(f"Prompt file not found: {path}")
    text = path.read_text(encoding="utf-8")
    if not text.strip():
        raise SystemExit(f"Prompt file is empty: {path}")
    return text


# ---------------------------------------------------------------------------
# Oxford-IIIT Pet Dataset download and parsing
# ---------------------------------------------------------------------------


def _download_with_progress(url: str, dest: Path, label: str) -> None:
    """Download url to dest, printing a simple progress indicator."""
    log = logging.getLogger(__name__)
    log.info("Downloading %s from %s ...", label, url)

    def _hook(block_num: int, block_size: int, total_size: int) -> None:
        if total_size > 0 and block_num % 500 == 0:
            downloaded = block_num * block_size
            pct = min(100, downloaded * 100 // total_size)
            log.info("  %s: %d%%", label, pct)

    urllib.request.urlretrieve(url, dest, reporthook=_hook)
    log.info("Downloaded %s", label)


def _safe_extractall(tf: tarfile.TarFile, dest: Path) -> None:
    """
    Extract a tar archive using the 'data' filter when available (Python 3.12+).
    Falls back to unfiltered extraction on older Python without the filter kwarg.
    The 'data' filter blocks path-traversal entries and strips dangerous metadata.
    """
    try:
        tf.extractall(dest, filter="data")
    except TypeError:
        tf.extractall(dest)  # Python < 3.12


def ensure_oxford_dataset(source_dir: Path) -> None:
    """
    Download and extract the Oxford-IIIT Pet Dataset into source_dir if not
    already present. Idempotent: skips individual components that already exist.
    """
    log = logging.getLogger(__name__)
    source_dir.mkdir(parents=True, exist_ok=True)
    images_dir = source_dir / "images"
    annotations_dir = source_dir / "annotations"

    if not images_dir.exists():
        tar_path = source_dir / "images.tar.gz"
        _download_with_progress(
            OXFORD_IMAGES_URL, tar_path, "Oxford-IIIT images (~800MB)"
        )
        log.info("Extracting images...")
        with tarfile.open(tar_path) as tf:
            _safe_extractall(tf, source_dir)
        tar_path.unlink()
        log.info("Images extracted to %s", images_dir)
    else:
        log.info("Oxford images already present at %s", images_dir)

    if not annotations_dir.exists():
        tar_path = source_dir / "annotations.tar.gz"
        _download_with_progress(
            OXFORD_ANNOTATIONS_URL, tar_path, "Oxford-IIIT annotations"
        )
        log.info("Extracting annotations...")
        with tarfile.open(tar_path) as tf:
            _safe_extractall(tf, source_dir)
        tar_path.unlink()
        log.info("Annotations extracted to %s", annotations_dir)
    else:
        log.info("Oxford annotations already present at %s", annotations_dir)


def parse_oxford_annotations(source_dir: Path) -> dict[str, str]:
    """
    Parse annotations/list.txt and return {image_stem: "cat"|"dog"}.

    list.txt format (space-separated, lines starting with # are comments):
      Image  CLASS-ID  SPECIES  BREED-ID
    SPECIES: 1 = cat, 2 = dog

    Oxford labels are used ONLY for image organisation and balanced sampling.
    They are never used as training labels in training.jsonl.
    """
    list_path = source_dir / "annotations" / "list.txt"
    if not list_path.exists():
        raise SystemExit(f"Annotation file not found: {list_path}")

    result: dict[str, str] = {}
    for line in list_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 3:
            continue
        image_stem = parts[0]
        species = parts[2]
        if species == "1":
            result[image_stem] = "cat"
        elif species == "2":
            result[image_stem] = "dog"
    return result


def build_class_pools(
    annotations: dict[str, str], source_dir: Path
) -> dict[str, list[Path]]:
    """
    Build sorted cat and dog image path pools from Oxford annotations.
    Sorted by stem for deterministic ordering across runs.
    """
    pools: dict[str, list[Path]] = {"cat": [], "dog": []}
    images_dir = source_dir / "images"

    for stem in sorted(annotations.keys()):
        cls = annotations[stem]
        if cls not in VALID_CLASSES:
            continue
        for ext in (".jpg", ".jpeg", ".png"):
            img_path = images_dir / f"{stem}{ext}"
            if img_path.exists():
                pools[cls].append(img_path)
                break

    return pools


def sample_class(pool: list[Path], max_count: int, seed: int) -> list[Path]:
    """Select up to max_count images deterministically using seed."""
    rng = random.Random(seed)
    shuffled = pool.copy()
    rng.shuffle(shuffled)
    return shuffled[:max_count]


# ---------------------------------------------------------------------------
# Image handling
# ---------------------------------------------------------------------------


def make_local_image_path(cls: str, index: int, src_path: Path) -> str:
    """Return dataset-relative path: images/{cls}/{cls}_{index:06d}{ext}."""
    ext = src_path.suffix.lower()
    if ext not in IMAGE_EXTENSIONS:
        ext = ".jpg"
    return f"images/{cls}/{cls}_{index:06d}{ext}"


def save_image_to_dataset(src_path: Path, dataset_root: Path, local_rel: str) -> None:
    """Copy image from Oxford source to dataset folder. Creates parent dirs."""
    out_path = dataset_root / local_rel
    out_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src_path, out_path)


def load_image_bytes(path: Path) -> tuple[bytes, str]:
    """Read image file and return (bytes, mime_type)."""
    data = path.read_bytes()
    ext = path.suffix.lower()
    mime = EXT_TO_MIME.get(ext, "image/jpeg")
    return data, mime


def bytes_to_data_url(data: bytes, mime: str) -> str:
    """Base64-encode image bytes into a data URL for Azure OpenAI vision."""
    b64 = base64.b64encode(data).decode("ascii")
    return f"data:{mime};base64,{b64}"


# ---------------------------------------------------------------------------
# Teacher model
# ---------------------------------------------------------------------------


def call_teacher_vision(
    client: AzureOpenAI,
    model: str,
    prompt_text: str,
    image_data_url: str,
    timeout: int = 60,
) -> str:
    """Send image + prompt to Azure OpenAI teacher and return raw output."""
    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": image_data_url}},
                    {"type": "text", "text": prompt_text},
                ],
            }
        ],
        max_completion_tokens=50,
        temperature=0,
        timeout=timeout,
    )
    content = response.choices[0].message.content
    return content or ""


def parse_and_validate_response(raw_text: str) -> tuple[str | None, str]:
    """
    Parse and strictly validate teacher model response.

    Accepts:
      {"animal":"cat"}  or  {"animal":"dog"}  (with any internal whitespace)

    Rejects:
      - markdown fences or backticks
      - prose before or after the JSON object
      - any JSON that is not exactly {"animal": "cat"} or {"animal": "dog"}

    Returns (parsed_animal, normalized_json_str) or (None, "") if invalid.
    Normalized output is always compact: {"animal":"cat"} or {"animal":"dog"}.
    """
    text = raw_text.strip()

    # Reject fenced code blocks and markdown wrappers
    if "`" in text:
        return None, ""

    # Reject prose before or after JSON: must start with { and end with }
    if not (text.startswith("{") and text.endswith("}")):
        return None, ""

    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        return None, ""

    if not isinstance(obj, dict):
        return None, ""
    if set(obj.keys()) != {"animal"}:
        return None, ""
    val = obj["animal"]
    if val not in ("cat", "dog"):
        return None, ""

    normalized = json.dumps({"animal": val}, separators=(",", ":"))
    return val, normalized


# ---------------------------------------------------------------------------
# JSONL output
# ---------------------------------------------------------------------------


def build_training_row(
    local_image_path: str,
    prompt_text: str,
    assistant_json: str,
) -> dict:
    """
    Build one chat-format training JSONL row.

    Image paths are dataset-relative (e.g. images/cat/cat_000001.jpg).
    The training pipeline must convert these to file:// URIs before Qwen.
    The assistant content is the normalized teacher model output, never the
    Oxford source class.
    """
    return {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": local_image_path},
                    {"type": "text", "text": prompt_text},
                ],
            },
            {"role": "assistant", "content": assistant_json},
        ]
    }


def build_metadata_row(
    source_image: str,
    source_dataset: str,
    source_dataset_version: str,
    source_annotation_file: str,
    source_dataset_class: str,
    local_image_path: str | None,
    prompt_file: str,
    prompt_hash: str,
    model_deployment: str,
    raw_model_output: str,
    parsed_animal: str | None,
    teacher_label: str | None,
    label_source_for_training: str,
    teacher_dataset_match: bool | None,
    retry_count: int,
    status: str,
    error_message: str | None = None,
) -> dict:
    """
    Build one metadata record for auditing and resume support.

    source_dataset_class is the Oxford-assigned label (cat/dog) used for
    organisation and debugging only. label_source_for_training is always
    "teacher" — training labels come exclusively from the teacher model.
    """
    rec: dict = {
        "source_image": source_image,
        "source_dataset": source_dataset,
        "source_dataset_version": source_dataset_version,
        "source_annotation_file": source_annotation_file,
        "source_dataset_class": source_dataset_class,
        "prompt_file": prompt_file,
        "prompt_hash": prompt_hash,
        "model_deployment": model_deployment,
        "raw_model_output": raw_model_output,
        "parsed_animal": parsed_animal,
        "teacher_label": teacher_label,
        "label_source_for_training": label_source_for_training,
        "teacher_dataset_match": teacher_dataset_match,
        "retry_count": retry_count,
        "status": status,
        "error_message": error_message,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    if local_image_path is not None:
        rec["local_image_path"] = local_image_path
    return rec


def append_jsonl(path: Path, record: dict) -> None:
    """Append one JSON object as a single line."""
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# Resume support
# ---------------------------------------------------------------------------


def load_processed_with_paths(metadata_path: Path) -> dict[str, str]:
    """
    Load {source_image: local_image_path} for status=ok rows.
    Called once at startup when --resume is active.
    """
    result: dict[str, str] = {}
    if not metadata_path.exists():
        return result
    for line in metadata_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
            if rec.get("status") == "ok":
                img = rec.get("source_image", "")
                local = rec.get("local_image_path", "")
                if img and local:
                    result[img] = local
        except json.JSONDecodeError:
            continue
    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    log = logging.getLogger(__name__)

    args = parse_args()
    script_dir = Path(__file__).resolve().parent
    dataset_dir = script_dir / args.dataset_dir
    output_path = dataset_dir / "training.jsonl"
    metadata_path = dataset_dir / "training_metadata.jsonl"
    source_dir = dataset_dir / "source" / OXFORD_SOURCE_NAME

    max_per_class = parse_max_per_class(args.max_per_class, CLASSES)

    # Load prompt (fail fast if missing or empty)
    prompt_path = script_dir / args.prompt
    prompt_text = load_prompt(prompt_path)
    prompt_hash = hashlib.sha256(prompt_text.encode()).hexdigest()[:16]
    log.info("Loaded prompt from %s (hash=%s)", prompt_path, prompt_hash)

    # Ensure Oxford-IIIT dataset is present (downloads if needed)
    ensure_oxford_dataset(source_dir)

    # Parse Oxford annotations and build class pools
    annotations = parse_oxford_annotations(source_dir)
    all_pools = build_class_pools(annotations, source_dir)

    log.info(
        "Oxford annotation pools: cat=%d, dog=%d",
        len(all_pools["cat"]),
        len(all_pools["dog"]),
    )

    # Sample deterministically
    sampled_pools: dict[str, list[Path]] = {}
    for cls in CLASSES:
        sampled_pools[cls] = sample_class(all_pools[cls], max_per_class[cls], args.seed)
        log.info(
            "Class %s: %d available, sampling %d (seed=%d)",
            cls,
            len(all_pools[cls]),
            len(sampled_pools[cls]),
            args.seed,
        )

    if args.dry_run:
        log.info("--- DRY RUN: no API calls will be made ---")
        for cls in CLASSES:
            sampled = sampled_pools[cls]
            log.info("  %s: %d images sampled", cls, len(sampled))
            for idx, src_path in enumerate(sampled[:3], start=1):
                local_rel = make_local_image_path(cls, idx, src_path)
                preview_label = f'{{"animal":"{cls}"}}'
                row = build_training_row(
                    local_rel, prompt_text[:80] + "...", preview_label
                )
                log.info("  Preview row: %s", json.dumps(row)[:160])
        log.info("Dry run complete. Dataset dir: %s", dataset_dir)
        return

    # Full run: load config and prepare output files
    config = load_config(use_openai_endpoint=args.use_openai_endpoint)
    log.info(
        "Vision model: %s | dataset_dir: %s | max_per_class: %s",
        config["VISION_MODEL"],
        dataset_dir,
        max_per_class,
    )

    dataset_dir.mkdir(parents=True, exist_ok=True)

    processed: dict[str, str] = {}
    if args.resume and metadata_path.exists():
        processed = load_processed_with_paths(metadata_path)
        log.info("Resume mode: %d images already processed", len(processed))
    if not processed:
        output_path.write_text("", encoding="utf-8")
        metadata_path.write_text("", encoding="utf-8")

    openai_client = AzureOpenAI(
        azure_endpoint=config["VISION_BASE_URL"].rstrip("/"),
        api_key=config["VISION_API_KEY"],
        api_version=config["VISION_API_VERSION"],
    )
    model = config["VISION_MODEL"]

    total_ok = 0
    total_skip = 0

    for cls in CLASSES:
        sampled = sampled_pools[cls]
        (dataset_dir / "images" / cls).mkdir(parents=True, exist_ok=True)

        items = list(enumerate(sampled, start=1))
        if _tqdm:
            items = _tqdm.tqdm(items, desc=f"Processing {cls}", total=len(sampled))

        for idx, src_path in items:
            source_image = src_path.name
            local_rel = make_local_image_path(cls, idx, src_path)
            source_dataset_class = cls  # Oxford-assigned; for organisation only

            # Resume: skip if already successfully processed
            if source_image in processed:
                saved_path = processed[source_image]
                if (dataset_dir / saved_path).exists():
                    total_skip += 1
                    continue

            # Copy image from Oxford source into dataset folder
            try:
                save_image_to_dataset(src_path, dataset_dir, local_rel)
                data, mime = load_image_bytes(dataset_dir / local_rel)
            except Exception as exc:
                log.warning("Image copy/read failed %s: %s", source_image, exc)
                append_jsonl(
                    metadata_path,
                    build_metadata_row(
                        source_image=source_image,
                        source_dataset=OXFORD_SOURCE_NAME,
                        source_dataset_version=OXFORD_SOURCE_NAME,
                        source_annotation_file=OXFORD_ANNOTATION_FILE,
                        source_dataset_class=source_dataset_class,
                        local_image_path=None,
                        prompt_file=args.prompt,
                        prompt_hash=prompt_hash,
                        model_deployment=model,
                        raw_model_output="",
                        parsed_animal=None,
                        teacher_label=None,
                        label_source_for_training="teacher",
                        teacher_dataset_match=None,
                        retry_count=0,
                        status="image_read_failed",
                        error_message=str(exc),
                    ),
                )
                total_skip += 1
                continue

            data_url = bytes_to_data_url(data, mime)
            raw_output: str | None = None
            parsed_animal: str | None = None
            normalized = ""
            retry_count = 0
            last_error: str | None = None

            for attempt in range(args.max_retries + 1):
                retry_count = attempt
                try:
                    raw_output = call_teacher_vision(
                        openai_client, model, prompt_text, data_url
                    )
                    parsed_animal, normalized = parse_and_validate_response(raw_output)
                    if parsed_animal is not None:
                        break
                    last_error = f"Invalid response: {raw_output[:200]}"
                except APIStatusError as exc:
                    if exc.status_code == 404:
                        log.error(
                            "API returned 404 (DeploymentNotFound) for %s. "
                            "Check that deployment '%s' exists in your Azure OpenAI resource.",
                            source_image,
                            model,
                        )
                        raise SystemExit(1) from exc
                    last_error = str(exc)
                    if attempt < args.max_retries:
                        time.sleep(2**attempt)
                except Exception as exc:
                    last_error = str(exc)
                    if attempt < args.max_retries:
                        time.sleep(2**attempt)

            teacher_label = parsed_animal
            teacher_dataset_match = (
                (teacher_label == source_dataset_class)
                if teacher_label is not None
                else None
            )

            if parsed_animal is None:
                log.warning(
                    "Skipping %s after %d retries. Last: %s",
                    source_image,
                    retry_count,
                    (last_error or "")[:200],
                )
                append_jsonl(
                    metadata_path,
                    build_metadata_row(
                        source_image=source_image,
                        source_dataset=OXFORD_SOURCE_NAME,
                        source_dataset_version=OXFORD_SOURCE_NAME,
                        source_annotation_file=OXFORD_ANNOTATION_FILE,
                        source_dataset_class=source_dataset_class,
                        local_image_path=local_rel,
                        prompt_file=args.prompt,
                        prompt_hash=prompt_hash,
                        model_deployment=model,
                        raw_model_output=raw_output or "",
                        parsed_animal=None,
                        teacher_label=None,
                        label_source_for_training="teacher",
                        teacher_dataset_match=None,
                        retry_count=retry_count,
                        status="invalid_response",
                        error_message=last_error,
                    ),
                )
                total_skip += 1
                continue

            row = build_training_row(local_rel, prompt_text, normalized)
            append_jsonl(output_path, row)
            append_jsonl(
                metadata_path,
                build_metadata_row(
                    source_image=source_image,
                    source_dataset=OXFORD_SOURCE_NAME,
                    source_dataset_version=OXFORD_SOURCE_NAME,
                    source_annotation_file=OXFORD_ANNOTATION_FILE,
                    source_dataset_class=source_dataset_class,
                    local_image_path=local_rel,
                    prompt_file=args.prompt,
                    prompt_hash=prompt_hash,
                    model_deployment=model,
                    raw_model_output=raw_output or "",
                    parsed_animal=parsed_animal,
                    teacher_label=teacher_label,
                    label_source_for_training="teacher",
                    teacher_dataset_match=teacher_dataset_match,
                    retry_count=retry_count,
                    status="ok",
                ),
            )
            total_ok += 1

    log.info(
        "Done. Wrote %d rows to %s. Skipped %d.",
        total_ok,
        output_path,
        total_skip,
    )


if __name__ == "__main__":
    main()
