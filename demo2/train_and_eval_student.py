#!/usr/bin/env python3
"""
MLX-LM student model: LoRA fine-tune + smoke test + test-set evaluation.

Requires: pip install "mlx-lm[train]"
Run from this directory: python train_and_eval_student.py

Uses mlx_lm Python API (load, generate, lora.run). Data: data/train.jsonl,
data/valid.jsonl, data/test.jsonl. Each JSONL line: {"prompt": "...", "completion": "<JSON string>"}.
"""

import json
import os
import sys
import time
import types
from pathlib import Path
from typing import Any, Optional

from mlx_lm import generate, load
from mlx_lm.lora import run as lora_run
from mlx_lm.sample_utils import make_sampler


def format_duration(seconds: float) -> str:
    """Format seconds as human-readable string (e.g. '2m 34.5s' or '1h 5m 23s')."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    mins = int(seconds // 60)
    secs = seconds % 60
    if mins < 60:
        return f"{mins}m {secs:.1f}s"
    hours = mins // 60
    mins = mins % 60
    return f"{hours}h {mins}m {secs:.1f}s"


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
DATA_DIR = Path("data")
MODEL = "mlx-community/Qwen2.5-7B-Instruct-4bit"
ADAPTER_DIR = Path("adapters_qwen25_7b_damage")

# Training
ITERS = 1200
BATCH_SIZE = 2
LEARNING_RATE = 7e-5
MASK_PROMPT = True
GRAD_CHECKPOINT = True

# Generation
MAX_TOKENS = 256
TEMP = 0.0

# Schema
DAMAGE_FIELDS = [
    "broken_plaster",
    "mould",
    "floor_water_damage",
    "electrical_damage",
    "ceiling_damage",
    "structural_crack",
    "carpet_damage",
    "cabinet_damage",
    "appliance_damage",
    "odor_present",
]
SEVERITY_FIELD = "overall_severity"


def train(script_dir: Path) -> None:
    """Verify data files exist and run LoRA training via mlx_lm.lora.run()."""
    data_dir = script_dir / DATA_DIR
    adapter_path = script_dir / ADAPTER_DIR
    for name in ("train.jsonl", "valid.jsonl", "test.jsonl"):
        p = data_dir / name
        if not p.exists():
            print(f"Error: required file not found: {p}", file=sys.stderr)
            sys.exit(1)

    args = types.SimpleNamespace(
        model=MODEL,
        train=True,
        data=str(data_dir),
        iters=ITERS,
        batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        adapter_path=str(adapter_path),
        mask_prompt=MASK_PROMPT,
        grad_checkpoint=GRAD_CHECKPOINT,
        fine_tune_type="lora",
        optimizer="adam",
        optimizer_config={
            "adam": {},
            "adamw": {},
            "muon": {},
            "sgd": {},
            "adafactor": {},
        },
        seed=0,
        num_layers=16,
        val_batches=25,
        steps_per_report=10,
        steps_per_eval=200,
        resume_adapter_file=None,
        save_every=100,
        test=False,
        test_batches=500,
        max_seq_length=2048,
        config=None,
        grad_accumulation_steps=1,
        lr_schedule=None,
        lora_parameters={"rank": 8, "dropout": 0.0, "scale": 20.0},
        report_to=None,
        project_name=None,
    )
    lora_run(args)


def generate_with_model(
    prompt: str,
    model: Any,
    tokenizer: Any,
    sampler: Any,
    quiet: bool = False,
) -> str:
    """Generate text using loaded model and tokenizer."""
    return generate(
        model,
        tokenizer,
        prompt,
        max_tokens=MAX_TOKENS,
        sampler=sampler,
    )


def smoke_test(model: Any, tokenizer: Any, sampler: Any) -> None:
    """Generate on a short narrative, parse JSON, print result."""
    prompt = (
        "You are an insurance damage extraction system.\n\n"
        "Extract damage indicators from the narrative and return ONLY valid JSON "
        "matching the schema.\n\nNARRATIVE:\n"
        "Small amount of water in the basement corner. Carpet is slightly damp. "
        "No mold or odor. Walls and ceiling look fine. Electrical and appliances OK."
    )
    raw = generate_with_model(prompt, model, tokenizer, sampler)
    print("\n--- Raw output ---")
    print(raw)
    block = extract_first_json_block(raw)
    ok, obj = safe_parse_json(block)
    print("\nJSON parse: " + ("OK" if ok else "FAILED"))
    if ok and obj:
        print(json.dumps(obj, indent=2))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    """Load JSONL file; return list of dicts (skip empty lines)."""
    out = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        out.append(json.loads(line))
    return out


def safe_parse_json(text: str) -> tuple[bool, Optional[dict]]:
    """Try to parse JSON; return (True, obj) or (False, None)."""
    try:
        return True, json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return False, None


def extract_first_json_block(text: str) -> str:
    """Return first {...} substring; empty string if not found."""
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end < start:
        return ""
    return text[start: end + 1]


def eval_on_test(
    script_dir: Path,
    model: Any,
    tokenizer: Any,
    sampler: Any,
    max_examples: Optional[int] = 100,
) -> None:
    """Evaluate on test.jsonl: validity rate, per-field accuracy, severity accuracy."""
    data_dir = script_dir / DATA_DIR
    test_path = data_dir / "test.jsonl"
    if not test_path.exists():
        print(f"Error: {test_path} not found", file=sys.stderr)
        sys.exit(1)

    examples = load_jsonl(test_path)
    if max_examples is not None:
        examples = examples[:max_examples]
    total = len(examples)

    valid_count = 0
    damage_correct: dict[str, int] = {f: 0 for f in DAMAGE_FIELDS}
    severity_correct = 0

    for i, ex in enumerate(examples):
        if (i + 1) % 20 == 0:
            print(f"  Evaluated {i + 1} / {total} ...")

        prompt = ex.get("prompt", "")
        completion_raw = ex.get("completion", "")
        ok_gold, gold = safe_parse_json(completion_raw)
        if not ok_gold or not gold:
            continue

        raw_out = generate_with_model(prompt, model, tokenizer, sampler, quiet=True)
        block = extract_first_json_block(raw_out)
        ok_pred, pred = safe_parse_json(block)
        if not ok_pred or not pred:
            continue
        valid_count += 1

        pred_damage = pred.get("damage") or {}
        gold_damage = gold.get("damage") or {}
        for f in DAMAGE_FIELDS:
            if pred_damage.get(f) == gold_damage.get(f):
                damage_correct[f] += 1
        if pred.get(SEVERITY_FIELD) == gold.get(SEVERITY_FIELD):
            severity_correct += 1

    # Report (per-field and severity accuracy over examples with valid JSON only)
    print("\n--- EVAL RESULTS ---")
    print(
        f"JSON validity rate: {valid_count}/{total} = {100.0 * valid_count / total:.1f}%"
    )
    n = valid_count if valid_count else 1
    print("Per-field accuracy (damage booleans, on valid predictions):")
    for f in DAMAGE_FIELDS:
        acc = 100.0 * damage_correct[f] / n
        print(f"  {f}: {damage_correct[f]}/{valid_count} = {acc:.1f}%")
    sev_acc = 100.0 * severity_correct / n
    print(f"{SEVERITY_FIELD}: {severity_correct}/{valid_count} = {sev_acc:.1f}%")


def main() -> None:
    script_dir = Path(__file__).resolve().parent
    os.chdir(script_dir)

    data_dir = script_dir / DATA_DIR
    adapter_path = script_dir / ADAPTER_DIR

    print("=== CONFIG ===")
    print(f"  Model:       {MODEL}")
    print(f"  Data dir:    {data_dir}")
    print(f"  Adapter dir: {adapter_path}")
    print()

    total_start = time.perf_counter()

    print("=== TRAIN ===")
    t0 = time.perf_counter()
    train(script_dir)
    train_elapsed = time.perf_counter() - t0
    print(f"  Training completed in {format_duration(train_elapsed)}")
    print()

    print("=== LOAD MODEL + ADAPTERS ===")
    model, tokenizer = load(MODEL, adapter_path=str(adapter_path))
    sampler = make_sampler(temp=TEMP)
    print()

    print("=== SMOKE TEST ===")
    t0 = time.perf_counter()
    smoke_test(model, tokenizer, sampler)
    smoke_elapsed = time.perf_counter() - t0
    print(f"  Smoke test completed in {format_duration(smoke_elapsed)}")
    print()

    # Set max_examples=None for full test set evaluation
    print("=== EVAL (test set) ===")
    t0 = time.perf_counter()
    eval_on_test(script_dir, model, tokenizer, sampler, max_examples=100)
    eval_elapsed = time.perf_counter() - t0
    print(f"  Eval completed in {format_duration(eval_elapsed)}")
    print()

    total_elapsed = time.perf_counter() - total_start
    print("--- TIMING ---")
    print(f"  Train:      {format_duration(train_elapsed)}")
    print(f"  Smoke test: {format_duration(smoke_elapsed)}")
    print(f"  Eval:       {format_duration(eval_elapsed)}")
    print(f"  Total:      {format_duration(total_elapsed)}")


if __name__ == "__main__":
    main()
