#!/usr/bin/env python3
"""
Split training_data.jsonl into MLX-style train/valid/test under data/.
Uses a fixed seed for reproducible splits: 800 train, 100 valid, 100 test.
"""

import random
import sys
from pathlib import Path

TRAIN_SIZE = 800
VALID_SIZE = 100
TEST_SIZE = 100
TOTAL = TRAIN_SIZE + VALID_SIZE + TEST_SIZE
SEED = 42


def main() -> int:
    base_dir = Path(__file__).resolve().parent
    input_path = base_dir / "training_data.jsonl"
    data_dir = base_dir / "data"

    if not input_path.exists():
        print(f"Error: {input_path} not found", file=sys.stderr)
        return 1

    lines = [line for line in input_path.read_text().splitlines() if line.strip()]
    if len(lines) != TOTAL:
        print(
            f"Error: expected {TOTAL} records, found {len(lines)}. "
            "Run generate_training_data.py first.",
            file=sys.stderr,
        )
        return 1

    rng = random.Random(SEED)
    rng.shuffle(lines)

    data_dir.mkdir(parents=True, exist_ok=True)
    train_lines = lines[:TRAIN_SIZE]
    valid_lines = lines[TRAIN_SIZE : TRAIN_SIZE + VALID_SIZE]
    test_lines = lines[TRAIN_SIZE + VALID_SIZE :]

    (data_dir / "train.jsonl").write_text("\n".join(train_lines) + "\n")
    (data_dir / "valid.jsonl").write_text("\n".join(valid_lines) + "\n")
    (data_dir / "test.jsonl").write_text("\n".join(test_lines) + "\n")

    print(f"Wrote {data_dir}/")
    print(f"  train.jsonl: {len(train_lines)}")
    print(f"  valid.jsonl: {len(valid_lines)}")
    print(f"  test.jsonl:  {len(test_lines)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
