#!/usr/bin/env python3
"""
Demo 3: Same test as demo1 (prompt + narrative) but run against the local MLX model
with LoRA adapters (Qwen2.5-7B + adapters_qwen25_7b_damage).

Uses shared/demo1_prompt.txt and shared/demo1_narrative.txt from the demos/ directory.
Requires: pip install "mlx-lm"

Run from this directory: python run.py
Or from demos/: python demo3/run.py
"""

import json
import sys
from pathlib import Path

from mlx_lm import generate, load
from mlx_lm.sample_utils import make_sampler

MODEL = "mlx-community/Qwen2.5-7B-Instruct-4bit"
MAX_TOKENS = 256
TEMP = 0.0


def extract_first_json_block(text: str) -> str:
    """Return first complete {...} substring; empty string if not found."""
    start = text.find("{")
    if start == -1:
        return ""
    depth = 0
    for i, c in enumerate(text[start:], start):
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return ""


def run_generate(prompt: str, adapter_path: Path) -> str:
    """Load model + adapters and generate; return generated text."""
    model, tokenizer = load(MODEL, adapter_path=str(adapter_path))
    sampler = make_sampler(temp=TEMP)
    return generate(
        model,
        tokenizer,
        prompt,
        max_tokens=MAX_TOKENS,
        sampler=sampler,
    )


def main() -> None:
    demo3_dir = Path(__file__).resolve().parent
    demos_root = demo3_dir.parent

    shared_dir = demos_root / "shared"
    prompt_path = shared_dir / "demo1_prompt.txt"
    narrative_path = shared_dir / "demo1_narrative.txt"
    adapter_path = demos_root / "demo2" / "adapters_qwen25_7b_damage"

    if not prompt_path.exists():
        print(f"Error: Prompt file not found: {prompt_path}", file=sys.stderr)
        sys.exit(1)
    if not narrative_path.exists():
        print(f"Error: Narrative file not found: {narrative_path}", file=sys.stderr)
        sys.exit(1)
    if not adapter_path.exists():
        print(
            f"Error: Adapter directory not found: {adapter_path}\n"
            "Run demo2/train_and_eval_student.py first to train the adapters.",
            file=sys.stderr,
        )
        sys.exit(1)

    prompt_template = prompt_path.read_text()
    narrative_text = narrative_path.read_text().strip()
    if not narrative_text:
        print("Error: demo1_narrative.txt is empty.", file=sys.stderr)
        sys.exit(1)

    prompt_text = prompt_template.replace("{narrative_text}", narrative_text)

    content = run_generate(prompt_text, adapter_path)
    # Extract and print only the JSON (pretty-printed like demo1)
    json_block = extract_first_json_block(content)
    if json_block:
        try:
            parsed = json.loads(json_block)
            print(json.dumps(parsed, indent=2))
        except json.JSONDecodeError:
            print(json_block)
    else:
        print(content)


if __name__ == "__main__":
    main()
