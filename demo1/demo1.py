#!/usr/bin/env python3
"""
Demo 1: Load demo1_prompt.txt (template), inject demo1_narrative.txt into {narrative_text},
send the combined prompt to an OpenAI GPT endpoint, and print the model's response.
Uses .env for OPENAI_API_KEY; optional OPENAI_BASE_URL (Azure endpoint), OPENAI_API_VERSION, OPENAI_MODEL.
"""

import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from openai import AzureOpenAI, OpenAI


def main() -> None:
    # Load .env from demos root (no-op if file missing; env vars can be set externally)
    demos_dir = Path(__file__).resolve().parent.parent
    load_dotenv(demos_dir / ".env")

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print(
            "Error: OPENAI_API_KEY is not set. Add it to .env or set the env var.",
            file=sys.stderr,
        )
        sys.exit(1)

    shared_dir = demos_dir / "shared"
    prompt_path = shared_dir / "demo1_prompt.txt"
    narrative_path = shared_dir / "demo1_narrative.txt"

    if not prompt_path.exists():
        print(f"Error: Prompt file not found: {prompt_path}", file=sys.stderr)
        sys.exit(1)
    if not narrative_path.exists():
        print(f"Error: Narrative file not found: {narrative_path}", file=sys.stderr)
        sys.exit(1)

    prompt_template = prompt_path.read_text()
    narrative_text = narrative_path.read_text().strip()
    if not narrative_text:
        print("Error: demo1_narrative.txt is empty.", file=sys.stderr)
        sys.exit(1)

    prompt_text = prompt_template.replace("{narrative_text}", narrative_text)
    if not prompt_text.strip():
        print("Error: demo1_prompt.txt is empty.", file=sys.stderr)
        sys.exit(1)

    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    base_url = os.getenv("OPENAI_BASE_URL")

    if base_url:
        client = AzureOpenAI(
            api_version=os.getenv("OPENAI_API_VERSION", "2024-12-01-preview"),
            azure_endpoint=base_url.rstrip("/"),
            api_key=api_key,
        )
    else:
        client = OpenAI(api_key=api_key)

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "user", "content": prompt_text},
            ],
        )
    except Exception as e:
        print(f"Error calling OpenAI API: {e}", file=sys.stderr)
        sys.exit(1)

    content = response.choices[0].message.content
    if content is None:
        content = ""
    print(content)


if __name__ == "__main__":
    main()
