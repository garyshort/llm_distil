#!/usr/bin/env python3
"""
Validate training_data.jsonl: required keys (prompt, completion), well-formed JSON
per line and inside completion, and completion schema (damage keys + overall_severity).
"""

import json
import sys
from pathlib import Path

REQUIRED_KEYS = ("prompt", "completion")
DAMAGE_KEYS = (
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
)
VALID_SEVERITIES = ("low", "moderate", "high")


def validate_line(line: str, line_num: int) -> list[str]:
    errors = []
    stripped = line.strip()
    if not stripped:
        return []

    try:
        obj = json.loads(stripped)
    except json.JSONDecodeError as e:
        return [f"Line {line_num}: invalid JSON - {e}"]

    if not isinstance(obj, dict):
        errors.append(f"Line {line_num}: root value must be an object")
        return errors

    for key in REQUIRED_KEYS:
        if key not in obj:
            errors.append(f"Line {line_num}: missing required key '{key}'")
            return errors
        if not isinstance(obj[key], str):
            errors.append(f"Line {line_num}: '{key}' must be a string")
        elif not obj[key].strip():
            errors.append(f"Line {line_num}: '{key}' must be non-empty")

    if "completion" in obj and isinstance(obj["completion"], str):
        try:
            completion = json.loads(obj["completion"])
        except json.JSONDecodeError as e:
            errors.append(f"Line {line_num}: completion is not valid JSON - {e}")
            return errors

        if not isinstance(completion, dict):
            errors.append(f"Line {line_num}: completion must be a JSON object")
            return errors

        if "damage" not in completion:
            errors.append(f"Line {line_num}: completion missing 'damage'")
        else:
            damage = completion["damage"]
            if not isinstance(damage, dict):
                errors.append(f"Line {line_num}: completion.damage must be an object")
            else:
                for dk in DAMAGE_KEYS:
                    if dk not in damage:
                        errors.append(f"Line {line_num}: completion.damage missing '{dk}'")
                    elif not isinstance(damage[dk], bool):
                        errors.append(
                            f"Line {line_num}: completion.damage.{dk} must be boolean"
                        )

        if "overall_severity" not in completion:
            errors.append(f"Line {line_num}: completion missing 'overall_severity'")
        elif completion["overall_severity"] not in VALID_SEVERITIES:
            errors.append(
                f"Line {line_num}: overall_severity must be one of {VALID_SEVERITIES}"
            )

    return errors


def main() -> int:
    base_dir = Path(__file__).resolve().parent
    path = base_dir / "training_data.jsonl"

    if not path.exists():
        print(f"Error: {path} not found", file=sys.stderr)
        return 1

    all_errors = []
    line_count = 0

    with open(path) as f:
        for line_num, line in enumerate(f, start=1):
            errs = validate_line(line, line_num)
            if errs:
                all_errors.extend(errs)
            if line.strip():
                line_count += 1

    if all_errors:
        for err in all_errors:
            print(err, file=sys.stderr)
        print(f"\nValidation failed: {len(all_errors)} error(s), {line_count} line(s)", file=sys.stderr)
        return 1

    print(f"Valid: {line_count} record(s) in {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
