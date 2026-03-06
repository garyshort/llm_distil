#!/usr/bin/env bash
#
# Compare demo1.py (OpenAI) vs run.py (local MLX + LoRA) outputs.
# Runs both, displays their output, and reports % key-value match.
#
# Run from demo3/: ./compare_demo1_and_run.sh
# Or from demos/: ./demo3/compare_demo1_and_run.sh
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEMOS_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

cd "$DEMOS_ROOT"

echo "=========================================="
echo "Demo 1 (OpenAI) output:"
echo "=========================================="
OUTPUT_DEMO1=$(python demo1/demo1.py) || { echo "demo1.py failed (check OPENAI_API_KEY in .env)" >&2; exit 1; }
echo "$OUTPUT_DEMO1"

echo ""
echo "=========================================="
echo "Demo 3 run.py (MLX + LoRA) output:"
echo "=========================================="
OUTPUT_RUN=$(python demo3/run.py) || { echo "run.py failed" >&2; exit 1; }
echo "$OUTPUT_RUN"

echo ""
echo "=========================================="
echo "Match comparison:"
echo "=========================================="

TMP1=$(mktemp)
TMP2=$(mktemp)
trap 'rm -f "$TMP1" "$TMP2"' EXIT
echo "$OUTPUT_DEMO1" > "$TMP1"
echo "$OUTPUT_RUN" > "$TMP2"

python3 - "$TMP1" "$TMP2" << 'PYEOF'
import json
import sys

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

def flatten(obj, prefix=""):
    """Flatten nested dict to dot-separated key paths -> value."""
    result = {}
    for k, v in obj.items():
        path = f"{prefix}.{k}" if prefix else k
        if isinstance(v, dict):
            if all(not isinstance(x, (dict, list)) for x in v.values()):
                for k2, v2 in v.items():
                    result[f"{path}.{k2}"] = v2
            else:
                result.update(flatten(v, path))
        else:
            result[path] = v
    return result

def main():
    with open(sys.argv[1]) as f:
        output1 = f.read()
    with open(sys.argv[2]) as f:
        output2 = f.read()

    json1_str = extract_first_json_block(output1)
    json2_str = extract_first_json_block(output2)

    if not json1_str or not json2_str:
        print("Error: Could not extract valid JSON from one or both outputs.", file=sys.stderr)
        sys.exit(1)

    try:
        d1 = json.loads(json1_str)
        d2 = json.loads(json2_str)
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON - {e}", file=sys.stderr)
        sys.exit(1)

    flat1 = flatten(d1)
    flat2 = flatten(d2)

    all_keys = sorted(set(flat1.keys()) | set(flat2.keys()))
    if not all_keys:
        print("No keys to compare.")
        return

    matches = sum(1 for k in all_keys if flat1.get(k) == flat2.get(k))
    pct = 100.0 * matches / len(all_keys)

    print(f"Keys compared: {len(all_keys)}")
    print(f"Matches: {matches}")
    print(f"Match: {pct:.1f}%")

    mismatches = [k for k in all_keys if flat1.get(k) != flat2.get(k)]
    if mismatches:
        print("\nMismatches:")
        for k in mismatches:
            v1 = flat1.get(k, "<missing>")
            v2 = flat2.get(k, "<missing>")
            print(f"  {k}: demo1={v1!r} vs run.py={v2!r}")

if __name__ == "__main__":
    main()
PYEOF
