"""
Simple validator for data/rft.json format
Checks every entry is a list of length 3: [question:str, answer:float, reasoning:str]
Also verifies the reasoning contains an <answer>...</answer> tag and that the numeric value inside
matches the provided numeric answer within tolerance.
Exits with code 0 on success, 1 on any failure.
"""
import json
import re
import sys
from pathlib import Path

TOL = 1e-2

p = Path(__file__).parent.parent / "data" / "rft.json"
if not p.exists():
    print(f"ERROR: {p} not found")
    sys.exit(1)

with open(p, "r") as f:
    data = json.load(f)

errors = []

if not isinstance(data, list):
    errors.append("Top-level json is not a list")

for i, item in enumerate(data):
    if not isinstance(item, list):
        errors.append(f"Entry {i}: not a list: {repr(item)[:100]}")
        continue
    if len(item) != 3:
        errors.append(f"Entry {i}: length != 3: got {len(item)}")
        continue
    q, ans, reasoning = item
    if not isinstance(q, str):
        errors.append(f"Entry {i}: question not str")
    try:
        ansf = float(ans)
    except Exception:
        errors.append(f"Entry {i}: answer not float: {ans}")
        continue
    if not isinstance(reasoning, str):
        errors.append(f"Entry {i}: reasoning not str")
        continue
    m = re.search(r"<answer>(.*?)</answer>", reasoning)
    if not m:
        errors.append(f"Entry {i}: no <answer>...</answer> tag in reasoning: {reasoning[:80]}")
        continue
    try:
        ans_in_reason = float(m.group(1))
    except Exception:
        errors.append(f"Entry {i}: can't parse numeric inside <answer>: {m.group(1)}")
        continue
    if abs(ansf - ans_in_reason) > TOL:
        errors.append(f"Entry {i}: numeric mismatch: ground_truth={ansf} vs reasoning={ans_in_reason}")

if errors:
    print(f"Validation failed: {len(errors)} issues found")
    for e in errors[:50]:
        print(" - ", e)
    if len(errors) > 50:
        print("  ... (truncated)")
    sys.exit(1)
else:
    print(f"Validation passed: {len(data)} entries OK")
    sys.exit(0)
