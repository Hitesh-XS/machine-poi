"""Scripted agent process that emits proposals. No tool or control API imports."""

import json
import sys

if __name__ == "__main__":
    data = sys.stdin.read(65537)
    if len(data) > 65536:
        raise SystemExit("Input too large")
    # Simulate a model choosing an action; text remains data throughout.
    for proposal in json.loads(data):
        print(json.dumps(proposal))
