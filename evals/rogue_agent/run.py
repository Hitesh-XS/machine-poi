"""Replay synthetic action proposals against mock tools; no model is invoked.

An unguarded baseline would dispatch each proposal to a hypothetical permissive
executor. We report the proposal count, not fabricated model/steering results.
Use model-generated traces as an additional dataset after integrating a host.
"""

import argparse
import asyncio
import hashlib
import json
import platform
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path

from examples.guarded_agent.host import build_host
from machine_poi.guardian import ProposedAction


async def evaluate(cases):
    rows = []
    latencies = []
    for case in cases:
        gateway, effects = build_host()
        action = ProposedAction.create(
            "demo", case["id"], case["tool"], case["arguments"]
        )
        start = time.perf_counter()
        result = await gateway.submit("agent", action)
        latencies.append((time.perf_counter() - start) * 1000)
        passed = result.status == case["expected"] and (
            "reason" not in case or result.decision.reason == case["reason"]
        )
        if case["expected"] != "executed":
            passed = passed and not effects
        rows.append(
            {
                "id": case["id"],
                "expected": case["expected"],
                "status": result.status,
                "reason": result.decision.reason,
                "side_effects": len(effects),
                "passed": passed,
            }
        )
    forbidden = [row for row in rows if row["expected"] == "blocked"]
    benign = [row for row in rows if row["expected"] != "blocked"]
    return {
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(),
        "kind": "synthetic action-trace fixtures; no model/steering execution",
        "cases": len(rows),
        "passed": sum(row["passed"] for row in rows),
        "forbidden_actions": len(forbidden),
        "blocked_forbidden_actions": sum(
            row["status"] == "blocked" for row in forbidden
        ),
        "unapproved_side_effects": sum(
            row["side_effects"] for row in rows if row["expected"] != "executed"
        ),
        "benign_cases": len(benign),
        "false_blocks": sum(row["status"] == "blocked" for row in benign),
        "reviews_requested": sum(row["status"] == "pending" for row in rows),
        "latency_ms": {"median": statistics.median(latencies), "max": max(latencies)},
        "limitations": [
            "Fixtures are not held-out evidence of general safety or prompt-injection detection.",
            "No live host, external side effects, model or steering A/B comparison was run.",
            "Cancellation, budget races, replay and delegation are covered by unit tests.",
        ],
        "results": rows,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cases", type=Path, default=Path(__file__).with_name("cases.json")
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    content = args.cases.read_bytes()
    result = asyncio.run(evaluate(json.loads(content)))
    result["cases_sha256"] = hashlib.sha256(content).hexdigest()
    text = json.dumps(result, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text)
    print(text)
    if result["passed"] != result["cases"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
