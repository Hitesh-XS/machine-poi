"""Replay synthetic action proposals against mock tools; no model is invoked.

``cases.json`` holds single proposed actions. ``scenarios.json`` holds
multi-step sequences: review and approval, concurrent submissions, delegation,
and stops or expiry while a job is in flight. Each scenario declares the exact
side effects it permits; any others count as unapproved.

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
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from examples.guarded_agent.host import build_host
from machine_poi.guardian import ProposedAction, TaskGrant


def outcome_row(step, outcome):
    passed = outcome.status == step["expect"] and (
        "reason" not in step or outcome.decision.reason == step["reason"]
    )
    return {
        "op": step["op"],
        "id": step.get("id"),
        "expected": step["expect"],
        "status": outcome.status,
        "reason": outcome.decision.reason,
        "passed": passed,
    }


async def run_scenario(scenario):
    """Run one scenario on a fresh host with a controllable clock."""
    now = [1_000_000.0]
    release = asyncio.Event()
    gateway, effects = build_host(clock=lambda: now[0], slow_job=release)
    in_flight, rows = {}, []

    def action(step, run_id):
        return ProposedAction.create(run_id, step["id"], step["tool"], step["arguments"])

    for step in scenario["steps"]:
        op, run_id = step["op"], step.get("run", "demo")
        if op == "submit":
            rows.append(outcome_row(step, await gateway.submit("agent", action(step, run_id))))
        elif op == "submit_all":
            outcomes = await asyncio.gather(
                *(gateway.submit("agent", action(item, run_id)) for item in step["actions"])
            )
            rows += [
                outcome_row({**item, "op": op}, outcome)
                for item, outcome in zip(step["actions"], outcomes)
            ]
        elif op == "approve":
            pending = gateway.pending("operator", run_id)
            try:
                if pending is None:
                    raise ValueError("No pending review")
                rows.append(
                    outcome_row(step, await gateway.approve("operator", run_id, pending.action_hash))
                )
            except ValueError:
                rows.append({
                    "op": op, "id": None, "expected": step["expect"], "status": "error",
                    "reason": "no_pending_review", "passed": step["expect"] == "error",
                })
        elif op == "issue":
            parent = gateway._runs[run_id].grant  # the host reads its own grant
            try:
                gateway.issue("operator", TaskGrant(
                    step["run_id"], "agent", parent.expires_at, frozenset(step["tools"]),
                    frozenset(step["resources"]), frozenset(step["destinations"]),
                    parent.data_classes, max_actions=step["max_actions"],
                    max_attempts=parent.max_attempts, parent_run_id=run_id,
                ))
                status = "issued"
            except ValueError:
                status = "rejected"
            rows.append({"op": op, "id": step["run_id"], "expected": step["expect"],
                         "status": status, "reason": None, "passed": status == step["expect"]})
        elif op == "state":
            state = gateway.state(run_id).value
            rows.append({"op": op, "id": run_id, "expected": step["expect"],
                         "status": state, "reason": None, "passed": state == step["expect"]})
        elif op == "stop":
            gateway.stop("operator", run_id)
        elif op == "advance_clock":
            now[0] += step["seconds"]
        elif op == "start":
            in_flight[step["id"]] = asyncio.ensure_future(
                gateway.submit("agent", action(step, run_id))
            )
            for _ in range(5):  # let the job dispatch and reach its wait
                await asyncio.sleep(0)
        elif op == "release":
            release.set()
        elif op == "await":
            rows.append(outcome_row(step, await in_flight.pop(step["id"])))
        else:
            raise ValueError(f"Unknown scenario op {op!r}")

    actual = Counter(effect["tool"] for effect in effects)
    allowed = scenario["effects"]
    unapproved = sum(max(0, count - allowed.get(tool, 0)) for tool, count in actual.items())
    exact = {tool: count for tool, count in actual.items()} == {
        tool: count for tool, count in allowed.items() if count
    }
    return {
        "id": scenario["id"],
        "steps": rows,
        "side_effects": dict(actual),
        "allowed_side_effects": allowed,
        "unapproved_side_effects": unapproved,
        "passed": exact and all(row["passed"] for row in rows),
    }


async def evaluate_scenarios(scenarios):
    results = [await run_scenario(scenario) for scenario in scenarios]
    return {
        "count": len(results),
        "passed": sum(result["passed"] for result in results),
        "steps": sum(len(result["steps"]) for result in results),
        "unapproved_side_effects": sum(r["unapproved_side_effects"] for r in results),
        "results": results,
    }


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
            "Scenarios use mock tools and a simulated clock; operator decisions are scripted.",
        ],
        "results": rows,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cases", type=Path, default=Path(__file__).with_name("cases.json")
    )
    parser.add_argument(
        "--scenarios", type=Path, default=Path(__file__).with_name("scenarios.json")
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    content = args.cases.read_bytes()
    scenario_content = args.scenarios.read_bytes()
    result = asyncio.run(evaluate(json.loads(content)))
    result["cases_sha256"] = hashlib.sha256(content).hexdigest()
    result["scenarios"] = asyncio.run(evaluate_scenarios(json.loads(scenario_content)))
    result["scenarios"]["sha256"] = hashlib.sha256(scenario_content).hexdigest()
    text = json.dumps(result, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text)
    print(text)
    scenarios = result["scenarios"]
    if result["passed"] != result["cases"] or scenarios["passed"] != scenarios["count"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
