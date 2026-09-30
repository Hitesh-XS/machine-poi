"""The committed containment report must match a fresh run of its fixtures.

Docs cite evals/rogue_agent/results.json instead of hardcoding counts, so this
test keeps that file current. Regenerate it with:

    python -m evals.rogue_agent.run --output evals/rogue_agent/results.json
"""

import asyncio
import hashlib
import json
from pathlib import Path

from evals.rogue_agent import run as runner

EVAL_DIR = Path(runner.__file__).parent
SUMMARY = (
    "cases", "passed", "forbidden_actions", "blocked_forbidden_actions",
    "unapproved_side_effects", "benign_cases", "false_blocks", "reviews_requested",
)


def fresh_report():
    cases = (EVAL_DIR / "cases.json").read_bytes()
    scenarios = (EVAL_DIR / "scenarios.json").read_bytes()
    report = asyncio.run(runner.evaluate(json.loads(cases)))
    report["cases_sha256"] = hashlib.sha256(cases).hexdigest()
    report["scenarios"] = asyncio.run(runner.evaluate_scenarios(json.loads(scenarios)))
    report["scenarios"]["sha256"] = hashlib.sha256(scenarios).hexdigest()
    return report


def test_committed_report_matches_a_fresh_run():
    committed = json.loads((EVAL_DIR / "results.json").read_text())
    fresh = fresh_report()
    assert committed["cases_sha256"] == fresh["cases_sha256"], "regenerate results.json"
    assert {k: committed[k] for k in SUMMARY} == {k: fresh[k] for k in SUMMARY}
    assert committed["results"] == fresh["results"]
    assert committed["scenarios"] == fresh["scenarios"]


def test_every_fixture_passes_with_no_unapproved_effects():
    report = fresh_report()
    assert report["passed"] == report["cases"] and report["unapproved_side_effects"] == 0
    scenarios = report["scenarios"]
    assert scenarios["passed"] == scenarios["count"] and scenarios["unapproved_side_effects"] == 0
