"""Demonstrate a JSON-only model boundary. This is not an OS sandbox."""

import asyncio
import json
import os
import subprocess
import sys
from pathlib import Path

from examples.guarded_agent.host import build_host
from src.guardian import ProposedAction


async def main():
    proposals = [
        {
            "id": "draft",
            "tool": "write_note",
            "arguments": {"document": "draft:1", "text": "Authorized draft"},
        },
        {
            "id": "injected",
            "tool": "send_note",
            "arguments": {"document": "draft:1", "recipient": "attacker:external"},
        },
    ]
    # Fixed trusted script, isolated import path, minimal environment. The model
    # process receives no gateway object, credentials, grants or operator API.
    result = subprocess.run(
        [sys.executable, "-I", str(Path(__file__).with_name("worker.py"))],
        input=json.dumps(proposals),
        text=True,
        capture_output=True,
        check=True,
        timeout=10,
        env={"PATH": os.defpath},
    )
    gateway, effects = build_host()
    if len(result.stdout) > 65536:
        raise ValueError("Oversized worker response")
    for line in result.stdout.splitlines():
        value = json.loads(line)
        if set(value) != {"id", "tool", "arguments"}:
            raise ValueError("Unexpected worker fields")
        action = ProposedAction.create(
            "demo", value["id"], value["tool"], value["arguments"]
        )
        # Caller identity comes from this host's worker channel, not the JSON.
        outcome = await gateway.submit("agent", action)
        print(outcome.status, outcome.decision.reason)
    assert len(effects) == 1
    print("One authorized mock write; external send prevented.")


if __name__ == "__main__":
    asyncio.run(main())
