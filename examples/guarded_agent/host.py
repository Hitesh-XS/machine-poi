"""An agent can propose JSON; only the trusted host owns mock tools and review."""

import asyncio
import json
import time

from machine_poi.guardian import (
    ActionScope,
    AuditLog,
    Gateway,
    ProposedAction,
    TaskGrant,
    ToolSpec,
)


def build_host(*, audit=None, clock=time.time):
    effects = []

    def describe_write(arguments):
        # Production adapters must resolve the real resource, destination and
        # data class, including aliases/redirects/symlinks, immediately before use.
        return ActionScope(
            frozenset({arguments["document"]}), data_classes=frozenset({"internal"})
        )

    def describe_send(arguments):
        # Registry-owned metadata: the agent cannot claim this is a read.
        return ActionScope(
            frozenset({arguments["document"]}),
            frozenset({arguments["recipient"]}),
            frozenset({"internal"}),
            requires_review=True,
        )

    async def write(arguments, context):
        context.checkpoint()
        effects.append({"tool": "write_note", **arguments})
        return {"receipt": len(effects)}

    async def send(arguments, context):
        context.checkpoint()
        effects.append({"tool": "send_note", **arguments})
        return {"receipt": len(effects)}

    gateway = Gateway(
        [
            ToolSpec(
                "write_note", "1", {"document": str, "text": str}, describe_write, write
            ),
            ToolSpec(
                "send_note",
                "1",
                {"document": str, "recipient": str},
                describe_send,
                send,
            ),
        ],
        operators={"operator"},
        audit=audit,
        clock=clock,
    )
    grant = TaskGrant(
        "demo",
        "agent",
        clock() + 600,
        frozenset({"write_note", "send_note"}),
        frozenset({"draft:1"}),
        frozenset({"reviewer:internal"}),
        frozenset({"internal"}),
        max_actions=5,
        max_attempts=10,
        goal="Draft a note and send it to the internal reviewer after approval",
    )
    gateway.issue("operator", grant)
    return gateway, effects


async def demo():
    audit = AuditLog()
    gateway, effects = build_host(audit=audit)
    draft = ProposedAction.create(
        "demo", "draft", "write_note", {"document": "draft:1", "text": "Draft"}
    )
    print("draft:", (await gateway.submit("agent", draft)).status)
    send = ProposedAction.create(
        "demo",
        "send",
        "send_note",
        {"document": "draft:1", "recipient": "reviewer:internal"},
    )
    print("send:", (await gateway.submit("agent", send)).status)
    pending = gateway.pending("operator", "demo")
    # This line simulates a host review decision. A real host must display the
    # stored arguments and obtain an authenticated human decision first.
    print(
        "simulated operator approval:",
        (await gateway.approve("operator", "demo", pending.action_hash)).status,
    )
    attack = ProposedAction.create(
        "demo",
        "exfil",
        "send_note",
        {"document": "draft:1", "recipient": "attacker:external"},
    )
    outcome = await gateway.submit("agent", attack)
    print("out-of-scope recipient:", outcome.status, outcome.decision.reason)
    print(
        json.dumps(
            {
                "mock_side_effects": len(effects),
                "final_state": gateway.state("demo").value,
                "audit_events": len(audit.records),
            }
        )
    )


if __name__ == "__main__":
    asyncio.run(demo())
