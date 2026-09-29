"""Action-level tests with a trusted mock executor, including async failure paths."""

import asyncio
from dataclasses import replace

import pytest

from machine_poi.guardian import (
    ActionScope,
    AuditLog,
    Gateway,
    Policy,
    ProposedAction,
    RunState,
    TaskGrant,
    ToolSpec,
    Verdict,
    verify_records,
)


def run(coro):
    return asyncio.run(coro)


class Rig:
    def __init__(
        self,
        *,
        review=False,
        execute=None,
        describe=None,
        audit=None,
        policy=None,
        max_actions=10,
        on_stop=None,
    ):
        self.now = 1000.0
        self.effects = []
        self.audit = audit or AuditLog()
        self.revoked = []

        async def default_execute(args, context):
            context.checkpoint()
            self.effects.append(args.copy())
            return {"receipt": len(self.effects)}

        def default_describe(args):
            return ActionScope(
                frozenset({args["resource"]}),
                frozenset({args["destination"]}),
                requires_review=review,
                cost_units=2,
                tokens=3,
            )

        tool = ToolSpec(
            "write",
            "1",
            {"resource": str, "destination": str, "body": str},
            describe or default_describe,
            execute or default_execute,
        )
        self.gateway = Gateway(
            [tool],
            operators={"operator"},
            audit=self.audit,
            policy=policy,
            clock=lambda: self.now,
            on_stop=on_stop or self.revoked.append,
        )
        self.grant = TaskGrant(
            "run",
            "agent",
            2000,
            frozenset({"write"}),
            frozenset({"document:1"}),
            frozenset({"internal"}),
            max_actions=max_actions,
            max_cost_units=20,
            max_tokens=30,
        )
        self.gateway.issue("operator", self.grant)

    def action(self, name="a", **kwargs):
        return ProposedAction.create(
            "run",
            name,
            "write",
            {
                "resource": "document:1",
                "destination": "internal",
                "body": "hello",
                **kwargs,
            },
        )


def test_only_host_can_issue_approve_or_stop():
    rig = Rig()
    with pytest.raises(PermissionError):
        rig.gateway.issue("agent", replace(rig.grant, run_id="forged"))
    with pytest.raises(PermissionError):
        rig.gateway.stop("agent", "run")
    assert run(rig.gateway.submit("imposter", rig.action())).status == "blocked"
    assert rig.gateway.state("run") == RunState.RUNNING
    assert not rig.effects


@pytest.mark.parametrize(
    "change,reason",
    [
        ({"resource": "document:secret"}, "resource_scope"),
        ({"destination": "attacker.example"}, "destination_scope"),
        ({"approved": True}, "invalid_arguments"),
        ({"resource": {"safe": "document:1"}}, "invalid_arguments"),
    ],
)
def test_argument_scope_and_spoofed_approval(change, reason):
    rig = Rig()
    outcome = run(rig.gateway.submit("agent", rig.action(**change)))
    assert outcome.decision.reason == reason
    assert rig.gateway.state("run") == RunState.STOPPED
    assert not rig.effects


def test_policy_uses_adapter_data_class_not_agent_claim():
    rig = Rig(
        describe=lambda args: ActionScope(
            frozenset({args["resource"]}), data_classes=frozenset({"secret"})
        )
    )
    assert (
        run(rig.gateway.submit("agent", rig.action())).decision.reason == "data_scope"
    )
    assert not rig.effects


def test_missing_expired_and_unknown_tool_grants():
    async def scenario():
        rig = Rig()
        missing = ProposedAction.create("missing", "a", "write", {})
        assert (
            await rig.gateway.submit("agent", missing)
        ).decision.reason == "identity_or_grant"
        rig.now = 2000
        assert (
            await rig.gateway.submit("agent", rig.action())
        ).decision.reason == "grant_expired"
        assert not rig.effects
        other = Rig()
        unknown = ProposedAction.create("run", "a", "shell", {"command": "anything"})
        assert (
            await other.gateway.submit("agent", unknown)
        ).decision.reason == "tool_scope"

    run(scenario())


def test_approval_binds_immutable_stored_arguments_and_is_once_only():
    async def scenario():
        rig = Rig(review=True)
        action = rig.action(body="original")
        outcome = await rig.gateway.submit("agent", action)
        assert outcome.status == "pending" and not rig.effects
        pending = rig.gateway.pending("operator", "run")
        pending.action.arguments["body"] = "changed"  # Fresh copy, no mutation.
        with pytest.raises(PermissionError):
            await rig.gateway.approve("agent", "run", pending.action_hash)
        with pytest.raises(ValueError):
            await rig.gateway.approve("operator", "run", "forged-hash")
        assert (
            await rig.gateway.approve("operator", "run", pending.action_hash)
        ).status == "executed"
        assert rig.effects == [
            {"resource": "document:1", "destination": "internal", "body": "original"}
        ]
        with pytest.raises(ValueError):
            await rig.gateway.approve("operator", "run", pending.action_hash)
        assert (await rig.gateway.submit("agent", action)).decision.reason == "replay"
        assert len(rig.effects) == 1

    run(scenario())


def test_pending_review_cannot_authorize_new_action():
    async def scenario():
        rig = Rig(review=True)
        await rig.gateway.submit("agent", rig.action())
        out = await rig.gateway.submit("agent", rig.action("b", destination="attacker"))
        assert out.decision.reason == "run_paused" and not rig.effects
        pending = rig.gateway.pending("operator", "run")
        rig.now += 301
        out = await rig.gateway.approve("operator", "run", pending.action_hash)
        assert out.decision.reason == "approval_expired" and not rig.effects

    run(scenario())


def test_changed_adapter_scope_invalidates_approval():
    async def scenario():
        mutable = {"review": True}
        rig = Rig(
            describe=lambda args: ActionScope(
                frozenset({"document:1"}), requires_review=mutable["review"]
            )
        )
        await rig.gateway.submit("agent", rig.action())
        pending = rig.gateway.pending("operator", "run")
        mutable["review"] = False
        out = await rig.gateway.approve("operator", "run", pending.action_hash)
        assert out.decision.reason == "approval_binding_changed"
        assert not rig.effects

    run(scenario())


def test_idempotency_key_replay_with_new_action_id():
    async def scenario():
        rig = Rig()
        action = rig.action()
        assert (await rig.gateway.submit("agent", action)).status == "executed"
        retry = replace(action, action_id="different")
        assert (await rig.gateway.submit("agent", retry)).decision.reason == "replay"
        assert len(rig.effects) == 1

    run(scenario())


def test_concurrent_calls_cannot_overspend():
    async def scenario():
        rig = Rig(max_actions=1)
        outcomes = await asyncio.gather(
            rig.gateway.submit("agent", rig.action("a")),
            rig.gateway.submit("agent", rig.action("b")),
        )
        assert any(o.decision.reason == "budget_exhausted" for o in outcomes)
        assert len(rig.effects) <= 1
        assert rig.gateway.state("run") == RunState.STOPPED

    run(scenario())


@pytest.mark.parametrize("budget", ["max_cost_units", "max_tokens", "max_attempts"])
def test_other_budgets_enforced(budget):
    rig = Rig()
    limited = replace(rig.grant, run_id="limited", **{budget: 0})
    rig.gateway.issue("operator", limited)
    action = replace(rig.action(), run_id="limited")
    out = run(rig.gateway.submit("agent", action))
    assert out.decision.reason in {"budget_exhausted", "attempt_limit"}
    assert not rig.effects


def test_revocation_cancels_inflight_and_denies_queued_work():
    async def scenario():
        started, release = asyncio.Event(), asyncio.Event()
        effects = []

        async def execute(args, context):
            started.set()
            await release.wait()
            context.checkpoint()
            effects.append(args)

        rig = Rig(execute=execute)
        pending = asyncio.create_task(rig.gateway.submit("agent", rig.action()))
        await started.wait()
        rig.gateway.stop("operator", "run")
        release.set()
        out = await pending
        assert out.status == "interrupted"
        assert (
            await rig.gateway.submit("agent", rig.action("later"))
        ).decision.reason == "run_stopped"
        assert not effects and rig.revoked == ["run"]
        with pytest.raises(ValueError):
            rig.gateway.issue("operator", rig.grant)

    run(scenario())


def test_delegation_can_only_narrow_and_charges_parent_budget():
    async def scenario():
        rig = Rig(max_actions=1)
        child = replace(
            rig.grant, run_id="child", principal="child-agent", parent_run_id="run"
        )
        with pytest.raises(ValueError):
            rig.gateway.issue(
                "operator", replace(child, resources=frozenset({"other"}))
            )
        rig.gateway.issue("operator", child)
        assert (
            await rig.gateway.submit(
                "child-agent", replace(rig.action(), run_id="child")
            )
        ).status == "executed"
        assert (
            await rig.gateway.submit("agent", rig.action())
        ).decision.reason == "budget_exhausted"
        assert rig.gateway.state("child") == RunState.STOPPED
        assert len(rig.effects) == 1

    run(scenario())


def test_ancestor_stop_and_pending_approval_revoked():
    async def scenario():
        rig = Rig(review=True)
        child = replace(
            rig.grant, run_id="child", principal="child-agent", parent_run_id="run"
        )
        rig.gateway.issue("operator", child)
        await rig.gateway.submit("child-agent", replace(rig.action(), run_id="child"))
        pending = rig.gateway.pending("operator", "child")
        rig.gateway.stop("operator", "run")
        with pytest.raises(ValueError):
            await rig.gateway.approve("operator", "child", pending.action_hash)
        assert not rig.effects

    run(scenario())


def test_policy_unavailable_and_audit_unavailable_fail_closed():
    class FailedPolicy(Policy):
        def evaluate(self, *args):
            raise RuntimeError("private service URL")

    rig = Rig(policy=FailedPolicy())
    out = run(rig.gateway.submit("agent", rig.action()))
    assert out.decision.reason == "policy_unavailable" and not rig.effects
    rig = Rig()
    rig.audit.append = lambda **kwargs: (_ for _ in ()).throw(OSError("full"))
    out = run(rig.gateway.submit("agent", rig.action()))
    assert out.decision.verdict == Verdict.STOP and not rig.effects


def test_outcome_audit_failure_is_uncertain_and_stops_run():
    async def scenario():
        rig = Rig()
        append = rig.audit.append

        def broken(**kwargs):
            if kwargs["event"] == "outcome":
                raise OSError("full")
            append(**kwargs)

        rig.audit.append = broken
        out = await rig.gateway.submit("agent", rig.action())
        assert len(rig.effects) == 1
        assert out.status == "uncertain" and out.decision.reason == "audit_unavailable"
        assert rig.gateway.state("run") == RunState.STOPPED

    run(scenario())


def test_tool_exception_does_not_leak_and_cannot_retry():
    async def bad(args, context):
        raise RuntimeError("SECRET_TOKEN")

    rig = Rig(execute=bad)
    out = run(rig.gateway.submit("agent", rig.action()))
    assert out.status == "uncertain"
    assert "SECRET_TOKEN" not in str(out) + str(rig.audit.records)
    assert rig.gateway.state("run") == RunState.STOPPED


def test_audit_chain_redacts_arguments_and_detects_edits(tmp_path):
    audit = AuditLog(tmp_path / "audit.jsonl")
    rig = Rig(audit=audit)
    assert (
        run(rig.gateway.submit("agent", rig.action(body="private-body"))).status
        == "executed"
    )
    assert "private-body" not in (tmp_path / "audit.jsonl").read_text()
    assert verify_records(AuditLog(tmp_path / "audit.jsonl").records)
    changed = audit.records
    changed[0]["event"] = "altered"
    assert not verify_records(changed)


def test_shadow_mode_and_observation_never_dispatch_or_add_authority():
    async def scenario():
        rig = Rig()
        rig.gateway.observe("operator", "run", "injection_indicator")
        assert rig.gateway.state("run") == RunState.OBSERVE
        assert rig.gateway.preview("agent", rig.action()).verdict == Verdict.ALLOW
        assert not rig.effects
        assert (await rig.gateway.submit("agent", rig.action())).status == "executed"

    run(scenario())


@pytest.mark.parametrize("arguments", [{"n": float("nan")}, ["not", "object"]])
def test_non_json_or_nonfinite_arguments_rejected(arguments):
    with pytest.raises(ValueError):
        ProposedAction.create("run", "a", "write", arguments)


def test_scope_rechecked_after_queueing_before_executor_entry():
    async def scenario():
        mutable = {"target": "document:1"}
        rig = Rig(describe=lambda args: ActionScope(frozenset({mutable["target"]})))
        task = asyncio.create_task(rig.gateway.submit("agent", rig.action()))
        await asyncio.sleep(0)  # submit reserves; executor has not entered yet.
        mutable["target"] = "document:outside"
        result = await task
        assert (
            result.status == "blocked"
            and result.decision.reason == "authorization_changed"
        )
        assert not rig.effects

    run(scenario())


def test_concurrent_review_holds_new_work_without_stopping_dispatched_action():
    async def scenario():
        def describe(args):
            return ActionScope(
                frozenset({args["resource"]}),
                frozenset({args["destination"]}),
                requires_review=args["body"] == "needs review",
            )

        rig = Rig(describe=describe)
        allowed, reviewed = await asyncio.gather(
            rig.gateway.submit("agent", rig.action("allowed")),
            rig.gateway.submit("agent", rig.action("reviewed", body="needs review")),
        )
        assert allowed.status == "executed" and reviewed.status == "pending"
        assert rig.gateway.state("run") == RunState.PAUSED and not rig.revoked
        held = await rig.gateway.submit("agent", rig.action("held"))
        assert held.status == "blocked" and held.decision.reason == "run_paused"
        assert rig.gateway.state("run") == RunState.PAUSED
        pending = rig.gateway.pending("operator", "run")
        out = await rig.gateway.approve("operator", "run", pending.action_hash)
        assert out.status == "executed"
        assert [effect["body"] for effect in rig.effects] == ["hello", "needs review"]

    run(scenario())


def test_paused_attempts_cannot_bypass_attempt_budget():
    async def scenario():
        rig = Rig(review=True)
        grant = replace(rig.grant, run_id="limited", max_attempts=1)
        rig.gateway.issue("operator", grant)
        action = replace(rig.action(), run_id="limited")
        assert (await rig.gateway.submit("agent", action)).status == "pending"
        result = await rig.gateway.submit("agent", replace(action, action_id="again"))
        assert result.decision.reason == "attempt_limit"
        assert not rig.effects

    run(scenario())


def test_revocation_callback_failure_stops_other_inflight_runs():
    async def scenario():
        started, release = asyncio.Event(), asyncio.Event()
        effects = []

        async def execute(args, context):
            started.set()
            await release.wait()
            context.checkpoint()
            effects.append(args)

        def revoke(run_id):
            raise OSError("credential service down")

        rig = Rig(execute=execute, on_stop=revoke)
        rig.gateway.issue("operator", replace(rig.grant, run_id="other"))
        task = asyncio.create_task(
            rig.gateway.submit("agent", replace(rig.action(), run_id="other"))
        )
        await started.wait()
        with pytest.raises(RuntimeError, match="revocation failed"):
            rig.gateway.stop("operator", "run")
        release.set()
        assert (await task).status == "interrupted"
        assert rig.gateway.state("other") == RunState.STOPPED
        assert not effects

    run(scenario())


def test_tool_deadline_cancels_unfinished_async_work():
    import time

    async def scenario():
        effects = []

        async def execute(args, context):
            await asyncio.Event().wait()
            effects.append("must not happen")

        tool = ToolSpec("wait", "1", {}, lambda args: ActionScope(frozenset()), execute)
        gateway = Gateway([tool], operators={"operator"})
        gateway.issue(
            "operator",
            TaskGrant(
                "short", "agent", time.time() + 0.1, frozenset({"wait"}), frozenset()
            ),
        )
        result = await asyncio.wait_for(
            gateway.submit("agent", ProposedAction.create("short", "a", "wait", {})), 2
        )
        assert (
            result.status == "interrupted"
            and result.decision.reason == "execution_expired"
        )
        assert not effects and gateway.state("short") == RunState.STOPPED

    run(scenario())
