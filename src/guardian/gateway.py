"""Reference host gateway: complete mediation is a deployment responsibility.

Only submit/preview should be reachable from an untrusted agent. The authenticated
caller is injected by the host transport. Never run arbitrary agent Python in this
process or expose issue/approve/stop to it. One gateway owns one tool registry and
one process; distributed coordination and crash recovery require a host store.
"""

import asyncio
import math
import time
from dataclasses import replace
from threading import RLock
from types import MappingProxyType

from .audit import AuditLog
from .contracts import (
    ActionOutcome,
    Decision,
    ProposedAction,
    RunState,
    TaskGrant,
    Verdict,
)
from .policy import Policy, ToolSpec
from .recovery import ExecutionContext
from .review import PendingReview
from .state import RunRecord


class AuthorizationChanged(Exception):
    """The action became invalid while queued, before executor entry."""


class Gateway:
    def __init__(
        self,
        tools,
        *,
        operators,
        audit=None,
        policy=None,
        clock=time.time,
        review_ttl=300.0,
        on_stop=None,
        max_runs=1000,
    ):
        specs = list(tools)
        if any(not isinstance(spec, ToolSpec) for spec in specs):
            raise TypeError("Expected trusted ToolSpec instances")
        if len({tool.name for tool in specs}) != len(specs):
            raise ValueError("Duplicate tool name")
        if not operators or not math.isfinite(review_ttl) or review_ttl <= 0:
            raise ValueError("Operators and a finite positive review TTL are required")
        if type(max_runs) is not int or max_runs < 1:
            raise ValueError("Invalid run limit")
        self._tools = MappingProxyType({tool.name: tool for tool in specs})
        self._operators = frozenset(operators)
        self._audit = audit if audit is not None else AuditLog()
        self._policy = policy if policy is not None else Policy()
        self._clock = clock
        self._review_ttl = review_ttl
        self._on_stop = on_stop
        self._max_runs = max_runs
        self._lock = RLock()
        self._runs = {}
        self._failed = False
        self._loop = None

    def _operator(self, operator):
        if operator not in self._operators:
            raise PermissionError("An authenticated operator is required")

    def _bind_loop(self):
        loop = asyncio.get_running_loop()
        if self._loop is None:
            self._loop = loop
        elif self._loop is not loop:
            raise RuntimeError("A gateway belongs to one event loop")
        return loop

    def _event(self, run_id, action_hash, event, reason):
        try:
            self._audit.append(
                timestamp=self._clock(),
                run_id=run_id,
                action_hash=action_hash,
                event=event,
                reason=reason,
            )
        except Exception:
            self._failed = True
            for failed_run_id in tuple(self._runs):
                self._stop_locked(failed_run_id)
            raise

    def _lineage(self, record):
        chain = [record]
        while chain[-1].grant.parent_run_id is not None:
            chain.append(self._runs[chain[-1].grant.parent_run_id])
        return chain

    def issue(self, operator: str, grant: TaskGrant):
        """Host-only. Transport authentication must precede this call."""
        self._operator(operator)
        if type(grant) is not TaskGrant:
            raise TypeError("Expected TaskGrant")
        with self._lock:
            if (
                self._failed
                or grant.run_id in self._runs
                or len(self._runs) >= self._max_runs
            ):
                raise ValueError(
                    "Gateway unavailable or run ID already used/capacity reached"
                )
            if grant.principal in self._operators or self._clock() >= grant.expires_at:
                raise ValueError(
                    "Agent must be distinct from operators and grant unexpired"
                )
            if (
                grant.policy_version != self._policy.version
                or not grant.tools <= self._tools.keys()
            ):
                raise ValueError("Unknown tools or policy")
            if grant.parent_run_id is not None:
                parent = self._runs.get(grant.parent_run_id)
                if parent is None or parent.state not in (
                    RunState.RUNNING,
                    RunState.OBSERVE,
                ):
                    raise ValueError("Parent is not active")
                if any(
                    getattr(grant, key) > getattr(parent.grant, key)
                    for key in (
                        "expires_at",
                        "max_actions",
                        "max_attempts",
                        "max_cost_units",
                        "max_tokens",
                    )
                ):
                    raise ValueError("Delegation cannot broaden lifetime or budgets")
                if any(
                    not getattr(grant, key) <= getattr(parent.grant, key)
                    for key in ("tools", "resources", "destinations", "data_classes")
                ):
                    raise ValueError("Delegation cannot broaden scope")
            self._event(grant.run_id, "", "grant_issued", "host_authorized")
            self._runs[grant.run_id] = RunRecord(grant)

    def state(self, run_id):
        with self._lock:
            return self._runs[run_id].state

    def _stop_locked(self, run_id):
        # Revoke descendants too; do this before audit/callback errors can occur.
        affected = [
            record
            for record in self._runs.values()
            if any(item.grant.run_id == run_id for item in self._lineage(record))
        ]
        for record in affected:
            if record.state == RunState.STOPPED:
                continue
            record.state = RunState.STOPPED
            record.pending = None
            for context in record.contexts:
                context.cancel()
            for task in record.tasks:
                task.get_loop().call_soon_threadsafe(task.cancel)
            if self._on_stop is not None:
                try:
                    self._on_stop(record.grant.run_id)
                except Exception:
                    # The logical stop remains effective; host must reconcile
                    # failed credential revocation via its incident runbook.
                    self._failed = True
                    # Freeze every local run when host credential revocation is
                    # unreliable. Do not keep unrelated in-flight work running.
                    for other in self._runs.values():
                        other.state = RunState.STOPPED
                        other.pending = None
                        for context in other.contexts:
                            context.cancel()
                        for task in other.tasks:
                            task.get_loop().call_soon_threadsafe(task.cancel)

    def stop(self, operator, run_id):
        """Host kill switch. A stopped run cannot be resumed/reissued."""
        self._operator(operator)
        with self._lock:
            if run_id not in self._runs:
                raise KeyError(run_id)
            self._stop_locked(run_id)
            reason = "recovery_failed" if self._failed else "operator_stop"
            self._event(run_id, "", "run_stopped", reason)
            if self._failed:
                raise RuntimeError("Host revocation failed; gateway stopped")

    def observe(self, operator, run_id, signal):
        """Record calibrated host signals; never use them to add authority."""
        self._operator(operator)
        if signal not in {"goal_drift", "injection_indicator", "loop_indicator"}:
            raise ValueError("Unknown signal")
        with self._lock:
            record = self._runs[run_id]
            if record.state == RunState.RUNNING:
                record.state = RunState.OBSERVE
            self._event(run_id, "", "risk_signal", signal)

    def pending(self, operator, run_id):
        self._operator(operator)
        with self._lock:
            return self._runs[run_id].pending

    def _decision(self, action, verdict, reason):
        return Decision(verdict, reason, action.fingerprint, self._policy.version)

    def _assess(self, caller, action, *, approved=False, dispatched=False):
        # PAUSED holds new dispatch only. An action that was authorized and
        # reserved before a concurrent review paused the run may still start;
        # stop, expiry and scope/binding changes are rechecked regardless.
        record = self._runs.get(action.run_id)
        if self._failed:
            return (
                None,
                None,
                None,
                self._decision(action, Verdict.STOP, "gateway_unavailable"),
            )
        if record is None or record.grant.principal != caller:
            return (
                None,
                None,
                None,
                self._decision(action, Verdict.DENY, "identity_or_grant"),
            )
        for item in self._lineage(record):
            if item.state == RunState.STOPPED:
                return (
                    record,
                    None,
                    None,
                    self._decision(action, Verdict.STOP, "run_stopped"),
                )
            if (
                item.state == RunState.PAUSED
                and not dispatched
                and not (approved and item is record)
            ):
                return (
                    record,
                    None,
                    None,
                    self._decision(action, Verdict.REVIEW, "run_paused"),
                )
            if self._clock() >= item.grant.expires_at:
                return (
                    record,
                    None,
                    None,
                    self._decision(action, Verdict.STOP, "grant_expired"),
                )
        tool = self._tools.get(action.tool)
        if tool is None or action.tool not in record.grant.tools:
            return (
                record,
                None,
                None,
                self._decision(action, Verdict.DENY, "tool_scope"),
            )
        try:
            scope = tool.resolve(action)
        except Exception:
            return (
                record,
                None,
                None,
                self._decision(action, Verdict.DENY, "invalid_arguments"),
            )
        try:
            decision = self._policy.evaluate(
                record.grant, action, scope, tool, self._clock()
            )
            if type(decision) is not Decision or decision.verdict not in Verdict:
                raise ValueError("Invalid policy decision")
        except Exception:
            return (
                record,
                None,
                None,
                self._decision(action, Verdict.STOP, "policy_unavailable"),
            )
        return record, tool, scope, decision

    def preview(self, caller, action):
        """Shadow evaluation only: never dispatch, issue approval or reserve budget."""
        if type(action) is not ProposedAction:
            raise TypeError("Expected ProposedAction")
        with self._lock:
            record, _, scope, decision = self._assess(caller, action)
            if record and scope and decision.verdict in (Verdict.ALLOW, Verdict.REVIEW):
                if self._over_budget(record, scope):
                    decision = replace(
                        decision, verdict=Verdict.STOP, reason="budget_exhausted"
                    )
            self._event(
                action.run_id, decision.action_hash, "shadow_decision", decision.reason
            )
            return decision

    def _over_budget(self, record, scope):
        return any(
            item.actions + 1 > item.grant.max_actions
            or item.cost_units + scope.cost_units > item.grant.max_cost_units
            or item.tokens + scope.tokens > item.grant.max_tokens
            for item in self._lineage(record)
        )

    def _reject(self, action, decision, record=None):
        if record and decision.verdict in (Verdict.DENY, Verdict.STOP):
            self._stop_locked(action.run_id)
        self._event(action.run_id, decision.action_hash, "decision", decision.reason)
        return ActionOutcome(decision, "blocked")

    async def submit(self, caller, action):
        if type(action) is not ProposedAction:
            raise TypeError("Expected ProposedAction")
        with self._lock:
            self._bind_loop()
            candidate = self._runs.get(action.run_id)
            if (
                candidate
                and candidate.grant.principal == caller
                and candidate.state != RunState.STOPPED
            ):
                for item in self._lineage(candidate):
                    item.attempts += 1
            record, tool, scope, decision = self._assess(caller, action)
            try:
                if record is not None and any(
                    item.attempts > item.grant.max_attempts
                    for item in self._lineage(record)
                ):
                    return self._reject(
                        action,
                        replace(decision, verdict=Verdict.STOP, reason="attempt_limit"),
                        record,
                    )
                if (
                    record is None
                    or scope is None
                    or decision.verdict in (Verdict.DENY, Verdict.STOP)
                ):
                    return self._reject(action, decision, record)
                if (
                    action.action_id in record.seen_ids
                    or action.idempotency_key in record.seen_keys
                ):
                    return self._reject(
                        action,
                        replace(decision, verdict=Verdict.DENY, reason="replay"),
                        record,
                    )
                if self._over_budget(record, scope):
                    return self._reject(
                        action,
                        replace(
                            decision, verdict=Verdict.STOP, reason="budget_exhausted"
                        ),
                        record,
                    )
                record.seen_ids.add(action.action_id)
                record.seen_keys.add(action.idempotency_key)
                if decision.verdict == Verdict.REVIEW:
                    record.pending = PendingReview(
                        action,
                        decision.action_hash,
                        min(record.grant.expires_at, self._clock() + self._review_ttl),
                        scope,
                        tool.version,
                    )
                    record.state = RunState.PAUSED
                    self._event(
                        action.run_id,
                        decision.action_hash,
                        "review_requested",
                        decision.reason,
                    )
                    return ActionOutcome(decision, "pending")
                task, context = self._dispatch(record, action, tool, scope, decision)
            except Exception:
                self._stop_locked(action.run_id)
                return ActionOutcome(
                    self._decision(action, Verdict.STOP, "gateway_unavailable"),
                    "blocked",
                )
        return await self._finish(record, action, task, context, decision)

    async def approve(self, operator, run_id, action_hash):
        """Host-only: approve and execute the stored action once, after revalidation."""
        self._operator(operator)
        with self._lock:
            self._bind_loop()
            record = self._runs[run_id]
            pending = record.pending
            if record.state != RunState.PAUSED or pending is None:
                raise ValueError("No pending review")
            if action_hash != pending.action_hash:
                raise ValueError("Approval does not match pending action")
            action = pending.action
            if self._clock() >= pending.expires_at:
                return self._reject(
                    action,
                    self._decision(action, Verdict.STOP, "approval_expired"),
                    record,
                )
            _, tool, scope, decision = self._assess(
                record.grant.principal, action, approved=True
            )
            if scope is None or decision.verdict not in (Verdict.ALLOW, Verdict.REVIEW):
                return self._reject(action, decision, record)
            if decision.action_hash != pending.action_hash:
                return self._reject(
                    action,
                    replace(
                        decision,
                        verdict=Verdict.DENY,
                        reason="approval_binding_changed",
                    ),
                    record,
                )
            if self._over_budget(record, scope):
                return self._reject(
                    action,
                    replace(decision, verdict=Verdict.STOP, reason="budget_exhausted"),
                    record,
                )
            decision = replace(
                decision, verdict=Verdict.ALLOW, reason="operator_approved"
            )
            try:
                self._event(
                    run_id, decision.action_hash, "approved", "operator_approved"
                )
                record.pending = None
                record.state = RunState.RUNNING
                task, context = self._dispatch(record, action, tool, scope, decision)
            except Exception:
                self._stop_locked(run_id)
                return ActionOutcome(
                    self._decision(action, Verdict.STOP, "gateway_unavailable"),
                    "blocked",
                )
        return await self._finish(record, action, task, context, decision)

    def _dispatch(self, record, action, tool, scope, decision):
        # Called under the same lock as evaluation, so reservation is atomic
        # across concurrent requests and every ancestor's shared budget.
        if self._over_budget(record, scope):
            self._stop_locked(action.run_id)
            self._event(
                action.run_id, decision.action_hash, "decision", "budget_exhausted"
            )
            raise ValueError("Budget exhausted")
        self._event(action.run_id, decision.action_hash, "dispatch", decision.reason)
        for item in self._lineage(record):
            item.actions += 1
            item.cost_units += scope.cost_units
            item.tokens += scope.tokens
        deadline = min(item.grant.expires_at for item in self._lineage(record))
        context = ExecutionContext(deadline, self._clock)

        async def execute():
            context.checkpoint()
            with self._lock:
                _, current_tool, current_scope, current = self._assess(
                    record.grant.principal, action, dispatched=True
                )
                permitted = current.verdict == Verdict.ALLOW or (
                    current.verdict == Verdict.REVIEW
                    and decision.reason == "operator_approved"
                )
                if (
                    current_tool is not tool
                    or current_scope is None
                    or not permitted
                    or current.action_hash != decision.action_hash
                ):
                    raise AuthorizationChanged
                context.started = True
            result = await asyncio.wait_for(
                tool.execute(action.arguments, context),
                timeout=max(0.0, deadline - self._clock()),
            )
            context.checkpoint()
            return result

        task = self._loop.create_task(execute())
        record.tasks.add(task)
        record.contexts.add(context)
        return task, context

    async def _finish(self, record, action, task, context, decision):
        result = None
        status, reason = "executed", "tool_completed"
        try:
            result = await task
        except AuthorizationChanged:
            status, reason = "blocked", "authorization_changed"
        except asyncio.CancelledError:
            status, reason = "interrupted", "execution_cancelled"
        except asyncio.TimeoutError:
            status, reason = "interrupted", "execution_expired"
        except Exception:
            # An external tool may have partially completed before failing.
            status, reason = "uncertain", "tool_failed"
        with self._lock:
            record.tasks.discard(task)
            record.contexts.discard(context)
            if status != "executed":
                self._stop_locked(action.run_id)
            try:
                self._event(action.run_id, decision.action_hash, "outcome", reason)
            except Exception:
                return ActionOutcome(
                    replace(decision, verdict=Verdict.STOP, reason="audit_unavailable"),
                    "uncertain",
                )
        verdict = decision.verdict if status == "executed" else Verdict.STOP
        return ActionOutcome(
            replace(decision, verdict=verdict, reason=reason), status, result
        )
